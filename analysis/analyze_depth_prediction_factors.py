import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_csv_data(path):
    df = pd.read_csv(path)
    mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
    if mat_cols:
        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
        return df[mat_cols].values.astype(np.float32)
    return df.iloc[:, -96:].values.astype(np.float32)


def normalize_frame(frame):
    dmin = float(frame.min())
    dmax = float(frame.max())
    if dmax - dmin > 1e-6:
        return (frame - dmin) / (dmax - dmin)
    return frame - dmin


def extract_window_features(data_values, end_row, seq_len=10):
    st = max(0, end_row - seq_len + 1)
    ed = end_row + 1
    seq = data_values[st:ed]
    if seq.shape[0] < seq_len:
        pad = np.repeat(seq[:1], seq_len - seq.shape[0], axis=0)
        seq = np.concatenate([pad, seq], axis=0)
    means = np.mean(seq, axis=1)
    maxs = np.max(seq, axis=1)
    stds = np.std(seq, axis=1)
    rep = normalize_frame(seq[-2].reshape(12, 8))
    rep_max = float(rep.max())
    thr = rep_max * 0.7
    hotspot_area_ratio = float(np.mean(rep >= thr))
    gx = np.abs(np.diff(rep, axis=1))
    gy = np.abs(np.diff(rep, axis=0))
    grad_energy = float((gx.mean() + gy.mean()) / 2.0)
    center = rep[4:8, 2:6]
    center_mean = float(np.mean(center))
    border = np.concatenate([rep[:2, :].reshape(-1), rep[-2:, :].reshape(-1), rep[:, :2].reshape(-1), rep[:, -2:].reshape(-1)])
    border_mean = float(np.mean(border))
    center_border_contrast = float(center_mean - border_mean)
    temporal_max_slope = float(maxs[-1] - maxs[0])
    temporal_mean_slope = float(means[-1] - means[0])
    temporal_std_slope = float(stds[-1] - stds[0])
    return {
        'feat_mean_mean': float(np.mean(means)),
        'feat_mean_max': float(np.mean(maxs)),
        'feat_mean_std': float(np.mean(stds)),
        'feat_peak_max': float(np.max(maxs)),
        'feat_hotspot_area_ratio': hotspot_area_ratio,
        'feat_gradient_energy': grad_energy,
        'feat_center_border_contrast': center_border_contrast,
        'feat_temporal_max_slope': temporal_max_slope,
        'feat_temporal_mean_slope': temporal_mean_slope,
        'feat_temporal_std_slope': temporal_std_slope,
    }


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den < 1e-12:
        return 0.0
    return float(np.sum(x * y) / den)


def standardize(X):
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu.squeeze(), sd.squeeze()


def linear_coef(X, y):
    Xs, _, _ = standardize(X)
    ys = (y - np.mean(y)) / (np.std(y) + 1e-8)
    X1 = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
    beta, *_ = np.linalg.lstsq(X1, ys, rcond=None)
    return beta[1:]


def concordance_ccc(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    den = vx + vy + (mx - my) ** 2
    if den < 1e-12:
        return 0.0
    return float((2.0 * cov) / den)


def load_records(summary_path):
    s = load_json(summary_path)
    data_root = s['data_root']
    per_file = None
    for r in s['ablation_results']:
        if r['condition'] == 'dual_stream':
            per_file = r['per_file_records']
            break
    rows = []
    cache = {}
    for rec in per_file:
        rel = rec['file_path']
        full = os.path.join(data_root, rel.replace('\\', os.sep))
        if full not in cache:
            cache[full] = read_csv_data(full)
        feats = extract_window_features(cache[full], int(rec['best_end_row']), seq_len=10)
        row = {
            'file_path': rel,
            'true_depth': float(rec['true_depth']),
            'pred_depth': float(rec['pred_depth']),
            'err_depth': float(rec['err_depth']),
            'true_size': float(rec['true_size']),
            'pred_size': float(rec['pred_size']),
            'err_size': float(rec['err_size']),
            'best_prob': float(rec['best_prob']),
        }
        row.update(feats)
        rows.append(row)
    return rows


def make_plots(rows, out_dir):
    feat_names = [k for k in rows[0].keys() if k.startswith('feat_')]
    y_true = np.array([r['true_depth'] for r in rows], dtype=float)
    y_pred = np.array([r['pred_depth'] for r in rows], dtype=float)
    X = np.array([[r[f] for f in feat_names] for r in rows], dtype=float)

    corr_true = [pearson(X[:, i], y_true) for i in range(len(feat_names))]
    corr_pred = [pearson(X[:, i], y_pred) for i in range(len(feat_names))]
    coef = linear_coef(X, y_pred)

    x = np.arange(len(feat_names))
    plt.figure(figsize=(12, 4.8))
    plt.bar(x - 0.2, corr_true, width=0.4, label='Corr with True Depth')
    plt.bar(x + 0.2, corr_pred, width=0.4, label='Corr with Pred Depth')
    plt.axhline(0.0, color='black', linewidth=1)
    plt.xticks(x, [n.replace('feat_', '') for n in feat_names], rotation=35, ha='right')
    plt.title('Depth-Relevant Feature Correlations')
    plt.grid(axis='y', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_depth_feature_correlation.png'), dpi=240, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(11.5, 4.6))
    order = np.argsort(np.abs(coef))[::-1]
    names_sorted = [feat_names[i].replace('feat_', '') for i in order]
    coef_sorted = coef[order]
    plt.bar(np.arange(len(names_sorted)), coef_sorted)
    plt.axhline(0.0, color='black', linewidth=1)
    plt.xticks(np.arange(len(names_sorted)), names_sorted, rotation=35, ha='right')
    plt.title('Linear Sensitivity to Predicted Depth (Standardized Coefficients)')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_depth_linear_sensitivity.png'), dpi=240, bbox_inches='tight')
    plt.close()

    diff = y_pred - y_true
    mean_depth = (y_pred + y_true) / 2.0
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    loa_up = md + 1.96 * sd
    loa_dn = md - 1.96 * sd
    bins = [0.0, 0.75, 1.25, 1.75, 2.25, 4.0]
    idx = np.digitize(y_true, bins, right=True)
    bin_labels = ['<=0.75', '0.75-1.25', '1.25-1.75', '1.75-2.25', '>=2.25']
    bin_mae = []
    for i in range(1, len(bins)):
        m = idx == i
        if np.sum(m) == 0:
            bin_mae.append(np.nan)
        else:
            bin_mae.append(float(np.mean(np.abs(diff[m]))))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].scatter(mean_depth, diff, alpha=0.75)
    axes[0].axhline(md, color='blue', linewidth=1.3, label=f'mean={md:.3f}')
    axes[0].axhline(loa_up, color='red', linestyle='--', linewidth=1.1, label=f'+1.96SD={loa_up:.3f}')
    axes[0].axhline(loa_dn, color='red', linestyle='--', linewidth=1.1, label=f'-1.96SD={loa_dn:.3f}')
    axes[0].set_xlabel('Mean(True, Pred) Depth')
    axes[0].set_ylabel('Pred - True Depth')
    axes[0].set_title('Bland-Altman Consistency')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    x2 = np.arange(len(bin_labels))
    axes[1].bar(x2, bin_mae)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(bin_labels, rotation=20)
    axes[1].set_title('Depth-bin MAE Consistency')
    axes[1].set_ylabel('MAE')
    axes[1].grid(axis='y', alpha=0.25)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_depth_consistency.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)

    consistency = {
        'mean_diff': md,
        'loa_upper': float(loa_up),
        'loa_lower': float(loa_dn),
        'ccc': concordance_ccc(y_true, y_pred),
        'depth_bin_labels': bin_labels,
        'depth_bin_mae': [None if np.isnan(v) else float(v) for v in bin_mae]
    }
    return feat_names, corr_true, corr_pred, coef, consistency


def write_summary(rows, feat_names, corr_true, corr_pred, coef, consistency, out_dir):
    top_corr_true = sorted(zip(feat_names, corr_true), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_corr_pred = sorted(zip(feat_names, corr_pred), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_coef = sorted(zip(feat_names, coef), key=lambda x: abs(x[1]), reverse=True)[:5]
    err_depth = np.array([r['err_depth'] for r in rows], dtype=float)
    summary = {
        'sample_count': int(len(rows)),
        'depth_mae': float(np.mean(np.abs(err_depth))),
        'depth_bias': float(np.mean(np.array([r['pred_depth'] - r['true_depth'] for r in rows], dtype=float))),
        'top_corr_true_depth': [{'feature': n, 'corr': float(v)} for n, v in top_corr_true],
        'top_corr_pred_depth': [{'feature': n, 'corr': float(v)} for n, v in top_corr_pred],
        'top_linear_sensitivity': [{'feature': n, 'coef': float(v)} for n, v in top_coef],
        'consistency': consistency,
    }
    with open(os.path.join(out_dir, 'depth_factor_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    guide = f"""# 深度预测影响因素（数据驱动）

样本数：{summary['sample_count']}
深度MAE：{summary['depth_mae']:.4f}
深度Bias：{summary['depth_bias']:.4f}

## 与真实深度相关性最高的特征
{os.linesep.join([f"- {d['feature']}: corr={d['corr']:.4f}" for d in summary['top_corr_true_depth']])}

## 与模型预测深度相关性最高的特征
{os.linesep.join([f"- {d['feature']}: corr={d['corr']:.4f}" for d in summary['top_corr_pred_depth']])}

## 线性敏感度最高的特征
{os.linesep.join([f"- {d['feature']}: coef={d['coef']:.4f}" for d in summary['top_linear_sensitivity']])}

## 一致性分析
- CCC（Concordance Correlation Coefficient）: {summary['consistency']['ccc']:.4f}
- Bland-Altman mean diff: {summary['consistency']['mean_diff']:.4f}
- Bland-Altman LoA: [{summary['consistency']['loa_lower']:.4f}, {summary['consistency']['loa_upper']:.4f}]

## 如何解释
- mean_max 与 peak_max、temporal_max_slope 反映接触强度与时序增长速度。
- hotspot_area_ratio 与 gradient_energy、center_border_contrast 反映空间扩散与边缘陡峭度。
- 若深度相关的高权重特征集中在“空间扩散+时序斜率”，说明深度主要由力场形态变化而非单点峰值决定。
"""
    with open(os.path.join(out_dir, '深度影响因素解读.md'), 'w', encoding='utf-8') as f:
        f.write(guide)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'paper_results', 'depth_factor_analysis')
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(base_dir, 'paper_results', '2_csv_causal_orig_seed2026', 'causal_explainability_summary.json'),
        os.path.join(base_dir, 'paper_results', '3_csv_causal_orig_seed2026', 'causal_explainability_summary.json'),
    ]
    rows = []
    for p in paths:
        rows.extend(load_records(p))
    feat_names, corr_true, corr_pred, coef, consistency = make_plots(rows, out_dir)
    write_summary(rows, feat_names, corr_true, corr_pred, coef, consistency, out_dir)
    print(f"Saved depth factor analysis to: {out_dir}")


if __name__ == '__main__':
    main()
