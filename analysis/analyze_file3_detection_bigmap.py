import json
import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dualstream_3dcnn_lstm import DualStream3DCNNLSTM


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)
    data_root = os.environ.get('TACTILE_DATA_ROOT', '').strip()
    if not data_root:
        data_root = os.path.join(repo_root, '整理好的数据集', '建表数据')
    labels_path = os.environ.get('FILE3_LABELS_JSON', '').strip()
    if not labels_path:
        labels_path = os.path.join(data_root, 'manual_keyframe_labels_file3.json')
    model_path = os.environ.get('MODEL_PATH', '').strip()
    if not model_path:
        model_path = os.path.join(base_dir, 'paper_results', 'fair_ablation_seed2026', 'models', 'dual_stream_fair_retrain.pth')
    out_dir = os.path.join(base_dir, 'paper_results', 'file3_detection_bigmap')
    os.makedirs(out_dir, exist_ok=True)
    return base_dir, data_root, labels_path, model_path, out_dir


def read_csv_data(path):
    df = pd.read_csv(path)
    mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
    if mat_cols:
        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
        return df[mat_cols].values.astype(np.float32)
    return df.iloc[:, -96:].values.astype(np.float32)


def normalize_sequence(seq_raw):
    seq_len = len(seq_raw)
    seq = np.zeros((seq_len, 1, 12, 8), dtype=np.float32)
    for i in range(seq_len):
        fr = seq_raw[i]
        dmin, dmax = fr.min(), fr.max()
        if dmax - dmin > 1e-6:
            fr = (fr - dmin) / (dmax - dmin)
        else:
            fr = fr - dmin
        seq[i, 0] = fr.reshape(12, 8)
    return seq


def stats_sequence(seq_raw):
    seq_len = len(seq_raw)
    s = np.zeros((seq_len, 3), dtype=np.float32)
    for i in range(seq_len):
        fr = seq_raw[i]
        s[i, 0] = float(np.mean(fr))
        s[i, 1] = float(np.max(fr))
        s[i, 2] = float(np.std(fr))
    return s


def parse_size_depth_from_path(rel_path):
    sm = re.search(r'(\d+(?:\.\d+)?)cm大', rel_path)
    dm = re.search(r'(\d+(?:\.\d+)?)cm深', rel_path)
    size = float(sm.group(1)) if sm else 0.0
    depth = float(dm.group(1)) if dm else 0.0
    return size, depth


def sanitize_segments(segments, n):
    out = []
    for seg in segments or []:
        if not isinstance(seg, (list, tuple)) or len(seg) == 0:
            continue
        if len(seg) == 1:
            st = int(seg[0])
            ed = st + 1
        else:
            st = int(seg[0])
            ed = int(seg[1])
            if ed <= st:
                ed = st + 1
        st = max(0, min(st, n))
        ed = max(0, min(ed, n))
        if ed > st:
            out.append((st, ed))
    return out


def overlap_positive(ws, we, segments):
    for st, ed in segments:
        if ws < ed and we >= st:
            return True
    return False


def compute_metrics(y_true, y_score, thr):
    yt = np.asarray(y_true, dtype=np.int32)
    ys = np.asarray(y_score, dtype=np.float64)
    yp = (ys >= float(thr)).astype(np.int32)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def select_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.0, 1.0, 1001):
        m = compute_metrics(y_true, y_prob, t)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_thr = float(t)
    return best_thr


def infer_all_windows(model, seq_np, stats_np, device, bs=512):
    probs, sizes, depths = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, seq_np.shape[0], bs):
            x = torch.tensor(seq_np[i:i + bs], dtype=torch.float32, device=device)
            s = torch.tensor(stats_np[i:i + bs], dtype=torch.float32, device=device)
            p, sz, dp = model(x, s)
            probs.append(torch.sigmoid(p).squeeze(1).cpu().numpy())
            sizes.append(sz.squeeze(1).cpu().numpy())
            depths.append(dp.squeeze(1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(sizes), np.concatenate(depths)


def build_bigmap_and_summary(records, threshold, save_png, save_json):
    max_len = max(len(r['probs']) for r in records)
    n = len(records)
    prob_map = np.full((n, max_len), np.nan, dtype=np.float32)
    gt_map = np.zeros((n, max_len), dtype=np.float32)
    hit_vec = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(records):
        L = len(r['probs'])
        prob_map[i, :L] = r['probs']
        gt_map[i, :L] = r['gt'].astype(np.float32)
        hit_vec[i] = 1.0 if (np.max(r['probs']) >= threshold and np.max(r['gt']) > 0) else 0.0

    order = sorted(range(n), key=lambda i: (records[i]['true_depth'], records[i]['true_size'], records[i]['best_prob']), reverse=False)
    prob_map = prob_map[order]
    gt_map = gt_map[order]
    hit_vec = hit_vec[order]
    ordered = [records[i] for i in order]

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.8), gridspec_kw={'height_ratios': [4.6, 2.0, 0.7]})
    ax1, ax2, ax3 = axes
    im1 = ax1.imshow(prob_map, aspect='auto', cmap='turbo', vmin=0.0, vmax=1.0)
    ax1.set_title('File3 Detection BigMap - Predicted Nodule Probability per Window', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cases (sorted by true depth,size)')
    cbar = fig.colorbar(im1, ax=ax1, fraction=0.015, pad=0.01)
    cbar.set_label('Pred Probability')

    im2 = ax2.imshow(gt_map, aspect='auto', cmap='gray_r', vmin=0.0, vmax=1.0)
    ax2.set_title('Ground-Truth Positive Window Mask', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cases')
    fig.colorbar(im2, ax=ax2, fraction=0.015, pad=0.01)

    ax3.imshow(hit_vec.reshape(-1, 1).T, aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0)
    ax3.set_title(f'Hit@thr={threshold:.3f} (green=hit)', fontsize=11, fontweight='bold')
    ax3.set_yticks([])
    ax3.set_xlabel('Cases (same order)')
    ax3.set_xticks([])

    plt.tight_layout()
    plt.savefig(save_png, dpi=260, bbox_inches='tight')
    plt.close(fig)

    miss_cases = []
    fp_cases = []
    for r in ordered:
        has_gt = int(np.max(r['gt']) > 0)
        pred_pos = int(np.max(r['probs']) >= threshold)
        if has_gt == 1 and pred_pos == 0:
            miss_cases.append({
                'file_path': r['file_path'],
                'true_size': r['true_size'],
                'true_depth': r['true_depth'],
                'best_prob': float(r['best_prob']),
            })
        if has_gt == 0 and pred_pos == 1:
            fp_cases.append({
                'file_path': r['file_path'],
                'true_size': r['true_size'],
                'true_depth': r['true_depth'],
                'best_prob': float(r['best_prob']),
            })
    summary = {
        'threshold': float(threshold),
        'case_count': int(len(ordered)),
        'hit_rate': float(np.mean(hit_vec)),
        'miss_count': int(len(miss_cases)),
        'false_positive_count': int(len(fp_cases)),
        'miss_cases_top': miss_cases[:20],
        'false_positive_cases_top': fp_cases[:20],
        'ordered_case_meta': [
            {
                'file_path': r['file_path'],
                'true_size': r['true_size'],
                'true_depth': r['true_depth'],
                'best_prob': float(r['best_prob']),
                'best_row': int(r['best_row']),
                'pred_size_at_peak': float(r['pred_size_at_peak']),
                'pred_depth_at_peak': float(r['pred_depth_at_peak']),
            }
            for r in ordered
        ]
    }
    with open(save_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def write_report(path, metrics, summary):
    md = f"""# File3 双流模型检出大图分析

- 全局最佳阈值（按全窗口F1）：{metrics['threshold']:.4f}
- F1：{metrics['f1']:.4f}
- Precision：{metrics['precision']:.4f}
- Recall：{metrics['recall']:.4f}
- TP/TN/FP/FN：{metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}

- Case数：{summary['case_count']}
- Hit率（case级）：{summary['hit_rate']:.4f}
- 漏检case数：{summary['miss_count']}
- 误检case数：{summary['false_positive_count']}

## 读图方法
- 上图（彩色）每一行是一个样本文件，横轴是窗口序号，颜色表示双流模型结节概率。
- 中图（灰度）是对应的真值阳性窗口分布。
- 下图（红绿）表示该样本是否在阈值下被命中（绿色=命中）。
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)


def main():
    _, data_root, labels_path, model_path, out_dir = resolve_paths()
    labels = load_json(labels_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream3DCNNLSTM(seq_len=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_y, all_p = [], []
    records = []
    file_paths = []
    for root, _, files in os.walk(data_root):
        for fn in files:
            if fn.upper() == '3.CSV':
                file_paths.append(os.path.join(root, fn))
    file_paths.sort()
    for fp in file_paths:
        rel = os.path.relpath(fp, data_root).replace('/', '\\')
        if rel not in labels:
            continue
        values = read_csv_data(fp)
        n = len(values)
        segments = sanitize_segments(labels[rel].get('segments', []), n)
        seqs, stats, y = [], [], []
        for end_row in range(9, n):
            raw = values[end_row - 9:end_row + 1]
            seqs.append(normalize_sequence(raw))
            stats.append(stats_sequence(raw))
            y.append(1 if overlap_positive(end_row - 9, end_row, segments) else 0)
        if len(seqs) == 0:
            continue
        seq_np = np.asarray(seqs, dtype=np.float32)
        st_np = np.asarray(stats, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)
        probs, ps, pd = infer_all_windows(model, seq_np, st_np, device)
        best_i = int(np.argmax(probs))
        ts, td = parse_size_depth_from_path(rel)
        records.append({
            'file_path': rel,
            'true_size': float(ts),
            'true_depth': float(td),
            'probs': probs,
            'gt': y_np,
            'best_prob': float(probs[best_i]),
            'best_row': int(best_i + 9),
            'pred_size_at_peak': float(ps[best_i]),
            'pred_depth_at_peak': float(pd[best_i]),
        })
        all_y.extend(y_np.tolist())
        all_p.extend(probs.tolist())

    thr = select_best_threshold(all_y, all_p)
    m = compute_metrics(all_y, all_p, thr)
    m['threshold'] = thr
    save_png = os.path.join(out_dir, 'file3_dualstream_detection_bigmap.png')
    save_json = os.path.join(out_dir, 'file3_dualstream_detection_bigmap_summary.json')
    summary = build_bigmap_and_summary(records, thr, save_png, save_json)
    write_report(os.path.join(out_dir, 'README_检出大图分析.md'), m, summary)
    print(f"Saved: {out_dir}")


if __name__ == '__main__':
    main()
