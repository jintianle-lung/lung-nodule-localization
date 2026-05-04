from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
LATEST_ALGO = ROOT / "latest_algorithm"
MODELS = ROOT / "models"
for path in (ROOT, TMP, LATEST_ALGO, MODELS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from input_normalization_v1 import normalize_raw_frames_window_minmax
from task_protocol_v1 import COARSE_DEPTH_ORDER, SIZE_VALUES_CM
from run_r5_feature_baseline_auc import load_locked_splits
from v5_feature_learning_analysis import (
    FEATURE_TARGETS,
    forward_with_features,
    load_r5_model,
    raw_window_for_sample,
)


IN_DIR = ROOT / "deliverables" / "v5_feature_learning_final6_20260504"
OUT_DIR = ROOT / "deliverables" / "fig5_causal_intervention_20260504"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = IN_DIR / "file3_v5_outputs_plus_descriptors.csv"
PERTURB_CSV = IN_DIR / "v5_targeted_perturbation_response.csv"
RUN_SUMMARY = IN_DIR / "run_summary.json"
AUDIT_CSV = OUT_DIR / "Fig5F_cue_family_descriptor_audit.csv"
RESPONSE_CSV = OUT_DIR / "Fig5F_cue_family_model_response.csv"


TEXT = "#101828"
MUTED = "#536070"
GRID = "#DDE3EA"
BLUE = "#1687A7"
GREEN = "#119C77"
ORANGE = "#E79B17"
RED = "#D6533C"
GRAY = "#8C96A3"

INTERVENTIONS = [
    ("amplitude_mask", "Amplitude\nmask"),
    ("contrast_mask", "Contrast\nmask"),
    ("spread_contract", "Spread\ncompact"),
    ("temporal_ctrl", "Temporal\nctrl."),
]

FEATURE_COLUMNS = [
    ("raw_max_mean", "Peak"),
    ("center_border_contrast_center", "Contrast"),
    ("raw_p95_mean", "P95"),
    ("center_border_contrast_center", "Contrast"),
    ("hotspot_radius_max", "Radius"),
    ("second_moment_spread_max", "Spread"),
]

TASK_BANDS = [
    ("Detection", 0, 1, BLUE),
    ("Size", 2, 3, GREEN),
    ("Depth", 4, 5, ORANGE),
]

RESPONSE_BARS = [
    ("Amplitude", "amplitude_mask", "delta_size_pred_prob", GREEN),
    ("Contrast", "contrast_mask", "delta_size_pred_prob", GREEN),
    ("Spread", "spread_contract", "delta_depth_pred_prob", ORANGE),
    ("Ctrl.", "temporal_ctrl", "delta_det_prob", GRAY),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "savefig.dpi": 650,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def weighted_spread(norm_frame: np.ndarray) -> tuple[float, float]:
    weights = np.clip(np.asarray(norm_frame, dtype=np.float64), 0.0, None)
    total = float(weights.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    rows, cols = np.indices(weights.shape, dtype=np.float64)
    row_c = float((rows * weights).sum() / total)
    col_c = float((cols * weights).sum() / total)
    second = float((weights * ((rows - row_c) ** 2 + (cols - col_c) ** 2)).sum() / total)
    return float(math.sqrt(max(second, 0.0))), second


def normalized_descriptor_row(x: np.ndarray) -> dict[str, float]:
    frames = np.asarray(x[:, 0], dtype=np.float32)
    center_mask = np.zeros((12, 8), dtype=bool)
    center_mask[3:9, 2:6] = True
    border_mask = ~center_mask
    center_idx = frames.shape[0] // 2

    raw_max = []
    raw_p95 = []
    radius = []
    spread = []
    for frame in frames:
        raw_max.append(float(frame.max()))
        raw_p95.append(float(np.percentile(frame, 95)))
        r, s = weighted_spread(frame)
        radius.append(r)
        spread.append(s)

    center_frame = frames[center_idx]
    contrast = float(center_frame[center_mask].mean() - center_frame[border_mask].mean())
    return {
        "raw_max_mean": float(np.mean(raw_max)),
        "raw_p95_mean": float(np.mean(raw_p95)),
        "center_border_contrast_center": contrast,
        "hotspot_radius_max": float(np.max(radius)),
        "second_moment_spread_max": float(np.max(spread)),
    }


def hotspot_mask(x: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    mean_frame = np.asarray(x[:, 0], dtype=np.float32).mean(axis=0)
    peak = np.unravel_index(int(np.argmax(mean_frame)), mean_frame.shape)
    rows, cols = np.indices(mean_frame.shape, dtype=np.float32)
    dist2 = (rows - float(peak[0])) ** 2 + (cols - float(peak[1])) ** 2
    mask = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    mask /= max(float(mask.max()), 1e-6)
    return mask[None, None]


def cue_perturb(x: np.ndarray, mode: str) -> np.ndarray:
    xp = np.asarray(x, dtype=np.float32).copy()
    if mode == "amplitude_mask":
        mask = hotspot_mask(xp, sigma=1.15)
        floor = float(np.percentile(xp, 50))
        xp = xp * (1.0 - 0.78 * mask) + floor * (0.78 * mask)
    elif mode == "contrast_mask":
        center = np.zeros((12, 8), dtype=bool)
        center[3:9, 2:6] = True
        border_value = float(np.median(xp[:, 0, ~center]))
        xp[:, 0, center] = 0.38 * xp[:, 0, center] + 0.62 * border_value
    elif mode == "spread_contract":
        mean_frame = np.asarray(xp[:, 0], dtype=np.float32).mean(axis=0)
        peak = np.unravel_index(int(np.argmax(mean_frame)), mean_frame.shape)
        rows, cols = np.indices(mean_frame.shape, dtype=np.float32)
        dist = (rows - float(peak[0])) ** 2 + (cols - float(peak[1])) ** 2
        compact_order = np.argsort(dist.reshape(-1))
        for t in range(xp.shape[0]):
            values = np.sort(xp[t, 0].reshape(-1))[::-1]
            compacted = np.empty_like(values)
            compacted[compact_order] = values
            xp[t, 0] = compacted.reshape(12, 8)
    elif mode == "temporal_ctrl":
        order = np.arange(xp.shape[0])
        order = np.concatenate([order[1::2], order[::2]])
        xp = xp[order]
    else:
        raise ValueError(f"Unknown cue perturbation: {mode}")
    return np.clip(xp, 0.0, 1.0).astype(np.float32)


def load_file3_samples() -> tuple[dict, list[dict]]:
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads((Path(summary["detector_run"]) / "manifest.json").read_text(encoding="utf-8"))
    _train_records, _val_records, test_records, _train_samples_all, _train_samples_det, _val_samples, test_samples = load_locked_splits(manifest)
    return test_records, list(test_samples)


def compute_descriptor_audit() -> pd.DataFrame:
    if AUDIT_CSV.exists() and RESPONSE_CSV.exists():
        return pd.read_csv(AUDIT_CSV)

    test_records, test_samples = load_file3_samples()
    out_df = pd.read_csv(OUTPUT_CSV)
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    gated = out_df.index[(out_df["label"].astype(int) == 1) & (out_df["det_prob"].astype(float) >= threshold)].to_numpy()

    base_rows: list[dict[str, float]] = []
    pert_rows: dict[str, list[dict[str, float]]] = {key: [] for key, _label in INTERVENTIONS}
    for idx in gated:
        raw = raw_window_for_sample(test_records, test_samples[int(idx)])
        x = normalize_raw_frames_window_minmax(raw).astype(np.float32)[:, None, :, :]
        base_rows.append(normalized_descriptor_row(x))
        for mode, _label in INTERVENTIONS:
            pert_rows[mode].append(normalized_descriptor_row(cue_perturb(x, mode)))

    base_df = pd.DataFrame(base_rows)
    scale = base_df.std(axis=0, ddof=1).replace(0.0, np.nan)
    rows = []
    for mode, label in INTERVENTIONS:
        p_df = pd.DataFrame(pert_rows[mode])
        for feature, feature_label in FEATURE_TARGETS:
            deltas = ((p_df[feature] - base_df[feature]) / scale[feature]).replace([np.inf, -np.inf], np.nan).dropna()
            mean = float(deltas.mean())
            sem = float(deltas.std(ddof=1) / math.sqrt(max(len(deltas), 1))) if len(deltas) > 1 else 0.0
            rows.append(
                {
                    "perturbation": mode,
                    "perturbation_label": label,
                    "feature": feature,
                    "feature_label": feature_label,
                    "mean_z_delta": mean,
                    "sem": sem,
                    "ci95": 1.96 * sem,
                    "n_gated_positive": int(len(gated)),
                    "domain": "normalized_input",
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    return audit


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_model_response() -> pd.DataFrame:
    if RESPONSE_CSV.exists():
        return pd.read_csv(RESPONSE_CSV)

    test_records, test_samples = load_file3_samples()
    out_df = pd.read_csv(OUTPUT_CSV)
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    gated = out_df.index[(out_df["label"].astype(int) == 1) & (out_df["det_prob"].astype(float) >= threshold)].to_numpy()

    device = choose_device()
    model, _threshold, _summary = load_r5_model(Path(summary["residual_run"]), device)

    size_cols = [f"v5_size_prob_{value:g}cm" for value in SIZE_VALUES_CM]
    depth_cols = [f"v5_depth_prob_{name}" for name in COARSE_DEPTH_ORDER]
    base_det = out_df["det_prob"].to_numpy(dtype=np.float64)
    base_size_probs = out_df[size_cols].to_numpy(dtype=np.float64)
    base_depth_probs = out_df[depth_cols].to_numpy(dtype=np.float64)
    base_size_top = np.argmax(base_size_probs[gated], axis=1)
    base_depth_top = np.argmax(base_depth_probs[gated], axis=1)
    base_size_top_prob = base_size_probs[gated][np.arange(len(gated)), base_size_top]
    base_depth_top_prob = base_depth_probs[gated][np.arange(len(gated)), base_depth_top]

    rows = []
    batch_size = 192
    for mode, label in INTERVENTIONS:
        deltas = {
            "delta_det_prob": [],
            "delta_size_pred_prob": [],
            "delta_depth_pred_prob": [],
        }
        cursor = 0
        for start in range(0, len(gated), batch_size):
            idxs = gated[start : start + batch_size]
            xs = []
            for idx in idxs:
                raw = raw_window_for_sample(test_records, test_samples[int(idx)])
                x = normalize_raw_frames_window_minmax(raw).astype(np.float32)[:, None, :, :]
                xs.append(cue_perturb(x, mode))
            x_tensor = torch.from_numpy(np.stack(xs, axis=0)).to(device)
            out = forward_with_features(model, x_tensor)
            pert_det = out["det_prob"].detach().cpu().numpy().reshape(-1)
            pert_size_probs = out["size_probs"].detach().cpu().numpy()
            pert_depth_probs = out["depth_probs"].detach().cpu().numpy()
            local_n = len(idxs)
            local = slice(cursor, cursor + local_n)
            deltas["delta_det_prob"].extend((pert_det - base_det[idxs]).tolist())
            deltas["delta_size_pred_prob"].extend(
                (pert_size_probs[np.arange(local_n), base_size_top[local]] - base_size_top_prob[local]).tolist()
            )
            deltas["delta_depth_pred_prob"].extend(
                (pert_depth_probs[np.arange(local_n), base_depth_top[local]] - base_depth_top_prob[local]).tolist()
            )
            cursor += local_n
        for metric, values in deltas.items():
            arr = np.asarray(values, dtype=np.float64)
            sem = float(arr.std(ddof=1) / math.sqrt(max(len(arr), 1))) if len(arr) > 1 else 0.0
            rows.append(
                {
                    "perturbation": mode,
                    "perturbation_label": label,
                    "metric": metric,
                    "mean_delta": float(arr.mean()),
                    "sem": sem,
                    "ci95": 1.96 * sem,
                    "n_gated_positive": int(len(gated)),
                }
            )
    response = pd.DataFrame(rows)
    RESPONSE_CSV.write_text(response.to_csv(index=False), encoding="utf-8-sig")
    return response


def value_from(df: pd.DataFrame, intervention: str, metric: str) -> tuple[float, float]:
    row = df[(df["perturbation"] == intervention) & (df["metric"] == metric)].iloc[0]
    return float(row["mean_delta"]), float(row["ci95"])


def matrix_from_audit(audit: pd.DataFrame) -> np.ndarray:
    values = []
    for intervention, _label in INTERVENTIONS:
        row = []
        sub = audit[audit["perturbation"] == intervention]
        for feature, _name in FEATURE_COLUMNS:
            row.append(float(sub[sub["feature"] == feature]["mean_z_delta"].iloc[0]))
        values.append(row)
    return np.asarray(values, dtype=np.float64)


def response_matrix(response: pd.DataFrame) -> np.ndarray:
    metrics = ["delta_det_prob", "delta_size_pred_prob", "delta_depth_pred_prob"]
    values = []
    for intervention, _label in INTERVENTIONS:
        row = []
        sub = response[response["perturbation"] == intervention]
        for metric in metrics:
            row.append(float(sub[sub["metric"] == metric]["mean_delta"].iloc[0]))
        values.append(row)
    return np.asarray(values, dtype=np.float64)


def render_panel(audit: pd.DataFrame) -> list[Path]:
    apply_style()
    perturb_df = compute_model_response()
    n = int(audit["n_gated_positive"].iloc[0])
    mat = matrix_from_audit(audit)
    vmax = max(2.2, float(np.nanmax(np.abs(mat))) * 0.82)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cue_delta",
        ["#B2182B", "#F7A582", "#F7F7F7", "#92C5DE", "#2166AC"],
    )

    resp = response_matrix(perturb_df)

    fig = plt.figure(figsize=(4.72, 1.88), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.55, 0.78],
        left=0.080,
        right=0.985,
        bottom=0.205,
        top=0.695,
        wspace=0.135,
    )
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_resp = fig.add_subplot(gs[0, 1])

    im = ax_hm.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax_hm.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax_hm.set_xticklabels([name for _feature, name in FEATURE_COLUMNS], rotation=0, ha="center")
    ax_hm.set_yticks(np.arange(len(INTERVENTIONS)))
    ax_hm.set_yticklabels([label for _key, label in INTERVENTIONS])
    ax_hm.tick_params(axis="both", length=0, pad=2)
    for side in ax_hm.spines.values():
        side.set_visible(False)

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            value = mat[r, c]
            txt_color = "white" if abs(value) > vmax * 0.55 else TEXT
            ax_hm.text(c, r, f"{value:+.1f}", ha="center", va="center", fontsize=5.6, color=txt_color, fontweight="semibold")

    for x in [1.5, 3.5]:
        ax_hm.axvline(x, color="white", lw=1.8)
    for label, start, end, color in TASK_BANDS:
        x0 = start - 0.48
        width = end - start + 0.96
        ax_hm.add_patch(
            plt.Rectangle(
                (x0, -0.74),
                width,
                0.075,
                transform=ax_hm.transData,
                color=color,
                alpha=0.95,
                clip_on=False,
                linewidth=0,
            )
        )
        ax_hm.text((start + end) / 2, -0.86, label, ha="center", va="bottom", fontsize=4.8, color=color, fontweight="bold")
    ax_hm.set_title("Descriptor change (z)", loc="left", fontsize=6.2, fontweight="bold", color=TEXT, pad=16)

    cax = fig.add_axes([0.092, 0.075, 0.285, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-round(vmax, 1), 0, round(vmax, 1)])
    cb.ax.tick_params(labelsize=4.8, length=2, pad=1, colors=MUTED)
    cb.outline.set_visible(False)

    resp_v = max(0.60, float(np.nanmax(np.abs(resp))) * 1.03)
    im2 = ax_resp.imshow(resp, cmap=cmap, vmin=-resp_v, vmax=resp_v, aspect="auto", interpolation="nearest")
    ax_resp.set_xticks([0, 1, 2])
    ax_resp.set_xticklabels(["Det.", "Size", "Depth"])
    ax_resp.set_yticks(np.arange(len(INTERVENTIONS)))
    ax_resp.set_yticklabels([])
    ax_resp.tick_params(axis="both", length=0, pad=2)
    for side in ax_resp.spines.values():
        side.set_visible(False)
    for r in range(resp.shape[0]):
        for c in range(resp.shape[1]):
            value = resp[r, c]
            txt_color = "white" if abs(value) > resp_v * 0.48 else TEXT
            ax_resp.text(c, r, f"{value:+.2f}", ha="center", va="center", fontsize=5.8, color=txt_color, fontweight="semibold")
    ax_resp.set_title("Output drop (Delta p)", loc="left", fontsize=6.2, fontweight="bold", color=TEXT, pad=16)

    cax2 = fig.add_axes([0.640, 0.075, 0.205, 0.022])
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_ticks([-0.60, 0.0])
    cb2.ax.tick_params(labelsize=4.8, length=2, pad=1, colors=MUTED)
    cb2.outline.set_visible(False)

    fig.text(0.080, 0.955, "Counterfactual cue-family masking", ha="left", va="top", fontsize=7.2, fontweight="bold", color=TEXT)
    fig.text(
        0.080,
        0.875,
        f"Cue perturbations are audited against six FEM-guided descriptors before V5 response, n={n:,}.",
        ha="left",
        va="top",
        fontsize=5.0,
        color=MUTED,
    )
    stem = OUT_DIR / "Fig5F_counterfactual_cue_family_matrix"
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.018)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    audit = compute_descriptor_audit()
    paths = render_panel(audit)
    print(json.dumps({"audit_csv": str(AUDIT_CSV), "outputs": [str(p) for p in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
