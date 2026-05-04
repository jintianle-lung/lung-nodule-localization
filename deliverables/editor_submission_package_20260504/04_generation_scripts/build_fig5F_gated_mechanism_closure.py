from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from v5_feature_learning_analysis import forward_with_features, load_r5_model, raw_window_for_sample
from build_fig5F_cue_family_counterfactual import INTERVENTIONS, RUN_SUMMARY, cue_perturb, load_file3_samples


IN_DIR = ROOT / "deliverables" / "v5_feature_learning_final6_20260504"
OUT_DIR = ROOT / "deliverables" / "fig5_causal_intervention_20260504"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = IN_DIR / "file3_v5_outputs_plus_descriptors.csv"
GATED_CSV = OUT_DIR / "Fig5F_gated_mechanism_closure.csv"

TEXT = "#111827"
MUTED = "#586475"
GRID = "#DCE3EA"
BLUE = "#1687A7"
GREEN = "#119C77"
ORANGE = "#E79B17"
GRAY = "#8E99A6"

ROWS = [
    ("amplitude_mask", "Stress hotspot", "Peak intensity\nP95 amplitude"),
    ("contrast_mask", "Boundary contrast", "Center-border\ncenter contrast"),
    ("spread_contract", "Spatial spread", "Hotspot radius\nmoment spread"),
    ("temporal_ctrl", "Temporal control", "Order shuffle\nnegative ctrl."),
]

METRICS = [
    ("gate_loss", "Gate", BLUE),
    ("size_gatepos_loss", "Size|gate+", GREEN),
    ("depth_gatepos_loss", "Depth|gate+", ORANGE),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.3,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.75,
            "legend.fontsize": 5.15,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "savefig.dpi": 650,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wilson_ci(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4 * n * n))) / denom
    return float(max(abs(center - p), half))


def compute_gated_metrics() -> pd.DataFrame:
    if GATED_CSV.exists():
        return pd.read_csv(GATED_CSV)

    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    test_records, test_samples = load_file3_samples()
    out_df = pd.read_csv(OUTPUT_CSV)

    size_cols = [f"v5_size_prob_{value:g}cm" for value in SIZE_VALUES_CM]
    depth_cols = [f"v5_depth_prob_{name}" for name in COARSE_DEPTH_ORDER]
    label = out_df["label"].to_numpy(dtype=np.int64)
    true_size = out_df["size_class_index"].to_numpy(dtype=np.int64)
    true_depth = out_df["depth_coarse_index"].to_numpy(dtype=np.int64)
    base_det = out_df["det_prob"].to_numpy(dtype=np.float64) >= threshold
    base_size = np.argmax(out_df[size_cols].to_numpy(dtype=np.float64), axis=1)
    base_depth = np.argmax(out_df[depth_cols].to_numpy(dtype=np.float64), axis=1)

    pos_idx = np.flatnonzero(label == 1)
    gate_eval = pos_idx[base_det[pos_idx]]
    size_eval = pos_idx[base_det[pos_idx] & (base_size[pos_idx] == true_size[pos_idx])]
    depth_eval = pos_idx[base_det[pos_idx] & (base_depth[pos_idx] == true_depth[pos_idx])]
    needed = np.unique(np.concatenate([gate_eval, size_eval, depth_eval]))

    device = choose_device()
    model, _threshold, _summary = load_r5_model(Path(summary["residual_run"]), device)
    batch_size = 192
    rows: list[dict[str, object]] = []

    for mode, display, _features in ROWS:
        pert_det: dict[int, bool] = {}
        pert_size: dict[int, int] = {}
        pert_depth: dict[int, int] = {}
        for start in range(0, len(needed), batch_size):
            idxs = needed[start : start + batch_size]
            xs = []
            for idx in idxs:
                raw = raw_window_for_sample(test_records, test_samples[int(idx)])
                x = normalize_raw_frames_window_minmax(raw).astype(np.float32)[:, None, :, :]
                xs.append(cue_perturb(x, mode))
            x_tensor = torch.from_numpy(np.stack(xs, axis=0)).to(device)
            out = forward_with_features(model, x_tensor)
            det = out["det_prob"].detach().cpu().numpy().reshape(-1) >= threshold
            size = np.argmax(out["size_probs"].detach().cpu().numpy(), axis=1)
            depth = np.argmax(out["depth_probs"].detach().cpu().numpy(), axis=1)
            for idx, det_value, size_value, depth_value in zip(idxs, det, size, depth):
                key = int(idx)
                pert_det[key] = bool(det_value)
                pert_size[key] = int(size_value)
                pert_depth[key] = int(depth_value)

        gate_lost = np.asarray([not pert_det[int(i)] for i in gate_eval], dtype=bool)
        size_gate_kept = np.asarray([pert_det[int(i)] for i in size_eval], dtype=bool)
        depth_gate_kept = np.asarray([pert_det[int(i)] for i in depth_eval], dtype=bool)
        size_eval_kept = size_eval[size_gate_kept]
        depth_eval_kept = depth_eval[depth_gate_kept]
        size_lost = np.asarray([pert_size[int(i)] != true_size[int(i)] for i in size_eval_kept], dtype=bool)
        depth_lost = np.asarray([pert_depth[int(i)] != true_depth[int(i)] for i in depth_eval_kept], dtype=bool)

        metric_arrays = [
            ("gate_loss", "Gate", gate_lost, len(gate_eval), len(gate_eval)),
            ("size_gatepos_loss", "Size|gate+", size_lost, len(size_eval_kept), len(size_eval)),
            ("depth_gatepos_loss", "Depth|gate+", depth_lost, len(depth_eval_kept), len(depth_eval)),
        ]
        for metric, task, arr, denom, baseline_n in metric_arrays:
            rate = float(arr.mean()) if denom else 0.0
            rows.append(
                {
                    "perturbation": mode,
                    "cue_family": display,
                    "metric": metric,
                    "task": task,
                    "loss_rate": rate,
                    "loss_percent": 100.0 * rate,
                    "ci95_percent": 100.0 * wilson_ci(rate, denom),
                    "n_evaluable": int(denom),
                    "n_baseline_correct": int(baseline_n),
                    "n_lost": int(arr.sum()) if denom else 0,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(GATED_CSV, index=False, encoding="utf-8-sig")
    return df


def render(df: pd.DataFrame) -> list[Path]:
    apply_style()
    y = np.arange(len(ROWS), dtype=float)
    offsets = np.asarray([-0.21, 0.0, 0.21])
    height = 0.17

    fig, ax = plt.subplots(figsize=(2.42, 2.20), facecolor="white")
    fig.subplots_adjust(left=0.365, right=0.985, top=0.705, bottom=0.175)

    for offset, (metric, label, color) in zip(offsets, METRICS):
        values = []
        cis = []
        for mode, _display, _features in ROWS:
            row = df[(df["perturbation"] == mode) & (df["metric"] == metric)].iloc[0]
            values.append(float(row["loss_percent"]))
            cis.append(float(row["ci95_percent"]))
        bars = ax.barh(
            y + offset,
            values,
            xerr=cis,
            height=height,
            color=color,
            edgecolor="none",
            alpha=0.97,
            error_kw={"elinewidth": 0.58, "capthick": 0.58, "capsize": 1.45, "ecolor": TEXT},
            label=label,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            yc = bar.get_y() + bar.get_height() / 2
            if value >= 16:
                ax.text(value - 1.2, yc, f"{value:.0f}", ha="right", va="center", fontsize=5.0, color="white", fontweight="bold")
            elif value >= 2.5:
                ax.text(value + 1.0, yc, f"{value:.0f}", ha="left", va="center", fontsize=4.9, color=TEXT, fontweight="bold")

    labels = [f"{display}\n{features}" for _mode, display, features in ROWS]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 78)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xticklabels(["0", "20", "40", "60"])
    ax.grid(axis="x", color=GRID, lw=0.55, alpha=0.92, zorder=0)
    ax.set_xlabel("Baseline-correct cases lost (%)", labelpad=2)
    ax.text(0.0, 1.255, "Gated cue ablation", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9, fontweight="bold", color=TEXT)
    ax.text(0.0, 1.185, "Mechanistic closure with FEM features", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.05, color=MUTED)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2.1, pad=2)

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.55, 1.120),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.32,
        columnspacing=0.65,
    )
    for text in leg.get_texts():
        text.set_color(TEXT)

    stem = OUT_DIR / "Fig5F_gated_mechanism_closure_bar"
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.018)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    df = compute_gated_metrics()
    paths = render(df)
    print(json.dumps({"gated_csv": str(GATED_CSV), "outputs": [str(p) for p in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
