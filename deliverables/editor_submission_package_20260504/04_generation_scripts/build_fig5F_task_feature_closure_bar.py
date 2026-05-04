from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables" / "fig5_causal_intervention_20260504"
SRC = OUT_DIR / "Fig5F_gated_mechanism_closure.csv"

TEXT = "#111827"
MUTED = "#596579"
GRID = "#DCE3EA"
BLUE = "#1687A7"
GREEN = "#119C77"
ORANGE = "#E79B17"
GRAY = "#A0A8B4"

ROWS = [
    {
        "label": "Detection\nstress hotspot\nPeak + contrast",
        "perturbation": "contrast_mask",
        "metric": "gate_loss",
        "control_metric": "gate_loss",
        "color": BLUE,
    },
    {
        "label": "Size\nedge contour\nP95 + contrast",
        "perturbation": "contrast_mask",
        "metric": "size_gatepos_loss",
        "control_metric": "size_gatepos_loss",
        "color": GREEN,
    },
    {
        "label": "Depth\nspatial diffusion\nRadius + spread",
        "perturbation": "spread_contract",
        "metric": "depth_gatepos_loss",
        "control_metric": "depth_gatepos_loss",
        "color": ORANGE,
    },
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.85,
            "legend.fontsize": 5.25,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "savefig.dpi": 650,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def pick(df: pd.DataFrame, perturbation: str, metric: str) -> tuple[float, float]:
    row = df[(df["perturbation"] == perturbation) & (df["metric"] == metric)].iloc[0]
    return float(row["loss_percent"]), float(row["ci95_percent"])


def main() -> None:
    apply_style()
    df = pd.read_csv(SRC)

    y = np.arange(len(ROWS), dtype=float)
    cue_values = []
    cue_cis = []
    ctrl_values = []
    ctrl_cis = []
    colors = []
    labels = []
    for row in ROWS:
        value, ci = pick(df, row["perturbation"], row["metric"])
        ctrl_value, ctrl_ci = pick(df, "temporal_ctrl", row["control_metric"])
        cue_values.append(value)
        cue_cis.append(ci)
        ctrl_values.append(ctrl_value)
        ctrl_cis.append(ctrl_ci)
        colors.append(row["color"])
        labels.append(row["label"])

    fig, ax = plt.subplots(figsize=(2.28, 2.02), facecolor="white")
    fig.subplots_adjust(left=0.400, right=0.985, top=0.735, bottom=0.175)

    ax.barh(
        y + 0.135,
        ctrl_values,
        xerr=ctrl_cis,
        height=0.18,
        color=GRAY,
        edgecolor="none",
        alpha=0.55,
        error_kw={"elinewidth": 0.55, "capthick": 0.55, "capsize": 1.35, "ecolor": MUTED},
        label="_nolegend_",
        zorder=3,
    )
    bars = ax.barh(
        y - 0.135,
        cue_values,
        xerr=cue_cis,
        height=0.25,
        color=colors,
        edgecolor="none",
        alpha=0.97,
        error_kw={"elinewidth": 0.62, "capthick": 0.62, "capsize": 1.55, "ecolor": TEXT},
        label="_nolegend_",
        zorder=4,
    )

    for bar, value in zip(bars, cue_values):
        yc = bar.get_y() + bar.get_height() / 2
        ax.text(value + 1.05, yc, f"{value:.0f}%", ha="left", va="center", fontsize=5.1, color=TEXT, fontweight="bold")
    for yy, value in zip(y, ctrl_values):
        text = f"{value:.1f}%" if value < 10 else f"{value:.0f}%"
        ax.text(value + 0.85, yy + 0.135, text, ha="left", va="center", fontsize=4.8, color=MUTED, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 57)
    ax.set_xticks([0, 15, 30, 45])
    ax.set_xticklabels(["0", "15", "30", "45"])
    ax.grid(axis="x", color=GRID, lw=0.55, alpha=0.92, zorder=0)
    ax.set_xlabel("Gated cases lost (%)", labelpad=2)
    ax.text(0.0, 1.245, "Feature-guided cue ablation", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.75, fontweight="bold", color=TEXT)
    ax.text(0.0, 1.172, "Gate first; size/depth only if gate+", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.95, color=MUTED)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2.1, pad=2)

    stem = OUT_DIR / "Fig5F_feature_guided_gated_ablation_bar"
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.018)
        paths.append(str(path))
    plt.close(fig)
    print(json.dumps({"outputs": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
