from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "deliverables" / "v5_feature_learning_final6_20260504"
OUT = ROOT / "deliverables" / "error_mechanism_global_20260504"
OUT.mkdir(parents=True, exist_ok=True)

IN_CSV = IN_DIR / "file3_v5_outputs_plus_descriptors.csv"
RUN_SUMMARY = IN_DIR / "run_summary.json"

TEXT = "#111827"
MUTED = "#5A6577"
GRID = "#DEE5EC"
BLUE = "#1687A7"
GREEN = "#119C77"
ORANGE = "#E79B17"
RED = "#D6533C"
GRAY = "#8F99A6"
LIGHT = "#F2F5F8"

SIZE_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=float)
DEPTH_NAMES = ["shallow", "middle", "deep"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.65,
            "savefig.dpi": 650,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def size_value_from_col(col: str) -> float:
    return float(col.replace("v5_size_prob_", "").replace("cm", ""))


def add_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    size_cols = sorted([c for c in out.columns if c.startswith("v5_size_prob_")], key=size_value_from_col)
    size_values = np.asarray([size_value_from_col(c) for c in size_cols], dtype=float)
    size_prob = out[size_cols].to_numpy(dtype=float)
    size_order = np.argsort(size_prob, axis=1)[:, ::-1]
    out["pred_size_idx"] = size_order[:, 0]
    out["pred_size_bin_cm"] = size_values[out["pred_size_idx"].to_numpy(dtype=int)]
    out["size_top2_contains_true"] = [
        int(int(t) in size_order[i, :2]) for i, t in enumerate(out["size_class_index"].astype(int).to_numpy())
    ]
    depth_cols = [f"v5_depth_prob_{name}" for name in DEPTH_NAMES]
    depth_prob = out[depth_cols].to_numpy(dtype=float)
    depth_order = np.argsort(depth_prob, axis=1)[:, ::-1]
    out["pred_depth_idx"] = depth_order[:, 0]
    out["depth_top2_contains_true"] = [
        int(int(t) in depth_order[i, :2]) for i, t in enumerate(out["depth_coarse_index"].astype(int).to_numpy())
    ]
    out["det_pred"] = out["det_prob"] >= threshold
    out["size_abs_idx_err"] = (out["pred_size_idx"] - out["size_class_index"]).abs()
    out["size_abs_reg_err_cm"] = (out["size_reg_cm"] - out["size_cm"]).abs()
    out["depth_abs_idx_err"] = (out["pred_depth_idx"] - out["depth_coarse_index"]).abs()
    return out


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    fp = df[(df["label"] == 0) & df["det_pred"]]
    tn = df[(df["label"] == 0) & ~df["det_pred"]]
    tp = df[(df["label"] == 1) & df["det_pred"]]
    fn = df[(df["label"] == 1) & ~df["det_pred"]]
    gated = tp
    size_exact = (gated["size_abs_idx_err"] == 0).mean()
    size_top2 = gated["size_top2_contains_true"].mean()
    depth_exact = (gated["depth_abs_idx_err"] == 0).mean()
    depth_top2 = gated["depth_top2_contains_true"].mean()
    spread_rho = spearmanr(gated["second_moment_spread_max"], gated["depth_cm"]).statistic
    radius_rho = spearmanr(gated["hotspot_radius_max"], gated["depth_cm"]).statistic
    large = gated[gated["size_cm"] >= 1.0]
    small = gated[gated["size_cm"] <= 0.75]
    rows = [
        {
            "Claim": "Detection FP resembles hard heterogeneous contact",
            "Dataset evidence": (
                f"FP n={len(fp)}; median P95 {fp['raw_p95_mean'].median():.1f} vs TN {tn['raw_p95_mean'].median():.1f}; "
                f"median contrast {fp['center_border_contrast_center'].median():.1f} vs TN {tn['center_border_contrast_center'].median():.1f}."
            ),
            "Bounded interpretation": "Consistent with non-nodule hard/heterogeneous objects mimicking a focal hotspot; not direct anatomic proof.",
        },
        {
            "Claim": "Secondary morphology can help triage detection FP",
            "Dataset evidence": (
                f"FP morphology differs from nominal size or depth Top-2 in "
                f"{100*((~fp['size_top2_contains_true'].astype(bool)) | (~fp['depth_top2_contains_true'].astype(bool))).mean():.1f}% of FP windows."
            ),
            "Bounded interpretation": "Use size/depth/shape as a surgeon-facing cross-check with CT and operative view, not as automatic FP removal.",
        },
        {
            "Claim": "Size errors concentrate in deformation-prone larger contacts",
            "Dataset evidence": (
                f"Exact size: small<=0.75 cm {pct((small['size_abs_idx_err']==0).mean())}, "
                f">=1.0 cm {pct((large['size_abs_idx_err']==0).mean())}; overall Top-2 {pct(size_top2)}; MAE {gated['size_abs_reg_err_cm'].mean():.2f} cm."
            ),
            "Bounded interpretation": "Large nodules are more deformation-prone; most errors remain clinically nearby rather than arbitrary.",
        },
        {
            "Claim": "Depth errors reflect weak/coupled physics and possible sliding",
            "Dataset evidence": (
                f"Depth exact {pct(depth_exact)}, Top-2 {pct(depth_top2)}; spread-depth rho={spread_rho:.2f}, radius-depth rho={radius_rho:.2f}."
            ),
            "Bounded interpretation": "Depth is an exploratory aid because depth cues are weak and may be perturbed by sliding/displacement during palpation.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "global_error_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    return out


def draw_detection(ax: plt.Axes, df: pd.DataFrame) -> None:
    groups = {
        "TN": df[(df["label"] == 0) & ~df["det_pred"]],
        "FP": df[(df["label"] == 0) & df["det_pred"]],
        "TP": df[(df["label"] == 1) & df["det_pred"]],
    }
    metrics = [
        ("P95", "raw_p95_mean"),
        ("Contrast", "center_border_contrast_center"),
    ]
    x = np.arange(len(metrics), dtype=float)
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = [GRAY, RED, BLUE]
    for offset, (name, sub), color in zip(offsets, groups.items(), colors):
        vals = [float(sub[col].median()) for _label, col in metrics]
        ax.bar(x + offset, vals, width=width, color=color, edgecolor="none", alpha=0.95, label=name, zorder=3)
        for xx, val in zip(x + offset, vals):
            ax.text(xx, val + max(vals) * 0.035, f"{val:.0f}", ha="center", va="bottom", fontsize=5.1, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _col in metrics])
    ax.set_ylabel("Median descriptor")
    ax.set_title("Detection FP: nodule-like hard hotspot", loc="left", fontsize=6.7, fontweight="bold", color=TEXT, pad=12)
    ax.text(0.0, 1.03, "FP approaches TP-like amplitude/contrast", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color=MUTED)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=3, handlelength=0.9, columnspacing=0.75)
    ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_size(ax: plt.Axes, gated: pd.DataFrame) -> None:
    rows = []
    for size in SIZE_VALUES:
        sub = gated[gated["size_cm"] == size]
        rows.append(
            {
                "size": size,
                "exact": 100.0 * (sub["size_abs_idx_err"] == 0).mean(),
                "top2": 100.0 * sub["size_top2_contains_true"].mean(),
            }
        )
    data = pd.DataFrame(rows)
    ax.plot(data["size"], data["top2"], color=GREEN, lw=1.55, marker="o", ms=3.2, label="Top-2", zorder=4)
    ax.plot(data["size"], data["exact"], color=ORANGE, lw=1.35, marker="o", ms=3.0, label="Exact", zorder=4)
    ax.fill_between(data["size"], data["exact"], data["top2"], color=GREEN, alpha=0.10, linewidth=0)
    ax.axvspan(1.0, 1.75, color=ORANGE, alpha=0.08, lw=0)
    ax.set_ylim(25, 103)
    ax.set_xticks(SIZE_VALUES)
    ax.set_xticklabels([f"{v:g}" for v in SIZE_VALUES], rotation=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("True size (cm)")
    ax.set_title("Size errors: larger/deformable contacts", loc="left", fontsize=6.7, fontweight="bold", color=TEXT, pad=12)
    ax.text(0.0, 1.03, "Top-2 remains high despite exact-bin drops", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color=MUTED)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02), ncol=2, handlelength=1.3, columnspacing=0.8)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_depth(ax: plt.Axes, gated: pd.DataFrame) -> None:
    depth_labels = ["Shallow", "Middle", "Deep"]
    rows = []
    for idx, label in enumerate(depth_labels):
        sub = gated[gated["depth_coarse_index"] == idx]
        rows.append(
            {
                "depth": label,
                "exact": 100.0 * (sub["depth_abs_idx_err"] == 0).mean(),
                "top2": 100.0 * sub["depth_top2_contains_true"].mean(),
            }
        )
    data = pd.DataFrame(rows)
    x = np.arange(len(data), dtype=float)
    ax.bar(x - 0.16, data["exact"], width=0.30, color=ORANGE, edgecolor="none", alpha=0.92, label="Exact", zorder=3)
    ax.bar(x + 0.16, data["top2"], width=0.30, color=GREEN, edgecolor="none", alpha=0.92, label="Top-2", zorder=3)
    for xx, val in zip(x - 0.16, data["exact"]):
        ax.text(xx, val + 1.3, f"{val:.0f}", ha="center", va="bottom", fontsize=5.1, color=TEXT)
    for xx, val in zip(x + 0.16, data["top2"]):
        ax.text(xx, val + 1.3, f"{val:.0f}", ha="center", va="bottom", fontsize=5.1, color=TEXT)
    rho = spearmanr(gated["second_moment_spread_max"], gated["depth_cm"]).statistic
    ax.set_xticks(x)
    ax.set_xticklabels(data["depth"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Depth errors: weak diffusion cue", loc="left", fontsize=6.7, fontweight="bold", color=TEXT, pad=12)
    ax.text(0.0, 1.03, f"Spread-depth rho={rho:.2f}; sliding can distort depth", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color=MUTED)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02), ncol=2, handlelength=1.3, columnspacing=0.8)
    ax.grid(axis="y", color=GRID, lw=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render(df: pd.DataFrame) -> list[Path]:
    apply_style()
    gated = df[(df["label"] == 1) & df["det_pred"]].copy()
    fig = plt.figure(figsize=(6.65, 2.55), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.060, right=0.987, top=0.655, bottom=0.205, wspace=0.350)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    draw_detection(axes[0], df)
    draw_size(axes[1], gated)
    draw_depth(axes[2], gated)
    fig.text(0.060, 0.955, "Error mechanisms are structured by tactile physics", ha="left", va="top", fontsize=8.3, fontweight="bold", color=TEXT)
    fig.text(
        0.060,
        0.880,
        "Detection errors resemble hard heterogeneous contacts; size errors stay near adjacent bins; depth remains limited by weak/coupled diffusion cues.",
        ha="left",
        va="top",
        fontsize=5.3,
        color=MUTED,
    )
    stem = OUT / "Fig_error_mechanism_global_summary"
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    df = pd.read_csv(IN_CSV)
    df = add_predictions(df, float(summary["threshold"]))
    summary_df = build_summary(df)
    paths = render(df)
    print(json.dumps({"summary_csv": str(OUT / "global_error_mechanism_summary.csv"), "outputs": [str(p) for p in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
