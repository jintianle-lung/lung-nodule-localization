import json
import re
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "deliverables" / "v5_feature_learning_20260503" / "file3_v5_outputs_plus_descriptors.csv"
RUN_SUMMARY = ROOT / "deliverables" / "v5_feature_learning_20260503" / "run_summary.json"
OUT = ROOT / "deliverables" / "failure_mode_analysis_20260503"

DEPTH_NAMES = ["shallow", "middle", "deep"]
COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}


def size_value_from_col(name: str) -> float:
    text = name.replace("v5_size_prob_", "").replace("cm", "")
    return float(text)


def hedges_g(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    g = (a.mean() - b.mean()) / pooled
    correction = 1.0 - 3.0 / max(4 * (len(a) + len(b)) - 9, 1)
    return float(g * correction)


def mean_or_nan(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def add_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    size_cols = [c for c in out.columns if c.startswith("v5_size_prob_")]
    size_cols = sorted(size_cols, key=size_value_from_col)
    size_values = np.array([size_value_from_col(c) for c in size_cols], dtype=float)
    size_probs = out[size_cols].to_numpy(dtype=float)
    size_order = np.argsort(size_probs, axis=1)[:, ::-1]
    out["pred_size_idx"] = size_order[:, 0]
    out["pred_size_bin_cm"] = size_values[out["pred_size_idx"].to_numpy(dtype=int)]
    out["size_top2_contains_true"] = [
        int(int(t) in size_order[i, :2]) for i, t in enumerate(out["size_class_index"].astype(int).to_numpy())
    ]

    depth_cols = [f"v5_depth_prob_{name}" for name in DEPTH_NAMES]
    depth_probs = out[depth_cols].to_numpy(dtype=float)
    depth_order = np.argsort(depth_probs, axis=1)[:, ::-1]
    out["pred_depth_idx"] = depth_order[:, 0]
    out["pred_depth_name"] = [DEPTH_NAMES[i] for i in out["pred_depth_idx"].astype(int)]
    out["depth_top2_contains_true"] = [
        int(int(t) in depth_order[i, :2]) for i, t in enumerate(out["depth_coarse_index"].astype(int).to_numpy())
    ]

    out["det_pred"] = out["det_prob"] >= threshold
    out["size_idx_err"] = out["pred_size_idx"] - out["size_class_index"]
    out["size_abs_idx_err"] = out["size_idx_err"].abs()
    out["size_reg_err_cm"] = out["size_reg_cm"] - out["size_cm"]
    out["depth_idx_err"] = out["pred_depth_idx"] - out["depth_coarse_index"]
    out["depth_abs_idx_err"] = out["depth_idx_err"].abs()
    return out


def build_summary(df: pd.DataFrame) -> dict:
    pos_gated = df[(df["label"] == 1) & df["det_pred"]].copy()
    det_counts = {
        "TP": int(((df["label"] == 1) & df["det_pred"]).sum()),
        "FN": int(((df["label"] == 1) & ~df["det_pred"]).sum()),
        "TN": int(((df["label"] == 0) & ~df["det_pred"]).sum()),
        "FP": int(((df["label"] == 0) & df["det_pred"]).sum()),
    }
    size_counts = {
        "Exact": int((pos_gated["size_abs_idx_err"] == 0).sum()),
        "Adjacent": int((pos_gated["size_abs_idx_err"] == 1).sum()),
        "Severe": int((pos_gated["size_abs_idx_err"] >= 2).sum()),
    }
    depth_counts = {
        "Exact": int((pos_gated["depth_abs_idx_err"] == 0).sum()),
        "Adjacent": int((pos_gated["depth_abs_idx_err"] == 1).sum()),
        "Non-adjacent": int((pos_gated["depth_abs_idx_err"] >= 2).sum()),
    }
    metrics = {
        "detection_sensitivity": det_counts["TP"] / max(det_counts["TP"] + det_counts["FN"], 1),
        "detection_specificity": det_counts["TN"] / max(det_counts["TN"] + det_counts["FP"], 1),
        "size_exact": size_counts["Exact"] / max(len(pos_gated), 1),
        "size_adjacent_or_exact": (size_counts["Exact"] + size_counts["Adjacent"]) / max(len(pos_gated), 1),
        "size_top2_contains_true": mean_or_nan(pos_gated["size_top2_contains_true"]),
        "depth_exact": depth_counts["Exact"] / max(len(pos_gated), 1),
        "depth_adjacent_or_exact": (depth_counts["Exact"] + depth_counts["Adjacent"]) / max(len(pos_gated), 1),
        "depth_top2_contains_true": mean_or_nan(pos_gated["depth_top2_contains_true"]),
        "n_all": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_gated_positive": int(len(pos_gated)),
    }
    return {"detection": det_counts, "size": size_counts, "depth": depth_counts, "metrics": metrics}


def build_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        ("Amplitude", "raw_p95_mean"),
        ("Center contrast", "center_border_contrast_center"),
        ("Temporal fluct.", "window_raw_global_std"),
        ("Spatial spread", "second_moment_spread_max"),
        ("Hotspot radius", "hotspot_radius_max"),
        ("Entropy", "spatial_entropy_mean"),
        ("Centroid drift", "centroid_drift"),
        ("Persistence", "peak_persistence_ratio"),
    ]
    comparisons = [
        (
            "Detection FP",
            "FP vs TN",
            df[(df["label"] == 0) & df["det_pred"]],
            df[(df["label"] == 0) & ~df["det_pred"]],
            "false positive minus true negative",
        ),
        (
            "Detection FN",
            "FN vs TP",
            df[(df["label"] == 1) & ~df["det_pred"]],
            df[(df["label"] == 1) & df["det_pred"]],
            "false negative minus true positive",
        ),
        (
            "Size error",
            "Size error vs exact",
            df[(df["label"] == 1) & df["det_pred"] & (df["size_abs_idx_err"] > 0)],
            df[(df["label"] == 1) & df["det_pred"] & (df["size_abs_idx_err"] == 0)],
            "wrong size bin minus exact size bin",
        ),
        (
            "Deep under-call",
            "Deep under-call vs deep exact",
            df[
                (df["label"] == 1)
                & df["det_pred"]
                & (df["depth_coarse_index"] == 2)
                & (df["pred_depth_idx"] < 2)
            ],
            df[
                (df["label"] == 1)
                & df["det_pred"]
                & (df["depth_coarse_index"] == 2)
                & (df["pred_depth_idx"] == 2)
            ],
            "deep predicted shallower minus deep exact",
        ),
    ]
    rows = []
    for family, comp, fail, success, meaning in comparisons:
        for feature, col in features:
            rows.append(
                {
                    "family": family,
                    "comparison": comp,
                    "meaning": meaning,
                    "feature": feature,
                    "column": col,
                    "n_failure": int(len(fail)),
                    "n_reference": int(len(success)),
                    "failure_mean": mean_or_nan(fail[col]),
                    "reference_mean": mean_or_nan(success[col]),
                    "hedges_g_failure_minus_reference": hedges_g(fail[col], success[col]),
                }
            )
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111827")
    ax.spines["bottom"].set_color("#111827")
    ax.tick_params(colors="#111827", labelsize=8)


def draw_stacked(ax, summary: dict, compact: bool = False):
    rows = [
        ("Detection", [("TP", summary["detection"]["TP"], COLORS["green"]), ("FP", summary["detection"]["FP"], COLORS["vermillion"]), ("FN", summary["detection"]["FN"], COLORS["orange"])]),
        ("Size", [("Exact", summary["size"]["Exact"], COLORS["green"]), ("Adjacent", summary["size"]["Adjacent"], COLORS["orange"]), ("Severe", summary["size"]["Severe"], COLORS["vermillion"])]),
        ("Depth", [("Exact", summary["depth"]["Exact"], COLORS["green"]), ("Adjacent", summary["depth"]["Adjacent"], COLORS["orange"]), ("Non-adj.", summary["depth"]["Non-adjacent"], COLORS["vermillion"])]),
    ]
    for y, (name, parts) in enumerate(rows):
        total = sum(v for _, v, _ in parts)
        left = 0.0
        for label, count, color in parts:
            width = 100 * count / max(total, 1)
            ax.barh(y, width, left=left, color=color, height=0.55, edgecolor="white", linewidth=0.8)
            if compact and width < 8:
                pass
            elif width >= 9:
                label_text = f"{label}\n{count}"
                if compact:
                    label_text = {
                        "Adjacent": f"Adj.\n{count}",
                        "Severe": f"Sev.\n{count}",
                        "Non-adj.": f"Far\n{count}",
                    }.get(label, label_text if len(label) <= 5 else f"{label[:5]}\n{count}")
                ax.text(left + width / 2, y, label_text, ha="center", va="center", fontsize=7, color="white" if color != COLORS["orange"] else "#111827", linespacing=0.95)
            elif not compact:
                ax.text(left + width + 1.0, y, f"{label} {count}", ha="left", va="center", fontsize=6.6, color="#111827")
            left += width
        if not compact:
            ax.text(103, y, f"n={total}", ha="left", va="center", fontsize=7.5, color=COLORS["gray"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 117)
    ax.set_xlabel("Composition (%)", fontsize=8)
    ax.set_title("A  Outcome spectrum is structured, not random", loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.invert_yaxis()
    despine(ax)


def draw_detection_heatmap(ax, effects: pd.DataFrame):
    features = ["Amplitude", "Center contrast", "Temporal fluct.", "Spatial spread", "Entropy", "Centroid drift"]
    comps = ["Detection FP", "Detection FN"]
    mat = np.zeros((len(features), len(comps)))
    for i, feature in enumerate(features):
        for j, comp in enumerate(comps):
            value = effects[(effects["family"] == comp) & (effects["feature"] == feature)][
                "hedges_g_failure_minus_reference"
            ].iloc[0]
            mat[i, j] = value
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(comps)), ["FP vs TN", "FN vs TP"])
    ax.set_yticks(range(len(features)), features)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(val) > 1.15 else "#111827")
    ax.set_title("B  Detection descriptor shifts", loc="left", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_effect_bars(ax, effects: pd.DataFrame, family: str, title: str, features: list[str], xlim=(-1.0, 1.0)):
    sub = effects[(effects["family"] == family) & effects["feature"].isin(features)].copy()
    sub["feature"] = pd.Categorical(sub["feature"], categories=features, ordered=True)
    sub = sub.sort_values("feature")
    vals = sub["hedges_g_failure_minus_reference"].to_numpy(dtype=float)
    y = np.arange(len(sub))
    colors = [COLORS["vermillion"] if v > 0 else COLORS["blue"] for v in vals]
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.barh(y, vals, color=colors, alpha=0.88, height=0.62)
    ax.set_yticks(y, sub["feature"].astype(str).tolist())
    ax.set_xlim(*xlim)
    ax.set_xlabel("Effect size: failure - reference", fontsize=8)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    for yi, val in zip(y, vals):
        ax.text(val + (0.035 if val >= 0 else -0.035), yi, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=7)
    ax.invert_yaxis()
    despine(ax)


def draw_tolerance(ax, summary: dict):
    metrics = summary["metrics"]
    groups = [
        ("Detection", [("Sensitivity", metrics["detection_sensitivity"]), ("Specificity", metrics["detection_specificity"])]),
        ("Size", [("Exact", metrics["size_exact"]), ("Adj./exact", metrics["size_adjacent_or_exact"]), ("Top-2", metrics["size_top2_contains_true"])]),
        ("Depth", [("Exact", metrics["depth_exact"]), ("Adj./exact", metrics["depth_adjacent_or_exact"]), ("Top-2", metrics["depth_top2_contains_true"])]),
    ]
    x = []
    labels = []
    vals = []
    colors = []
    pos = 0
    for group_name, items in groups:
        start = pos
        for label, val in items:
            x.append(pos)
            labels.append(label)
            vals.append(val)
            colors.append(COLORS["green"] if label in {"Sensitivity", "Exact"} else COLORS["blue"] if label == "Top-2" else COLORS["orange"])
            pos += 1
        ax.text((start + pos - 1) / 2, -0.17, group_name, ha="center", va="top", fontsize=7.5, transform=ax.get_xaxis_transform())
        pos += 0.7
    ax.bar(x, vals, color=colors, width=0.68)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.025, pct(val), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate", fontsize=8)
    ax.set_title("E  Residual errors still preserve neighborhood information", loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    despine(ax)


def draw_summary_boxes(ax):
    ax.axis("off")
    ax.set_title("F  Suggested manuscript logic", loc="left", fontsize=9, fontweight="bold")
    boxes = [
        ("False detections", "Non-nodule hard contacts can mimic focal tactile hotspots; missed positives show weak or drifting responses."),
        ("Size errors", "Wrong bins occur when amplitude and morphology decouple under compression; neighboring size range is usually retained."),
        ("Depth errors", "Depth has weaker physical separability and is sensitive to sliding; report it as auxiliary guidance."),
    ]
    y = 0.92
    for title, body in boxes:
        rect = plt.Rectangle((0.02, y - 0.24), 0.96, 0.20, transform=ax.transAxes, facecolor="#F9FAFB", edgecolor="#D1D5DB", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(0.05, y - 0.07, title, transform=ax.transAxes, fontsize=7.7, fontweight="bold", va="top", color=COLORS["dark"])
        ax.text(0.05, y - 0.135, body, transform=ax.transAxes, fontsize=6.8, va="top", color="#374151", wrap=True)
        y -= 0.285


def save_detailed(df: pd.DataFrame, effects: pd.DataFrame, summary: dict, out_dir: Path):
    fig = plt.figure(figsize=(13.4, 7.35), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.985, top=0.84, bottom=0.09, wspace=0.42, hspace=0.50)
    fig.text(0.06, 0.975, "Dataset-level failure mode analysis of V5/R5 tactile inversion", ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(
        0.06,
        0.935,
        "Failure causes are inferred from population-level tactile descriptors; CAM should be used as representative visualization, not standalone proof.",
        ha="left",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )
    ax_a = fig.add_subplot(gs[0, 0])
    draw_stacked(ax_a, summary)
    ax_b = fig.add_subplot(gs[0, 1])
    im = draw_detection_heatmap(ax_b, effects)
    cax = fig.add_axes([0.636, 0.575, 0.008, 0.255])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Hedges g", fontsize=7.5)
    ax_c = fig.add_subplot(gs[0, 2])
    draw_effect_bars(
        ax_c,
        effects,
        "Size error",
        "C  Size-error descriptor shifts",
        ["Amplitude", "Temporal fluct.", "Center contrast", "Spatial spread", "Hotspot radius", "Entropy", "Centroid drift"],
        xlim=(-0.85, 0.85),
    )
    ax_d = fig.add_subplot(gs[1, 0])
    draw_effect_bars(
        ax_d,
        effects,
        "Deep under-call",
        "D  Deep under-call descriptors",
        ["Temporal fluct.", "Amplitude", "Center contrast", "Spatial spread", "Hotspot radius", "Entropy", "Persistence", "Centroid drift"],
        xlim=(-0.9, 0.9),
    )
    ax_e = fig.add_subplot(gs[1, 1])
    draw_tolerance(ax_e, summary)
    ax_f = fig.add_subplot(gs[1, 2])
    draw_summary_boxes(ax_f)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"G_failure_mode_dataset_analysis_detailed.{ext}", dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_compact(effects: pd.DataFrame, summary: dict, out_dir: Path):
    fig = plt.figure(figsize=(7.35, 2.75), facecolor="white")
    gs = fig.add_gridspec(1, 4, left=0.04, right=0.99, top=0.82, bottom=0.19, wspace=0.42, width_ratios=[1.2, 1.05, 1.0, 1.15])
    fig.text(0.04, 0.965, "Dataset-level failure modes", ha="left", va="top", fontsize=7.6, fontweight="bold")
    fig.text(0.04, 0.91, "Errors are linked to tactile descriptors and clinical tolerance, not inferred from CAM alone.", ha="left", va="top", fontsize=5.3, color="#4B5563")

    ax0 = fig.add_subplot(gs[0, 0])
    draw_stacked(ax0, summary, compact=True)
    ax0.set_xlim(0, 105)
    ax0.set_title("A  Outcome spectrum", loc="left", fontsize=6.1, fontweight="bold")
    ax0.set_xlabel("%", fontsize=5.6)
    ax0.tick_params(labelsize=5.4)
    for t in ax0.texts:
        t.set_fontsize(4.7)

    ax1 = fig.add_subplot(gs[0, 1])
    selected = effects[
        effects["feature"].isin(["Amplitude", "Temporal fluct.", "Spatial spread", "Entropy", "Centroid drift"])
        & effects["family"].isin(["Detection FP", "Size error", "Deep under-call"])
    ].copy()
    features = ["Amplitude", "Temporal fluct.", "Spatial spread", "Entropy", "Centroid drift"]
    families = ["Detection FP", "Size error", "Deep under-call"]
    mat = np.zeros((len(features), len(families)))
    for i, feat in enumerate(features):
        for j, fam in enumerate(families):
            mat[i, j] = selected[(selected["feature"] == feat) & (selected["family"] == fam)][
                "hedges_g_failure_minus_reference"
            ].iloc[0]
    ax1.imshow(mat, cmap="RdBu_r", vmin=-2.1, vmax=2.1, aspect="auto")
    ax1.set_xticks(range(len(families)), ["FP", "Size", "Deep"], rotation=35, ha="right")
    ax1.set_yticks(range(len(features)), ["Amp.", "Fluct.", "Spread", "Entropy", "Drift"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax1.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=4.5, color="white" if abs(mat[i, j]) > 1.15 else "#111827")
    ax1.set_title("B  Descriptor shifts", loc="left", fontsize=6.1, fontweight="bold")
    ax1.tick_params(labelsize=5.1)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2 = fig.add_subplot(gs[0, 2])
    metrics = summary["metrics"]
    labels = ["Det\nsens.", "Size\nexact", "Size\nTop-2", "Depth\nexact", "Depth\nTop-2"]
    vals = [
        metrics["detection_sensitivity"],
        metrics["size_exact"],
        metrics["size_top2_contains_true"],
        metrics["depth_exact"],
        metrics["depth_top2_contains_true"],
    ]
    cols = [COLORS["green"], COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["blue"]]
    x = np.arange(len(vals))
    ax2.bar(x, vals, color=cols, width=0.68)
    for xi, val in zip(x, vals):
        ax2.text(xi, val + 0.025, f"{100 * val:.0f}", ha="center", va="bottom", fontsize=4.9)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Rate", fontsize=5.5)
    ax2.set_title("C  Clinical tolerance", loc="left", fontsize=6.1, fontweight="bold")
    ax2.grid(axis="y", color="#E5E7EB", linewidth=0.5)
    despine(ax2)
    ax2.tick_params(labelsize=4.9)

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    ax3.set_title("D  Interpretation", loc="left", fontsize=6.1, fontweight="bold")
    text = (
        "False positives: hard non-nodule contacts mimic focal hotspots.\n\n"
        "Size errors: compression decouples amplitude from footprint; nearby bin is usually retained.\n\n"
        "Depth errors: weak separability plus sliding; use as auxiliary guidance."
    )
    ax3.text(0.0, 0.95, text, ha="left", va="top", fontsize=5.2, color="#111827", linespacing=1.12, wrap=True)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"G_failure_mode_dataset_analysis_compact.{ext}", dpi=360, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    threshold = 0.463
    if RUN_SUMMARY.exists():
        threshold = float(json.loads(RUN_SUMMARY.read_text(encoding="utf-8")).get("threshold", threshold))
    df = add_predictions(pd.read_csv(IN_CSV), threshold)
    summary = build_summary(df)
    effects = build_effect_table(df)

    df.to_csv(OUT / "failure_mode_source_rows_with_predictions.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT / "failure_mode_effect_sizes.csv", index=False, encoding="utf-8-sig")
    (OUT / "failure_mode_counts_and_rates.json").write_text(
        json.dumps({"threshold": threshold, **summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_detailed(df, effects, summary, OUT)
    save_compact(effects, summary, OUT)
    print(f"Wrote failure-mode analysis to {OUT}")


if __name__ == "__main__":
    main()
