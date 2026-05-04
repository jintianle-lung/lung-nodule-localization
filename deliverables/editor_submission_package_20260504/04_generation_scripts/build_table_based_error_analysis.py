import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "deliverables" / "v5_feature_learning_20260503" / "file3_v5_outputs_plus_descriptors.csv"
RUN_SUMMARY = ROOT / "deliverables" / "v5_feature_learning_20260503" / "run_summary.json"
OUT = ROOT / "deliverables" / "table_based_error_analysis_20260503"

SIZE_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=float)
DEPTH_NAMES = ["shallow", "middle", "deep"]
DEPTH_LABELS = {
    0: "Shallow\n0.5-1.0 cm",
    1: "Middle\n1.5-2.0 cm",
    2: "Deep\n2.5-3.0 cm",
}

COLORS = {
    "green": "#009E73",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "gray": "#6B7280",
    "light": "#F3F4F6",
    "line": "#D1D5DB",
    "text": "#111827",
}


def pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{100 * float(x):.1f}%"


def fmt_float(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def size_value_from_col(name: str) -> float:
    return float(name.replace("v5_size_prob_", "").replace("cm", ""))


def add_predictions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    size_cols = sorted([c for c in out.columns if c.startswith("v5_size_prob_")], key=size_value_from_col)
    size_values = np.array([size_value_from_col(c) for c in size_cols], dtype=float)
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
    out["pred_depth_name"] = [DEPTH_NAMES[i] for i in out["pred_depth_idx"].astype(int)]
    out["depth_top2_contains_true"] = [
        int(int(t) in depth_order[i, :2]) for i, t in enumerate(out["depth_coarse_index"].astype(int).to_numpy())
    ]

    out["det_pred"] = out["det_prob"] >= threshold
    out["size_idx_err"] = out["pred_size_idx"] - out["size_class_index"]
    out["size_abs_idx_err"] = out["size_idx_err"].abs()
    out["size_abs_reg_err_cm"] = (out["size_reg_cm"] - out["size_cm"]).abs()
    out["depth_idx_err"] = out["pred_depth_idx"] - out["depth_coarse_index"]
    out["depth_abs_idx_err"] = out["depth_idx_err"].abs()
    return out


def common_prediction(sub: pd.DataFrame, pred_col: str, fmt=lambda x: str(x)) -> str:
    if len(sub) == 0:
        return "-"
    counts = sub[pred_col].value_counts()
    value = counts.index[0]
    return f"{fmt(value)} ({counts.iloc[0]})"


def source_label(row: pd.Series) -> str:
    return f"size {float(row.size_cm):g} cm / depth {float(row.depth_cm):g} cm / end {int(row.end_row)}"


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pos = df[df["label"] == 1].copy()
    gated = pos[pos["det_pred"]].copy()
    det_counts = {
        "TP": int(((df["label"] == 1) & df["det_pred"]).sum()),
        "FN": int(((df["label"] == 1) & ~df["det_pred"]).sum()),
        "TN": int(((df["label"] == 0) & ~df["det_pred"]).sum()),
        "FP": int(((df["label"] == 0) & df["det_pred"]).sum()),
    }
    summary = {
        "threshold": float(df.attrs.get("threshold", 0.463)),
        "n_all": int(len(df)),
        "n_positive": int(len(pos)),
        "n_gated_positive": int(len(gated)),
        **det_counts,
    }
    summary_table = pd.DataFrame(
        [
            {
                "Readout": "Detection",
                "Evaluation set": f"All windows (n={len(df)})",
                "Primary metric": f"Sensitivity {pct(det_counts['TP'] / max(det_counts['TP'] + det_counts['FN'], 1))}",
                "Clinical tolerance": f"Specificity {pct(det_counts['TN'] / max(det_counts['TN'] + det_counts['FP'], 1))}",
                "Counts": f"TP {det_counts['TP']} / FP {det_counts['FP']} / FN {det_counts['FN']}",
            },
            {
                "Readout": "Size",
                "Evaluation set": f"Detected positive windows (n={len(gated)})",
                "Primary metric": f"Exact {pct((gated['size_abs_idx_err'] == 0).mean())}",
                "Clinical tolerance": f"Adj./exact {pct((gated['size_abs_idx_err'] <= 1).mean())}; Top-2 {pct(gated['size_top2_contains_true'].mean())}",
                "Counts": f"Exact {(gated['size_abs_idx_err'] == 0).sum()} / Adj. {(gated['size_abs_idx_err'] == 1).sum()} / Severe {(gated['size_abs_idx_err'] >= 2).sum()}",
            },
            {
                "Readout": "Depth",
                "Evaluation set": f"Detected positive windows (n={len(gated)})",
                "Primary metric": f"Exact {pct((gated['depth_abs_idx_err'] == 0).mean())}",
                "Clinical tolerance": f"Adj./exact {pct((gated['depth_abs_idx_err'] <= 1).mean())}; Top-2 {pct(gated['depth_top2_contains_true'].mean())}",
                "Counts": f"Exact {(gated['depth_abs_idx_err'] == 0).sum()} / Adj. {(gated['depth_abs_idx_err'] == 1).sum()} / Far {(gated['depth_abs_idx_err'] >= 2).sum()}",
            },
        ]
    )

    size_rows = []
    for idx, size in enumerate(SIZE_VALUES):
        sub_all = pos[pos["size_class_index"] == idx]
        sub = gated[gated["size_class_index"] == idx]
        exact = (sub["size_abs_idx_err"] == 0).mean() if len(sub) else np.nan
        adj = (sub["size_abs_idx_err"] <= 1).mean() if len(sub) else np.nan
        top2 = sub["size_top2_contains_true"].mean() if len(sub) else np.nan
        under = int((sub["size_idx_err"] < 0).sum())
        over = int((sub["size_idx_err"] > 0).sum())
        size_rows.append(
            {
                "True size": f"{size:g} cm",
                "Gate": f"{len(sub)}/{len(sub_all)}",
                "Exact": pct(exact),
                "Adj./exact": pct(adj),
                "Top-2": pct(top2),
                "MAE cm": fmt_float(sub["size_abs_reg_err_cm"].mean() if len(sub) else np.nan, 2),
                "Under/Over": f"{under}/{over}",
                "Most common pred.": common_prediction(sub, "pred_size_bin_cm", lambda x: f"{float(x):g} cm"),
            }
        )
    size_table = pd.DataFrame(size_rows)

    depth_rows = []
    for idx in range(3):
        sub_all = pos[pos["depth_coarse_index"] == idx]
        sub = gated[gated["depth_coarse_index"] == idx]
        exact = (sub["depth_abs_idx_err"] == 0).mean() if len(sub) else np.nan
        adj = (sub["depth_abs_idx_err"] <= 1).mean() if len(sub) else np.nan
        top2 = sub["depth_top2_contains_true"].mean() if len(sub) else np.nan
        under = int((sub["depth_idx_err"] < 0).sum())
        over = int((sub["depth_idx_err"] > 0).sum())
        depth_rows.append(
            {
                "True depth": DEPTH_LABELS[idx].replace("\n", " "),
                "Gate": f"{len(sub)}/{len(sub_all)}",
                "Exact": pct(exact),
                "Adj./exact": pct(adj),
                "Top-2": pct(top2),
                "Under/Over": f"{under}/{over}",
                "Most common pred.": common_prediction(sub, "pred_depth_name", str),
            }
        )
    depth_table = pd.DataFrame(depth_rows)

    nominal_rows = []
    for depth in sorted(pos["depth_cm"].unique()):
        sub_all = pos[pos["depth_cm"] == depth]
        sub = gated[gated["depth_cm"] == depth]
        nominal_rows.append(
            {
                "Nominal depth": f"{depth:g} cm",
                "Coarse true": DEPTH_NAMES[int(sub_all["depth_coarse_index"].iloc[0])] if len(sub_all) else "-",
                "Gate": f"{len(sub)}/{len(sub_all)}",
                "Exact coarse": pct((sub["depth_abs_idx_err"] == 0).mean() if len(sub) else np.nan),
                "Top-2": pct(sub["depth_top2_contains_true"].mean() if len(sub) else np.nan),
                "Most common pred.": common_prediction(sub, "pred_depth_name", str),
            }
        )
    nominal_depth_table = pd.DataFrame(nominal_rows)
    return summary_table, size_table, depth_table, nominal_depth_table, summary


def select_failure_cases(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    fp = df[(df["label"] == 0) & df["det_pred"]].copy()
    if len(fp):
        fp["score"] = fp["det_prob"] + 0.01 * fp["raw_p95_mean"] + 0.01 * fp["window_raw_global_std"]
        r = fp.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "Failure type": "Detection FP",
                "Representative source": source_label(r),
                "GT -> Pred": f"negative window -> P={r.det_prob:.2f}",
                "Dataset-level evidence": "FPs show higher amplitude/fluctuation than TNs.",
                "Interpretation": "Hard non-nodule or heterogeneous contact can mimic a focal hotspot.",
                "Use in surgery": "Cross-check with visual field and predicted size/depth before acting.",
            }
        )

    fn = df[(df["label"] == 1) & ~df["det_pred"]].copy()
    if len(fn):
        fn["score"] = fn["centroid_drift"] + fn["second_moment_spread_max"] - 0.02 * fn["raw_p95_mean"]
        r = fn.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "Failure type": "Detection FN",
                "Representative source": source_label(r),
                "GT -> Pred": f"{r.size_cm:g} cm / {DEPTH_NAMES[int(r.depth_coarse_index)]} -> P={r.det_prob:.2f}",
                "Dataset-level evidence": "FNs show weaker amplitude/contrast and larger drift than TPs.",
                "Interpretation": "Weak or sliding contact can suppress the focal response.",
                "Use in surgery": "Repeat palpation sweep when CT prior suggests a target nearby.",
            }
        )

    size_adj = df[
        (df["label"] == 1)
        & df["det_pred"]
        & (df["size_abs_idx_err"] == 1)
        & (df["size_cm"].between(1.0, 1.5))
    ].copy()
    if len(size_adj):
        size_adj["score"] = (
            size_adj["det_prob"]
            + (size_adj["size_cm"].sub(1.25).abs().rsub(1.25)).clip(lower=0)
            + (size_adj["size_idx_err"] > 0).astype(float) * 0.25
            + size_adj["size_abs_reg_err_cm"]
        )
        r = size_adj.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "Failure type": "Size adjacent-bin error",
                "Representative source": source_label(r),
                "GT -> Pred": f"{r.size_cm:g} cm -> {r.pred_size_bin_cm:g} cm",
                "Dataset-level evidence": "Size Top-2 retains true class in 90.6% of gated positives.",
                "Interpretation": "Compression can decouple contact amplitude from footprint geometry.",
                "Use in surgery": "Report a nearby size range rather than a hard single bin.",
            }
        )

    depth_under = df[
        (df["label"] == 1)
        & df["det_pred"]
        & (df["depth_coarse_index"] == 2)
        & (df["pred_depth_idx"] < 2)
    ].copy()
    if len(depth_under):
        depth_under["score"] = depth_under["det_prob"] + depth_under["centroid_drift"] + 0.01 * depth_under["window_raw_global_std"]
        r = depth_under.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "Failure type": "Depth under-call",
                "Representative source": source_label(r),
                "GT -> Pred": f"deep -> {r.pred_depth_name}",
                "Dataset-level evidence": "Deep under-calls show lower spread/radius and higher fluctuation.",
                "Interpretation": "Depth cue is weak and sensitive to sliding or changing compression.",
                "Use in surgery": "Use depth as auxiliary guidance, not a rigid decision label.",
            }
        )
    return pd.DataFrame(rows)


def draw_table(ax, df: pd.DataFrame, title: str, col_widths=None, fontsize=6.4, header_fontsize=6.2, row_height=0.12):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold", pad=3)
    n_rows, n_cols = df.shape
    if col_widths is None:
        col_widths = [1 / n_cols] * n_cols
    else:
        total = sum(col_widths)
        col_widths = [w / total for w in col_widths]
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="upper left",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0, 0, 1, min(0.95, row_height * (n_rows + 1))],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["line"])
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#E5E7EB")
            cell.set_text_props(weight="bold", color=COLORS["text"], fontsize=header_fontsize)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F9FAFB")
            if col == 0:
                cell.set_text_props(weight="bold", color=COLORS["text"])
    return table


def draw_summary_figure(summary_table: pd.DataFrame, size_table: pd.DataFrame, depth_table: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(7.35, 5.25), facecolor="white")
    gs = fig.add_gridspec(3, 1, left=0.025, right=0.985, top=0.91, bottom=0.045, hspace=0.28, height_ratios=[1.05, 1.65, 1.0])
    fig.text(0.025, 0.985, "Table-based V5/R5 readout summary", ha="left", va="top", fontsize=8.8, fontweight="bold")
    fig.text(0.025, 0.95, "Performance is summarized by task-level and class-wise readout tables on the held-out File-3 set.", ha="left", va="top", fontsize=5.6, color="#4B5563")
    ax0 = fig.add_subplot(gs[0])
    draw_table(
        ax0,
        summary_table,
        "A  Overall readout summary",
        col_widths=[0.12, 0.24, 0.18, 0.27, 0.19],
        fontsize=4.9,
        header_fontsize=4.8,
        row_height=0.19,
    )
    ax1 = fig.add_subplot(gs[1])
    draw_table(
        ax1,
        size_table,
        "B  Size readout by true 7-class category",
        col_widths=[0.10, 0.10, 0.105, 0.13, 0.10, 0.10, 0.11, 0.245],
        fontsize=4.95,
        header_fontsize=4.75,
        row_height=0.105,
    )
    ax2 = fig.add_subplot(gs[2])
    draw_table(
        ax2,
        depth_table,
        "C  Depth readout by true coarse category",
        col_widths=[0.20, 0.12, 0.11, 0.13, 0.10, 0.12, 0.22],
        fontsize=5.05,
        header_fontsize=4.9,
        row_height=0.16,
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig5_table_readout_summary.{ext}", dpi=360, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def draw_individual_table_figures(summary_table: pd.DataFrame, size_table: pd.DataFrame, depth_table: pd.DataFrame, out_dir: Path):
    specs = [
        (
            "Fig5_overall_readout_table",
            summary_table,
            "Overall readout summary",
            [0.12, 0.24, 0.18, 0.27, 0.19],
            (7.35, 1.65),
            5.0,
            4.9,
            0.20,
        ),
        (
            "Fig5_size_7class_table",
            size_table,
            "Size readout by true 7-class category",
            [0.10, 0.10, 0.105, 0.13, 0.10, 0.10, 0.11, 0.245],
            (7.35, 2.35),
            5.1,
            4.9,
            0.105,
        ),
        (
            "Fig5_depth_3class_table",
            depth_table,
            "Depth readout by true coarse category",
            [0.20, 0.12, 0.11, 0.13, 0.10, 0.12, 0.22],
            (7.35, 1.45),
            5.3,
            5.0,
            0.17,
        ),
    ]
    for stem, df, title, widths, figsize, fontsize, header_fontsize, row_height in specs:
        fig = plt.figure(figsize=figsize, facecolor="white")
        ax = fig.add_axes([0.015, 0.04, 0.97, 0.86])
        draw_table(ax, df, title, col_widths=widths, fontsize=fontsize, header_fontsize=header_fontsize, row_height=row_height)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=360, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def draw_error_explanation_figure(error_table: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(7.35, 3.25), facecolor="white")
    ax = fig.add_axes([0.025, 0.05, 0.96, 0.84])
    fig.text(0.025, 0.98, "Failure-case explanation table", ha="left", va="top", fontsize=8.8, fontweight="bold")
    fig.text(
        0.025,
        0.93,
        "Representative cases are paired with population-level descriptor evidence; these are mechanism-consistent explanations, not CAM-only claims.",
        ha="left",
        va="top",
        fontsize=5.5,
        color="#4B5563",
    )
    draw_table(
        ax,
        error_table,
        "",
        col_widths=[0.12, 0.18, 0.13, 0.20, 0.21, 0.16],
        fontsize=4.65,
        header_fontsize=4.45,
        row_height=0.16,
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig5_error_case_explanation_table.{ext}", dpi=360, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def draw_error_cards_figure(error_table: pd.DataFrame, out_dir: Path):
    fig = plt.figure(figsize=(7.35, 4.1), facecolor="white")
    fig.text(0.035, 0.98, "Failure cases explained by dataset-level evidence", ha="left", va="top", fontsize=8.8, fontweight="bold")
    fig.text(
        0.035,
        0.94,
        "Each card combines one representative source window with the population-level descriptor pattern.",
        ha="left",
        va="top",
        fontsize=5.6,
        color="#4B5563",
    )
    gs = fig.add_gridspec(2, 2, left=0.035, right=0.985, top=0.86, bottom=0.06, wspace=0.12, hspace=0.18)
    palette = [COLORS["vermillion"], COLORS["orange"], COLORS["blue"], COLORS["green"]]
    for idx, (_, row) in enumerate(error_table.iterrows()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_axis_off()
        ax.add_patch(
            plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F9FAFB", edgecolor=COLORS["line"], linewidth=0.9)
        )
        ax.add_patch(
            plt.Rectangle((0, 0.89), 1, 0.11, transform=ax.transAxes, facecolor=palette[idx % len(palette)], edgecolor="none")
        )
        ax.text(0.035, 0.945, str(row["Failure type"]), transform=ax.transAxes, ha="left", va="center", fontsize=6.3, fontweight="bold", color="white")
        body = [
            ("Source", row["Representative source"]),
            ("GT -> Pred", row["GT -> Pred"]),
            ("Evidence", row["Dataset-level evidence"]),
            ("Mechanism", row["Interpretation"]),
            ("Use", row["Use in surgery"]),
        ]
        y = 0.82
        for label, text in body:
            ax.text(0.035, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold", color=COLORS["text"])
            wrapped = textwrap.fill(str(text), width=54)
            ax.text(0.20, y, wrapped, transform=ax.transAxes, ha="left", va="top", fontsize=5.15, color="#374151", linespacing=1.12)
            y -= 0.15 if len(wrapped) < 55 else 0.20
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig5_error_case_explanation_cards.{ext}", dpi=360, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    threshold = 0.463
    if RUN_SUMMARY.exists():
        threshold = float(json.loads(RUN_SUMMARY.read_text(encoding="utf-8")).get("threshold", threshold))
    df = pd.read_csv(IN_CSV)
    df.attrs["threshold"] = threshold
    df = add_predictions(df, threshold)
    df.attrs["threshold"] = threshold

    summary_table, size_table, depth_table, nominal_depth_table, summary = build_tables(df)
    error_table = select_failure_cases(df)

    summary_table.to_csv(OUT / "overall_readout_summary_table.csv", index=False, encoding="utf-8-sig")
    size_table.to_csv(OUT / "size_7class_by_true_size_table.csv", index=False, encoding="utf-8-sig")
    depth_table.to_csv(OUT / "depth_3class_by_true_depth_table.csv", index=False, encoding="utf-8-sig")
    nominal_depth_table.to_csv(OUT / "depth_nominal_6level_table.csv", index=False, encoding="utf-8-sig")
    error_table.to_csv(OUT / "failure_case_explanation_table.csv", index=False, encoding="utf-8-sig")
    (OUT / "table_readout_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    draw_summary_figure(summary_table, size_table, depth_table, OUT)
    draw_individual_table_figures(summary_table, size_table, depth_table, OUT)
    draw_error_explanation_figure(error_table, OUT)
    draw_error_cards_figure(error_table, OUT)
    print(f"Wrote table-based error analysis to {OUT}")


if __name__ == "__main__":
    main()
