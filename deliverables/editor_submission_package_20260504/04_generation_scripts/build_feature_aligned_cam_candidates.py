from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
LATEST = ROOT / "latest_algorithm"
for path in (ROOT, TMP, LATEST):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from build_task_guided_scorecam_overlay import (  # noqa: E402
    CAM_OVERLAY_CMAP,
    TASK_COLORS,
    apply_style,
    build_case,
    choose_device,
    cleanup_ax,
    display_frame,
    draw_cam_heatmap_overlay,
    find_sample,
    load_model,
)
from run_r5_feature_baseline_auc import load_locked_splits, sample_feature_row  # noqa: E402
from task_protocol_v1 import COARSE_DEPTH_ORDER, depth_to_coarse_index  # noqa: E402


OUT = ROOT / "deliverables" / "cam_feature_alignment_20260504"


CASE_DEFS = [
    {
        "letter": "A",
        "title": "Detection cue: true positive",
        "group_key": "1cm大|1.0cm深|3.CSV",
        "end_row": 175,
        "task": "detection",
        "anchor": "Feature anchor: center-border contrast and peak intensity are higher in nodule windows.",
        "observation": "CAM follows a focal high-response hotspot, matching the FEM-guided contrast/amplitude cue.",
        "role": "mechanism_support",
    },
    {
        "letter": "B",
        "title": "Detection cue: pseudo-hotspot FP",
        "group_key": "1.5cm大|2.0cm深|3.CSV",
        "end_row": 583,
        "task": "detection",
        "anchor": "Feature anchor: the detector is intentionally sensitive to localized contrast.",
        "observation": "A non-nodule heterogeneous region creates a nodule-like CAM focus; size/depth readouts can help exclude it.",
        "role": "failure_explanation",
    },
    {
        "letter": "C",
        "title": "Size cue: large footprint",
        "group_key": "1.75cm大|2.5cm深|3.CSV",
        "end_row": 203,
        "task": "size",
        "anchor": "Feature anchor: late contrast and temporal hotspot emergence increase with nodule size.",
        "observation": "Size CAM covers an extended footprint on the late high-response frame, consistent with morphology-based sizing.",
        "role": "mechanism_support",
    },
    {
        "letter": "D",
        "title": "Size error: deformed footprint",
        "group_key": "1.25cm大|1.5cm深|3.CSV",
        "end_row": 249,
        "task": "size",
        "anchor": "Feature anchor: size uses footprint morphology, not only the peak value.",
        "observation": "The size CAM follows a broad/deformed response and overcalls an adjacent bin, which fits the deformation failure mode.",
        "role": "failure_explanation",
    },
    {
        "letter": "E",
        "title": "Depth cue: spread/position",
        "group_key": "1.75cm大|2.5cm深|3.CSV",
        "end_row": 203,
        "task": "depth",
        "anchor": "Feature anchor: hotspot radius and spatial spread weakly increase with depth.",
        "observation": "Depth CAM shifts to the peripheral/spread component; this is consistent but weaker than detection and size.",
        "role": "auxiliary_support",
    },
    {
        "letter": "F",
        "title": "Depth error: weak separability",
        "group_key": "0.5cm大|3.0cm深|3.CSV",
        "end_row": 259,
        "task": "depth",
        "anchor": "Feature anchor: depth effects are modest and strongly affected by contact state.",
        "observation": "Detection and size remain correct, but depth CAM is shifted/multifocal and predicts middle for a deep target.",
        "role": "failure_explanation",
    },
]


FEATURE_SUMMARY = {
    "center_border_contrast_center": "center-border contrast",
    "raw_max_mean": "peak intensity",
    "center_border_contrast_last": "late center-border contrast",
    "window_raw_global_std": "temporal fluctuation",
    "hotspot_radius_max": "max hotspot radius",
    "second_moment_spread_max": "max spatial spread",
}


def truth_text(sample: dict) -> str:
    if int(sample["label"]) == 0:
        return "GT: negative window"
    size = float(sample["size_cm"])
    depth_cm = float(sample["depth_cm"])
    depth = COARSE_DEPTH_ORDER[depth_to_coarse_index(depth_cm)]
    return f"GT: {size:g} cm, {depth} ({depth_cm:g} cm)"


def pred_text(pred: dict) -> str:
    return f"Pred: P={pred['p_det']:.2f}, size={pred['size_reg_cm']:.2f} cm, depth={pred['depth_name']}"


def compact_feature_metrics(records_by_key: dict, sample: dict) -> dict[str, str]:
    if int(sample["label"]) == 0:
        return {}
    row = sample_feature_row(records_by_key, sample)
    out = {}
    for key in FEATURE_SUMMARY:
        value = row.get(key)
        if value is None:
            continue
        out[key] = f"{float(value):.2f}"
    return out


def wrapped(text: str, width: int = 48) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def plot_card(ax, ax_note, case: dict, spec: dict, metrics: dict[str, str]) -> None:
    task = spec["task"]
    raw = case["raw"]
    cam = case["cams"][task]
    frame, cam_frame, frame_idx = display_frame(raw, cam)
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    if cam_frame is not None:
        draw_cam_heatmap_overlay(ax, cam_frame)
    cleanup_ax(ax, edge=TASK_COLORS[task])
    ax.set_title(f"{spec['letter']}. {spec['title']}", fontsize=8.8, fontweight="bold", loc="left", pad=4)
    ax.text(
        0.98,
        0.98,
        task.upper(),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.18", facecolor=TASK_COLORS[task], edgecolor="#111827", linewidth=0.55),
    )
    sample = case["sample"]
    pred = case["pred"]
    metric_line = ""
    if metrics:
        if task == "detection":
            metric_line = (
                f"contrast={metrics.get('center_border_contrast_center', 'NA')}; "
                f"peak={metrics.get('raw_max_mean', 'NA')}"
            )
        elif task == "size":
            metric_line = (
                f"late contrast={metrics.get('center_border_contrast_last', 'NA')}; "
                f"temporal std={metrics.get('window_raw_global_std', 'NA')}"
            )
        else:
            metric_line = (
                f"radius={metrics.get('hotspot_radius_max', 'NA')}; "
                f"spread={metrics.get('second_moment_spread_max', 'NA')}"
            )
    ax_note.axis("off")
    ax_note.text(
        0.0,
        0.98,
        f"{truth_text(sample)}\n{pred_text(pred)}",
        transform=ax_note.transAxes,
        ha="left",
        va="top",
        fontsize=5.85,
        color="#111827",
        linespacing=1.08,
        fontweight="bold",
    )
    y = 0.58
    if metric_line:
        ax_note.text(
            0.0,
            y,
            f"Frame {frame_idx + 1}; {metric_line}",
            transform=ax_note.transAxes,
            ha="left",
            va="top",
            fontsize=5.35,
            color="#374151",
        )
        y -= 0.18
    ax_note.text(
        0.0,
        y,
        wrapped(spec["anchor"].replace("Feature anchor: ", "Feature: "), 50),
        transform=ax_note.transAxes,
        ha="left",
        va="top",
        fontsize=5.18,
        color="#1F2937",
        linespacing=1.10,
    )
    ax_note.text(
        0.0,
        0.04,
        wrapped(spec["observation"].replace("CAM ", "CAM: ", 1), 50),
        transform=ax_note.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.18,
        color="#1F2937",
        linespacing=1.10,
    )


def write_mapping_csv(rows: list[dict]) -> None:
    fields = [
        "letter",
        "title",
        "role",
        "task_cam",
        "group_key",
        "end_row",
        "ground_truth",
        "prediction",
        "feature_anchor",
        "cam_observation",
        *FEATURE_SUMMARY.keys(),
    ]
    with (OUT / "CAM_feature_alignment_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.titlesize": 8.4,
            "savefig.dpi": 420,
        }
    )
    device = choose_device()
    model, threshold, detector_run = load_model(device)
    manifest = json.loads((detector_run / "manifest.json").read_text(encoding="utf-8"))
    _train_records, _val_records, test_records, _train_all, _train_det, _val_samples, test_samples = load_locked_splits(manifest)

    built = []
    rows = []
    for spec in CASE_DEFS:
        sample = find_sample(test_samples, spec["group_key"], spec["end_row"])
        case = build_case(model, test_records, sample, spec["title"], device)
        metrics = compact_feature_metrics(test_records, sample)
        built.append((spec, case, metrics))
        pred = case["pred"]
        row = {
            "letter": spec["letter"],
            "title": spec["title"],
            "role": spec["role"],
            "task_cam": spec["task"],
            "group_key": spec["group_key"],
            "end_row": int(spec["end_row"]),
            "ground_truth": truth_text(sample),
            "prediction": pred_text(pred),
            "feature_anchor": spec["anchor"],
            "cam_observation": spec["observation"],
        }
        row.update(metrics)
        rows.append(row)
        print(f"{spec['letter']} {spec['title']}: {spec['group_key']} end={spec['end_row']} | {pred_text(pred)}")

    fig = plt.figure(figsize=(10.8, 8.2))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.985,
        top=0.86,
        bottom=0.055,
        wspace=0.25,
        hspace=0.34,
    )
    fig.text(
        0.045,
        0.965,
        "Feature-aligned CAM candidates for the V5/R5 explainability loop",
        ha="left",
        va="top",
        fontsize=14.0,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.925,
        "Selection rule: each CAM is tied to a prior FEM/experimental descriptor conclusion; failure examples are used only as mechanism checks.",
        ha="left",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )
    fig.text(
        0.045,
        0.895,
        f"Method: task-guided Score-CAM from the last spatial activation layer; CAM heatmap = {CAM_OVERLAY_CMAP}; gate threshold={threshold:.3f}.",
        ha="left",
        va="top",
        fontsize=6.8,
        color="#4B5563",
    )
    for idx, (spec, case, metrics) in enumerate(built):
        sub = gs[idx // 3, idx % 3].subgridspec(2, 1, height_ratios=[1.0, 0.52], hspace=0.05)
        ax = fig.add_subplot(sub[0, 0])
        ax_note = fig.add_subplot(sub[1, 0])
        plot_card(ax, ax_note, case, spec, metrics)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"CAM_feature_aligned_candidates.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    write_mapping_csv(rows)
    (OUT / "CAM_feature_alignment_notes.md").write_text(
        "\n".join(
            [
                "# CAM feature-alignment notes",
                "",
                "These examples were selected after checking the prior feature-analysis conclusions.",
                "",
                "- Detection: use true-positive focal CAM as support; false-positive pseudo-hotspot as a failure counterpart.",
                "- Size: use large-footprint size CAM as support; deformed footprint as adjacent-bin failure.",
                "- Depth: use spread/position CAM only as auxiliary support; depth failure shows why the depth endpoint should be described as weak/effective tactile depth.",
                "",
                "Single-frame CAM cannot directly prove temporal fluctuation. Temporal fluctuation should be supported by the descriptor/probe/perturbation analysis and, if needed, by a short temporal CAM strip.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote feature-aligned CAM candidates to {OUT}")


if __name__ == "__main__":
    main()
