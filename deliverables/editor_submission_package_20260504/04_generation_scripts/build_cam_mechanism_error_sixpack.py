from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
LATEST = ROOT / "latest_algorithm"
for path in (ROOT, TMP, LATEST):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from build_cam_oldstyle_temporal import (  # noqa: E402
    CAM_CMAP,
    PEAK_FRAME_COLOR,
    build_case_for_display,
    cam_frame_relevance,
    find_sample,
    keyframe_indices,
    load_model,
    normalize01,
    upsample_frames,
)
from run_r5_feature_baseline_auc import load_locked_splits  # noqa: E402


OUT = ROOT / "deliverables" / "cam_feature_alignment_20260504"


PANELS = [
    {
        "panel": "A",
        "row_group": "Mechanism examples",
        "label": "Detection learns hotspot",
        "group_key": "1cm大|1.0cm深|3.CSV",
        "end_row": 175,
        "task": "detection",
        "depth_target": "true",
        "caption": "GT: 1.0 cm shallow | Pred: 1.00 cm shallow\nCAM locks onto a focal high-response hotspot.\nMatches contrast + peak-intensity descriptors.",
    },
    {
        "panel": "B",
        "row_group": "Mechanism examples",
        "label": "Size learns contour",
        "group_key": "1.75cm大|2.5cm深|3.CSV",
        "end_row": 203,
        "task": "size",
        "depth_target": "true",
        "caption": "GT: 1.75 cm deep | Pred: 1.75 cm deep\nCAM follows the footprint/contour rather than one pixel.\nMatches morphology + late-contrast descriptors.",
    },
    {
        "panel": "C",
        "row_group": "Mechanism examples",
        "label": "Depth uses diffusion",
        "group_key": "1.75cm大|2.5cm深|3.CSV",
        "end_row": 203,
        "task": "depth",
        "depth_target": "true",
        "caption": "GT: deep | Pred: deep\nDepth CAM emphasizes broader spread and position.\nConsistent with weak radius/spread descriptors.",
    },
    {
        "panel": "D",
        "row_group": "Residual error examples",
        "label": "Detection error: heterogeneity",
        "group_key": "1.75cm大|2cm深|3.CSV",
        "end_row": 123,
        "task": "detection",
        "depth_target": "pred",
        "caption": "GT: negative window | Pred: P>gate, 1.25 cm\nDetected hotspot differs from the planted 1.75 cm group.\nLikely solid/bronchus-like heterogeneity; cross-check can exclude.",
    },
    {
        "panel": "E",
        "row_group": "Residual error examples",
        "label": "Size error: deformation",
        "group_key": "1.25cm大|1.5cm深|3.CSV",
        "end_row": 249,
        "task": "size",
        "depth_target": "true",
        "caption": "GT: 1.25 cm | Pred: 1.49 -> 1.5 cm\nCAM expands over a broad deformed footprint.\nAdjacent-bin error is deformation-consistent.",
    },
    {
        "panel": "F",
        "row_group": "Residual error examples",
        "label": "Depth error: weak separability",
        "group_key": "0.5cm大|3.0cm深|3.CSV",
        "end_row": 259,
        "task": "depth",
        "depth_target": "pred",
        "caption": "GT: 0.5 cm deep | Pred: 0.51 cm middle\nDetection and size remain correct, but depth CAM is shifted.\nSuggests weak spread cue and contact/sliding ambiguity.",
    },
]


SINGLE_PANEL_META = {
    "A": {
        "file": "A_detection_hotspot",
        "short_title": "Detection: hotspot",
        "status": "Cue: focal high-response hotspot",
        "gt_pred": "GT 1.0 cm shallow | Pred 1.00 cm shallow",
    },
    "B": {
        "file": "B_size_contour",
        "short_title": "Size: contour",
        "status": "Cue: footprint / contour",
        "gt_pred": "GT 1.75 cm deep | Pred 1.75 cm deep",
    },
    "C": {
        "file": "C_depth_diffusion",
        "short_title": "Depth: diffusion",
        "status": "Cue: broader spread / position",
        "gt_pred": "GT deep | Pred deep",
    },
    "D": {
        "file": "D_detection_heterogeneity",
        "short_title": "Detection error",
        "status": "Error cue: heterogeneous pseudo-hotspot",
        "gt_pred": "GT negative | Pred 1.25 cm",
    },
    "E": {
        "file": "E_size_deformation",
        "short_title": "Size error",
        "status": "Error cue: deformed footprint",
        "gt_pred": "GT 1.25 cm | Pred 1.49 cm",
    },
    "F": {
        "file": "F_depth_weak_separability",
        "short_title": "Depth error",
        "status": "Error cue: weak spread separability",
        "gt_pred": "GT 0.5 cm deep | Pred middle",
    },
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 380,
        }
    )


def render_cam_triplet(axs_raw, axs_cam, case: dict, task: str, mark_peak: bool = True) -> list[int]:
    raw_up = upsample_frames(normalize01(case["raw"]))
    cam_up = upsample_frames(normalize01(case["cams"][task]))
    frames = keyframe_indices(cam_up, n_frames=3)
    peak = int(np.argmax(cam_frame_relevance(cam_up)))
    for j, frame_idx in enumerate(frames):
        raw = raw_up[frame_idx]
        cam = normalize01(cam_up[frame_idx])
        ax_raw = axs_raw[j]
        ax_cam = axs_cam[j]
        ax_raw.imshow(raw, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax_cam.imshow(raw, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax_cam.imshow(cam, cmap=CAM_CMAP, vmin=0, vmax=1, alpha=np.clip((cam**0.78) * 0.90, 0.0, 0.90), interpolation="nearest")
        ax_raw.set_title(f"F{frame_idx + 1}", fontsize=5.4, pad=1.1)
        if j == 0:
            ax_raw.set_ylabel("Raw", fontsize=5.4, labelpad=2.0)
            ax_cam.set_ylabel("CAM", fontsize=5.4, labelpad=2.0)
        for ax in (ax_raw, ax_cam):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("black")
        if mark_peak and frame_idx == peak:
            for ax in (ax_raw, ax_cam):
                ax.add_patch(Rectangle((-0.5, -0.5), raw_up.shape[2], raw_up.shape[1], fill=False, ec=PEAK_FRAME_COLOR, lw=1.25))
    return frames


def source_label(group_key: str, end_row: int) -> str:
    size_match = re.search(r"([0-9.]+)cm大", group_key)
    depth_match = re.search(r"\|([0-9.]+)cm深", group_key)
    file_match = re.search(r"\|([^|]+\.CSV)$", group_key, flags=re.IGNORECASE)
    size = size_match.group(1) if size_match else "NA"
    depth = depth_match.group(1) if depth_match else "NA"
    file_name = file_match.group(1).replace(".CSV", "") if file_match else "file"
    return f"Sample: group {size} cm / {depth} cm, {file_name}, end {end_row}"


def draw_panel(cases: list[tuple[dict, dict]], out_stem: Path) -> None:
    fig = plt.figure(figsize=(9.2, 5.55), facecolor="white")
    outer = fig.add_gridspec(2, 3, left=0.025, right=0.99, top=0.84, bottom=0.055, wspace=0.16, hspace=0.24)
    fig.text(0.025, 0.985, "Task-guided CAM closes the feature loop and explains residual errors", ha="left", va="top", fontsize=9.8, fontweight="bold")
    fig.text(0.025, 0.952, "Gray: raw tactile frame. Jet overlay: task-specific CAM. Red box: highest CAM-relevance frame.", ha="left", va="top", fontsize=5.8, color="#4B5563")
    fig.text(0.025, 0.875, "Mechanism examples", ha="left", va="center", fontsize=6.4, fontweight="bold", color="#111827")
    fig.text(0.025, 0.452, "Residual error examples", ha="left", va="center", fontsize=6.4, fontweight="bold", color="#111827")

    source_rows = []
    for idx, (spec, case) in enumerate(cases):
        r = idx // 3
        c = idx % 3
        cell = outer[r, c].subgridspec(4, 3, height_ratios=[0.22, 1.0, 1.0, 0.62], hspace=0.13, wspace=0.09)
        title_ax = fig.add_subplot(cell[0, :])
        title_ax.axis("off")
        title_ax.text(0.5, 0.60, spec["label"], ha="center", va="center", fontsize=6.85, fontweight="bold")

        raw_axes = [fig.add_subplot(cell[1, j]) for j in range(3)]
        cam_axes = [fig.add_subplot(cell[2, j]) for j in range(3)]
        frames = render_cam_triplet(raw_axes, cam_axes, case, spec["task"], mark_peak=True)

        cap_ax = fig.add_subplot(cell[3, :])
        cap_ax.axis("off")
        pred = case["pred"]
        caption = (
            f"{source_label(spec['group_key'], int(spec['end_row']))}\n"
            f"{spec['caption']}"
        )
        cap_ax.text(0.0, 0.98, caption, ha="left", va="top", fontsize=3.9, linespacing=1.04, color="#111827")
        source_rows.append(
            {
                "panel": spec["panel"],
                "label": spec["label"],
                "row_group": spec["row_group"],
                "task_cam": spec["task"],
                "group_key": spec["group_key"],
                "end_row": int(spec["end_row"]),
                "frames": [int(v + 1) for v in frames],
                "pred_p_det": float(pred["p_det"]),
                "pred_size_reg_cm": float(pred["size_reg_cm"]),
                "pred_depth": str(pred["depth_name"]),
                "caption": spec["caption"],
            }
        )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    (out_stem.with_name(out_stem.name + "_sources.json")).write_text(json.dumps(source_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_single_panels(cases: list[tuple[dict, dict]], out_dir: Path) -> None:
    single_dir = out_dir / "single_cam_panels"
    single_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for spec, case in cases:
        meta = SINGLE_PANEL_META[spec["panel"]]
        raw_up = upsample_frames(normalize01(case["raw"]))
        cam_up = upsample_frames(normalize01(case["cams"][spec["task"]]))
        frames = keyframe_indices(cam_up, n_frames=3)
        peak = int(np.argmax(cam_frame_relevance(cam_up)))

        fig = plt.figure(figsize=(3.05, 2.38), facecolor="white")
        fig.text(0.5, 0.980, meta["short_title"], ha="center", va="top", fontsize=7.35, fontweight="bold", color="#000000")
        fig.text(0.5, 0.918, meta["status"], ha="center", va="top", fontsize=4.85, color="#4B5563")
        gs = fig.add_gridspec(
            3,
            3,
            height_ratios=[1.0, 1.0, 0.20],
            left=0.10,
            right=0.985,
            top=0.835,
            bottom=0.080,
            hspace=0.13,
            wspace=0.10,
        )

        for j, frame_idx in enumerate(frames):
            raw = raw_up[frame_idx]
            cam = normalize01(cam_up[frame_idx])
            ax_raw = fig.add_subplot(gs[0, j])
            ax_cam = fig.add_subplot(gs[1, j])
            ax_raw.imshow(raw, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax_cam.imshow(raw, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax_cam.imshow(cam, cmap=CAM_CMAP, vmin=0, vmax=1, alpha=np.clip((cam**0.78) * 0.90, 0.0, 0.90), interpolation="nearest")
            if j == 0:
                ax_raw.set_ylabel("Raw", fontsize=5.4, labelpad=2.1)
                ax_cam.set_ylabel("CAM", fontsize=5.4, labelpad=2.1)
            for ax in (ax_raw, ax_cam):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.45)
                    spine.set_color("black")
            if frame_idx == peak:
                for ax in (ax_raw, ax_cam):
                    ax.add_patch(Rectangle((-0.5, -0.5), raw_up.shape[2], raw_up.shape[1], fill=False, ec=PEAK_FRAME_COLOR, lw=1.25))

        cap_ax = fig.add_subplot(gs[2, :])
        cap_ax.axis("off")
        fig.text(0.5, 0.062, meta["gt_pred"], ha="center", va="top", fontsize=4.95, color="#111827")

        out_stem = single_dir / f"G_cam_{meta['file']}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.018)
        plt.close(fig)
        source_rows.append(
            {
                "panel": spec["panel"],
                "file_stem": str(out_stem),
                "title": meta["short_title"],
                "task_cam": spec["task"],
                "group_key": spec["group_key"],
                "end_row": int(spec["end_row"]),
                "frames": [int(frame + 1) for frame in frames],
                "peak_frame": int(peak + 1),
                "gt_pred": meta["gt_pred"],
                "short_annotation": meta["status"],
            }
        )
    (single_dir / "single_cam_panel_sources.json").write_text(json.dumps(source_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _threshold, detector_run = load_model(device)
    manifest = json.loads((detector_run / "manifest.json").read_text(encoding="utf-8"))
    _train_records, _val_records, test_records, _train_all, _train_det, _val_samples, test_samples = load_locked_splits(manifest)

    cases = []
    for spec in PANELS:
        sample = find_sample(test_samples, spec["group_key"], int(spec["end_row"]))
        case = build_case_for_display(model, test_records, sample, spec["label"], device, depth_target=spec["depth_target"])
        cases.append((spec, case))
        pred = case["pred"]
        print(
            f"{spec['panel']} {spec['label']}: {spec['group_key']} end={spec['end_row']} "
            f"task={spec['task']} P={pred['p_det']:.3f} size={pred['size_reg_cm']:.3f} depth={pred['depth_name']}"
        )

    draw_panel(cases, OUT / "G_cam_mechanism_error_sixpack")
    draw_single_panels(cases, OUT)
    print(f"Wrote six-case CAM panel to {OUT}")


if __name__ == "__main__":
    main()
