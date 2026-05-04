from __future__ import annotations

import json
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

from build_cam_mechanism_error_sixpack import PANELS, SINGLE_PANEL_META, style  # noqa: E402
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


OUT = ROOT / "deliverables" / "cam_feature_alignment_20260504" / "single_cam_panels_clean"
CANONICAL_OUT = ROOT / "deliverables" / "cam_feature_alignment_20260504" / "single_cam_panels"


def draw_clean_single(spec: dict, case: dict) -> dict:
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

    out_stem = OUT / f"G_cam_{meta['file']}_clean"
    canonical_stem = CANONICAL_OUT / f"G_cam_{meta['file']}"
    for stem in (out_stem, canonical_stem):
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)

    return {
        "panel_key": spec["panel"],
        "file_stem": str(canonical_stem),
        "clean_file_stem": str(out_stem),
        "title": meta["short_title"],
        "task_cam": spec["task"],
        "group_key": spec["group_key"],
        "end_row": int(spec["end_row"]),
        "frames": [int(frame + 1) for frame in frames],
        "peak_frame": int(peak + 1),
        "gt_pred": meta["gt_pred"],
        "short_annotation": meta["status"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CANONICAL_OUT.mkdir(parents=True, exist_ok=True)
    style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _threshold, detector_run = load_model(device)
    manifest = json.loads((detector_run / "manifest.json").read_text(encoding="utf-8"))
    _train_records, _val_records, test_records, _train_all, _train_det, _val_samples, test_samples = load_locked_splits(manifest)

    rows = []
    for spec in PANELS:
        sample = find_sample(test_samples, spec["group_key"], int(spec["end_row"]))
        case = build_case_for_display(model, test_records, sample, spec["label"], device, depth_target=spec["depth_target"])
        rows.append(draw_clean_single(spec, case))
        print(f"{spec['panel']} clean single: {SINGLE_PANEL_META[spec['panel']]['short_title']} -> end {spec['end_row']}")

    (OUT / "single_cam_panel_sources_clean.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (CANONICAL_OUT / "single_cam_panel_sources.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote clean single CAM panels to {OUT}")
    print(f"Updated canonical single CAM panels in {CANONICAL_OUT}")


if __name__ == "__main__":
    main()
