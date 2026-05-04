from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))

from build_global_error_mechanism_summary import (
    IN_CSV,
    OUT,
    RUN_SUMMARY,
    add_predictions,
    apply_style,
    draw_depth,
    draw_detection,
    draw_size,
)


def save_single_panel(name: str, draw_fn, df: pd.DataFrame, figsize: tuple[float, float]) -> Path:
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.subplots_adjust(left=0.215, right=0.985, top=0.770, bottom=0.210)
    draw_fn(ax, df)
    out_path = OUT / f"{name}.svg"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return out_path


def main() -> None:
    apply_style()
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    df = add_predictions(pd.read_csv(IN_CSV), float(summary["threshold"]))
    gated = df[(df["label"] == 1) & df["det_pred"]].copy()

    paths = [
        save_single_panel("Error_detection_FP_hard_hotspot", draw_detection, df, (2.35, 1.95)),
        save_single_panel("Error_size_larger_deformable_contacts", draw_size, gated, (2.35, 1.95)),
        save_single_panel("Error_depth_weak_diffusion_cue", draw_depth, gated, (2.35, 1.95)),
    ]
    print(json.dumps({"svg_outputs": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
