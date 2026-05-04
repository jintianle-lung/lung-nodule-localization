from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LATEST_ALGO = ROOT / "latest_algorithm"
MODELS = ROOT / "models"
for path in (ROOT, LATEST_ALGO, MODELS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from input_normalization_v1 import normalize_raw_frames_window_minmax
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM
from train_frozen_detector_residual_inversion import FrozenDetectorResidualInversion, load_frozen_detector, sigmoid_np, softmax_np
from run_r5_feature_baseline_auc import (
    FEATURE_COLS,
    build_feature_frame,
    load_locked_splits,
    sample_feature_row,
)


FEATURE_TARGETS = [
    ("raw_max_mean", "Peak\nintensity"),
    ("center_border_contrast_center", "Center-border\ncontrast"),
    ("raw_p95_mean", "High-response\namplitude"),
    ("center_border_contrast_center", "Center contrast\n(size)"),
    ("hotspot_radius_max", "Max hotspot\nradius"),
    ("second_moment_spread_max", "Max spatial\nspread"),
]

OUTPUT_TARGETS = [
    ("det_prob", "Detection\nprobability", "all"),
    ("size_reg_cm", "Size\nestimate", "positive"),
    ("depth_expected", "Depth\nexpected", "positive"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether V5/R5 learned FEM-guided experimental tactile descriptors.")
    parser.add_argument("--detector-run", default=str(LATEST_ALGO / "runs" / "shared_cnn_mstcn_cascade_file3_20260426_active_best"))
    parser.add_argument("--residual-run", default=str(LATEST_ALGO / "runs" / "residual_tune_R5_hybrid_20260427"))
    parser.add_argument("--out-dir", default=str(ROOT / "deliverables" / "v5_feature_learning_20260503"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-probe-train", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260503)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def load_r5_model(residual_run: Path, device: torch.device):
    summary = json.loads((residual_run / "summary.json").read_text(encoding="utf-8"))
    cfg = summary["config"]
    detector_run = Path(cfg["run_dir"])
    _detector_cfg, frozen, threshold = load_frozen_detector(detector_run, device)
    model = FrozenDetectorResidualInversion(
        frozen,
        int(cfg.get("morph_dim", 64)),
        int(cfg.get("hidden_dim", 160)),
        float(cfg.get("dropout", 0.28)),
        str(cfg.get("size_reg_mode", "expected_residual")),
        float(cfg.get("size_residual_span", 0.35)),
        str(cfg.get("depth_conditioning", "size7_coarse")),
    ).to(device)
    model.load_state_dict(torch.load(residual_run / "best_model.pth", map_location=device))
    model.eval()
    model.detector.eval()
    return model, threshold, summary


def batched_indices(n: int, batch_size: int) -> Iterable[range]:
    for start in range(0, int(n), int(batch_size)):
        yield range(start, min(start + int(batch_size), int(n)))


def raw_window_for_sample(records_by_key: dict, sample: dict) -> np.ndarray:
    rec = records_by_key[sample["group_key"]]
    end_row = int(sample["end_row"])
    seq_len = int(rec.get("seq_len", INPUT_SEQ_LEN))
    st = end_row - seq_len + 1
    raw_window = np.asarray(rec["raw_frames"][st : end_row + 1], dtype=np.float32)
    if raw_window.ndim == 2 and raw_window.shape[1] == 96:
        raw_window = raw_window.reshape(raw_window.shape[0], 12, 8)
    return raw_window.astype(np.float32, copy=False)


def normalized_x_batch(records_by_key: dict, samples: Sequence[dict]) -> torch.Tensor:
    windows = []
    for sample in samples:
        raw = raw_window_for_sample(records_by_key, sample)
        windows.append(normalize_raw_frames_window_minmax(raw).astype(np.float32))
    arr = np.stack(windows, axis=0)[:, :, None, :, :]
    return torch.from_numpy(arr)


def forward_with_features(model: FrozenDetectorResidualInversion, x: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        det_logit, _size_logits0, _size_reg0, _depth_logits0, extra = model.detector(x, return_features=True)
        detector_z = extra["shared_features"]
        det_prob = torch.sigmoid(det_logit)
        morph_z = model.morph(x)
        fused_h = model.fuse(torch.cat([detector_z, det_prob, morph_z], dim=1))
        size_logits = model.size_head(fused_h)
        size_coarse_logits = model.size_coarse_head(fused_h)
        size_probs = torch.softmax(size_logits, dim=1)
        size_expected = torch.sum(size_probs * model.size_values.to(size_probs.device), dim=1, keepdim=True)
        if model.size_reg_mode == "expected_residual":
            residual = torch.tanh(model.size_reg_head(fused_h)) * float(max(model.size_residual_span, 0.0))
            size_reg_cm = torch.clamp(size_expected + residual, model.size_min, float(max(SIZE_VALUES_CM)))
        else:
            size_reg_norm = torch.sigmoid(model.size_reg_head(fused_h))
            size_reg_cm = model.size_min + size_reg_norm * max(model.size_span, 1e-6)
        depth_parts = [fused_h, size_probs, size_expected / float(max(SIZE_VALUES_CM))]
        if model.depth_conditioning == "size7_coarse":
            depth_parts.append(torch.softmax(size_coarse_logits, dim=1))
        depth_input = torch.cat(depth_parts, dim=1)
        depth_logits = model.depth_head(depth_input)
        deep_logit = model.deep_head(depth_input)
    return {
        "det_logit": det_logit,
        "det_prob": det_prob,
        "detector_z": detector_z,
        "morph_z": morph_z,
        "fused_h": fused_h,
        "size_logits": size_logits,
        "size_probs": size_probs,
        "size_reg_cm": size_reg_cm,
        "depth_logits": depth_logits,
        "depth_probs": torch.softmax(depth_logits, dim=1),
        "deep_score": torch.sigmoid(deep_logit),
    }


def extract_model_table(model, records_by_key: dict, samples: Sequence[dict], device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {
        "det_prob": [],
        "size_reg_cm": [],
        "size_probs": [],
        "depth_probs": [],
        "deep_score": [],
        "detector_z": [],
        "morph_z": [],
        "fused_h": [],
    }
    for idx_range in batched_indices(len(samples), batch_size):
        batch_samples = [samples[i] for i in idx_range]
        x = normalized_x_batch(records_by_key, batch_samples).to(device)
        out = forward_with_features(model, x)
        for key in buckets:
            buckets[key].append(out[key].detach().cpu().numpy())
    result = {key: np.concatenate(values, axis=0) for key, values in buckets.items()}
    result["size_expected"] = result["size_probs"] @ np.asarray(SIZE_VALUES_CM, dtype=np.float32)
    result["size_conf"] = result["size_probs"].max(axis=1)
    result["depth_expected"] = result["depth_probs"] @ np.arange(len(COARSE_DEPTH_ORDER), dtype=np.float32)
    result["depth_conf"] = result["depth_probs"].max(axis=1)
    return result


def attach_outputs(feature_df: pd.DataFrame, outputs: dict[str, np.ndarray]) -> pd.DataFrame:
    out = feature_df.copy()
    out["det_prob"] = outputs["det_prob"].reshape(-1)
    out["size_reg_cm"] = outputs["size_reg_cm"].reshape(-1)
    out["size_expected"] = outputs["size_expected"].reshape(-1)
    out["size_conf"] = outputs["size_conf"].reshape(-1)
    out["depth_expected"] = outputs["depth_expected"].reshape(-1)
    out["depth_conf"] = outputs["depth_conf"].reshape(-1)
    out["deep_score"] = outputs["deep_score"].reshape(-1)
    for i, value in enumerate(SIZE_VALUES_CM):
        out[f"v5_size_prob_{value:g}cm"] = outputs["size_probs"][:, i]
    for i, name in enumerate(COARSE_DEPTH_ORDER):
        out[f"v5_depth_prob_{name}"] = outputs["depth_probs"][:, i]
    return out


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        value = min(prev, p[idx] * n / max(original_rank, 1))
        q[idx] = value
        prev = value
    return np.clip(q, 0.0, 1.0)


def finite_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5 or np.nanstd(x[mask]) <= 1e-12 or np.nanstd(y[mask]) <= 1e-12:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def output_alignment(test_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    pvals = []
    for feat_col, feat_name in FEATURE_TARGETS:
        for out_col, out_name, subset in OUTPUT_TARGETS:
            sub = test_df if subset == "all" else test_df[test_df["label"] == 1]
            rho, p = finite_spearman(sub[feat_col].to_numpy(), sub[out_col].to_numpy())
            rows.append(
                {
                    "feature": feat_col,
                    "feature_label": feat_name,
                    "output": out_col,
                    "output_label": out_name,
                    "subset": subset,
                    "rho": rho,
                    "p": p,
                    "n": int(len(sub)),
                }
            )
            pvals.append(1.0 if not np.isfinite(p) else p)
    qvals = bh_fdr(pvals)
    for row, q in zip(rows, qvals):
        row["q"] = float(q)
    row_order = [label for _col, label in FEATURE_TARGETS]
    col_order = [label for _col, label, _subset in OUTPUT_TARGETS]
    matrix = (
        pd.DataFrame(rows)
        .pivot(index="feature_label", columns="output_label", values="rho")
        .reindex(index=row_order, columns=col_order)
    )
    return pd.DataFrame(rows), matrix.to_numpy(dtype=np.float64)


def subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def latent_probe(
    train_df: pd.DataFrame,
    train_outputs: dict[str, np.ndarray],
    test_df: pd.DataFrame,
    test_outputs: dict[str, np.ndarray],
    max_train: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_mask = train_df["label"].to_numpy(dtype=np.int32) == 1
    test_mask = test_df["label"].to_numpy(dtype=np.int32) == 1
    train_idx_all = np.flatnonzero(train_mask)
    train_idx = train_idx_all[subsample_indices(len(train_idx_all), max_train, seed)]
    test_idx = np.flatnonzero(test_mask)
    embeddings = [
        ("detector_z", "Frozen detector\nembedding"),
        ("morph_z", "Residual morphology\nembedding"),
        ("fused_h", "Fused inversion\nembedding"),
    ]
    rows = []
    pvals = []
    for emb_key, emb_label in embeddings:
        x_train = train_outputs[emb_key][train_idx]
        x_test = test_outputs[emb_key][test_idx]
        for feat_col, feat_label in FEATURE_TARGETS:
            y_train = train_df[feat_col].to_numpy(dtype=np.float64)[train_idx]
            y_test = test_df[feat_col].to_numpy(dtype=np.float64)[test_idx]
            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-3, 3, 13)),
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            rho, p = finite_spearman(pred, y_test)
            r2 = float(r2_score(y_test, pred)) if np.isfinite(pred).all() and np.nanstd(y_test) > 1e-12 else float("nan")
            rows.append(
                {
                    "embedding": emb_key,
                    "embedding_label": emb_label,
                    "feature": feat_col,
                    "feature_label": feat_label,
                    "spearman_rho": rho,
                    "p": p,
                    "test_r2": r2,
                    "n_train_positive": int(len(train_idx)),
                    "n_test_positive": int(len(test_idx)),
                }
            )
            pvals.append(1.0 if not np.isfinite(p) else p)
    qvals = bh_fdr(pvals)
    for row, q in zip(rows, qvals):
        row["q"] = float(q)
    row_order = [label for _col, label in FEATURE_TARGETS]
    col_order = [
        "Frozen detector\nembedding",
        "Residual morphology\nembedding",
        "Fused inversion\nembedding",
    ]
    matrix = (
        pd.DataFrame(rows)
        .pivot(index="feature_label", columns="embedding_label", values="spearman_rho")
        .reindex(index=row_order, columns=col_order)
    )
    return pd.DataFrame(rows), matrix.to_numpy(dtype=np.float64)


def gaussian_hotspot_mask(x: np.ndarray, sigma: float = 1.25) -> np.ndarray:
    mean_frame = np.asarray(x[:, 0], dtype=np.float32).mean(axis=0)
    peak = np.unravel_index(int(np.argmax(mean_frame)), mean_frame.shape)
    rows, cols = np.indices(mean_frame.shape, dtype=np.float32)
    dist2 = (rows - float(peak[0])) ** 2 + (cols - float(peak[1])) ** 2
    mask2 = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    mask2 /= max(float(mask2.max()), 1e-6)
    return mask2[None, None]


def gaussian_distance_mask_from_peak(x: np.ndarray, sigma: float) -> np.ndarray:
    mean_frame = np.asarray(x[:, 0], dtype=np.float32).mean(axis=0)
    peak = np.unravel_index(int(np.argmax(mean_frame)), mean_frame.shape)
    rows, cols = np.indices(mean_frame.shape, dtype=np.float32)
    dist2 = (rows - float(peak[0])) ** 2 + (cols - float(peak[1])) ** 2
    mask2 = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    mask2 /= max(float(mask2.max()), 1e-6)
    return mask2[None, None]


def perturb_tensor(x: np.ndarray, mode: str) -> np.ndarray:
    xp = np.asarray(x, dtype=np.float32).copy()
    if mode == "hotspot_flatten":
        mask = gaussian_hotspot_mask(xp)
        local_floor = np.percentile(xp, 55)
        xp = xp * (1.0 - 0.75 * mask) + float(local_floor) * (0.75 * mask)
    elif mode == "center_contrast_drop":
        border = xp.copy()
        center = np.zeros((12, 8), dtype=bool)
        center[3:9, 2:6] = True
        border_value = float(np.median(border[:, 0, ~center]))
        xp[:, 0, center] = 0.45 * xp[:, 0, center] + 0.55 * border_value
    elif mode == "spatial_blur":
        for t in range(xp.shape[0]):
            xp[t, 0] = gaussian_filter(xp[t, 0], sigma=1.0)
    elif mode == "peripheral_spread_drop":
        core = gaussian_distance_mask_from_peak(xp, sigma=1.05)
        peripheral = np.clip(1.0 - core, 0.0, 1.0)
        for t in range(xp.shape[0]):
            frame = xp[t, 0]
            floor = float(np.percentile(frame, 35))
            xp[t, 0] = frame * (1.0 - 0.70 * peripheral[0, 0]) + floor * (0.70 * peripheral[0, 0])
    elif mode == "temporal_shuffle":
        order = np.arange(xp.shape[0])
        # Deterministic high-disruption order avoids stochastic figure drift.
        order = np.concatenate([order[1::2], order[::2]])
        xp = xp[order]
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return np.clip(xp, 0.0, 1.0).astype(np.float32)


def perturbation_analysis(
    model,
    records_by_key: dict,
    samples: Sequence[dict],
    base_outputs: dict[str, np.ndarray],
    threshold: float,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    positive_indices = [
        i
        for i, sample in enumerate(samples)
        if int(sample["label"]) == 1 and float(base_outputs["det_prob"][i]) >= float(threshold)
    ]
    modes = [
        ("hotspot_flatten", "Flatten\nhotspot"),
        ("center_contrast_drop", "Drop center-border\ncontrast"),
        ("peripheral_spread_drop", "Suppress peripheral\nspread"),
        ("temporal_shuffle", "Shuffle temporal\norder"),
    ]
    rows = []
    for mode, label in modes:
        deltas = {
            "delta_det_prob": [],
            "delta_size_pred_prob": [],
            "delta_size_conf": [],
            "delta_size_cm": [],
            "delta_depth_pred_prob": [],
            "delta_depth_conf": [],
            "delta_depth_expected": [],
        }
        for chunk_start in range(0, len(positive_indices), batch_size):
            idxs = positive_indices[chunk_start : chunk_start + batch_size]
            xs = []
            for i in idxs:
                raw = raw_window_for_sample(records_by_key, samples[i])
                x = normalize_raw_frames_window_minmax(raw).astype(np.float32)[:, None, :, :]
                xs.append(perturb_tensor(x, mode))
            x_tensor = torch.from_numpy(np.stack(xs, axis=0)).to(device)
            out = forward_with_features(model, x_tensor)
            pert_det = out["det_prob"].detach().cpu().numpy().reshape(-1)
            pert_size_probs = out["size_probs"].detach().cpu().numpy()
            pert_size_conf = pert_size_probs.max(axis=1)
            pert_size = out["size_reg_cm"].detach().cpu().numpy().reshape(-1)
            pert_depth_probs = out["depth_probs"].detach().cpu().numpy()
            pert_depth_conf = pert_depth_probs.max(axis=1)
            pert_depth_exp = pert_depth_probs @ np.arange(len(COARSE_DEPTH_ORDER), dtype=np.float32)
            base_det = base_outputs["det_prob"][idxs].reshape(-1)
            base_size_probs = base_outputs["size_probs"][idxs]
            base_size_top = np.argmax(base_size_probs, axis=1)
            base_size_pred_prob = base_size_probs[np.arange(len(base_size_top)), base_size_top]
            pert_size_pred_prob = pert_size_probs[np.arange(len(base_size_top)), base_size_top]
            base_size_conf = base_outputs["size_conf"][idxs].reshape(-1)
            base_size = base_outputs["size_reg_cm"][idxs].reshape(-1)
            base_depth_probs = base_outputs["depth_probs"][idxs]
            base_depth_top = np.argmax(base_depth_probs, axis=1)
            base_depth_pred_prob = base_depth_probs[np.arange(len(base_depth_top)), base_depth_top]
            pert_depth_pred_prob = pert_depth_probs[np.arange(len(base_depth_top)), base_depth_top]
            base_depth_conf = base_outputs["depth_conf"][idxs].reshape(-1)
            base_depth_exp = (base_outputs["depth_probs"][idxs] @ np.arange(len(COARSE_DEPTH_ORDER), dtype=np.float32)).reshape(-1)
            deltas["delta_det_prob"].extend((pert_det - base_det).tolist())
            deltas["delta_size_pred_prob"].extend((pert_size_pred_prob - base_size_pred_prob).tolist())
            deltas["delta_size_conf"].extend((pert_size_conf - base_size_conf).tolist())
            deltas["delta_size_cm"].extend((pert_size - base_size).tolist())
            deltas["delta_depth_pred_prob"].extend((pert_depth_pred_prob - base_depth_pred_prob).tolist())
            deltas["delta_depth_conf"].extend((pert_depth_conf - base_depth_conf).tolist())
            deltas["delta_depth_expected"].extend((pert_depth_exp - base_depth_exp).tolist())
        for metric, values in deltas.items():
            values_arr = np.asarray(values, dtype=np.float64)
            mean = float(np.mean(values_arr))
            sem = float(np.std(values_arr, ddof=1) / math.sqrt(max(len(values_arr), 1))) if len(values_arr) > 1 else 0.0
            rows.append(
                {
                    "perturbation": mode,
                    "perturbation_label": label,
                    "metric": metric,
                    "mean_delta": mean,
                    "sem": sem,
                    "ci95": 1.96 * sem,
                    "n_gated_positive": int(len(positive_indices)),
                }
            )
    return pd.DataFrame(rows)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def heatmap(ax, matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, vmin=-1.0, vmax=1.0):
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = "" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
    return im


def render_figure(
    out_dir: Path,
    alignment_df: pd.DataFrame,
    alignment_matrix: np.ndarray,
    probe_df: pd.DataFrame,
    probe_matrix: np.ndarray,
    perturb_df: pd.DataFrame,
) -> None:
    apply_style()
    feature_labels = [label for _col, label in FEATURE_TARGETS]
    output_labels = [label for _col, label, _subset in OUTPUT_TARGETS]
    embedding_labels = ["Frozen detector\nembedding", "Residual morphology\nembedding", "Fused inversion\nembedding"]

    fig = plt.figure(figsize=(10.8, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    im_a = heatmap(
        ax_a,
        alignment_matrix,
        feature_labels,
        output_labels,
        "A  V5/R5 outputs align with experimental descriptors",
    )
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.02)
    cbar_a.set_label("Spearman rho")

    im_b = heatmap(
        ax_b,
        probe_matrix,
        feature_labels,
        embedding_labels,
        "B  Linear probes decode descriptors from frozen V5/R5 latents",
    )
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.02)
    cbar_b.set_label("Probe rho")

    metrics = [
        ("delta_det_prob", "Detection probability"),
        ("delta_size_pred_prob", "Size class probability"),
        ("delta_depth_pred_prob", "Depth class probability"),
    ]
    colors = ["#0072B2", "#009E73", "#D55E00"]
    perturb_order = ["hotspot_flatten", "center_contrast_drop", "peripheral_spread_drop", "temporal_shuffle"]
    labels = (
        perturb_df[["perturbation", "perturbation_label"]]
        .drop_duplicates()
        .set_index("perturbation")
        .loc[perturb_order, "perturbation_label"]
        .tolist()
    )
    x = np.arange(len(perturb_order), dtype=np.float64)
    width = 0.22
    for k, (metric, label) in enumerate(metrics):
        sub = perturb_df[perturb_df["metric"] == metric].set_index("perturbation").loc[perturb_order]
        offset = (k - 1) * width
        ax_c.bar(
            x + offset,
            sub["mean_delta"].to_numpy(),
            yerr=sub["ci95"].to_numpy(),
            width=width,
            color=colors[k],
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
            label=label,
        )
    ax_c.axhline(0, color="#333333", linewidth=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Mean change after perturbation")
    ax_c.set_title("C  Targeted removal of validated tactile cues changes V5/R5 outputs", fontweight="bold", pad=18)
    ax_c.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax_c.legend(frameon=False, ncol=3, loc="upper left")

    fig.suptitle(
        "Frozen V5/R5 encodes the FEM-guided experimental tactile descriptor family",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"figure5_v5_learned_features_candidate{ext}", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    detector_run = Path(args.detector_run)
    residual_run = Path(args.residual_run)
    manifest = json.loads((detector_run / "manifest.json").read_text(encoding="utf-8"))
    train_records, val_records, test_records, train_samples_all, _train_samples_det, val_samples, test_samples = load_locked_splits(manifest)
    train_val_records = {**train_records, **val_records}
    train_val_samples = list(train_samples_all) + list(val_samples)

    model, threshold, summary = load_r5_model(residual_run, device)
    print(f"Device: {device}; threshold={threshold:.3f}; train/val samples={len(train_val_samples)}; test samples={len(test_samples)}")

    print("Computing FEM-guided descriptors...")
    train_df = build_feature_frame(train_val_records, train_val_samples, "train_val")
    test_df = build_feature_frame(test_records, test_samples, "file3_test")
    train_df.to_csv(out_dir / "train_val_fem_guided_descriptors.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "file3_fem_guided_descriptors.csv", index=False, encoding="utf-8-sig")

    print("Extracting frozen V5/R5 outputs and latent embeddings...")
    train_outputs = extract_model_table(model, train_val_records, train_val_samples, device, int(args.batch_size))
    test_outputs = extract_model_table(model, test_records, test_samples, device, int(args.batch_size))
    train_with_outputs = attach_outputs(train_df, train_outputs)
    test_with_outputs = attach_outputs(test_df, test_outputs)
    train_with_outputs.to_csv(out_dir / "train_val_v5_outputs_plus_descriptors.csv", index=False, encoding="utf-8-sig")
    test_with_outputs.to_csv(out_dir / "file3_v5_outputs_plus_descriptors.csv", index=False, encoding="utf-8-sig")

    print("Running output-feature alignment...")
    alignment_df, alignment_matrix = output_alignment(test_with_outputs)
    alignment_df.to_csv(out_dir / "v5_output_descriptor_alignment.csv", index=False, encoding="utf-8-sig")

    print("Running latent descriptor probes...")
    probe_df, probe_matrix = latent_probe(
        train_with_outputs,
        train_outputs,
        test_with_outputs,
        test_outputs,
        max_train=int(args.max_probe_train),
        seed=int(args.seed),
    )
    probe_df.to_csv(out_dir / "v5_latent_descriptor_probe.csv", index=False, encoding="utf-8-sig")

    print("Running targeted perturbations...")
    perturb_df = perturbation_analysis(
        model,
        test_records,
        test_samples,
        test_outputs,
        threshold,
        device,
        int(args.batch_size),
    )
    perturb_df.to_csv(out_dir / "v5_targeted_perturbation_response.csv", index=False, encoding="utf-8-sig")

    render_figure(out_dir, alignment_df, alignment_matrix, probe_df, probe_matrix, perturb_df)

    best_probe = (
        probe_df.sort_values(["spearman_rho"], ascending=False)
        .head(8)[["embedding_label", "feature_label", "spearman_rho", "q", "test_r2"]]
        .to_dict(orient="records")
    )
    perturb_summary = perturb_df.pivot(index="perturbation_label", columns="metric", values="mean_delta").reset_index().to_dict(orient="records")
    run_summary = {
        "detector_run": str(detector_run.resolve()),
        "residual_run": str(residual_run.resolve()),
        "threshold": float(threshold),
        "device": str(device),
        "n_train_val": int(len(train_val_samples)),
        "n_train_val_positive": int((train_with_outputs["label"] == 1).sum()),
        "n_test": int(len(test_samples)),
        "n_test_positive": int((test_with_outputs["label"] == 1).sum()),
        "feature_targets": [{"column": col, "label": label} for col, label in FEATURE_TARGETS],
        "output_alignment_csv": str((out_dir / "v5_output_descriptor_alignment.csv").resolve()),
        "latent_probe_csv": str((out_dir / "v5_latent_descriptor_probe.csv").resolve()),
        "perturbation_csv": str((out_dir / "v5_targeted_perturbation_response.csv").resolve()),
        "figure_png": str((out_dir / "figure5_v5_learned_features_candidate.png").resolve()),
        "figure_pdf": str((out_dir / "figure5_v5_learned_features_candidate.pdf").resolve()),
        "top_probe_relationships": best_probe,
        "perturbation_mean_deltas": perturb_summary,
        "interpretation_boundary": (
            "These are post-hoc representational and perturbation tests on the frozen V5/R5 model. "
            "They support descriptor encoding and task-specific reliance, but should not be written as proof of a unique causal tissue law."
        ),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
