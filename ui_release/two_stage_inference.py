from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
CHECKPOINT_DIR = APP_DIR / "checkpoints"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(CHECKPOINT_DIR) not in sys.path:
    sys.path.insert(0, str(CHECKPOINT_DIR))
ROOT_MODELS_DIR = PROJECT_DIR / "models"
if str(ROOT_MODELS_DIR) not in sys.path:
    sys.path.append(str(ROOT_MODELS_DIR))

from dual_stream_mstcn_detection import DualStreamMSTCNDetector
from hierarchical_positive_inverter import HierarchicalPositiveInverter
from task_protocol_v1 import (
    INPUT_SEQ_LEN,
    SIZE_VALUES_CM,
    class_index_to_size,
    coarse_index_to_name,
    format_runtime_payload,
)
try:
    from dual_stream_mstcn_multitask import DualStreamMSTCNMultiTask
except Exception:
    DualStreamMSTCNMultiTask = None
try:
    from depth_analysis_utils import frame_physics_features, window_temporal_features
except Exception:
    CENTER_MASK = np.zeros((12, 8), dtype=bool)
    CENTER_MASK[3:9, 2:6] = True
    BORDER_MASK = ~CENTER_MASK

    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        mn = float(frame.min())
        mx = float(frame.max())
        if mx - mn <= 1e-8:
            return np.zeros_like(frame, dtype=np.float32)
        return ((frame - mn) / (mx - mn)).astype(np.float32)

    def weighted_centroid_and_cov(norm_frame: np.ndarray):
        weights = np.clip(norm_frame.astype(np.float64), 0.0, None)
        total = float(weights.sum())
        if total <= 1e-12:
            return 5.5, 3.5, np.eye(2, dtype=np.float64) * 1e-6
        rows, cols = np.indices(norm_frame.shape, dtype=np.float64)
        row_c = float((rows * weights).sum() / total)
        col_c = float((cols * weights).sum() / total)
        dr = rows - row_c
        dc = cols - col_c
        cov_rr = float((weights * dr * dr).sum() / total)
        cov_cc = float((weights * dc * dc).sum() / total)
        cov_rc = float((weights * dr * dc).sum() / total)
        cov = np.array([[cov_rr, cov_rc], [cov_rc, cov_cc]], dtype=np.float64)
        return row_c, col_c, cov

    def hotspot_radius_and_spread(norm_frame: np.ndarray):
        row_c, col_c, _cov = weighted_centroid_and_cov(norm_frame)
        rows, cols = np.indices(norm_frame.shape, dtype=np.float64)
        weights = np.clip(norm_frame.astype(np.float64), 0.0, None)
        total = float(weights.sum())
        if total <= 1e-12:
            return 0.0, 0.0
        dist2 = (rows - row_c) ** 2 + (cols - col_c) ** 2
        second_moment = float((weights * dist2).sum() / total)
        radius = float(math.sqrt(max(second_moment, 0.0)))
        return radius, second_moment

    def spatial_entropy(norm_frame: np.ndarray) -> float:
        weights = np.clip(norm_frame.astype(np.float64), 0.0, None).reshape(-1)
        total = float(weights.sum())
        if total <= 1e-12:
            return 0.0
        probs = weights / total
        probs = probs[probs > 1e-12]
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = math.log(max(len(weights), 1))
        return float(entropy / max(max_entropy, 1e-12))

    def anisotropy_ratio(norm_frame: np.ndarray) -> float:
        _row_c, _col_c, cov = weighted_centroid_and_cov(norm_frame)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        return float(math.sqrt(float(eigvals.max() / eigvals.min())))

    def local_peak_count(norm_frame: np.ndarray, threshold: float = 0.85) -> int:
        row_pad = np.pad(norm_frame, 1, mode="edge")
        local_max = np.zeros_like(norm_frame, dtype=bool)
        for r in range(norm_frame.shape[0]):
            for c in range(norm_frame.shape[1]):
                patch = row_pad[r : r + 3, c : c + 3]
                local_max[r, c] = norm_frame[r, c] >= threshold and norm_frame[r, c] >= patch.max() - 1e-8
        return int(local_max.sum())

    def frame_physics_features(frame):
        arr = np.asarray(frame, dtype=np.float32)
        norm = normalize_frame(arr)
        hotspot_area = float((norm >= 0.70).mean())
        radius, second_moment = hotspot_radius_and_spread(norm)
        centroid_row, centroid_col, _cov = weighted_centroid_and_cov(norm)
        center_mean = float(arr[CENTER_MASK].mean())
        border_mean = float(arr[BORDER_MASK].mean())
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "raw_mean": float(arr.mean()) if arr.size else 0.0,
            "raw_max": float(arr.max()) if arr.size else 0.0,
            "raw_sum": float(arr.sum()) if arr.size else 0.0,
            "raw_p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "center_mean": center_mean,
            "border_mean": border_mean,
            "center_border_contrast": float(center_mean - border_mean),
            "hotspot_area": hotspot_area,
            "hotspot_radius": radius,
            "second_moment_spread": second_moment,
            "spatial_entropy": float(spatial_entropy(norm)),
            "anisotropy_ratio": float(anisotropy_ratio(norm)),
            "peak_count": int(local_peak_count(norm)),
            "centroid_row": float(centroid_row),
            "centroid_col": float(centroid_col),
        }

    def window_temporal_features(frame_rows):
        raw_sum = np.asarray([row["raw_sum"] for row in frame_rows], dtype=np.float32)
        centroid_rows = np.asarray([row["centroid_row"] for row in frame_rows], dtype=np.float32)
        centroid_cols = np.asarray([row["centroid_col"] for row in frame_rows], dtype=np.float32)
        return {
            "length": float(len(frame_rows)),
            "raw_sum_slope": float(np.polyfit(np.arange(len(raw_sum), dtype=np.float32), raw_sum, 1)[0]) if len(raw_sum) > 1 else 0.0,
            "centroid_drift": float(np.mean(np.sqrt(np.diff(centroid_rows) ** 2 + np.diff(centroid_cols) ** 2))) if len(centroid_rows) > 1 else 0.0,
        }
try:
    from train_xgboost_baselines import window_feature_row
except Exception:
    def summarize_series(out: Dict[str, float], prefix: str, values: list) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            out[f"{prefix}_mean"] = 0.0
            out[f"{prefix}_std"] = 0.0
            out[f"{prefix}_min"] = 0.0
            out[f"{prefix}_max"] = 0.0
            out[f"{prefix}_first"] = 0.0
            out[f"{prefix}_center"] = 0.0
            out[f"{prefix}_last"] = 0.0
            out[f"{prefix}_delta"] = 0.0
            out[f"{prefix}_slope"] = 0.0
            return
        x = np.arange(arr.size, dtype=np.float32)
        x_mean = float(x.mean())
        y_mean = float(arr.mean())
        denom = float(np.sum((x - x_mean) ** 2))
        slope = float(np.sum((x - x_mean) * (arr - y_mean)) / denom) if denom > 1e-8 else 0.0
        center_idx = int(arr.size // 2)
        out[f"{prefix}_mean"] = float(arr.mean())
        out[f"{prefix}_std"] = float(arr.std())
        out[f"{prefix}_min"] = float(arr.min())
        out[f"{prefix}_max"] = float(arr.max())
        out[f"{prefix}_first"] = float(arr[0])
        out[f"{prefix}_center"] = float(arr[center_idx])
        out[f"{prefix}_last"] = float(arr[-1])
        out[f"{prefix}_delta"] = float(arr[-1] - arr[0])
        out[f"{prefix}_slope"] = slope

    def window_feature_row(records_by_key: Dict[str, dict], sample: dict) -> Dict[str, float]:
        rec = records_by_key[sample["group_key"]]
        end_row = int(sample["end_row"])
        seq_len = int(rec["seq_len"])
        start_row = end_row - seq_len + 1
        raw_window = rec["raw_frames"][start_row : end_row + 1]
        norm_window = rec["norm_frames"][start_row : end_row + 1]
        frame_rows = rec["frame_rows"][start_row : end_row + 1]

        out: Dict[str, float] = {
            "label": int(sample["label"]),
            "size_cm": float(sample["size_cm"]),
            "depth_cm": float(sample["depth_cm"]),
            "size_class_index": int(sample["size_class_index"]),
            "depth_coarse_index": int(sample["depth_coarse_index"]),
            "center_row": int(sample["center_row"]),
            "end_row": int(sample["end_row"]),
        }

        frame_keys = list(frame_rows[0].keys()) if frame_rows else []
        for key in frame_keys:
            summarize_series(out, key, [float(row[key]) for row in frame_rows])

        for key, value in window_temporal_features(frame_rows).items():
            out[f"window_{key}"] = float(value)

        mean_frame_raw = raw_window.mean(axis=0)
        max_frame_raw = raw_window.max(axis=0)
        center_frame_raw = raw_window[len(raw_window) // 2]
        mean_frame_norm = norm_window.mean(axis=0)

        for prefix, frame in {
            "meanframe_raw": mean_frame_raw,
            "maxframe_raw": max_frame_raw,
            "centerframe_raw": center_frame_raw,
            "meanframe_norm": mean_frame_norm,
        }.items():
            for key, value in frame_physics_features(frame).items():
                out[f"{prefix}_{key}"] = float(value)

        if len(raw_window) > 1:
            delta = np.diff(raw_window, axis=0)
            abs_delta = np.abs(delta)
            out["delta_abs_mean"] = float(abs_delta.mean())
            out["delta_abs_std"] = float(abs_delta.std())
            out["delta_abs_max"] = float(abs_delta.max())
            delta_mean_frame = abs_delta.mean(axis=0)
            for key, value in frame_physics_features(delta_mean_frame).items():
                out[f"deltaframe_{key}"] = float(value)
        else:
            out["delta_abs_mean"] = 0.0
            out["delta_abs_std"] = 0.0
            out["delta_abs_max"] = 0.0
            for key, value in frame_physics_features(np.zeros((12, 8), dtype=np.float32)).items():
                out[f"deltaframe_{key}"] = float(value)

        out["window_raw_global_mean"] = float(raw_window.mean())
        out["window_raw_global_std"] = float(raw_window.std())
        out["window_raw_global_max"] = float(raw_window.max())
        out["window_raw_global_p95"] = float(np.percentile(raw_window.reshape(-1), 95))
        out["window_norm_global_mean"] = float(norm_window.mean())
        out["window_norm_global_std"] = float(norm_window.std())
        return out


COARSE_DEPTH_DISPLAY = {
    "shallow": "浅层 (0.5-1.0 cm)",
    "middle": "中层 (1.5-2.0 cm)",
    "deep": "深层 (2.5-3.0 cm)",
}


def _torch_load_compat(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary_threshold(summary: Dict, fallback: float = 0.62) -> float:
    best_record = summary.get("best_record", {}) if isinstance(summary, dict) else {}
    threshold = best_record.get("val_best_threshold")
    if threshold is None:
        threshold = summary.get("stage1_reference_metrics", {}).get("stage1_val_best_threshold")
    try:
        return float(threshold)
    except Exception:
        return float(fallback)


def _coerce_frame_to_matrix(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.shape == (12, 8):
        return arr.astype(np.float32, copy=False)
    flat = arr.reshape(-1)
    if flat.size < 96:
        raise ValueError(f"Frame has {flat.size} values, expected at least 96.")
    if flat.size > 96:
        flat = flat[-96:]
    return flat.reshape(12, 8).astype(np.float32, copy=False)


def _normalize_sequence(seq_raw: np.ndarray) -> np.ndarray:
    seq_raw = np.asarray(seq_raw, dtype=np.float32)
    if seq_raw.shape != (INPUT_SEQ_LEN, 12, 8):
        raise ValueError(f"Expected sequence shape {(INPUT_SEQ_LEN, 12, 8)}, got {tuple(seq_raw.shape)}")
    seq_norm = np.zeros((INPUT_SEQ_LEN, 1, 12, 8), dtype=np.float32)
    for i in range(INPUT_SEQ_LEN):
        frame = seq_raw[i]
        mn = float(frame.min())
        mx = float(frame.max())
        if mx - mn > 1e-6:
            frame = (frame - mn) / (mx - mn)
        else:
            frame = frame - mn
        seq_norm[i, 0] = frame
    return seq_norm


def _size_norm_to_cm(size_norm: float) -> float:
    lo = float(min(SIZE_VALUES_CM))
    hi = float(max(SIZE_VALUES_CM))
    return float(lo + np.clip(float(size_norm), 0.0, 1.0) * (hi - lo))


def _compute_runtime_feature_vector(
    seq_raw: np.ndarray,
    seq_norm: np.ndarray,
    selected_features: list,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    frame_rows = [frame_physics_features(frame) for frame in seq_raw]
    records = {
        "runtime_window": {
            "raw_frames": seq_raw.astype(np.float32),
            "norm_frames": seq_norm[:, 0].astype(np.float32),
            "frame_rows": frame_rows,
            "seq_len": int(seq_raw.shape[0]),
        }
    }
    sample = {
        "group_key": "runtime_window",
        "label": 1,
        "size_cm": 1.0,
        "depth_cm": 1.5,
        "size_class_index": 3,
        "depth_coarse_index": 1,
        "center_row": int(seq_raw.shape[0] // 2),
        "end_row": int(seq_raw.shape[0] - 1),
    }
    row = window_feature_row(records, sample)
    feat = np.asarray([float(row[name]) for name in selected_features], dtype=np.float32)
    feat = (feat - feature_mean.astype(np.float32)) / np.maximum(feature_std.astype(np.float32), 1e-6)
    return feat


class TwoStageNoduleInference:
    def __init__(
        self,
        detector_ckpt: Optional[str] = None,
        detector_summary: Optional[str] = None,
        inverter_ckpt: Optional[str] = None,
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        default_stage1_dir = CHECKPOINT_DIR
        default_stage2_dir = CHECKPOINT_DIR

        detector_ckpt = detector_ckpt or os.environ.get(
            "PAPER_GUI_STAGE1_CKPT",
            str(default_stage1_dir / "paper_stage1_dualstream_mstcn_best.pth"),
        )
        detector_summary = detector_summary or os.environ.get(
            "PAPER_GUI_STAGE1_SUMMARY",
            str(CHECKPOINT_DIR / "stage1_detection_summary.json"),
        )
        inverter_ckpt = inverter_ckpt or os.environ.get(
            "PAPER_GUI_STAGE2_CKPT",
            str(default_stage2_dir / "paper_hierarchical_positive_inverter_best.pth"),
        )

        self.detector_ckpt = Path(detector_ckpt)
        self.detector_summary_path = Path(detector_summary)
        self.inverter_ckpt = Path(inverter_ckpt)

        if not self.detector_ckpt.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {self.detector_ckpt}")
        if not self.inverter_ckpt.exists():
            raise FileNotFoundError(f"Inverter checkpoint not found: {self.inverter_ckpt}")

        detector_summary_data = _load_json(self.detector_summary_path) if self.detector_summary_path.exists() else {}
        self.threshold = float(threshold) if threshold is not None else _summary_threshold(detector_summary_data)
        self.inverter_kind = "legacy_multitask"
        self.selected_features = None
        self.feature_mean = None
        self.feature_std = None
        self.raw_scale = 1.0

        self.detector = self._load_detector(self.detector_ckpt)
        self.inverter = self._load_inverter(self.inverter_ckpt)

    def _load_detector(self, ckpt_path: Path) -> DualStreamMSTCNDetector:
        ckpt = _torch_load_compat(str(ckpt_path), map_location=self.device)
        config = ckpt.get("config", {})
        model = DualStreamMSTCNDetector(
            seq_len=int(config.get("seq_len", INPUT_SEQ_LEN)),
            frame_feature_dim=int(config.get("frame_feature_dim", 32)),
            temporal_channels=int(config.get("temporal_channels", 64)),
            temporal_blocks=int(config.get("temporal_blocks", 3)),
            dropout=float(config.get("dropout", 0.35)),
            use_delta_branch=bool(config.get("use_delta_branch", True)),
        ).to(self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if any(str(k).startswith("pooling.attn") or str(k).startswith("pooling.proj") for k in state_dict.keys()):
            remapped = {}
            for key, value in state_dict.items():
                key = str(key)
                if key.startswith("pooling.") and not key.startswith("pooling.pool."):
                    remapped["pooling.pool." + key[len("pooling."):]] = value
                else:
                    remapped[key] = value
            state_dict = remapped
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_inverter(self, ckpt_path: Path) -> DualStreamMSTCNMultiTask:
        ckpt = _torch_load_compat(str(ckpt_path), map_location=self.device)
        router_name = str(ckpt.get("router_model_name", ckpt.get("model_name", "")))
        if router_name == "HierarchicalPositiveInverter":
            config = ckpt.get("model_config", {})
            model = HierarchicalPositiveInverter(
                seq_len=int(config.get("seq_len", INPUT_SEQ_LEN)),
                frame_feature_dim=int(config.get("frame_feature_dim", 24)),
                temporal_channels=int(config.get("temporal_channels", 48)),
                temporal_blocks=int(config.get("temporal_blocks", 3)),
                dropout=float(config.get("dropout", 0.28)),
                num_size_classes=int(config.get("num_size_classes", len(SIZE_VALUES_CM))),
                num_depth_classes=int(config.get("num_depth_classes", 3)),
                num_tabular_features=int(config.get("num_tabular_features", len(ckpt.get("selected_features", [])))),
                tabular_hidden_dim=int(config.get("tabular_hidden_dim", 64)),
            ).to(self.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            self.inverter_kind = "hierarchical_positive_inverter"
            self.selected_features = [str(x) for x in ckpt.get("selected_features", [])]
            self.feature_mean = np.asarray(ckpt.get("feature_mean", []), dtype=np.float32)
            self.feature_std = np.asarray(ckpt.get("feature_std", []), dtype=np.float32)
            self.raw_scale = float(ckpt.get("raw_scale", 1.0))
            return model

        config = ckpt.get("config", {})
        if DualStreamMSTCNMultiTask is None:
            raise ModuleNotFoundError(
                "DualStreamMSTCNMultiTask module is unavailable for legacy inverter loading"
            )
        model = DualStreamMSTCNMultiTask(
            seq_len=int(config.get("seq_len", INPUT_SEQ_LEN)),
            dropout=float(config.get("dropout", 0.35)),
            use_delta_branch=bool(config.get("use_delta_branch", False)),
            num_size_classes=7,
            num_depth_classes=3,
        ).to(self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        self.inverter_kind = "legacy_multitask"
        return model

    def predict_from_frames(self, frames: Iterable[np.ndarray]) -> Dict[str, object]:
        seq_raw = np.stack([_coerce_frame_to_matrix(frame) for frame in frames], axis=0).astype(np.float32)
        seq_norm = _normalize_sequence(seq_raw)
        input_tensor = torch.from_numpy(seq_norm).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            det_logit = self.detector(input_tensor)
            det_prob = float(torch.sigmoid(det_logit).item())

            payload = format_runtime_payload(det_prob=det_prob, threshold=self.threshold)
            size_logits = None
            depth_logits = None
            size_reg_cm = None

            if self.inverter_kind == "hierarchical_positive_inverter":
                raw_input = np.clip(seq_raw / max(float(self.raw_scale), 1e-6), 0.0, 3.0).astype(np.float32)
                raw_tensor = torch.from_numpy(raw_input[:, None, :, :]).unsqueeze(0).to(self.device)
                feat_np = _compute_runtime_feature_vector(
                    seq_raw,
                    seq_norm,
                    self.selected_features or [],
                    self.feature_mean if self.feature_mean is not None else np.zeros((0,), dtype=np.float32),
                    self.feature_std if self.feature_std is not None else np.ones((0,), dtype=np.float32),
                )
                feat_tensor = torch.from_numpy(feat_np[None, :]).to(self.device)
                size_logits_tensor, _size_ord_tensor, size_reg_norm_tensor, size_probs_tensor, feats = self.inverter(
                    raw_tensor, input_tensor, feat_tensor, return_features=True
                )
                depth_logits_tensor = self.inverter.route_depth_logits(
                    feats["depth_feat"], torch.argmax(size_probs_tensor, dim=1)
                )
                size_logits = size_probs_tensor.detach().cpu().numpy()[0]
                depth_logits = torch.softmax(depth_logits_tensor, dim=1).detach().cpu().numpy()[0]
                size_reg_cm = _size_norm_to_cm(float(size_reg_norm_tensor.item()))
            else:
                _det2, size_logits_tensor, size_reg_tensor, depth_logits_tensor = self.inverter(input_tensor)
                size_logits = torch.softmax(size_logits_tensor, dim=1).detach().cpu().numpy()[0]
                depth_logits = torch.softmax(depth_logits_tensor, dim=1).detach().cpu().numpy()[0]
                size_reg_cm = float(size_reg_tensor.item())

            if payload["gate_open"]:
                size_index = int(np.argmax(size_logits))
                depth_index = int(np.argmax(depth_logits))
                size_class_cm = float(class_index_to_size(size_index))
                depth_name = coarse_index_to_name(depth_index)
                payload = format_runtime_payload(
                    det_prob=det_prob,
                    threshold=self.threshold,
                    size_class=f"{size_class_cm:g}cm",
                    size_reg_cm=size_reg_cm,
                    depth_coarse=depth_name,
                )
                payload.update(
                    {
                        "size_class_index": size_index,
                        "size_class_cm": size_class_cm,
                        "size_probs": size_logits.tolist(),
                        "depth_coarse_index": depth_index,
                        "depth_coarse_probs": depth_logits.tolist(),
                        "depth_coarse_display": COARSE_DEPTH_DISPLAY.get(depth_name, depth_name),
                        "inverter_kind": self.inverter_kind,
                    }
                )
            else:
                size_index = int(np.argmax(size_logits))
                depth_index = int(np.argmax(depth_logits))
                size_class_cm = float(class_index_to_size(size_index))
                depth_name = coarse_index_to_name(depth_index)
                payload.update(
                    {
                        "size_class_index": size_index,
                        "size_class_cm": size_class_cm,
                        "size_probs": size_logits.tolist(),
                        "size_reg_cm_raw": float(size_reg_cm) if size_reg_cm is not None else None,
                        "depth_coarse_index": depth_index,
                        "depth_coarse_probs": depth_logits.tolist(),
                        "depth_coarse_raw": depth_name,
                        "depth_coarse_display_raw": COARSE_DEPTH_DISPLAY.get(depth_name, depth_name),
                        "inverter_kind": self.inverter_kind,
                    }
                )

        latency_ms = (time.perf_counter() - start) * 1000.0
        payload.setdefault("size_class_index", None)
        payload.setdefault("size_class_cm", None)
        payload.setdefault("size_probs", None)
        payload.setdefault("size_reg_cm_raw", None)
        payload.setdefault("depth_coarse_index", None)
        payload.setdefault("depth_coarse_probs", None)
        payload.setdefault("depth_coarse_display", None)
        payload.setdefault("depth_coarse_raw", None)
        payload.setdefault("depth_coarse_display_raw", None)
        payload["latency_ms"] = float(latency_ms)
        payload["ready"] = True
        payload["threshold"] = float(self.threshold)
        return payload
