import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Setup paths for the release package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

for path in [MODELS_DIR, UTILS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from raw_positive_size_model_v2 import RawPositiveSizeModelV2
from input_normalization_v1 import resolve_pressure_conversion, resolve_raw_norm_bounds
from task_protocol_v1 import INPUT_SEQ_LEN, INPUT_SHAPE, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
from train_stage3_raw_size_conditioned_depth import (
    PositiveDepthDataset,
    build_positive_depth_samples_for_file,
    compute_concept_stats,
    compute_raw_scale,
)
from train_triplet_repeat_classifier import load_json, set_seed


class PositiveSizeDataset(Dataset):
    def __init__(self, base_ds: PositiveDepthDataset, samples: List[dict]):
        self.base_ds = base_ds
        self.samples = samples

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        raw_x, norm_x, size_idx, _depth_idx, _concept_target, _sample_weight = self.base_ds[idx]
        group_key = str(self.samples[idx]["group_key"])
        size_cm = float(self.base_ds.records_by_key[group_key]["size_cm"])
        return raw_x, norm_x, size_idx, torch.tensor(size_cm, dtype=torch.float32)


def apply_input_mode(raw_x: torch.Tensor, norm_x: torch.Tensor, input_mode: str):
    if str(input_mode).lower() == "single_tied":
        norm_x = raw_x
    return raw_x, norm_x


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(1, int(num_classes), device=labels.device).view(1, -1)
    return (labels.view(-1, 1) >= thresholds).float()


def ordinal_logits_to_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    k = int(num_classes)
    if logits.dim() != 2 or logits.shape[1] != (k - 1):
        raise ValueError(f"logits must have shape (B, {k - 1}), got {tuple(logits.shape)}")
    q = torch.sigmoid(logits)
    probs = torch.zeros(logits.shape[0], k, dtype=logits.dtype, device=logits.device)
    probs[:, 0] = 1.0 - q[:, 0]
    if k > 2:
        probs[:, 1:-1] = q[:, :-1] - q[:, 1:]
    probs[:, -1] = q[:, -1]
    probs = torch.clamp(probs, min=1e-8)
    probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-8)
    return probs


def ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = ordinal_targets(labels, logits.shape[1] + 1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    if class_weight is not None:
        loss = loss * class_weight[labels.long()]
    if sample_weight is not None:
        loss = loss * sample_weight.view(-1)
        return loss.sum() / torch.clamp(sample_weight.sum(), min=1.0)
    return loss.mean()


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(probs, axis=1)
    top1 = float(np.mean(y_pred == y_true))
    top2 = float(np.mean(np.any(np.argsort(-probs, axis=1)[:, :2] == y_true[:, None], axis=1)))
    return {"top1": top1, "top2": top2}


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    ae = np.abs(pred - true)
    return {"mae": float(np.mean(ae)), "median_ae": float(np.median(ae))}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weight: Optional[torch.Tensor] = None,
    ordinal_loss_weight: float = 1.0,
    input_mode: str = "dual",
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    probs_all, reg_all, y_cls_all, y_reg_all = [], [], [], []
    size_values_t = torch.tensor(SIZE_VALUES_CM, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        for raw_x, norm_x, size_idx, size_cm in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            raw_x, norm_x = apply_input_mode(raw_x, norm_x, input_mode)
            size_idx = size_idx.to(device)
            size_cm = size_cm.to(device)

            _size_logits, size_ord_logits, _size_reg_norm, size_probs = model(raw_x, norm_x)
            loss = float(ordinal_loss_weight) * ordinal_loss(size_ord_logits, size_idx, class_weight=class_weight)
            size_pred_cm = torch.sum(size_probs * size_values_t.to(size_probs.dtype), dim=1)

            total_loss += float(loss.item()) * int(size_idx.shape[0])
            total_n += int(size_idx.shape[0])
            probs_all.append(size_probs.cpu().numpy().astype(np.float64))
            reg_all.append(size_pred_cm.cpu().numpy().astype(np.float64))
            y_cls_all.append(size_idx.cpu().numpy().astype(np.int32))
            y_reg_all.append(size_cm.cpu().numpy().astype(np.float64))

    probs_np = np.concatenate(probs_all, axis=0)
    reg_np = np.concatenate(reg_all, axis=0)
    y_cls_np = np.concatenate(y_cls_all, axis=0)
    y_reg_np = np.concatenate(y_reg_all, axis=0)
    cls = classification_metrics(y_cls_np, probs_np)
    reg = regression_metrics(reg_np, y_reg_np)
    return {
        "loss": float(total_loss / max(total_n, 1)),
        "top1": cls["top1"],
        "top2": cls["top2"],
        "mae": reg["mae"],
        "median_ae": reg["median_ae"],
        "count": int(total_n),
    }


def plot_curves(history: Dict[str, List[float]], output_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_top1"], label="val top1")
    axes[1].plot(epochs, history["test_top1"], label="test top1")
    axes[1].set_title("Top1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_top2"], label="val top2")
    axes[2].plot(epochs, history["test_top2"], label="test top2")
    axes[2].set_title("Top2")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, history["val_mae"], label="val mae")
    axes[3].plot(epochs, history["test_mae"], label="test mae")
    axes[3].set_title("MAE")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def warm_start_from_detector(model: nn.Module, ckpt_path: str) -> int:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model_state = model.state_dict()
    loaded = 0

    def copy_tensor(dst_key: str, src_key: str):
        nonlocal loaded
        if src_key in state and dst_key in model_state and state[src_key].shape == model_state[dst_key].shape:
            model_state[dst_key] = state[src_key]
            loaded += 1

    for key in list(model_state.keys()):
        if key.startswith("amplitude_encoder."):
            copy_tensor(key, key.replace("amplitude_encoder.", "raw_encoder."))
        elif key.startswith("shape_encoder."):
            copy_tensor(key, key.replace("shape_encoder.", "raw_encoder."))
        elif key.startswith("delta_encoder.") and getattr(model, "use_delta", True):
            copy_tensor(key, key.replace("delta_encoder.", "delta_encoder."))
        elif key.startswith("temporal_input.") or key.startswith("temporal_blocks."):
            copy_tensor(key, key)

    model.load_state_dict(model_state)
    return loaded


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERSIZEV2_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERSIZEV2_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERSIZEV2_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERSIZEV2_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERSIZEV2_EPOCHS", "120"))
    batch_size: int = int(os.environ.get("PAPERSIZEV2_BATCH_SIZE", "48"))
    eval_batch_size: int = int(os.environ.get("PAPERSIZEV2_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERSIZEV2_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("PAPERSIZEV2_WEIGHT_DECAY", "8e-4"))
    dropout: float = float(os.environ.get("PAPERSIZEV2_DROPOUT", "0.22"))
    frame_feature_dim: int = int(os.environ.get("PAPERSIZEV2_FRAME_DIM", "24"))
    temporal_channels: int = int(os.environ.get("PAPERSIZEV2_TEMPORAL_CHANNELS", "48"))
    temporal_blocks: int = int(os.environ.get("PAPERSIZEV2_TEMPORAL_BLOCKS", "2"))
    use_delta: bool = os.environ.get("PAPERSIZEV2_USE_DELTA", "0").strip() != "0"
    use_implicit_change: bool = os.environ.get("PAPERSIZEV2_USE_IMPLICIT_CHANGE", "0").strip() != "0"
    use_phase_pooling: bool = os.environ.get("PAPERSIZEV2_USE_PHASE_POOLING", "0").strip() != "0"
    use_residual_head: bool = os.environ.get("PAPERSIZEV2_USE_RESIDUAL", "1").strip() != "0"
    ordinal_loss_weight: float = float(os.environ.get("PAPERSIZEV2_ORD_LOSS_WEIGHT", "1.0"))
    patience: int = int(os.environ.get("PAPERSIZEV2_PATIENCE", "24"))
    grad_clip: float = float(os.environ.get("PAPERSIZEV2_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("PAPERSIZEV2_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERSIZEV2_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERSIZEV2_AUG_FRAME_DROPOUT", "0.02"))
    num_workers: int = int(os.environ.get("PAPERSIZEV2_NUM_WORKERS", "0"))
    input_mode: str = os.environ.get("PAPERSIZEV2_INPUT_MODE", "dual").strip().lower()

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        if self.input_mode not in {"dual", "single_tied"}:
            self.input_mode = "dual"
        self.data_root = os.environ.get("PAPERSIZEV2_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERSIZEV2_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERSIZEV2_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERSIZEV2_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERSIZEV2_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage2_raw_positive_size_v2"),
        )
        self.warm_start_ckpt = os.environ.get(
            "PAPERSIZEV2_WARM_START_CKPT",
            os.path.join(
                PROJECT_DIR,
                "experiments",
                "outputs_stage1_dualstream_mstcn_detection_raw_delta",
                "paper_stage1_dualstream_mstcn_best.pth",
            ),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    raw_norm_lo, raw_norm_hi = resolve_raw_norm_bounds()
    pressure_scale, pressure_offset = resolve_pressure_conversion()

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    rec1, samples1 = build_positive_depth_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, samples2 = build_positive_depth_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3 = build_positive_depth_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        )
    )
    common_set = set(common_base_groups)
    train_records = {k: v for k, v in rec1.items() if v["base_group"] in common_set}
    val_records = {k: v for k, v in rec2.items() if v["base_group"] in common_set}
    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_set}
    train_samples = [s for s in samples1 if s["base_group"] in common_set]
    val_samples = [s for s in samples2 if s["base_group"] in common_set]
    test_samples = [s for s in samples3 if s["base_group"] in common_set]

    raw_scale = compute_raw_scale(train_records)
    concept_mean, concept_std = compute_concept_stats(train_samples)

    base_train_ds = PositiveDepthDataset(
        train_records,
        train_samples,
        raw_scale=raw_scale,
        concept_mean=concept_mean,
        concept_std=concept_std,
        is_train=True,
        aug_noise_std=cfg.aug_noise_std,
        aug_scale_jitter=cfg.aug_scale_jitter,
        aug_frame_dropout=cfg.aug_frame_dropout,
    )
    base_val_ds = PositiveDepthDataset(val_records, val_samples, raw_scale, concept_mean, concept_std, is_train=False)
    base_test_ds = PositiveDepthDataset(test_records, test_samples, raw_scale, concept_mean, concept_std, is_train=False)
    ds_train = PositiveSizeDataset(base_train_ds, train_samples)
    ds_val = PositiveSizeDataset(base_val_ds, val_samples)
    ds_test = PositiveSizeDataset(base_test_ds, test_samples)

    train_size = np.array([int(s["size_class_index"]) for s in train_samples], dtype=np.int32)
    counts = np.bincount(train_size, minlength=len(SIZE_VALUES_CM)).astype(np.float32)
    class_weight_np = counts.sum() / np.maximum(counts * len(SIZE_VALUES_CM), 1.0)
    sampler_weight_np = class_weight_np[train_size]

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=WeightedRandomSampler(torch.tensor(sampler_weight_np, dtype=torch.float32), num_samples=len(train_samples), replacement=True),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(ds_val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(ds_test, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight_t = torch.tensor(class_weight_np, dtype=torch.float32, device=device)
    model = RawPositiveSizeModelV2(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        num_size_classes=len(SIZE_VALUES_CM),
        use_delta=cfg.use_delta,
        use_implicit_change=cfg.use_implicit_change,
        use_phase_pooling=cfg.use_phase_pooling,
        use_residual_head=cfg.use_residual_head,
    ).to(device)

    warm_loaded = warm_start_from_detector(model, cfg.warm_start_ckpt)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "val_top1": [], "test_top1": [], "val_top2": [], "test_top2": [], "val_mae": [], "test_mae": []}
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for raw_x, norm_x, size_idx, _size_cm in train_loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            raw_x, norm_x = apply_input_mode(raw_x, norm_x, cfg.input_mode)
            size_idx = size_idx.to(device)

            optimizer.zero_grad()
            _size_logits, size_ord_logits, _size_reg_norm, _size_probs = model(raw_x, norm_x)
            loss = float(cfg.ordinal_loss_weight) * ordinal_loss(size_ord_logits, size_idx, class_weight=class_weight_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

            total_loss += float(loss.item()) * int(size_idx.shape[0])
            total_n += int(size_idx.shape[0])
        scheduler.step()

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            class_weight_t,
            cfg.ordinal_loss_weight,
            cfg.input_mode,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            class_weight_t,
            cfg.ordinal_loss_weight,
            cfg.input_mode,
        )
        history["train_loss"].append(float(total_loss / max(total_n, 1)))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_top1"].append(float(val_metrics["top1"]))
        history["test_top1"].append(float(test_metrics["top1"]))
        history["val_top2"].append(float(val_metrics["top2"]))
        history["test_top2"].append(float(test_metrics["top2"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        history["test_mae"].append(float(test_metrics["mae"]))

        score = (-float(val_metrics["mae"]), float(val_metrics["top1"]), float(val_metrics["top2"]), -float(val_metrics["loss"]))
        if best is None or score > best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            patience_left = int(cfg.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    ckpt_path = os.path.join(cfg.output_dir, "paper_stage2_raw_positive_size_v2_best.pth")
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "input_shape": list(INPUT_SHAPE),
            "raw_scale": float(raw_scale),
            "raw_norm_lo": float(raw_norm_lo),
            "raw_norm_hi": float(raw_norm_hi),
            "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
            "class_weight": class_weight_np.tolist(),
            "router_model_name": "RawPositiveSizeModelV2",
            "warm_start_ckpt": cfg.warm_start_ckpt,
            "warm_loaded_tensors": int(warm_loaded),
            "model_config": {
                "seq_len": cfg.seq_len,
                "frame_feature_dim": cfg.frame_feature_dim,
                "temporal_channels": cfg.temporal_channels,
                "temporal_blocks": cfg.temporal_blocks,
                "dropout": cfg.dropout,
                "num_size_classes": len(SIZE_VALUES_CM),
                "use_delta": bool(cfg.use_delta),
                "use_implicit_change": bool(cfg.use_implicit_change),
                "use_phase_pooling": bool(cfg.use_phase_pooling),
                "use_residual_head": bool(cfg.use_residual_head),
            },
            "train_config": {
                "ordinal_loss_weight": cfg.ordinal_loss_weight,
                "input_mode": cfg.input_mode,
                "router_probability_source": "ordinal_head",
            },
        },
        ckpt_path,
    )

    plot_curves(history, os.path.join(cfg.output_dir, "paper_stage2_raw_positive_size_v2_curves.png"))

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "RawPositiveSizeModelV2",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "best_epoch": int(best["epoch"]),
        "warm_loaded_tensors": int(warm_loaded),
        "model_config": {
            "use_delta": bool(cfg.use_delta),
            "use_implicit_change": bool(cfg.use_implicit_change),
            "use_phase_pooling": bool(cfg.use_phase_pooling),
            "use_residual_head": bool(cfg.use_residual_head),
        },
        "train_config": {
            "ordinal_loss_weight": float(cfg.ordinal_loss_weight),
            "input_mode": cfg.input_mode,
            "router_probability_source": "ordinal_head",
        },
        "loss_formula_train": f"L_size = {float(cfg.ordinal_loss_weight):.2f}*Ordinal(size)",
        "input_normalization": {
            "type": "fixed_global_clipped_raw + per-frame_shape_branch",
            "input_mode": cfg.input_mode,
            "raw_norm_lo": float(raw_norm_lo),
            "raw_norm_hi": float(raw_norm_hi),
            "pressure_scale": float(pressure_scale),
            "pressure_offset": float(pressure_offset),
        },
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "curve_path": os.path.join(cfg.output_dir, "paper_stage2_raw_positive_size_v2_curves.png"),
    }
    with open(os.path.join(cfg.output_dir, "paper_stage2_raw_positive_size_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
