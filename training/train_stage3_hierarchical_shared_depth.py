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

from hierarchical_shared_window_mtl import HierarchicalSharedWindowMTL
from input_normalization_v1 import normalize_raw_frames_global, resolve_pressure_conversion, resolve_raw_norm_bounds
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
from train_stage3_raw_size_conditioned_depth import (
    balanced_accuracy_from_cm,
    build_positive_depth_samples_for_file,
    confusion_matrix_counts,
    depth_majority_baseline,
)
from train_triplet_repeat_classifier import load_json, set_seed


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(1, int(num_classes), device=labels.device).view(1, -1)
    return (labels.view(-1, 1) >= thresholds).float()


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


class PositiveDepthSingleInputDataset(Dataset):
    def __init__(
        self,
        records_by_key: Dict[str, dict],
        sample_records: List[dict],
        is_train: bool = False,
        aug_noise_std: float = 0.0,
        aug_scale_jitter: float = 0.0,
        aug_frame_dropout: float = 0.0,
        context_radius: int = 0,
        window_stride: int = WINDOW_STRIDE,
    ):
        self.records_by_key = records_by_key
        self.samples = sample_records
        self.is_train = bool(is_train)
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_scale_jitter = float(max(0.0, aug_scale_jitter))
        self.aug_frame_dropout = float(min(max(0.0, aug_frame_dropout), 0.5))
        self.context_radius = int(max(0, context_radius))
        self.window_stride = int(max(1, window_stride))
        self.raw_norm_lo, self.raw_norm_hi = resolve_raw_norm_bounds()

    def __len__(self):
        return len(self.samples)

    def _extract_window_from_end(self, rec: dict, end_row: int) -> np.ndarray:
        seq_len = int(rec["seq_len"])
        st = int(end_row) - seq_len + 1
        x = normalize_raw_frames_global(
            rec["raw_frames"][st : int(end_row) + 1],
            lo=self.raw_norm_lo,
            hi=self.raw_norm_hi,
            out_hi=1.0,
        )
        return np.expand_dims(x, axis=1)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rec = self.records_by_key[str(sample["group_key"])]
        end_row = int(sample["end_row"])
        if self.context_radius > 0:
            min_end = int(rec["seq_len"]) - 1
            max_end = int(rec["raw_frames"].shape[0]) - 1
            context_items = []
            for offset in range(-self.context_radius, self.context_radius + 1):
                neighbor_end = int(np.clip(end_row + offset * self.window_stride, min_end, max_end))
                context_items.append(self._extract_window_from_end(rec, neighbor_end))
            x = np.stack(context_items, axis=0)
        else:
            x = self._extract_window_from_end(rec, end_row)
        if self.is_train:
            if self.aug_scale_jitter > 0.0:
                scale = 1.0 + float(np.random.uniform(-self.aug_scale_jitter, self.aug_scale_jitter))
                x = x * scale
            if self.aug_noise_std > 0.0:
                x = x + np.random.normal(0.0, self.aug_noise_std, size=x.shape).astype(np.float32)
            if self.aug_frame_dropout > 0.0:
                keep_shape = (x.shape[0], 1, 1, 1) if x.ndim == 4 else (x.shape[0], x.shape[1], 1, 1, 1)
                keep = (np.random.rand(*keep_shape) >= self.aug_frame_dropout).astype(np.float32)
                x = x * keep
            x = np.clip(x, 0.0, 1.0)
        return (
            torch.from_numpy(x.astype(np.float32, copy=False)),
            torch.tensor(int(sample["size_class_index"]), dtype=torch.long),
            torch.tensor(int(sample["depth_coarse_index"]), dtype=torch.long),
            torch.tensor(float(sample["sample_weight"]), dtype=torch.float32),
        )


def build_top2_probs(size_probs: torch.Tensor) -> torch.Tensor:
    top2_probs = size_probs.clone()
    zero_idx = torch.argsort(top2_probs, dim=1, descending=True)[:, 2:]
    top2_probs.scatter_(1, zero_idx, 0.0)
    top2_probs = top2_probs / torch.clamp(top2_probs.sum(dim=1, keepdim=True), min=1e-8)
    return top2_probs


def summarize_depth(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix_counts(y_true, y_pred, len(COARSE_DEPTH_ORDER))
    return {
        "count": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_from_cm(cm)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    depth_class_weight: torch.Tensor,
    size_class_weight: torch.Tensor,
    gt_weight: float,
    hard_weight: float,
    soft_weight: float,
    top2_weight: float,
    size_aux_weight: float,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0

    y_true = []
    y_gt = []
    y_hard = []
    y_soft = []
    y_top2 = []
    gt_size = []
    pred_size = []
    size_probs_rows = []

    with torch.no_grad():
        for x, size_idx, depth_idx, sample_weight in loader:
            x = x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            sample_weight = sample_weight.to(device)

            _size_cls, size_ord_logits, _size_reg, size_probs = model.forward_size(x)
            size_pred_idx = torch.argmax(size_probs, dim=1)
            top2_probs = build_top2_probs(size_probs)

            gt_logits, gt_probs = model.forward_depth(x, size_idx)
            hard_logits, hard_probs = model.forward_depth(x, size_pred_idx)
            soft_logits, soft_probs = model.forward_depth_soft(x, size_probs)
            top2_logits, top2_depth_probs = model.forward_depth_soft(x, top2_probs)

            loss_size = ordinal_loss(size_ord_logits, size_idx, sample_weight=sample_weight, class_weight=size_class_weight)
            loss_gt = ordinal_loss(gt_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight)
            loss_hard = ordinal_loss(hard_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight)
            loss_soft = ordinal_loss(soft_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight)
            loss_top2 = ordinal_loss(top2_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight)
            loss = (
                float(size_aux_weight) * loss_size
                + float(gt_weight) * loss_gt
                + float(hard_weight) * loss_hard
                + float(soft_weight) * loss_soft
                + float(top2_weight) * loss_top2
            )

            batch_weight = float(sample_weight.sum().item())
            total_loss += float(loss.item()) * batch_weight
            total_weight += batch_weight

            y_true.append(depth_idx.cpu().numpy().astype(np.int32))
            y_gt.append(torch.argmax(gt_probs, dim=1).cpu().numpy().astype(np.int32))
            y_hard.append(torch.argmax(hard_probs, dim=1).cpu().numpy().astype(np.int32))
            y_soft.append(torch.argmax(soft_probs, dim=1).cpu().numpy().astype(np.int32))
            y_top2.append(torch.argmax(top2_depth_probs, dim=1).cpu().numpy().astype(np.int32))
            gt_size.append(size_idx.cpu().numpy().astype(np.int32))
            pred_size.append(size_pred_idx.cpu().numpy().astype(np.int32))
            size_probs_rows.append(size_probs.cpu().numpy().astype(np.float32))

    y_true_np = np.concatenate(y_true)
    y_gt_np = np.concatenate(y_gt)
    y_hard_np = np.concatenate(y_hard)
    y_soft_np = np.concatenate(y_soft)
    y_top2_np = np.concatenate(y_top2)
    gt_size_np = np.concatenate(gt_size)
    pred_size_np = np.concatenate(pred_size)
    size_probs_np = np.concatenate(size_probs_rows, axis=0)
    route_match = pred_size_np == gt_size_np
    top2_idx = np.argsort(-size_probs_np, axis=1)[:, :2]
    size_top2 = float(np.mean(np.any(top2_idx == gt_size_np[:, None], axis=1)))

    return {
        "loss": float(total_loss / max(total_weight, 1.0)),
        "count": int(len(y_true_np)),
        "size_top1": float(np.mean(route_match)),
        "size_top2": size_top2,
        "route_match_rate": float(np.mean(route_match)),
        "gt_route": summarize_depth(y_true_np, y_gt_np),
        "hard_route": summarize_depth(y_true_np, y_hard_np),
        "soft_route": summarize_depth(y_true_np, y_soft_np),
        "top2_soft_route": summarize_depth(y_true_np, y_top2_np),
        "hard_route_when_size_correct": summarize_depth(y_true_np[route_match], y_hard_np[route_match]) if np.any(route_match) else None,
        "hard_route_when_size_wrong": summarize_depth(y_true_np[~route_match], y_hard_np[~route_match]) if np.any(~route_match) else None,
    }


def plot_curves(history: Dict[str, List[float]], output_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_soft_bal_acc"], label="val soft")
    axes[1].plot(epochs, history["test_soft_bal_acc"], label="test soft")
    axes[1].set_title("Soft Route Balanced Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_hard_bal_acc"], label="val hard")
    axes[2].plot(epochs, history["test_hard_bal_acc"], label="test hard")
    axes[2].set_title("Hard Route Balanced Accuracy")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, history["val_gt_bal_acc"], label="val gt")
    axes[3].plot(epochs, history["test_gt_bal_acc"], label="test gt")
    axes[3].set_title("GT Route Balanced Accuracy")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    axes[4].plot(epochs, history["val_size_top1"], label="val size top1")
    axes[4].plot(epochs, history["test_size_top1"], label="test size top1")
    axes[4].set_title("Router Top1")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def freeze_detection_path(model: HierarchicalSharedWindowMTL):
    for p in model.det_adapter.parameters():
        p.requires_grad_(False)
    for p in model.det_head.parameters():
        p.requires_grad_(False)


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERHMTL_DEPTH_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERHMTL_DEPTH_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERHMTL_DEPTH_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERHMTL_DEPTH_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERHMTL_DEPTH_EPOCHS", "80"))
    batch_size: int = int(os.environ.get("PAPERHMTL_DEPTH_BATCH_SIZE", "40"))
    eval_batch_size: int = int(os.environ.get("PAPERHMTL_DEPTH_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERHMTL_DEPTH_LR", "2e-4"))
    encoder_lr: float = float(os.environ.get("PAPERHMTL_DEPTH_ENCODER_LR", "5e-5"))
    size_lr: float = float(os.environ.get("PAPERHMTL_DEPTH_SIZE_LR", "8e-5"))
    weight_decay: float = float(os.environ.get("PAPERHMTL_DEPTH_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("PAPERHMTL_DEPTH_DROPOUT", "0.30"))
    patience: int = int(os.environ.get("PAPERHMTL_DEPTH_PATIENCE", "16"))
    grad_clip: float = float(os.environ.get("PAPERHMTL_DEPTH_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("PAPERHMTL_DEPTH_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERHMTL_DEPTH_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERHMTL_DEPTH_AUG_FRAME_DROPOUT", "0.02"))
    gt_weight: float = float(os.environ.get("PAPERHMTL_DEPTH_GT_WEIGHT", "0.2"))
    hard_weight: float = float(os.environ.get("PAPERHMTL_DEPTH_HARD_WEIGHT", "0.2"))
    soft_weight: float = float(os.environ.get("PAPERHMTL_DEPTH_SOFT_WEIGHT", "1.0"))
    top2_weight: float = float(os.environ.get("PAPERHMTL_DEPTH_TOP2_WEIGHT", "0.0"))
    size_aux_weight: float = float(os.environ.get("PAPERHMTL_DEPTH_SIZE_AUX_WEIGHT", "0.25"))
    gt_pretrain_epochs: int = int(os.environ.get("PAPERHMTL_DEPTH_GT_PRETRAIN_EPOCHS", "0"))
    context_radius: int = int(os.environ.get("PAPERHMTL_DEPTH_CONTEXT_RADIUS", "1"))
    num_workers: int = int(os.environ.get("PAPERHMTL_DEPTH_NUM_WORKERS", "0"))

    def __post_init__(self):
        self.data_root = os.environ.get("PAPERHMTL_DEPTH_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERHMTL_DEPTH_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERHMTL_DEPTH_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERHMTL_DEPTH_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERHMTL_DEPTH_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_hierarchical_shared_depth"),
        )
        self.stage2_ckpt = os.environ.get(
            "PAPERHMTL_DEPTH_STAGE2_CKPT",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage2_hierarchical_shared_size", "paper_stage2_hierarchical_shared_size_best.pth"),
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

    common_groups = sorted(list(set(v["base_group"] for v in rec1.values()) & set(v["base_group"] for v in rec2.values()) & set(v["base_group"] for v in rec3.values())))
    common_set = set(common_groups)
    train_records = {k: v for k, v in rec1.items() if v["base_group"] in common_set}
    val_records = {k: v for k, v in rec2.items() if v["base_group"] in common_set}
    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_set}
    train_samples = [s for s in samples1 if s["base_group"] in common_set]
    val_samples = [s for s in samples2 if s["base_group"] in common_set]
    test_samples = [s for s in samples3 if s["base_group"] in common_set]

    ds_train = PositiveDepthSingleInputDataset(
        train_records,
        train_samples,
        is_train=True,
        aug_noise_std=cfg.aug_noise_std,
        aug_scale_jitter=cfg.aug_scale_jitter,
        aug_frame_dropout=cfg.aug_frame_dropout,
        context_radius=cfg.context_radius,
        window_stride=cfg.stride,
    )
    ds_val = PositiveDepthSingleInputDataset(val_records, val_samples, is_train=False, context_radius=cfg.context_radius, window_stride=cfg.stride)
    ds_test = PositiveDepthSingleInputDataset(test_records, test_samples, is_train=False, context_radius=cfg.context_radius, window_stride=cfg.stride)

    train_depth = np.array([int(s["depth_coarse_index"]) for s in train_samples], dtype=np.int32)
    depth_counts = np.bincount(train_depth, minlength=len(COARSE_DEPTH_ORDER)).astype(np.float32)
    depth_class_weight_np = depth_counts.sum() / np.maximum(depth_counts * len(COARSE_DEPTH_ORDER), 1.0)
    sample_weight_np = np.array([float(s["sample_weight"]) for s in train_samples], dtype=np.float32)
    sampler_weight_np = depth_class_weight_np[train_depth] * sample_weight_np

    train_size = np.array([int(s["size_class_index"]) for s in train_samples], dtype=np.int32)
    size_counts = np.bincount(train_size, minlength=len(SIZE_VALUES_CM)).astype(np.float32)
    size_class_weight_np = size_counts.sum() / np.maximum(size_counts * len(SIZE_VALUES_CM), 1.0)

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
    payload = torch.load(cfg.stage2_ckpt, map_location="cpu")
    model_cfg = payload.get("model_config", {})
    model = HierarchicalSharedWindowMTL(
        dropout=float(model_cfg.get("dropout", cfg.dropout)),
        size_residual_scale=float(model_cfg.get("size_residual_scale", 0.20)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    freeze_detection_path(model)

    depth_class_weight_t = torch.tensor(depth_class_weight_np, dtype=torch.float32, device=device)
    size_class_weight_t = torch.tensor(size_class_weight_np, dtype=torch.float32, device=device)

    trainable_params = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": float(cfg.encoder_lr)},
        {"params": [p for p in model.size_temporal_pool.parameters() if p.requires_grad], "lr": float(cfg.size_lr)},
        {"params": [p for p in model.size_phase_pool.parameters() if p.requires_grad], "lr": float(cfg.size_lr)},
        {"params": [p for p in model.depth_temporal_pool.parameters() if p.requires_grad], "lr": float(cfg.lr)},
        {"params": [p for p in model.depth_phase_pool.parameters() if p.requires_grad], "lr": float(cfg.lr)},
        {"params": [p for p in model.size_adapter.parameters() if p.requires_grad], "lr": float(cfg.size_lr)},
        {"params": [p for p in model.size_head.parameters() if p.requires_grad], "lr": float(cfg.size_lr)},
        {"params": [p for p in model.size_residual_head.parameters() if p.requires_grad], "lr": float(cfg.size_lr)},
        {"params": [p for p in model.depth_adapter.parameters() if p.requires_grad], "lr": float(cfg.lr)},
        {"params": [p for p in model.depth_experts.parameters() if p.requires_grad], "lr": float(cfg.lr)},
    ]
    clip_params = []
    for group in trainable_params:
        clip_params.extend(group["params"])
    optimizer = optim.AdamW(trainable_params, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_gt_bal_acc": [],
        "val_hard_bal_acc": [],
        "val_soft_bal_acc": [],
        "test_gt_bal_acc": [],
        "test_hard_bal_acc": [],
        "test_soft_bal_acc": [],
        "val_size_top1": [],
        "test_size_top1": [],
    }
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        if epoch <= int(cfg.gt_pretrain_epochs):
            train_gt_weight = 1.0
            train_hard_weight = 0.0
            train_soft_weight = 0.0
            train_top2_weight = 0.0
        else:
            train_gt_weight = float(cfg.gt_weight)
            train_hard_weight = float(cfg.hard_weight)
            train_soft_weight = float(cfg.soft_weight)
            train_top2_weight = float(cfg.top2_weight)
        for x, size_idx, depth_idx, sample_weight in train_loader:
            x = x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            sample_weight = sample_weight.to(device)

            optimizer.zero_grad()
            _size_cls, size_ord_logits, _size_reg, size_probs = model.forward_size(x)
            size_pred_idx = torch.argmax(size_probs, dim=1)
            top2_probs = build_top2_probs(size_probs)

            gt_logits, _gt_probs = model.forward_depth(x, size_idx)
            hard_logits, _hard_probs = model.forward_depth(x, size_pred_idx)
            soft_logits, _soft_probs = model.forward_depth_soft(x, size_probs)
            top2_logits, _top2_probs = model.forward_depth_soft(x, top2_probs)

            loss_size = ordinal_loss(size_ord_logits, size_idx, sample_weight=sample_weight, class_weight=size_class_weight_t)
            loss_gt = ordinal_loss(gt_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight_t)
            loss_hard = ordinal_loss(hard_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight_t)
            loss_soft = ordinal_loss(soft_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight_t)
            loss_top2 = ordinal_loss(top2_logits, depth_idx, sample_weight=sample_weight, class_weight=depth_class_weight_t)
            loss = (
                float(cfg.size_aux_weight) * loss_size
                + float(train_gt_weight) * loss_gt
                + float(train_hard_weight) * loss_hard
                + float(train_soft_weight) * loss_soft
                + float(train_top2_weight) * loss_top2
            )
            loss.backward()
            nn.utils.clip_grad_norm_(clip_params, float(cfg.grad_clip))
            optimizer.step()

            batch_weight = float(sample_weight.sum().item())
            total_loss += float(loss.item()) * batch_weight
            total_weight += batch_weight

        scheduler.step()
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            depth_class_weight_t,
            size_class_weight_t,
            gt_weight=cfg.gt_weight,
            hard_weight=cfg.hard_weight,
            soft_weight=cfg.soft_weight,
            top2_weight=cfg.top2_weight,
            size_aux_weight=cfg.size_aux_weight,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            depth_class_weight_t,
            size_class_weight_t,
            gt_weight=cfg.gt_weight,
            hard_weight=cfg.hard_weight,
            soft_weight=cfg.soft_weight,
            top2_weight=cfg.top2_weight,
            size_aux_weight=cfg.size_aux_weight,
        )

        history["train_loss"].append(float(total_loss / max(total_weight, 1.0)))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_gt_bal_acc"].append(float(val_metrics["gt_route"]["balanced_accuracy"]))
        history["val_hard_bal_acc"].append(float(val_metrics["hard_route"]["balanced_accuracy"]))
        history["val_soft_bal_acc"].append(float(val_metrics["soft_route"]["balanced_accuracy"]))
        history["test_gt_bal_acc"].append(float(test_metrics["gt_route"]["balanced_accuracy"]))
        history["test_hard_bal_acc"].append(float(test_metrics["hard_route"]["balanced_accuracy"]))
        history["test_soft_bal_acc"].append(float(test_metrics["soft_route"]["balanced_accuracy"]))
        history["val_size_top1"].append(float(val_metrics["size_top1"]))
        history["test_size_top1"].append(float(test_metrics["size_top1"]))

        score = (
            float(val_metrics["soft_route"]["balanced_accuracy"]),
            float(val_metrics["hard_route"]["balanced_accuracy"]),
            float(val_metrics["gt_route"]["balanced_accuracy"]),
            float(val_metrics["size_top1"]),
            -float(val_metrics["loss"]),
        )
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

    if best is None:
        raise RuntimeError("Stage3 hierarchical depth training did not produce a checkpoint.")

    ckpt_path = os.path.join(cfg.output_dir, "paper_stage3_hierarchical_shared_depth_best.pth")
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "model_name": "HierarchicalSharedWindowMTL-Depth",
            "stage2_ckpt": cfg.stage2_ckpt,
            "model_config": {
                "dropout": float(model_cfg.get("dropout", cfg.dropout)),
            },
            "loss_weights": {
                "size_aux": float(cfg.size_aux_weight),
                "gt": float(cfg.gt_weight),
                "hard": float(cfg.hard_weight),
                "soft": float(cfg.soft_weight),
                "top2": float(cfg.top2_weight),
            },
            "gt_pretrain_epochs": int(cfg.gt_pretrain_epochs),
            "raw_norm_lo": float(raw_norm_lo),
            "raw_norm_hi": float(raw_norm_hi),
            "pressure_scale": float(pressure_scale),
            "pressure_offset": float(pressure_offset),
            "depth_class_weight": depth_class_weight_np.tolist(),
            "size_class_weight": size_class_weight_np.tolist(),
            "train_mode": "freeze_detection_finetune_encoder_size_and_depth_with_route_losses",
            "context_radius": int(cfg.context_radius),
        },
        ckpt_path,
    )

    curve_path = os.path.join(cfg.output_dir, "paper_stage3_hierarchical_shared_depth_curves.png")
    plot_curves(history, curve_path)

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "HierarchicalSharedWindowMTL-Depth",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "training_mode": "freeze_detection_finetune_encoder_size_and_depth_with_route_losses",
        "stage2_ckpt": cfg.stage2_ckpt,
        "selection_policy": "max(val soft-route balanced_accuracy) -> max(val hard-route balanced_accuracy) -> max(val gt-route balanced_accuracy) -> max(val size_top1) -> min(val loss)",
        "loss_weights": {
            "size_aux": float(cfg.size_aux_weight),
            "gt": float(cfg.gt_weight),
            "hard": float(cfg.hard_weight),
            "soft": float(cfg.soft_weight),
            "top2": float(cfg.top2_weight),
        },
        "gt_pretrain_epochs": int(cfg.gt_pretrain_epochs),
        "context_radius": int(cfg.context_radius),
        "raw_norm_lo": float(raw_norm_lo),
        "raw_norm_hi": float(raw_norm_hi),
        "pressure_scale": float(pressure_scale),
        "pressure_offset": float(pressure_offset),
        "train_sample_count": int(len(train_samples)),
        "val_sample_count": int(len(val_samples)),
        "test_sample_count": int(len(test_samples)),
        "test_majority_baseline": depth_majority_baseline(train_samples, test_samples),
        "best_epoch": int(best["epoch"]),
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "curve_path": curve_path,
        "ckpt_path": ckpt_path,
    }
    with open(os.path.join(cfg.output_dir, "paper_stage3_hierarchical_shared_depth_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
