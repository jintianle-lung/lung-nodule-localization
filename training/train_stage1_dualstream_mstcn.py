import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup paths for the release package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

for path in [MODELS_DIR, UTILS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dual_stream_mstcn_detection import DualStreamMSTCNDetector
from task_protocol_v1 import INPUT_SEQ_LEN, WINDOW_STRIDE, protocol_summary
from train_stage1_detection import (
    CenterLabelSequenceDataset,
    build_detection_samples_for_file,
    merge_condition_stats,
)
from train_file12_holdout_file3 import (
    downsample_negatives,
    plot_curves,
    plot_curves_plotly,
    split_base_groups_train_val_balanced,
)
from train_triplet_repeat_classifier import (
    build_pr,
    build_roc,
    compute_cls_metrics,
    env_bool,
    load_json,
    select_best_f1_threshold,
    set_seed,
)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion_hard: nn.Module,
    optimizer: optim.Optimizer = None,
    grad_clip: float = 0.0,
    criterion_soft: nn.Module = None,
    soft_loss_weight: float = 0.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_y = []
    all_score = []
    loss_sum = 0.0
    hard_loss_sum = 0.0
    soft_loss_sum = 0.0
    n = 0
    with torch.set_grad_enabled(is_train):
        for x, y_hard, y_soft in loader:
            x = x.to(device)
            y_hard = y_hard.to(device).unsqueeze(1)
            y_soft = y_soft.to(device).unsqueeze(1)
            if is_train:
                optimizer.zero_grad()

            logit = model(x)
            hard_loss = criterion_hard(logit, y_hard)
            soft_loss = torch.zeros((), device=device)
            if is_train and criterion_soft is not None and float(soft_loss_weight) > 0.0:
                soft_loss = criterion_soft(logit, y_soft)
            loss = hard_loss + float(soft_loss_weight) * soft_loss

            if is_train:
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

            bs = x.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs
            hard_loss_sum += float(hard_loss.item()) * bs
            soft_loss_sum += float(soft_loss.item()) * bs
            all_y.append(y_hard.detach().cpu().numpy().reshape(-1))
            all_score.append(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_y, axis=0).astype(np.int32)
    y_score = np.concatenate(all_score, axis=0).astype(np.float64)
    return {
        "loss": float(loss_sum / max(n, 1)),
        "hard_loss": float(hard_loss_sum / max(n, 1)),
        "soft_loss": float(soft_loss_sum / max(n, 1)),
        "y_true": y_true,
        "y_score": y_score,
    }


def build_checkpoint(
    model: DualStreamMSTCNDetector,
    cfg,
    epoch: int,
    best_record: dict = None,
):
    return {
        "epoch": int(epoch),
        "best_record": best_record,
        "config": {
            "seq_len": cfg.seq_len,
            "stride": cfg.stride,
            "dropout": cfg.dropout,
            "frame_feature_dim": cfg.frame_feature_dim,
            "temporal_channels": cfg.temporal_channels,
            "temporal_blocks": cfg.temporal_blocks,
            "use_delta_branch": bool(cfg.use_delta_branch),
            "label_mode": cfg.label_mode,
            "input_normalization": cfg.input_normalization,
        },
        "protocol_v1": protocol_summary(),
        "feature_dim": int(model.feature_dim),
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": {
            "raw_encoder": model.raw_encoder.state_dict(),
            "delta_encoder": model.delta_encoder.state_dict() if model.delta_encoder is not None else None,
        },
        "temporal_state_dict": {
            "temporal_input": model.temporal_input.state_dict(),
            "temporal_blocks": model.temporal_blocks.state_dict(),
        },
        "pooling_state_dict": model.pooling.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
    }


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERDSM_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERDSM_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERDSM_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERDSM_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERDSM_EPOCHS", "90"))
    batch_size: int = int(os.environ.get("PAPERDSM_BATCH_SIZE", "64"))
    lr: float = float(os.environ.get("PAPERDSM_LR", "1e-4"))
    weight_decay: float = float(os.environ.get("PAPERDSM_WEIGHT_DECAY", "1.2e-3"))
    dropout: float = float(os.environ.get("PAPERDSM_DROPOUT", "0.35"))
    frame_feature_dim: int = int(os.environ.get("PAPERDSM_FRAME_FEATURE_DIM", "32"))
    temporal_channels: int = int(os.environ.get("PAPERDSM_TEMPORAL_CHANNELS", "64"))
    temporal_blocks: int = int(os.environ.get("PAPERDSM_TEMPORAL_BLOCKS", "3"))
    patience: int = int(os.environ.get("PAPERDSM_PATIENCE", "18"))
    grad_clip: float = float(os.environ.get("PAPERDSM_GRAD_CLIP", "1.0"))
    max_neg_pos_ratio: float = float(os.environ.get("PAPERDSM_MAX_NEG_POS_RATIO", "2.5"))
    train_pos_weight_scale: float = float(os.environ.get("PAPERDSM_POS_WEIGHT_SCALE", "1.0"))
    soft_loss_weight: float = float(os.environ.get("PAPERDSM_SOFT_LOSS_WEIGHT", "0.10"))
    aug_noise_std: float = float(os.environ.get("PAPERDSM_AUG_NOISE_STD", "0.015"))
    aug_scale_jitter: float = float(os.environ.get("PAPERDSM_AUG_SCALE_JITTER", "0.10"))
    aug_frame_dropout: float = float(os.environ.get("PAPERDSM_AUG_FRAME_DROPOUT", "0.03"))
    min_lr_ratio: float = float(os.environ.get("PAPERDSM_MIN_LR_RATIO", "0.05"))
    num_workers: int = int(os.environ.get("PAPERDSM_NUM_WORKERS", "0"))
    save_plotly_html: bool = env_bool("PAPERDSM_SAVE_PLOTLY_HTML", True)
    use_delta_branch: bool = env_bool("PAPERDSM_USE_DELTA_BRANCH", False)
    label_mode: str = os.environ.get("PAPERDSM_LABEL_MODE", "window_overlap_positive").strip().lower()
    input_normalization: str = os.environ.get("PAPERDSM_INPUT_NORMALIZATION", "window_minmax").strip().lower()

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        if self.label_mode not in {"center_frame_positive", "window_overlap_positive"}:
            raise ValueError(f"Unsupported PAPERDSM_LABEL_MODE={self.label_mode}")
        if self.input_normalization not in {"fixed_global_clipped", "window_minmax"}:
            raise ValueError(f"Unsupported PAPERDSM_INPUT_NORMALIZATION={self.input_normalization}")

        self.variant_name = "raw_delta" if self.use_delta_branch else "raw_only"
        self.data_root = os.environ.get(
            "PAPERDSM_DATA_ROOT",
            os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"),
        )
        self.file1_labels = os.environ.get(
            "PAPERDSM_FILE1_LABELS",
            os.path.join(REPO_ROOT, "manual_keyframe_labels.json"),
        )
        self.file2_labels = os.environ.get(
            "PAPERDSM_FILE2_LABELS",
            os.path.join(self.data_root, "manual_keyframe_labels_file2.json"),
        )
        self.file3_labels = os.environ.get(
            "PAPERDSM_FILE3_LABELS",
            os.path.join(self.data_root, "manual_keyframe_labels_file3.json"),
        )
        self.output_dir = os.environ.get(
            "PAPERDSM_OUTPUT_DIR",
            os.path.join(
                PROJECT_DIR,
                "experiments",
                f"outputs_stage1_dualstream_mstcn_detection_{self.variant_name}_windowlabel",
            ),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    label_rule_text = (
        "center_frame_inside_positive_segment"
        if cfg.label_mode == "center_frame_positive"
        else "window_overlaps_positive_segment"
    )

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    rec1, samples1 = build_detection_samples_for_file(
        file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap, cfg.label_mode, cfg.input_normalization
    )
    rec2, samples2 = build_detection_samples_for_file(
        file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap, cfg.label_mode, cfg.input_normalization
    )
    rec3, samples3 = build_detection_samples_for_file(
        file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap, cfg.label_mode, cfg.input_normalization
    )

    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        )
    )
    common_base_group_set = set(common_base_groups)

    train_base_groups, val_base_groups = split_base_groups_train_val_balanced(common_base_groups)
    train_base_group_set = set(train_base_groups)
    val_base_group_set = set(val_base_groups)

    train_records = {}
    train_records.update({k: v for k, v in rec1.items() if v["base_group"] in train_base_group_set})
    train_records.update({k: v for k, v in rec2.items() if v["base_group"] in train_base_group_set})

    val_records = {}
    val_records.update({k: v for k, v in rec1.items() if v["base_group"] in val_base_group_set})
    val_records.update({k: v for k, v in rec2.items() if v["base_group"] in val_base_group_set})

    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_base_group_set}

    train_samples_all = [s for s in (samples1 + samples2) if s["base_group"] in train_base_group_set]
    train_samples = downsample_negatives(train_samples_all, cfg.max_neg_pos_ratio, cfg.seed)
    val_samples = [s for s in (samples1 + samples2) if s["base_group"] in val_base_group_set]
    test_samples = [s for s in samples3 if s["base_group"] in common_base_group_set]

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Empty train, val, or test samples.")

    ds_train = CenterLabelSequenceDataset(
        train_records,
        train_samples,
        is_train=True,
        aug_noise_std=cfg.aug_noise_std,
        aug_scale_jitter=cfg.aug_scale_jitter,
        aug_frame_dropout=cfg.aug_frame_dropout,
        input_normalization=cfg.input_normalization,
    )
    ds_train_eval = CenterLabelSequenceDataset(
        train_records,
        train_samples_all,
        is_train=False,
        input_normalization=cfg.input_normalization,
    )
    ds_val = CenterLabelSequenceDataset(
        val_records,
        val_samples,
        is_train=False,
        input_normalization=cfg.input_normalization,
    )
    ds_test = CenterLabelSequenceDataset(
        test_records,
        test_samples,
        is_train=False,
        input_normalization=cfg.input_normalization,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_train_eval = DataLoader(
        ds_train_eval,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_train = np.array([int(s["label"]) for s in train_samples], dtype=np.int32)
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    raw_pos_weight = float(neg / max(pos, 1))
    pos_weight = float(raw_pos_weight * cfg.train_pos_weight_scale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamMSTCNDetector(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        use_delta_branch=cfg.use_delta_branch,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.lr * cfg.min_lr_ratio,
    )
    criterion_train = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    criterion_soft = nn.BCEWithLogitsLoss()
    criterion_eval = nn.BCEWithLogitsLoss()

    best_path = os.path.join(cfg.output_dir, "paper_stage1_dualstream_mstcn_best.pth")
    last_path = os.path.join(cfg.output_dir, "paper_stage1_dualstream_mstcn_last.pth")
    curve_path = os.path.join(cfg.output_dir, "paper_stage1_dualstream_mstcn_curves.png")
    curve_html_path = os.path.join(cfg.output_dir, "paper_stage1_dualstream_mstcn_curves.html")
    summary_path = os.path.join(cfg.output_dir, "paper_stage1_dualstream_mstcn_summary.json")
    manifest_path = os.path.join(cfg.output_dir, "paper_stage1_dualstream_mstcn_split_manifest.json")

    best_key = None
    best_rec = None
    history = []
    no_improve = 0
    plotly_saved = False

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(
            model,
            loader_train,
            device,
            criterion_train,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
            criterion_soft=criterion_soft,
            soft_loss_weight=cfg.soft_loss_weight,
        )
        tr_eval = run_epoch(model, loader_train_eval, device, criterion_eval, optimizer=None)
        val_eval = run_epoch(model, loader_val, device, criterion_eval, optimizer=None)
        scheduler.step()

        best_thr_val = select_best_f1_threshold(val_eval["y_true"], val_eval["y_score"])
        rec = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(tr["loss"]),
            "train_hard_loss": float(tr["hard_loss"]),
            "train_soft_loss": float(tr["soft_loss"]),
            "train_eval_loss": float(tr_eval["loss"]),
            "val_loss": float(val_eval["loss"]),
            "val_f1": float(best_thr_val["f1"]),
            "val_precision": float(best_thr_val["precision"]),
            "val_recall": float(best_thr_val["recall"]),
            "val_best_threshold": float(best_thr_val["threshold"]),
            "val_auc": float(build_roc(val_eval["y_true"], val_eval["y_score"])),
            "val_ap": float(build_pr(val_eval["y_true"], val_eval["y_score"])),
        }
        history.append(rec)
        print(
            f"[{epoch:03d}/{cfg.epochs}] train={rec['train_loss']:.4f} "
            f"hard={rec['train_hard_loss']:.4f} soft={rec['train_soft_loss']:.4f} "
            f"val_loss={rec['val_loss']:.4f} val_f1={rec['val_f1']:.4f} "
            f"val_auc={rec['val_auc']:.4f} val_ap={rec['val_ap']:.4f} thr={rec['val_best_threshold']:.3f}"
        )

        cur_key = (rec["val_auc"], rec["val_ap"], rec["val_f1"], -rec["val_loss"])
        if best_key is None or cur_key > best_key:
            best_key = cur_key
            best_rec = rec
            no_improve = 0
            torch.save(build_checkpoint(model, cfg, epoch, best_record=rec), best_path)
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            print(f"Early stop at epoch {epoch} (patience={cfg.patience})")
            break

    torch.save(build_checkpoint(model, cfg, history[-1]["epoch"], best_record=best_rec), last_path)
    plot_curves(history, curve_path)
    if cfg.save_plotly_html:
        plotly_saved = plot_curves_plotly(history, curve_html_path)

    payload = torch.load(best_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    train_eval_best = run_epoch(model, loader_train_eval, device, criterion_eval, optimizer=None)
    val_best = run_epoch(model, loader_val, device, criterion_eval, optimizer=None)
    test_best = run_epoch(model, loader_test, device, criterion_eval, optimizer=None)
    best_thr_val = select_best_f1_threshold(val_best["y_true"], val_best["y_score"])
    thr = float(best_thr_val["threshold"])
    train_metrics_val_thr = compute_cls_metrics(train_eval_best["y_true"], train_eval_best["y_score"], thr)
    val_metrics_val_thr = compute_cls_metrics(val_best["y_true"], val_best["y_score"], thr)
    test_metrics_val_thr = compute_cls_metrics(test_best["y_true"], test_best["y_score"], thr)
    test_metrics_best_thr = select_best_f1_threshold(test_best["y_true"], test_best["y_score"])

    y_train_eval = np.array([int(s["label"]) for s in train_samples_all], dtype=np.int32)
    y_val = np.array([int(s["label"]) for s in val_samples], dtype=np.int32)
    y_test = np.array([int(s["label"]) for s in test_samples], dtype=np.int32)

    manifest = {
        "common_base_groups_detail": common_base_groups,
        "train_base_groups_detail": train_base_groups,
        "val_base_groups_detail": val_base_groups,
        "condition_stats": merge_condition_stats(
            common_groups=common_base_groups,
            train_samples_all=train_samples_all,
            train_samples_used=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary = {
        "task_name": "paper_stage1_dualstream_mstcn_detection",
        "task_definition": {
            "primary_task": "window-level nodule presence detection",
            "label_rule": label_rule_text,
            "soft_target": "positive_frame_fraction_in_window",
            "split_protocol": "1+2 development, 3 final test, group-level split only",
        },
        "config": {
            "seed": cfg.seed,
            "seq_len": cfg.seq_len,
            "stride": cfg.stride,
            "dedup_gap": cfg.dedup_gap,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "frame_feature_dim": cfg.frame_feature_dim,
            "temporal_channels": cfg.temporal_channels,
            "temporal_blocks": cfg.temporal_blocks,
            "patience": cfg.patience,
            "grad_clip": cfg.grad_clip,
            "max_neg_pos_ratio": cfg.max_neg_pos_ratio,
            "raw_pos_weight": raw_pos_weight,
            "train_pos_weight_used": pos_weight,
            "train_pos_weight_scale": cfg.train_pos_weight_scale,
            "soft_loss_weight": cfg.soft_loss_weight,
            "use_delta_branch": bool(cfg.use_delta_branch),
            "label_mode": cfg.label_mode,
            "input_normalization": cfg.input_normalization,
            "data_root": os.path.abspath(cfg.data_root),
            "output_dir": os.path.abspath(cfg.output_dir),
        },
        "split": {
            "common_base_groups": int(len(common_base_groups)),
            "train_base_groups": int(len(train_base_groups)),
            "val_base_groups": int(len(val_base_groups)),
            "train_samples": int(len(train_samples)),
            "train_eval_samples": int(len(train_samples_all)),
            "val_samples": int(len(val_samples)),
            "test_samples": int(len(test_samples)),
            "train_positive": int(np.sum(y_train == 1)),
            "train_negative": int(np.sum(y_train == 0)),
            "train_eval_positive": int(np.sum(y_train_eval == 1)),
            "val_positive": int(np.sum(y_val == 1)),
            "test_positive": int(np.sum(y_test == 1)),
        },
        "protocol_v1": protocol_summary(),
        "best_record": best_rec,
        "history": history,
        "train_metrics_at_val_best_threshold": train_metrics_val_thr,
        "val_metrics_at_val_best_threshold": val_metrics_val_thr,
        "test_metrics_at_val_best_threshold": test_metrics_val_thr,
        "test_metrics_at_test_best_threshold": test_metrics_best_thr,
        "val_auc": float(build_roc(val_best["y_true"], val_best["y_score"])),
        "val_ap": float(build_pr(val_best["y_true"], val_best["y_score"])),
        "test_auc": float(build_roc(test_best["y_true"], test_best["y_score"])),
        "test_ap": float(build_pr(test_best["y_true"], test_best["y_score"])),
        "curve_path": os.path.abspath(curve_path),
        "curve_html_path": os.path.abspath(curve_html_path) if plotly_saved else None,
        "model_path_best": os.path.abspath(best_path),
        "model_path_last": os.path.abspath(last_path),
        "split_manifest_path": os.path.abspath(manifest_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(os.path.abspath(best_path))
    print(os.path.abspath(summary_path))


if __name__ == "__main__":
    main()
