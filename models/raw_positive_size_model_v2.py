import torch
import torch.nn as nn

from concept_guided_depth_model import PhaseAwarePooling
from dual_stream_mstcn_detection import FrameEncoder2D, MultiScaleTemporalBlock, TemporalAttentionPooling
from implicit_temporal_change import ImplicitTemporalChangeEncoder
from task_protocol_v1 import SIZE_VALUES_CM


class RawPositiveSizeModelV2(nn.Module):
    """Pure raw-input size model with ordinal + expectation + residual heads.

    This model keeps the scientific role of the raw-input branch intact:
    - input: raw amplitude + normalized shape + delta
    - no handcrafted tabular physics features
    - stronger size head tailored for ordered size estimation
    """

    def __init__(
        self,
        seq_len: int = 10,
        frame_feature_dim: int = 32,
        temporal_channels: int = 64,
        temporal_blocks: int = 4,
        dropout: float = 0.22,
        num_size_classes: int = 7,
        residual_scale: float = 0.35,
        use_delta: bool = False,
        use_implicit_change: bool = True,
        use_phase_pooling: bool = True,
        use_residual_head: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_size_classes = int(num_size_classes)
        self.residual_scale = float(residual_scale)
        self.use_delta = bool(use_delta)
        self.use_implicit_change = bool(use_implicit_change)
        self.use_phase_pooling = bool(use_phase_pooling)
        self.use_residual_head = bool(use_residual_head)

        self.amplitude_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.20, dropout * 0.5),
        )
        self.shape_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.20, dropout * 0.5),
        )
        if self.use_implicit_change:
            self.implicit_change_encoder = ImplicitTemporalChangeEncoder(
                in_dim=frame_feature_dim * 3,
                out_dim=frame_feature_dim,
                dropout=min(0.20, dropout * 0.5),
            )
        else:
            self.implicit_change_encoder = None
        if self.use_delta:
            self.delta_encoder = FrameEncoder2D(
                in_channels=1,
                base_channels=max(16, frame_feature_dim),
                out_dim=frame_feature_dim,
                dropout=min(0.20, dropout * 0.5),
            )
        else:
            self.delta_encoder = None

        temporal_input_dim = frame_feature_dim * (2 + int(self.use_implicit_change) + int(self.use_delta))
        self.temporal_input = nn.Sequential(
            nn.Conv1d(temporal_input_dim, temporal_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                MultiScaleTemporalBlock(
                    channels=temporal_channels,
                    kernel_size=3,
                    dilations=(1, 2, 4),
                    dropout=min(0.28, dropout),
                )
                for _ in range(int(temporal_blocks))
            ]
        )
        self.global_pooling = TemporalAttentionPooling(temporal_channels, dropout=min(0.28, dropout))
        if self.use_phase_pooling:
            self.phase_pooling = PhaseAwarePooling(temporal_channels, dropout=min(0.28, dropout))
        else:
            self.phase_pooling = None

        fused_dim = int(self.global_pooling.out_dim)
        if self.use_phase_pooling and self.phase_pooling is not None:
            fused_dim += int(self.phase_pooling.out_dim)
        trunk_dim = max(96, temporal_channels * 2)
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        cls_hidden = max(96, trunk_dim)
        ord_hidden = max(64, trunk_dim // 2)
        self.size_cls_head = nn.Sequential(
            nn.Linear(trunk_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, int(num_size_classes)),
        )
        self.size_ord_head = nn.Sequential(
            nn.Linear(trunk_dim, ord_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ord_hidden, int(num_size_classes - 1)),
        )
        if self.use_residual_head:
            self.size_residual_head = nn.Sequential(
                nn.Linear(trunk_dim, cls_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(cls_hidden, 1),
            )
        else:
            self.size_residual_head = None

        if int(num_size_classes) == len(SIZE_VALUES_CM):
            size_values = torch.tensor(SIZE_VALUES_CM, dtype=torch.float32)
            lo = float(size_values.min().item())
            hi = float(size_values.max().item())
            size_values_norm = (size_values - lo) / max(hi - lo, 1e-6)
        else:
            size_values_norm = torch.linspace(0.0, 1.0, steps=int(num_size_classes), dtype=torch.float32)
        self.register_buffer("size_values_norm", size_values_norm.view(1, -1))
        self.feature_dim = int(trunk_dim)

    @staticmethod
    def ordinal_logits_to_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert ordinal logits (K-1) into class probabilities (K)."""
        k = int(num_classes)
        if k <= 1:
            raise ValueError(f"num_classes must be >=2, got {k}")
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

    @staticmethod
    def compute_delta(x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    def encode(self, raw_window: torch.Tensor, norm_window: torch.Tensor):
        raw_seq = self.amplitude_encoder(raw_window)
        shape_seq = self.shape_encoder(norm_window)
        seq_parts = [raw_seq, shape_seq]
        if self.use_implicit_change and self.implicit_change_encoder is not None:
            feature_delta = self.compute_delta(shape_seq)
            implicit_change_seq = self.implicit_change_encoder(torch.cat([raw_seq, shape_seq, feature_delta], dim=-1))
            seq_parts.append(implicit_change_seq)
        else:
            feature_delta = torch.zeros_like(shape_seq)
            implicit_change_seq = None

        delta_seq = None
        if self.use_delta and self.delta_encoder is not None:
            delta_seq = self.delta_encoder(self.compute_delta(norm_window))
            seq_parts.append(delta_seq)

        seq = torch.cat(seq_parts, dim=-1).transpose(1, 2)
        seq = self.temporal_input(seq)
        for block in self.temporal_blocks:
            seq = block(seq)
        temporal_seq = seq.transpose(1, 2)
        global_feat, attn_weights = self.global_pooling(temporal_seq)
        if self.use_phase_pooling and self.phase_pooling is not None:
            phase_feat, phase_masks = self.phase_pooling(temporal_seq, raw_window)
            fused = torch.cat([global_feat, phase_feat], dim=1)
        else:
            phase_feat, phase_masks = None, None
            fused = global_feat
        trunk_feat = self.trunk(fused)
        return {
            "temporal_seq": temporal_seq,
            "global_feat": global_feat,
            "phase_feat": phase_feat,
            "feature_delta_seq": feature_delta,
            "implicit_change_feat": implicit_change_seq,
            "delta_seq": delta_seq,
            "fused_feat": fused,
            "trunk_feat": trunk_feat,
            "attn_weights": attn_weights,
            "phase_masks": phase_masks,
        }

    def forward(self, raw_window: torch.Tensor, norm_window: torch.Tensor, return_features: bool = False):
        feats = self.encode(raw_window, norm_window)
        trunk_feat = feats["trunk_feat"]
        size_logits = self.size_cls_head(trunk_feat)
        size_ord_logits = self.size_ord_head(trunk_feat)
        size_probs = self.ordinal_logits_to_probs(size_ord_logits, self.num_size_classes)
        expected_norm = torch.sum(size_probs * self.size_values_norm.to(size_probs.dtype), dim=1, keepdim=True)
        if self.use_residual_head and self.size_residual_head is not None:
            residual = self.residual_scale * torch.tanh(self.size_residual_head(trunk_feat))
        else:
            residual = torch.zeros_like(expected_norm)
        size_reg_norm = torch.clamp(expected_norm + residual, 0.0, 1.0)
        if return_features:
            return size_logits, size_ord_logits, size_reg_norm, size_probs, feats
        return size_logits, size_ord_logits, size_reg_norm, size_probs
