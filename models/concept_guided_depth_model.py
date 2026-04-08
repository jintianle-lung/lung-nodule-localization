import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_stream_mstcn_detection import (
    FrameEncoder2D,
    MultiScaleTemporalBlock,
    TemporalAttentionPooling,
)


CONCEPT_NAMES = (
    "peak_strength",
    "integrated_response",
    "spread_extent",
    "shape_contrast",
    "temporal_prominence",
    "phase_early_response",
    "phase_release_response",
)


class PhaseAwarePooling(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = int(channels)

    @staticmethod
    def compute_phase_masks(raw_window: torch.Tensor) -> torch.Tensor:
        # raw_window: (B,T,1,H,W)
        energy = raw_window.mean(dim=(2, 3, 4))
        batch, seq_len = energy.shape
        device = energy.device
        time_index = torch.arange(seq_len, device=device).float().unsqueeze(0).expand(batch, -1)
        peak_idx = torch.argmax(energy, dim=1).float().unsqueeze(1)

        safe_peak = torch.clamp(peak_idx, min=1.0)
        pre_ratio = time_index / safe_peak

        early = ((time_index <= peak_idx) & (pre_ratio <= 0.5)).float()
        late = ((time_index <= peak_idx) & (pre_ratio > 0.5)).float()
        peak = (torch.abs(time_index - peak_idx) <= 1.0).float()
        release = (time_index > peak_idx).float()

        # Guarantee at least one active step per phase per sample.
        if seq_len > 0:
            early[:, 0] = 1.0
            peak.scatter_(1, torch.argmax(energy, dim=1, keepdim=True), 1.0)
            release[:, -1] = 1.0

        masks = torch.stack([early, late, peak, release], dim=1)
        return masks

    def forward(self, temporal_seq: torch.Tensor, raw_window: torch.Tensor):
        # temporal_seq: (B,T,C)
        masks = self.compute_phase_masks(raw_window)
        feats = []
        for phase_idx in range(masks.shape[1]):
            phase_mask = masks[:, phase_idx].unsqueeze(-1)
            denom = torch.clamp(phase_mask.sum(dim=1), min=1.0)
            phase_feat = (temporal_seq * phase_mask).sum(dim=1) / denom
            feats.append(phase_feat)
        stacked = torch.cat(feats, dim=1)
        return self.proj(stacked), masks


class ConceptGuidedDepthModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 10,
        frame_feature_dim: int = 32,
        temporal_channels: int = 64,
        temporal_blocks: int = 3,
        dropout: float = 0.35,
        size_embedding_dim: int = 12,
        num_size_classes: int = 7,
        num_concepts: int = len(CONCEPT_NAMES),
        num_depth_classes: int = 3,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_concepts = int(num_concepts)
        self.num_depth_classes = int(num_depth_classes)

        self.amplitude_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.2, dropout * 0.5),
        )
        self.shape_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.2, dropout * 0.5),
        )
        self.delta_encoder = FrameEncoder2D(
            in_channels=1,
            base_channels=max(16, frame_feature_dim),
            out_dim=frame_feature_dim,
            dropout=min(0.2, dropout * 0.5),
        )

        temporal_input_dim = frame_feature_dim * 3
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
                    dropout=min(0.35, dropout),
                )
                for _ in range(int(temporal_blocks))
            ]
        )
        self.global_pooling = TemporalAttentionPooling(temporal_channels, dropout=min(0.35, dropout))
        self.phase_pooling = PhaseAwarePooling(temporal_channels, dropout=min(0.35, dropout))

        fused_dim = self.global_pooling.out_dim + self.phase_pooling.out_dim
        concept_hidden = max(32, fused_dim // 2)
        self.concept_mlp = nn.Sequential(
            nn.Linear(fused_dim, concept_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(concept_hidden, int(num_concepts)),
        )

        self.size_embedding = nn.Embedding(int(num_size_classes), int(size_embedding_dim))
        depth_hidden = max(32, concept_hidden)
        self.depth_head = nn.Sequential(
            nn.Linear(int(num_concepts) + int(size_embedding_dim), depth_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(depth_hidden, int(num_depth_classes) - 1),
        )
        self.feature_dim = int(fused_dim)

    @staticmethod
    def compute_delta(x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    @staticmethod
    def ordinal_logits_to_probs(ordinal_logits: torch.Tensor) -> torch.Tensor:
        cumulative = torch.sigmoid(ordinal_logits)
        if cumulative.shape[1] == 1:
            probs0 = 1.0 - cumulative[:, 0]
            probs1 = cumulative[:, 0]
            return torch.stack([probs0, probs1], dim=1)
        probs = []
        probs.append(1.0 - cumulative[:, 0])
        for idx in range(cumulative.shape[1] - 1):
            probs.append(cumulative[:, idx] - cumulative[:, idx + 1])
        probs.append(cumulative[:, -1])
        return torch.stack(probs, dim=1)

    @staticmethod
    def ordinal_logits_to_class(ordinal_logits: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.sigmoid(ordinal_logits) >= 0.5, dim=1).long()

    def encode(self, raw_window: torch.Tensor, norm_window: torch.Tensor):
        raw_seq = self.amplitude_encoder(raw_window)
        shape_seq = self.shape_encoder(norm_window)
        delta_seq = self.delta_encoder(self.compute_delta(norm_window))

        seq = torch.cat([raw_seq, shape_seq, delta_seq], dim=-1).transpose(1, 2)
        seq = self.temporal_input(seq)
        for block in self.temporal_blocks:
            seq = block(seq)
        temporal_seq = seq.transpose(1, 2)
        global_feat, attn_weights = self.global_pooling(temporal_seq)
        phase_feat, phase_masks = self.phase_pooling(temporal_seq, raw_window)
        fused = torch.cat([global_feat, phase_feat], dim=1)
        return {
            "raw_seq": raw_seq,
            "shape_seq": shape_seq,
            "delta_seq": delta_seq,
            "temporal_seq": temporal_seq,
            "global_feat": global_feat,
            "phase_feat": phase_feat,
            "fused_feat": fused,
            "attn_weights": attn_weights,
            "phase_masks": phase_masks,
        }

    def forward(
        self,
        raw_window: torch.Tensor,
        norm_window: torch.Tensor,
        size_class_index: torch.Tensor,
        return_features: bool = False,
    ):
        feats = self.encode(raw_window, norm_window)
        concept_pred = self.concept_mlp(feats["fused_feat"])
        size_embed = self.size_embedding(size_class_index.long())
        depth_input = torch.cat([concept_pred, size_embed], dim=1)
        ordinal_depth_logits = self.depth_head(depth_input)
        depth_probs = self.ordinal_logits_to_probs(ordinal_depth_logits)
        if return_features:
            return ordinal_depth_logits, depth_probs, concept_pred, feats
        return ordinal_depth_logits, depth_probs, concept_pred
