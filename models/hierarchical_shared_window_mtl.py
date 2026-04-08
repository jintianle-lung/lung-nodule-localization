import torch
import torch.nn as nn

from cnn_stem_tinyvit_bigru_attn import SharedWindowEncoderTinyViTGRU
from concept_guided_depth_model import PhaseAwarePooling
from raw_positive_size_model_v2 import RawPositiveSizeModelV2
from task_protocol_v1 import COARSE_DEPTH_ORDER, SIZE_VALUES_CM


class ResidualTaskAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(hidden_dim)) if int(in_dim) != int(hidden_dim) else nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.out_norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        return self.out_norm(residual + self.net(residual))


class OrdinalClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        hidden = max(64, int(in_dim))
        self.cls_head = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, int(num_classes)),
        )
        self.ord_head = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, int(num_classes - 1)),
        )
        self.num_classes = int(num_classes)

    @staticmethod
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

    def forward(self, x: torch.Tensor):
        cls_logits = self.cls_head(x)
        ord_logits = self.ord_head(x)
        probs = self.ordinal_logits_to_probs(ord_logits, self.num_classes)
        return cls_logits, ord_logits, probs


class TaskTemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.20):
        super().__init__()
        hidden = max(16, int(channels) // 2)
        self.attn = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(int(channels) * 3, int(channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = int(channels)

    def forward(self, x: torch.Tensor):
        logits = self.attn(x)
        weights = torch.softmax(logits, dim=1)
        attn_feat = torch.sum(weights * x, dim=1)
        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        pooled = torch.cat([attn_feat, mean_feat, max_feat], dim=1)
        return self.proj(pooled), weights


class SizeConditionedDepthExperts(nn.Module):
    def __init__(self, in_dim: int, num_size_classes: int, num_depth_classes: int, dropout: float = 0.25):
        super().__init__()
        hidden = max(64, int(in_dim))
        self.num_size_classes = int(num_size_classes)
        self.num_depth_classes = int(num_depth_classes)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(in_dim), hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, int(num_depth_classes - 1)),
                )
                for _ in range(self.num_size_classes)
            ]
        )

    @staticmethod
    def ordinal_logits_to_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
        return OrdinalClassificationHead.ordinal_logits_to_probs(logits, num_classes)

    def forward(self, x: torch.Tensor, size_idx: torch.Tensor):
        logits = torch.zeros(
            x.shape[0],
            self.num_depth_classes - 1,
            dtype=x.dtype,
            device=x.device,
        )
        for expert_idx, expert in enumerate(self.experts):
            mask = size_idx.long() == int(expert_idx)
            if torch.any(mask):
                logits[mask] = expert(x[mask])
        probs = self.ordinal_logits_to_probs(logits, self.num_depth_classes)
        return logits, probs

    def forward_soft(self, x: torch.Tensor, size_probs: torch.Tensor):
        expert_logits = [expert(x).unsqueeze(1) for expert in self.experts]
        expert_logits = torch.cat(expert_logits, dim=1)
        mixed_logits = torch.sum(expert_logits * size_probs.unsqueeze(-1), dim=1)
        probs = self.ordinal_logits_to_probs(mixed_logits, self.num_depth_classes)
        return mixed_logits, probs


class LightweightWindowContextAttention(nn.Module):
    def __init__(self, channels: int, max_context_windows: int = 5, num_heads: int = 2, dropout: float = 0.15):
        super().__init__()
        self.channels = int(channels)
        self.max_context_windows = int(max_context_windows)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_context_windows, self.channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.channels * 3, self.channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.channels, self.channels),
        )
        self.out_norm = nn.LayerNorm(self.channels)

    def forward(self, window_tokens: torch.Tensor):
        if window_tokens.dim() != 3:
            raise ValueError(f"Expected window_tokens shape (B,K,C), got {tuple(window_tokens.shape)}")
        b, k, c = window_tokens.shape
        if c != self.channels:
            raise ValueError(f"Expected token dim {self.channels}, got {c}")
        if k > self.max_context_windows:
            raise ValueError(f"context windows {k} exceed max_context_windows={self.max_context_windows}")

        center_idx = k // 2
        center_token = window_tokens[:, center_idx]
        if k == 1:
            window_weights = torch.ones(b, 1, dtype=window_tokens.dtype, device=window_tokens.device)
            return center_token, torch.zeros_like(center_token), window_weights

        tokens_with_pos = self.norm(window_tokens + self.pos_embed[:, :k])
        query = tokens_with_pos[:, center_idx : center_idx + 1]
        context_token, window_weights = self.attn(query=query, key=tokens_with_pos, value=tokens_with_pos, need_weights=True)
        context_token = context_token.squeeze(1)
        window_weights = window_weights.squeeze(1)
        fused = self.fuse(torch.cat([center_token, context_token, context_token - center_token], dim=1))
        enhanced = self.out_norm(center_token + fused)
        return enhanced, context_token, window_weights


class HierarchicalSharedWindowMTL(nn.Module):
    """Unified hierarchical model built on the best-mainline local window encoder."""

    def __init__(
        self,
        seq_len: int = 10,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.30,
        use_delta_branch: bool = False,
        frame_feature_dim: int = 64,
        interp_h: int = 24,
        interp_w: int = 16,
        vit_depth: int = 2,
        vit_heads: int = 4,
        patch_size: int = 4,
        temporal_refine_blocks: int = 2,
        size_num_classes: int = len(SIZE_VALUES_CM),
        depth_num_classes: int = len(COARSE_DEPTH_ORDER),
        det_adapter_dim: int = 160,
        size_adapter_dim: int = 192,
        depth_adapter_dim: int = 192,
        size_residual_scale: float = 0.20,
        context_heads: int = 2,
        max_context_windows: int = 5,
    ):
        super().__init__()
        self.size_residual_scale = float(size_residual_scale)
        self.encoder = SharedWindowEncoderTinyViTGRU(
            seq_len=seq_len,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            use_delta_branch=use_delta_branch,
            frame_feature_dim=frame_feature_dim,
            interp_h=interp_h,
            interp_w=interp_w,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            patch_size=patch_size,
            temporal_refine_blocks=temporal_refine_blocks,
        )
        token_dim = int(self.encoder.window_token_dim)
        frame_dim = int(self.encoder.frame_feature_dim)
        temporal_dim = int(self.encoder.hidden_dim)
        self.size_temporal_pool = TaskTemporalAttentionPooling(temporal_dim, dropout=min(0.20, dropout))
        self.depth_temporal_pool = TaskTemporalAttentionPooling(temporal_dim, dropout=min(0.20, dropout))
        self.size_phase_pool = PhaseAwarePooling(temporal_dim, dropout=min(0.20, dropout))
        self.depth_phase_pool = PhaseAwarePooling(temporal_dim, dropout=min(0.20, dropout))
        self.size_legacy_branch = RawPositiveSizeModelV2(
            seq_len=seq_len,
            frame_feature_dim=max(32, frame_dim // 2),
            temporal_channels=64,
            temporal_blocks=4,
            dropout=min(0.25, dropout),
            num_size_classes=int(size_num_classes),
            residual_scale=float(size_residual_scale),
            use_delta=False,
            use_residual_head=False,
        )
        self.size_legacy_feat_dim = int(self.size_legacy_branch.feature_dim)
        self.size_context_dim = int((temporal_dim * 2) + frame_dim * 3 + self.encoder.center_proj_dim + 3 + self.size_legacy_feat_dim)
        self.depth_context_dim = int((temporal_dim * 2) + frame_dim * 2 + self.encoder.center_proj_dim + 3)
        self.window_context = LightweightWindowContextAttention(
            token_dim,
            max_context_windows=int(max_context_windows),
            num_heads=int(context_heads),
            dropout=min(0.20, dropout),
        )
        self.det_adapter = ResidualTaskAdapter(token_dim, int(det_adapter_dim), dropout=dropout)
        self.det_head = nn.Sequential(
            nn.Linear(int(det_adapter_dim), max(64, int(det_adapter_dim) // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(64, int(det_adapter_dim) // 2), 1),
        )

        self.size_adapter = ResidualTaskAdapter(token_dim + self.size_context_dim, int(size_adapter_dim), dropout=dropout)
        self.size_head = OrdinalClassificationHead(int(size_adapter_dim), int(size_num_classes), dropout=dropout)
        size_values = torch.tensor(SIZE_VALUES_CM, dtype=torch.float32)
        lo = float(size_values.min().item())
        hi = float(size_values.max().item())
        size_values_norm = (size_values - lo) / max(hi - lo, 1e-6)
        self.register_buffer("size_values_norm", size_values_norm.view(1, -1))
        self.size_residual_head = nn.Sequential(
            nn.Linear(int(size_adapter_dim), max(64, int(size_adapter_dim) // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(64, int(size_adapter_dim) // 2), 1),
        )

        self.depth_adapter = ResidualTaskAdapter(
            token_dim + self.depth_context_dim + int(size_num_classes),
            int(depth_adapter_dim),
            dropout=dropout,
        )
        self.depth_experts = SizeConditionedDepthExperts(
            int(depth_adapter_dim),
            int(size_num_classes),
            int(depth_num_classes),
            dropout=dropout,
        )

    @staticmethod
    def _extract_center_window_features(feats: dict, batch_size: int, num_windows: int, center_idx: int) -> dict:
        center_feats = {}
        for key, value in feats.items():
            if not torch.is_tensor(value):
                continue
            reshaped = value.reshape(batch_size, num_windows, *value.shape[1:])
            center_feats[key] = reshaped[:, center_idx]
        return center_feats

    def _encode_with_optional_context(self, x: torch.Tensor, return_features: bool = False):
        if x.dim() == 5:
            window_token, feats = self.encoder(x, return_features=True)
            feats["center_window_token"] = window_token
            feats["context_token"] = torch.zeros_like(window_token)
            feats["window_attention"] = torch.ones(window_token.shape[0], 1, dtype=window_token.dtype, device=window_token.device)
            feats["input_window"] = x
            if return_features:
                feats["window_token"] = window_token
                return window_token, feats
            return window_token

        if x.dim() != 6:
            raise ValueError(f"Expected input dims 5 or 6, got shape {tuple(x.shape)}")

        b, k, t, c, h, w = x.shape
        flat_x = x.reshape(b * k, t, c, h, w)
        flat_token, flat_feats = self.encoder(flat_x, return_features=True)
        window_tokens = flat_token.reshape(b, k, -1)
        enhanced_token, context_token, window_weights = self.window_context(window_tokens)
        center_idx = k // 2
        feats = self._extract_center_window_features(flat_feats, b, k, center_idx)
        feats["window_tokens"] = window_tokens
        feats["center_window_token"] = window_tokens[:, center_idx]
        feats["context_token"] = context_token
        feats["window_attention"] = window_weights
        feats["window_token"] = enhanced_token
        feats["input_window"] = x[:, center_idx]
        if return_features:
            return enhanced_token, feats
        return enhanced_token

    def _build_size_context(self, feats: dict):
        raw_seq = feats["raw_seq"]
        gru_out = feats["gru_out"]
        implicit_change_seq = feats["implicit_change_seq"]
        temporal_feat, temporal_attn = self.size_temporal_pool(gru_out)
        phase_feat, phase_masks = self.size_phase_pool(gru_out, feats["input_window"])
        shape_window = self._normalize_per_frame(feats["input_window"])
        legacy_feats = self.size_legacy_branch.encode(feats["input_window"], shape_window)
        private_feat = legacy_feats["trunk_feat"]
        raw_mean = raw_seq.mean(dim=1)
        raw_max = raw_seq.max(dim=1).values
        implicit_mean = implicit_change_seq.mean(dim=1)
        stability = feats["stability_stats"]
        center_feat = feats["center_feat"]
        context = torch.cat([temporal_feat, phase_feat, raw_mean, raw_max, implicit_mean, center_feat, stability, private_feat], dim=1)
        aux = {
            "size_temporal_attn": temporal_attn,
            "size_phase_masks": phase_masks,
            "size_private_attn": legacy_feats["attn_weights"],
            "size_private_phase_masks": legacy_feats["phase_masks"],
            "size_private_feat": private_feat,
        }
        return context, aux

    @staticmethod
    def _normalize_per_frame(x: torch.Tensor) -> torch.Tensor:
        frame_min = x.amin(dim=(3, 4), keepdim=True)
        frame_max = x.amax(dim=(3, 4), keepdim=True)
        denom = torch.clamp(frame_max - frame_min, min=1e-6)
        return torch.clamp((x - frame_min) / denom, 0.0, 1.0)

    def _build_depth_context(self, feats: dict):
        raw_seq = feats["raw_seq"]
        gru_out = feats["gru_out"]
        implicit_change_seq = feats["implicit_change_seq"]
        temporal_feat, temporal_attn = self.depth_temporal_pool(gru_out)
        phase_feat, phase_masks = self.depth_phase_pool(gru_out, feats["input_window"])
        raw_mean = raw_seq.mean(dim=1)
        implicit_mean = implicit_change_seq.mean(dim=1)
        stability = feats["stability_stats"]
        center_feat = feats["center_feat"]
        context = torch.cat([temporal_feat, phase_feat, raw_mean, implicit_mean, center_feat, stability], dim=1)
        return context, temporal_attn, phase_masks

    def encode_window(self, x: torch.Tensor, return_features: bool = False):
        return self._encode_with_optional_context(x, return_features=return_features)

    def forward_detection(self, x: torch.Tensor, return_features: bool = False):
        window_token, feats = self._encode_with_optional_context(x, return_features=True)
        det_feat = self.det_adapter(window_token)
        det_logit = self.det_head(det_feat)
        if return_features:
            feats["window_token"] = window_token
            feats["det_feat"] = det_feat
            return det_logit, feats
        return det_logit

    def forward_size(self, x: torch.Tensor, return_features: bool = False):
        window_token, feats = self._encode_with_optional_context(x, return_features=True)
        size_context, size_aux = self._build_size_context(feats)
        size_feat = self.size_adapter(torch.cat([window_token, size_context], dim=1))
        size_cls_logits, size_ord_logits, size_probs = self.size_head(size_feat)
        expected_norm = torch.sum(size_probs * self.size_values_norm.to(size_probs.dtype), dim=1, keepdim=True)
        residual = self.size_residual_scale * torch.tanh(self.size_residual_head(size_feat))
        size_reg_norm = torch.clamp(expected_norm + residual, 0.0, 1.0)
        if return_features:
            feats["window_token"] = window_token
            feats["size_context"] = size_context
            feats.update(size_aux)
            feats["size_feat"] = size_feat
            return size_cls_logits, size_ord_logits, size_reg_norm, size_probs, feats
        return size_cls_logits, size_ord_logits, size_reg_norm, size_probs

    def forward_depth(self, x: torch.Tensor, size_idx: torch.Tensor, return_features: bool = False):
        window_token, feats = self._encode_with_optional_context(x, return_features=True)
        size_cond = torch.nn.functional.one_hot(size_idx.long(), num_classes=len(SIZE_VALUES_CM)).to(window_token.dtype)
        depth_context, depth_temporal_attn, depth_phase_masks = self._build_depth_context(feats)
        depth_input = torch.cat([window_token, depth_context, size_cond.detach()], dim=1)
        depth_feat = self.depth_adapter(depth_input)
        depth_logits, depth_probs = self.depth_experts(depth_feat, size_idx)
        if return_features:
            feats["window_token"] = window_token
            feats["depth_context"] = depth_context
            feats["depth_temporal_attn"] = depth_temporal_attn
            feats["depth_phase_masks"] = depth_phase_masks
            feats["depth_feat"] = depth_feat
            return depth_logits, depth_probs, feats
        return depth_logits, depth_probs

    def forward_depth_soft(self, x: torch.Tensor, size_probs: torch.Tensor, return_features: bool = False):
        window_token, feats = self._encode_with_optional_context(x, return_features=True)
        depth_context, depth_temporal_attn, depth_phase_masks = self._build_depth_context(feats)
        depth_input = torch.cat([window_token, depth_context, size_probs.detach().to(window_token.dtype)], dim=1)
        depth_feat = self.depth_adapter(depth_input)
        depth_logits, depth_probs = self.depth_experts.forward_soft(depth_feat, size_probs)
        if return_features:
            feats["window_token"] = window_token
            feats["depth_context"] = depth_context
            feats["depth_temporal_attn"] = depth_temporal_attn
            feats["depth_phase_masks"] = depth_phase_masks
            feats["depth_feat"] = depth_feat
            return depth_logits, depth_probs, feats
        return depth_logits, depth_probs
