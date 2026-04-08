import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from implicit_temporal_change import ImplicitTemporalChangeEncoder


class TinyViTFrameEncoder(nn.Module):
    """
    Frame-wise spatial encoder:
    - bilinear interpolation to a modest canvas
    - shallow CNN stem to preserve local tactile inductive bias
    - tiny ViT token mixer to model global spatial relations
    """

    def __init__(
        self,
        in_channels=1,
        stem_channels=24,
        embed_dim=64,
        interp_size=(24, 16),
        patch_size=4,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.15,
    ):
        super().__init__()
        self.interp_size = tuple(int(x) for x in interp_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.patch_embed = nn.Conv2d(
            stem_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        grid_h = self.interp_size[0] // self.patch_size
        grid_w = self.interp_size[1] // self.patch_size
        self.num_tokens = int(grid_h * grid_w)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(embed_dim)

    def forward(self, x):
        # x: (B,T,1,H,W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = F.interpolate(
            x,
            size=self.interp_size,
            mode="bilinear",
            align_corners=False,
        )
        x = self.cnn_stem(x)
        x = self.patch_embed(x)  # (BT, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (BT, N, C)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return x.reshape(b, t, self.out_dim)


class CenterAwareTemporalAttentionPooling(nn.Module):
    def __init__(self, channels, dropout=0.2, center_prior_scale_init=0.5):
        super().__init__()
        hidden = max(16, channels // 2)
        self.attn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.center_logit_scale = nn.Parameter(
            torch.tensor(float(center_prior_scale_init), dtype=torch.float32)
        )
        self.proj = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = int(channels)

    def forward(self, x):
        # Center-aware temporal prior: compatible with center-frame supervision rule.
        b, t, _c = x.shape
        logits = self.attn(x)
        if t > 1:
            pos = torch.linspace(-1.0, 1.0, steps=t, device=x.device, dtype=x.dtype).view(1, t, 1)
            center_prior = -(pos ** 2)
            logits = logits + F.softplus(self.center_logit_scale) * center_prior
        weights = torch.softmax(logits, dim=1)
        attn_feat = torch.sum(weights * x, dim=1)
        mean_feat = x.mean(dim=1)
        max_feat = x.max(dim=1).values
        center_feat = x[:, t // 2]
        pooled = torch.cat([attn_feat, center_feat, mean_feat, max_feat], dim=1)
        return self.proj(pooled), weights


class TemporalConsistencyBlock(nn.Module):
    """Lightweight temporal refinement block to suppress isolated spike activations."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)
        pad = d * (k // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=d, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        z = x.transpose(1, 2)
        z = self.net(z).transpose(1, 2)
        return self.act(x + z)


class SharedWindowEncoderTinyViTGRU(nn.Module):
    """Reusable local window encoder for hierarchical tasks."""

    def __init__(
        self,
        seq_len=10,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.30,
        use_delta_branch=False,
        use_implicit_change=True,
        frame_feature_dim=64,
        interp_h=24,
        interp_w=16,
        vit_depth=2,
        vit_heads=4,
        patch_size=4,
        temporal_refine_blocks=2,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.frame_feature_dim = int(frame_feature_dim)
        self.hidden_dim = max(64, int(lstm_hidden) * 2)
        env_use_delta = os.environ.get("PAPERDET_USE_DELTA_BRANCH")
        if env_use_delta is not None:
            use_delta_branch = env_use_delta.strip().lower() in {"1", "true", "yes", "y", "on"}
        self.use_delta_branch = bool(use_delta_branch)
        env_use_implicit = os.environ.get("PAPERDET_USE_IMPLICIT_CHANGE")
        if env_use_implicit is not None:
            use_implicit_change = env_use_implicit.strip().lower() in {"1", "true", "yes", "y", "on"}
        self.use_implicit_change = bool(use_implicit_change)

        branch_dropout = min(0.18, dropout * 0.5)
        encoder_kwargs = dict(
            in_channels=1,
            stem_channels=max(16, frame_feature_dim // 2),
            embed_dim=frame_feature_dim,
            interp_size=(interp_h, interp_w),
            patch_size=patch_size,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=2.0,
            dropout=branch_dropout,
        )
        self.raw_encoder = TinyViTFrameEncoder(**encoder_kwargs)
        self.implicit_change_encoder = (
            ImplicitTemporalChangeEncoder(
                in_dim=frame_feature_dim * 2,
                out_dim=frame_feature_dim,
                dropout=min(0.18, dropout * 0.5),
            )
            if self.use_implicit_change
            else None
        )
        self.delta_encoder = (
            TinyViTFrameEncoder(**encoder_kwargs) if self.use_delta_branch else None
        )

        temporal_in = frame_feature_dim * (1 + int(self.use_implicit_change) + int(self.use_delta_branch))
        self.pre_gru = nn.Sequential(
            nn.LayerNorm(temporal_in),
            nn.Linear(temporal_in, temporal_in),
            nn.GELU(),
            nn.Dropout(min(0.20, dropout * 0.5)),
        )
        self.temporal_refine = nn.ModuleList(
            [
                TemporalConsistencyBlock(
                    channels=temporal_in,
                    kernel_size=3,
                    dilation=(1 if i % 2 == 0 else 2),
                    dropout=min(0.20, dropout * 0.5),
                )
                for i in range(int(max(0, temporal_refine_blocks)))
            ]
        )
        self.gru = nn.GRU(
            input_size=temporal_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.temporal_pool = CenterAwareTemporalAttentionPooling(
            lstm_hidden * 2,
            dropout=min(0.35, dropout),
        )
        self.stability_proj = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Dropout(min(0.20, dropout * 0.5)),
        )
        self.center_proj_dim = max(16, frame_feature_dim)
        self.center_proj = nn.Sequential(
            nn.LayerNorm(temporal_in),
            nn.Linear(temporal_in, self.center_proj_dim),
            nn.GELU(),
            nn.Dropout(min(0.20, dropout * 0.5)),
        )
        self.window_token_dim = int(self.temporal_pool.out_dim + 8 + self.center_proj_dim)

    @staticmethod
    def compute_delta(x):
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta

    def forward(self, x, stats_seq=None, return_features=False):
        raw_seq = self.raw_encoder(x)
        feature_delta = self.compute_delta(raw_seq)
        implicit_change_seq = (
            self.implicit_change_encoder(torch.cat([raw_seq, feature_delta], dim=-1))
            if self.use_implicit_change
            else torch.zeros_like(raw_seq)
        )
        streams = [raw_seq]
        if self.use_implicit_change:
            streams.append(implicit_change_seq)
        delta_seq = None
        if self.use_delta_branch:
            delta_seq = self.delta_encoder(self.compute_delta(x))
            streams.append(delta_seq)

        seq = torch.cat(streams, dim=-1)
        center_seq_feat = seq[:, seq.size(1) // 2]
        seq = self.pre_gru(seq)
        for block in self.temporal_refine:
            seq = block(seq)
        gru_out, _ = self.gru(seq)
        pooled, attn_weights = self.temporal_pool(gru_out)
        diff = torch.abs(seq[:, 1:] - seq[:, :-1])
        stability_stats = torch.stack(
            [
                diff.mean(dim=(1, 2)),
                seq.std(dim=1).mean(dim=1),
                seq.abs().mean(dim=(1, 2)),
            ],
            dim=1,
        )
        stability_feat = self.stability_proj(stability_stats)
        center_feat = self.center_proj(center_seq_feat)
        window_token = torch.cat([pooled, stability_feat, center_feat], dim=1)

        if return_features:
            return window_token, {
                "raw_seq": raw_seq,
                "feature_delta_seq": feature_delta,
                "implicit_change_seq": implicit_change_seq,
                "delta_seq": delta_seq,
                "gru_out": gru_out,
                "attn_weights": attn_weights,
                "stability_stats": stability_stats,
                "center_feat": center_feat,
                "window_token": window_token,
            }
        return window_token


class WindowEncoderDetectionHead(nn.Module):
    def __init__(self, encoder: SharedWindowEncoderTinyViTGRU, dropout: float = 0.30):
        super().__init__()
        self.encoder = encoder
        hidden_dim = int(encoder.hidden_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(int(encoder.window_token_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x, stats_seq=None, return_features=False):
        window_token, feats = self.encoder(x, stats_seq=stats_seq, return_features=True)
        features = self.fusion_fc(window_token)
        prob = self.prob_head(features)
        if return_features:
            feats["features"] = features
            return prob, feats
        return prob


class CNNStemTinyViTToGRUAttn(nn.Module):
    """
    Detection candidate:
    raw / delta dual streams
    -> CNN stem + tiny ViT per frame
    -> BiGRU temporal head
    -> temporal attention pooling
    """

    def __init__(
        self,
        seq_len=10,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.30,
        use_delta_branch=False,
        use_implicit_change=True,
        frame_feature_dim=64,
        interp_h=24,
        interp_w=16,
        vit_depth=2,
        vit_heads=4,
        patch_size=4,
        temporal_refine_blocks=2,
    ):
        super().__init__()
        self.encoder = SharedWindowEncoderTinyViTGRU(
            seq_len=seq_len,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            use_delta_branch=use_delta_branch,
            use_implicit_change=use_implicit_change,
            frame_feature_dim=frame_feature_dim,
            interp_h=interp_h,
            interp_w=interp_w,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            patch_size=patch_size,
            temporal_refine_blocks=temporal_refine_blocks,
        )
        self.det_head = WindowEncoderDetectionHead(self.encoder, dropout=dropout)

    def forward(self, x, stats_seq=None, return_features=False):
        return self.det_head(x, stats_seq=stats_seq, return_features=return_features)


class HierarchicalWindowEncoderDetector(nn.Module):
    """Stage1 detector built on the reusable best-mainline local encoder."""

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = SharedWindowEncoderTinyViTGRU(**kwargs)
        self.det_head = WindowEncoderDetectionHead(self.encoder, dropout=float(kwargs.get("dropout", 0.30)))

    def forward(self, x, stats_seq=None, return_features=False):
        return self.det_head(x, stats_seq=stats_seq, return_features=return_features)
