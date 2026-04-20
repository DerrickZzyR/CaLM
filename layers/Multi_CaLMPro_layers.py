from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PatchTokenEncoder(nn.Module):
    """Keep patch tokens as a sequence instead of collapsing them to one token."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.context_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self.out_ln = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, P, D_in]
        h = self.token_proj(patch_tokens)  # [B, P, D]
        h_conv = self.context_conv(h.transpose(1, 2)).transpose(1, 2)
        z = self.out_ln(h + h_conv)
        return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)


class MaskedAttentionPooling(nn.Module):
    """Pool a token sequence to one vector with learned attention weights."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, P, D], mask: [B, P] where True means valid
        scores = self.score(x).squeeze(-1)  # [B, P]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, P, 1]
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return torch.sum(weights * x, dim=1)  # [B, D]


class MultiPatchCrossAttentionFusionClassifier(nn.Module):
    """
    Multi-patch version of PatchTextFusionModel fusion head.

    Keep patch tokens as [B, P, D], let all patch tokens attend over text tokens,
    then pool after fusion for classification.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)

        self.raw_pool = MaskedAttentionPooling(embed_dim)
        self.fused_pool = MaskedAttentionPooling(embed_dim)
        self.txt_pool = MaskedAttentionPooling(embed_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(
        self,
        patch_feat: torch.Tensor,
        txt_tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # patch_feat: [B, P, D], txt_tokens: [B, T, D]
        key_padding_mask = None
        if txt_mask is not None:
            key_padding_mask = ~txt_mask.bool()

        attn_out, attn_weights = self.cross_attn(
            patch_feat,
            txt_tokens,
            txt_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )
        fused_tokens = self.attn_ln(patch_feat + attn_out)
        fused_tokens = self.ffn_ln(fused_tokens + self.ffn(fused_tokens))

        raw_vec = self.raw_pool(patch_feat, patch_mask)
        fused_vec = self.fused_pool(fused_tokens, patch_mask)
        txt_vec = self.txt_pool(txt_tokens, txt_mask)

        fused = torch.cat(
            [
                raw_vec,
                fused_vec,
                torch.abs(fused_vec - txt_vec),
                fused_vec * txt_vec,
            ],
            dim=-1,
        )
        hidden = self.fusion_mlp(fused)
        logits = self.classifier(hidden)
        if return_attn:
            return logits, attn_weights
        return logits


class MultiPatchTextFusionModel(nn.Module):
    """
    Multi-token fusion model.

    Compared with PatchTextFusionModel:
    - no global pooling before text fusion
    - keep patch tokens as [B, P, D]
    - run cross-attention between all patch tokens and text tokens
    - pool only after fusion, then classify
    """

    def __init__(
        self,
        patch_input_dim: int,
        embed_dim: int,
        num_classes: int = 3,
        patch_hidden_dim: int = 256,
        fusion_hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.patch_encoder = PatchTokenEncoder(
            input_dim=patch_input_dim,
            embed_dim=embed_dim,
            hidden_dim=patch_hidden_dim,
            dropout=dropout,
        )
        self.fusion_head = MultiPatchCrossAttentionFusionClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        patch_feat = self.patch_encoder(patch_tokens)
        return self.fusion_head(
            patch_feat,
            txt_tokens,
            patch_mask=patch_mask,
            txt_mask=txt_mask,
            return_attn=return_attn,
        )
    
class MultiPatchCrossAttentionFusionClassifier_ablation(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_raw: bool = True,
        use_fused: bool = True,
        use_diff: bool = True,
        use_prod: bool = True,
    ) -> None:
        super().__init__()

        self.use_raw = use_raw
        self.use_fused = use_fused
        self.use_diff = use_diff
        self.use_prod = use_prod

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)

        self.raw_pool = MaskedAttentionPooling(embed_dim)
        self.fused_pool = MaskedAttentionPooling(embed_dim)
        self.txt_pool = MaskedAttentionPooling(embed_dim)

        num_parts = sum([use_raw, use_fused, use_diff, use_prod])
        assert num_parts > 0, "At least one fusion component must be enabled."

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_parts, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(
        self,
        patch_feat: torch.Tensor,
        txt_tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        key_padding_mask = None
        if txt_mask is not None:
            key_padding_mask = ~txt_mask.bool()

        attn_out, attn_weights = self.cross_attn(
            patch_feat,
            txt_tokens,
            txt_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )
        fused_tokens = self.attn_ln(patch_feat + attn_out)
        fused_tokens = self.ffn_ln(fused_tokens + self.ffn(fused_tokens))

        raw_vec = self.raw_pool(patch_feat, patch_mask)
        fused_vec = self.fused_pool(fused_tokens, patch_mask)
        txt_vec = self.txt_pool(txt_tokens, txt_mask)

        parts = []
        if self.use_raw:
            parts.append(raw_vec)
        if self.use_fused:
            parts.append(fused_vec)
        if self.use_diff:
            parts.append(torch.abs(fused_vec - txt_vec))
        if self.use_prod:
            parts.append(fused_vec * txt_vec)

        fused = torch.cat(parts, dim=-1)
        hidden = self.fusion_mlp(fused)
        logits = self.classifier(hidden)

        if return_attn:
            return logits, attn_weights
        return logits


class MultiPatchCrossAttentionFusionClassifier_6way(nn.Module):
    """
    6-way ablation fusion classifier.

    Extends the original 4-component design with two extra self-comparison
    terms that measure how much cross-attention changed the patch representation:
        - self_diff:  |fused_vec - raw_vec|   (fusion delta)
        - self_prod:  fused_vec * raw_vec     (fusion retention)
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_raw: bool = True,
        use_fused: bool = True,
        use_cross_diff: bool = True,
        use_cross_prod: bool = True,
        use_self_diff: bool = True,
        use_self_prod: bool = True,
    ) -> None:
        super().__init__()

        self.use_raw = use_raw
        self.use_fused = use_fused
        self.use_cross_diff = use_cross_diff
        self.use_cross_prod = use_cross_prod
        self.use_self_diff = use_self_diff
        self.use_self_prod = use_self_prod

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)

        self.raw_pool = MaskedAttentionPooling(embed_dim)
        self.fused_pool = MaskedAttentionPooling(embed_dim)
        self.txt_pool = MaskedAttentionPooling(embed_dim)

        num_parts = sum([use_raw, use_fused, use_cross_diff, use_cross_prod,
                         use_self_diff, use_self_prod])
        assert num_parts > 0, "At least one fusion component must be enabled."

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_parts, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(
        self,
        patch_feat: torch.Tensor,
        txt_tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        key_padding_mask = None
        if txt_mask is not None:
            key_padding_mask = ~txt_mask.bool()

        attn_out, attn_weights = self.cross_attn(
            patch_feat,
            txt_tokens,
            txt_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )
        fused_tokens = self.attn_ln(patch_feat + attn_out)
        fused_tokens = self.ffn_ln(fused_tokens + self.ffn(fused_tokens))

        raw_vec = self.raw_pool(patch_feat, patch_mask)
        fused_vec = self.fused_pool(fused_tokens, patch_mask)
        txt_vec = self.txt_pool(txt_tokens, txt_mask)

        parts = []
        if self.use_raw:
            parts.append(raw_vec)
        if self.use_fused:
            parts.append(fused_vec)
        if self.use_cross_diff:
            parts.append(torch.abs(fused_vec - txt_vec))
        if self.use_cross_prod:
            parts.append(fused_vec * txt_vec)
        if self.use_self_diff:
            parts.append(torch.abs(fused_vec - raw_vec))
        if self.use_self_prod:
            parts.append(fused_vec * raw_vec)

        fused = torch.cat(parts, dim=-1)
        hidden = self.fusion_mlp(fused)
        logits = self.classifier(hidden)

        if return_attn:
            return logits, attn_weights
        return logits


__all__ = [
    "PatchTokenEncoder",
    "MaskedAttentionPooling",
    "MultiPatchCrossAttentionFusionClassifier",
    "MultiPatchCrossAttentionFusionClassifier_6way",
    "MultiPatchTextFusionModel",
]
