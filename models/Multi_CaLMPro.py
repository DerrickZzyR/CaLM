"""
Multi_CaLMPro_utils_direct.py
==============================
Direct fusion version — skip PatchTokenEncoder entirely.

When SoftShapeNet emb_dim is already 512 (same as OpenCLIP text dim),
patch tokens can directly cross-attend with text tokens without an
intermediate projection layer.

Compared with MultiPatchTextFusionModel:
    - No PatchTokenEncoder (no Linear 256→512, no Conv1d, no L2 norm)
    - patch_input_dim == embed_dim (both 512)
    - Fewer trainable parameters, no information distortion from re-encoding

Same forward signature — drop-in replacement for MultiPatchTextFusionModel.
Reuses MaskedAttentionPooling & MultiPatchCrossAttentionFusionClassifier
from the original file.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from layers.Multi_CaLMPro_layers import (
    MaskedAttentionPooling,
    MultiPatchCrossAttentionFusionClassifier,
    MultiPatchCrossAttentionFusionClassifier_6way,
)


class MultiPatchTextFusionModelDirect(nn.Module):
    """
    Direct fusion model — no PatchTokenEncoder.

    Assumes patch_input_dim == embed_dim (e.g. both 512).
    SoftShapeNet output goes directly into cross-attention fusion.

    Same __init__ params and forward signature as MultiPatchTextFusionModel.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 3,
        fusion_hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.patch_proj = nn.LayerNorm(embed_dim)

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
        patch_feat = self.patch_proj(patch_tokens)  # [B, P, 512] → [B, P, 512]
        return self.fusion_head(
            patch_feat,
            txt_tokens,
            patch_mask=patch_mask,
            txt_mask=txt_mask,
            return_attn=return_attn,
        )


__all__ = [
    "MultiPatchTextFusionModelDirect",
    "MultiPatchTextFusionModelDirect_6way",
]


class MultiPatchTextFusionModelDirect_6way(nn.Module):
    """
    Direct fusion model with 6-way ablation support.

    Same as MultiPatchTextFusionModelDirect but uses
    MultiPatchCrossAttentionFusionClassifier_6way with 6 boolean switches
    controlling which fusion components are active.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 3,
        fusion_hidden_dim: int = 512,
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

        self.patch_proj = nn.LayerNorm(embed_dim)

        self.fusion_head = MultiPatchCrossAttentionFusionClassifier_6way(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_raw=use_raw,
            use_fused=use_fused,
            use_cross_diff=use_cross_diff,
            use_cross_prod=use_cross_prod,
            use_self_diff=use_self_diff,
            use_self_prod=use_self_prod,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        patch_feat = self.patch_proj(patch_tokens)
        return self.fusion_head(
            patch_feat,
            txt_tokens,
            patch_mask=patch_mask,
            txt_mask=txt_mask,
            return_attn=return_attn,
        )
