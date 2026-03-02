"""Dual-branch decoder for Point-MAE reconstruction and anomaly scoring.

The decoder has two parallel branches:
    1. **Reconstruction branch**: Reconstructs masked point patches from
       visible patch representations (standard MAE objective).
    2. **Anomaly scoring branch**: Produces per-patch anomaly features
       that feed into the anomaly detection head.

Both branches share the same lightweight transformer architecture but
have independent weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from optiqual3d.config.settings import DecoderConfig, PointCloudConfig
from optiqual3d.models.encoder import TransformerBlock


# ---------------------------------------------------------------------------
# Reconstruction Branch
# ---------------------------------------------------------------------------


class ReconstructionDecoder(nn.Module):
    """Decoder branch for masked point patch reconstruction.

    Takes the encoded visible tokens, concatenates mask tokens for
    hidden patches, and reconstructs the full set of point patches.

    Attributes:
        config: Decoder hyperparameters.
    """

    def __init__(
        self,
        encoder_dim: int,
        config: DecoderConfig | None = None,
        patch_size: int = 32,
    ) -> None:
        """Initialise the reconstruction decoder.

        Args:
            encoder_dim: Encoder output dimension (for projection).
            config: Decoder hyperparameters.
            patch_size: Number of points per patch (output size).
        """
        super().__init__()
        self.config = config or DecoderConfig()
        self.patch_size = patch_size

        c = self.config
        # Project encoder features to decoder dimension
        self.embed_proj = nn.Linear(encoder_dim, c.embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, c.embed_dim) * 0.02
        )

        # Positional encoding projection for centroids
        self.pos_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, c.embed_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=c.embed_dim,
                    num_heads=c.num_heads,
                    mlp_ratio=c.mlp_ratio,
                )
                for _ in range(c.depth)
            ]
        )
        self.norm = nn.LayerNorm(c.embed_dim)

        # Head: predict (patch_size * 3) values per patch
        self.head = nn.Linear(c.embed_dim, patch_size * 3)

    def forward(
        self,
        visible_tokens: torch.Tensor,
        centroids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct masked patches.

        Args:
            visible_tokens: ``(B, G_vis, D_enc)`` encoded visible tokens.
            centroids: ``(B, G, 3)`` all patch centroids.
            mask: ``(B, G)`` boolean mask (``True`` = masked).

        Returns:
            ``(B, G_masked, P, 3)`` reconstructed point patches for
            the masked positions only.
        """
        b, g, _ = centroids.shape

        # Project to decoder dimension
        vis = self.embed_proj(visible_tokens)

        # Build full token sequence: visible + mask tokens
        num_masked = mask.sum(dim=1).max().item()
        mask_tokens = self.mask_token.expand(b, int(num_masked), -1)

        # Positional encoding for all patches
        pos = self.pos_proj(centroids)

        # Assemble full sequence (visible tokens first, then mask tokens)
        full_tokens = torch.cat([vis, mask_tokens], dim=1)

        # Add positional encodings (visible positions + masked positions)
        vis_pos = pos[~mask].reshape(b, -1, pos.shape[-1])
        mask_pos = pos[mask].reshape(b, -1, pos.shape[-1])
        full_pos = torch.cat([vis_pos, mask_pos], dim=1)
        full_tokens = full_tokens + full_pos

        # Transformer decoding
        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)

        # Extract only the masked positions (latter part)
        num_vis = vis.shape[1]
        masked_tokens = full_tokens[:, num_vis:]

        # Predict point coordinates: (B, G_masked, P*3) → (B, G_masked, P, 3)
        pred = self.head(masked_tokens)
        pred = pred.reshape(b, -1, self.patch_size, 3)

        return pred


# ---------------------------------------------------------------------------
# Anomaly Feature Branch
# ---------------------------------------------------------------------------


class AnomalyFeatureDecoder(nn.Module):
    """Decoder branch producing per-patch anomaly features.

    Processes all patch tokens (no masking) through a lightweight
    transformer and outputs feature vectors that feed into the
    anomaly detection head.

    Attributes:
        config: Decoder hyperparameters.
    """

    def __init__(
        self,
        encoder_dim: int,
        config: DecoderConfig | None = None,
    ) -> None:
        """Initialise the anomaly feature decoder.

        Args:
            encoder_dim: Encoder output dimension.
            config: Decoder hyperparameters.
        """
        super().__init__()
        self.config = config or DecoderConfig()

        c = self.config
        self.embed_proj = nn.Linear(encoder_dim, c.embed_dim)

        self.pos_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, c.embed_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=c.embed_dim,
                    num_heads=c.num_heads,
                    mlp_ratio=c.mlp_ratio,
                )
                for _ in range(c.depth)
            ]
        )
        self.norm = nn.LayerNorm(c.embed_dim)

    @property
    def output_dim(self) -> int:
        """Return the anomaly feature dimension."""
        return self.config.embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-patch anomaly features.

        During Phase 2 training and inference all patches are visible
        (no masking), so this decoder processes the full token set.

        Args:
            tokens: ``(B, G, D_enc)`` encoder output for all patches.
            centroids: ``(B, G, 3)`` patch centroids.

        Returns:
            ``(B, G, D_dec)`` per-patch anomaly feature vectors.
        """
        x = self.embed_proj(tokens)
        pos = self.pos_proj(centroids)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Combined Dual-Branch Decoder
# ---------------------------------------------------------------------------


class DualBranchDecoder(nn.Module):
    """Combined decoder with reconstruction and anomaly branches.

    During pre-training only the reconstruction branch is active.
    During Phase 2 both branches are active.

    Attributes:
        reconstruction: Reconstruction decoder branch.
        anomaly: Anomaly feature decoder branch.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_cfg: DecoderConfig | None = None,
        pc_cfg: PointCloudConfig | None = None,
    ) -> None:
        """Initialise the dual-branch decoder.

        Args:
            encoder_dim: Encoder output embedding dimension.
            decoder_cfg: Decoder hyperparameters.
            pc_cfg: Point cloud configuration (for patch_size).
        """
        super().__init__()
        decoder_cfg = decoder_cfg or DecoderConfig()
        pc_cfg = pc_cfg or PointCloudConfig()

        self.reconstruction = ReconstructionDecoder(
            encoder_dim=encoder_dim,
            config=decoder_cfg,
            patch_size=pc_cfg.patch_size,
        )
        self.anomaly = AnomalyFeatureDecoder(
            encoder_dim=encoder_dim,
            config=decoder_cfg,
        )

    @property
    def anomaly_feature_dim(self) -> int:
        """Dimension of anomaly feature vectors."""
        return self.anomaly.output_dim

    def forward(
        self,
        visible_tokens: torch.Tensor,
        all_tokens: torch.Tensor | None,
        centroids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = "both",
    ) -> dict[str, torch.Tensor]:
        """Forward through one or both decoder branches.

        Args:
            visible_tokens: ``(B, G_vis, D)`` encoded visible tokens.
            all_tokens: ``(B, G, D)`` full (unmasked) encoder output.
                Required when ``mode`` is ``"both"`` or ``"anomaly"``.
            centroids: ``(B, G, 3)`` patch centroids.
            mask: ``(B, G)`` boolean mask for reconstruction.
            mode: ``"reconstruct"``, ``"anomaly"``, or ``"both"``.

        Returns:
            Dictionary with:
                - ``"reconstruction"``: ``(B, G_masked, P, 3)`` predicted
                  patches (if reconstruction is active).
                - ``"anomaly_features"``: ``(B, G, D_dec)`` anomaly
                  features (if anomaly branch is active).
        """
        outputs: dict[str, torch.Tensor] = {}

        if mode in ("reconstruct", "both") and mask is not None:
            outputs["reconstruction"] = self.reconstruction(
                visible_tokens, centroids, mask
            )

        if mode in ("anomaly", "both") and all_tokens is not None:
            outputs["anomaly_features"] = self.anomaly(all_tokens, centroids)

        return outputs
