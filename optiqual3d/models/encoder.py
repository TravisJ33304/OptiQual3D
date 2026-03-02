"""Point-MAE encoder for self-supervised point cloud representation learning.

Implements a masked autoencoder for point clouds based on the Point-MAE
architecture.  The encoder is a standard Vision Transformer operating on
visible (unmasked) point patches.

Reference:
    Pang et al., "Masked Autoencoders for Point Cloud Self-supervised
    Learning", ECCV 2022.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from optiqual3d.config.settings import EncoderConfig, PointCloudConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PointPatchEmbedding(nn.Module):
    """Embed local point patches into token vectors.

    Groups the input point cloud into patches via mini-PointNet
    (shared MLP + max pooling) and projects each patch to the
    transformer embedding dimension.

    Attributes:
        patch_size: Number of points per patch.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        patch_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        """Initialise the patch embedding.

        Args:
            patch_size: Points per patch.
            in_channels: Input point feature dimension (typically 3 for XYZ).
            embed_dim: Transformer token dimension.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Mini-PointNet: shared MLP (3 → 64 → 128 → embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(
        self,
        patches: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed patches.

        Args:
            patches: ``(B, G, P, 3)`` local point patches (centroid-relative).
            centroids: ``(B, G, 3)`` patch centroid coordinates.

        Returns:
            ``(B, G, D)`` patch tokens.
        """
        # patches: (B, G, P, 3) → (B, G, P, D)
        tokens = self.mlp(patches)
        # Max pool over points in each patch → (B, G, D)
        tokens = tokens.max(dim=2).values
        return tokens


class PositionalEncoding(nn.Module):
    """Learnable positional encoding from patch centroid coordinates.

    Maps the 3-D centroid of each patch to the embedding dimension via
    a small MLP, then adds to the tokens.

    Attributes:
        embed_dim: Token embedding dimension.
    """

    def __init__(self, embed_dim: int = 384) -> None:
        """Initialise positional encoding.

        Args:
            embed_dim: Transformer embedding dimension.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """Compute positional encodings.

        Args:
            centroids: ``(B, G, 3)`` patch centroid coordinates.

        Returns:
            ``(B, G, D)`` positional encodings.
        """
        return self.mlp(centroids)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block.

    Consists of multi-head self-attention followed by a feed-forward
    network, both with residual connections and LayerNorm.

    Attributes:
        dim: Token dimension.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        """Initialise the transformer block.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension expansion factor.
            drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth drop rate.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, N, D)`` input tokens.

        Returns:
            ``(B, N, D)`` output tokens.
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)
        # Feed-forward with pre-norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation.

    During training, randomly drops entire residual branches with
    probability *drop_prob*.

    Args:
        drop_prob: Drop probability.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth.

        Args:
            x: Input tensor.

        Returns:
            Scaled tensor (or zeros for dropped paths).
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# Point-MAE Encoder
# ---------------------------------------------------------------------------


class PointMAEEncoder(nn.Module):
    """Point-MAE encoder: patches → masked → transformer → latent features.

    During pre-training, a fraction of patches are randomly masked and
    only the visible patches are processed by the transformer.  During
    inference (anomaly detection), all patches are processed.

    Attributes:
        config: Encoder hyperparameters.
        pc_cfg: Point cloud configuration.
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        pc_cfg: PointCloudConfig | None = None,
    ) -> None:
        """Initialise the Point-MAE encoder.

        Args:
            config: Encoder hyperparameters.
            pc_cfg: Point cloud configuration.
        """
        super().__init__()
        self.config = config or EncoderConfig()
        self.pc_cfg = pc_cfg or PointCloudConfig()

        c = self.config
        self.patch_embed = PointPatchEmbedding(
            patch_size=self.pc_cfg.patch_size,
            in_channels=3,
            embed_dim=c.embed_dim,
        )
        self.pos_embed = PositionalEncoding(embed_dim=c.embed_dim)

        # Stochastic depth schedule
        dpr = [
            x.item()
            for x in torch.linspace(0, c.drop_path_rate, c.depth)
        ]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=c.embed_dim,
                    num_heads=c.num_heads,
                    mlp_ratio=c.mlp_ratio,
                    drop=c.drop_rate,
                    attn_drop=c.attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(c.depth)
            ]
        )
        self.norm = nn.LayerNorm(c.embed_dim)

        # Mask token for masked patches (used during pre-training)
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, c.embed_dim) * 0.02
        )

        self._init_weights()

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self.config.embed_dim

    def forward(
        self,
        patches: torch.Tensor,
        centroids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode point cloud patches.

        Args:
            patches: ``(B, G, P, 3)`` local point patches.
            centroids: ``(B, G, 3)`` patch centroids.
            mask: ``(B, G)`` boolean mask (``True`` = masked/hidden).
                If ``None``, all patches are processed.

        Returns:
            Tuple of:
                - ``(B, G_vis, D)`` encoded visible patch features.
                - ``(B, G)`` mask tensor (``None`` if no masking).
        """
        # Embed patches → (B, G, D)
        tokens = self.patch_embed(patches, centroids)
        pos = self.pos_embed(centroids)
        tokens = tokens + pos

        if mask is not None:
            # Keep only visible (unmasked) tokens
            b, g, d = tokens.shape
            visible = tokens[~mask].reshape(b, -1, d)
        else:
            visible = tokens

        # Transformer encoding
        for block in self.blocks:
            visible = block(visible)
        visible = self.norm(visible)

        return visible, mask

    def generate_mask(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a random binary mask for pre-training.

        Args:
            batch_size: Number of samples.
            num_patches: Total number of patches per sample.
            device: Target device.

        Returns:
            ``(B, G)`` boolean tensor where ``True`` means masked.
        """
        num_masked = int(num_patches * self.config.mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        for i in range(batch_size):
            indices = torch.randperm(num_patches, device=device)[:num_masked]
            mask[i, indices] = True

        return mask

    def _init_weights(self) -> None:
        """Initialise model weights using truncated normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
