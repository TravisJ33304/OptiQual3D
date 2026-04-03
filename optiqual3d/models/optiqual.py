"""OptiQual3D: Full model assembly.

Composes the Point-MAE encoder, dual-branch decoder, anomaly
detection head, and contrastive normality module into a single
end-to-end model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from optiqual3d.config.settings import ModelConfig, PointCloudConfig
from optiqual3d.models.anomaly_head import (
    AnomalyDetectionHead,
    MultiScaleAnomalyHead,
    interpolate_scores_to_points,
)
from optiqual3d.models.contrastive import ContrastiveNormalityModule
from optiqual3d.models.decoder import AnomalyFeatureDecoder, DualBranchDecoder
from optiqual3d.models.encoder import PointMAEEncoder


class OptiQual3D(nn.Module):
    """End-to-end model for zero-shot 3D anomaly detection.

    Supports two operating modes:

    **Pre-training** (Phase 1):
        Masked point cloud reconstruction via the encoder +
        reconstruction decoder.

    **Anomaly detection** (Phase 2 / inference):
        Full encoder (no masking) → anomaly decoder → anomaly head
        + contrastive normality scoring.

    Attributes:
        encoder: Point-MAE transformer encoder.
        decoder: Dual-branch decoder.
        anomaly_head: Per-patch anomaly scoring MLP.
        contrastive: Contrastive normality learning module.
        pc_cfg: Point cloud configuration.
        model_cfg: Model configuration.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        pc_cfg: PointCloudConfig | None = None,
    ) -> None:
        """Initialise the OptiQual3D model.

        Args:
            model_cfg: Full model configuration.
            pc_cfg: Point cloud preprocessing configuration.
        """
        super().__init__()
        self.model_cfg = model_cfg or ModelConfig()
        self.pc_cfg = pc_cfg or PointCloudConfig()

        # Encoder
        self.encoder = PointMAEEncoder(
            config=self.model_cfg.encoder,
            pc_cfg=self.pc_cfg,
        )

        # Dual-branch decoder
        self.decoder = DualBranchDecoder(
            encoder_dim=self.encoder.embed_dim,
            decoder_cfg=self.model_cfg.decoder,
            pc_cfg=self.pc_cfg,
        )

        # Anomaly detection head
        if self.model_cfg.anomaly_head.use_multi_scale:
            # Tap intermediate encoder layers for multi-scale features
            self.multi_scale_layers = self.model_cfg.anomaly_head.multi_scale_layers
            # One anomaly decoder per scale
            self.scale_decoders = nn.ModuleList([
                AnomalyFeatureDecoder(
                    encoder_dim=self.encoder.embed_dim,
                    config=self.model_cfg.decoder,
                )
                for _ in self.multi_scale_layers
            ])
            scale_dim = self.model_cfg.decoder.embed_dim
            self.anomaly_head: nn.Module = MultiScaleAnomalyHead(
                in_dims=[scale_dim] * len(self.multi_scale_layers),
                config=self.model_cfg.anomaly_head,
            )
        else:
            self.multi_scale_layers = []
            self.scale_decoders = nn.ModuleList()
            self.anomaly_head = AnomalyDetectionHead(
                in_dim=self.decoder.anomaly_feature_dim,
                config=self.model_cfg.anomaly_head,
            )

        # Contrastive normality module
        self.contrastive = ContrastiveNormalityModule(
            feature_dim=self.encoder.embed_dim,
            config=self.model_cfg.contrastive,
        )

        # Global pooling for contrastive learning (patch features → global)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    # ------------------------------------------------------------------
    # Forward Modes
    # ------------------------------------------------------------------

    def forward_pretrain(
        self,
        patches: torch.Tensor,
        centroids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Phase 1: Self-supervised pre-training forward pass.

        Masks random patches, encodes visible ones, then reconstructs
        the masked patches.

        Args:
            patches: ``(B, G, P, 3)`` point patches.
            centroids: ``(B, G, 3)`` patch centroids.

        Returns:
            Dictionary with:
                - ``"reconstruction"``: ``(B, G_masked, P, 3)`` predicted
                  patches.
                - ``"mask"``: ``(B, G)`` boolean mask used.
                - ``"target_patches"``: ``(B, G_masked, P, 3)`` ground-truth
                  masked patches for loss computation.
        """
        b, g, p, _ = patches.shape

        # Generate mask
        mask = self.encoder.generate_mask(b, g, patches.device)

        # Encode visible patches only
        visible_tokens, _, _ = self.encoder(patches, centroids, mask=mask)

        # Decode to reconstruct masked patches
        decoder_out = self.decoder(
            visible_tokens=visible_tokens,
            all_tokens=None,
            centroids=centroids,
            mask=mask,
            mode="reconstruct",
        )

        # Extract ground-truth target patches at masked positions
        target_patches = patches[mask].reshape(b, -1, p, 3)

        return {
            "reconstruction": decoder_out["reconstruction"],
            "mask": mask,
            "target_patches": target_patches,
        }

    def forward_anomaly(
        self,
        patches: torch.Tensor,
        centroids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Phase 2: Anomaly detection forward pass.

        Encodes all patches (no masking), extracts anomaly features,
        and computes per-patch anomaly scores.

        Args:
            patches: ``(B, G, P, 3)`` point patches.
            centroids: ``(B, G, 3)`` patch centroids.
            labels: ``(B,)`` optional sample-level binary labels
                (for contrastive loss during training).

        Returns:
            Dictionary with:
                - ``"patch_scores"``: ``(B, G)`` per-patch anomaly scores.
                - ``"anomaly_features"``: ``(B, G, D)`` anomaly features.
                - ``"global_features"``: ``(B, D)`` global pooled features.
                - ``"contrastive"``: Contrastive module output dict
                  (only if *labels* provided).
        """
        b, g, p, _ = patches.shape

        # Encode all patches (no masking), optionally returning intermediates
        intermediates_indices = self.multi_scale_layers if self.multi_scale_layers else None
        all_tokens, _, intermediates = self.encoder(
            patches, centroids, mask=None,
            return_intermediates=intermediates_indices,
        )

        # Anomaly scoring
        if isinstance(self.anomaly_head, MultiScaleAnomalyHead) and intermediates:
            # Process each intermediate through its own anomaly decoder
            multi_scale_features = [
                decoder(inter, centroids)
                for decoder, inter in zip(self.scale_decoders, intermediates)
            ]
            patch_scores = self.anomaly_head(multi_scale_features)
            anomaly_features = multi_scale_features[-1]  # use deepest for output
        else:
            # Fallback: single-scale via the dual-branch decoder
            decoder_out = self.decoder(
                visible_tokens=all_tokens,
                all_tokens=all_tokens,
                centroids=centroids,
                mask=None,
                mode="anomaly",
            )
            anomaly_features = decoder_out["anomaly_features"]
            patch_scores = self.anomaly_head(anomaly_features)

        # Global features for contrastive learning
        # (B, G, D) → (B, D, G) → pool → (B, D)
        global_features = self.global_pool(
            all_tokens.transpose(1, 2)
        ).squeeze(-1)

        result: dict[str, torch.Tensor] = {
            "patch_scores": patch_scores,
            "anomaly_features": anomaly_features,
            "global_features": global_features,
        }

        # Contrastive loss (only during training with labels)
        if labels is not None:
            contrastive_out = self.contrastive(global_features, labels)
            result["contrastive"] = contrastive_out

        return result

    def forward(
        self,
        patches: torch.Tensor,
        centroids: torch.Tensor,
        mode: str = "anomaly",
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Unified forward pass dispatching to the appropriate mode.

        Args:
            patches: ``(B, G, P, 3)`` point patches.
            centroids: ``(B, G, 3)`` patch centroids.
            mode: ``"pretrain"`` or ``"anomaly"``.
            labels: Optional sample-level labels (Phase 2 only).

        Returns:
            Mode-specific output dictionary.

        Raises:
            ValueError: If *mode* is not recognised.
        """
        if mode == "pretrain":
            return self.forward_pretrain(patches, centroids)
        elif mode == "anomaly":
            return self.forward_anomaly(patches, centroids, labels)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'pretrain' or 'anomaly'.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        patches: torch.Tensor,
        centroids: torch.Tensor,
        points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run inference and produce per-point anomaly scores.

        Args:
            patches: ``(B, G, P, 3)`` point patches.
            centroids: ``(B, G, 3)`` patch centroids.
            points: ``(B, N, 3)`` full point cloud coordinates.

        Returns:
            Dictionary with:
                - ``"point_scores"``: ``(B, N)`` per-point anomaly scores.
                - ``"patch_scores"``: ``(B, G)`` per-patch anomaly scores.
                - ``"global_score"``: ``(B,)`` global anomaly scores.
        """
        self.eval()
        out = self.forward_anomaly(patches, centroids)

        # Interpolate patch scores → point scores
        point_scores = interpolate_scores_to_points(
            out["patch_scores"], centroids, points
        )

        # Global score: max over points
        global_score = point_scores.max(dim=-1).values

        return {
            "point_scores": torch.sigmoid(point_scores),
            "patch_scores": torch.sigmoid(out["patch_scores"]),
            "global_score": torch.sigmoid(global_score),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_parameters(self) -> dict[str, int]:
        """Count parameters by component.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        counts: dict[str, int] = {}
        for name, module in [
            ("encoder", self.encoder),
            ("decoder", self.decoder),
            ("anomaly_head", self.anomaly_head),
            ("contrastive", self.contrastive),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (for Phase 2 warm-up)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
