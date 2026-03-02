"""Combined multi-task loss for OptiQual3D.

Assembles the reconstruction (Chamfer), contrastive (InfoNCE), and
anomaly detection (BCE) losses with configurable weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from optiqual3d.config.settings import LossConfig
from optiqual3d.losses.chamfer import PatchChamferDistance


class PretrainLoss(nn.Module):
    """Phase 1 loss: masked patch reconstruction.

    Simply wraps :class:`PatchChamferDistance` for consistency with the
    training API.
    """

    def __init__(self) -> None:
        """Initialise the pre-training loss."""
        super().__init__()
        self.chamfer = PatchChamferDistance(reduction="mean")

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute pre-training loss.

        Args:
            model_output: Output from :meth:`OptiQual3D.forward_pretrain`
                containing ``"reconstruction"`` and ``"target_patches"``.

        Returns:
            Dictionary with:
                - ``"loss"``: Total scalar loss.
                - ``"reconstruction_loss"``: Chamfer distance value.
        """
        recon_loss = self.chamfer(
            model_output["reconstruction"],
            model_output["target_patches"],
        )

        return {
            "loss": recon_loss,
            "reconstruction_loss": recon_loss,
        }


class AnomalyDetectionLoss(nn.Module):
    """Phase 2 multi-task loss: reconstruction + contrastive + anomaly BCE.

    Total loss:

    .. math::

        \\mathcal{L} = \\lambda_r \\mathcal{L}_{\\text{recon}}
                     + \\lambda_c \\mathcal{L}_{\\text{contrast}}
                     + \\lambda_a \\mathcal{L}_{\\text{anomaly}}

    Attributes:
        config: Loss weight configuration.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Initialise the anomaly detection loss.

        Args:
            config: Loss weight configuration.
        """
        super().__init__()
        self.config = config or LossConfig()
        self.chamfer = PatchChamferDistance(reduction="mean")

    def forward(
        self,
        model_output: dict[str, Any],
        patch_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined multi-task loss.

        Args:
            model_output: Output from :meth:`OptiQual3D.forward_anomaly`
                containing ``"patch_scores"`` and optionally
                ``"contrastive"``.
            patch_labels: ``(B, G)`` per-patch binary ground truth.

        Returns:
            Dictionary with:
                - ``"loss"``: Total weighted loss.
                - ``"anomaly_loss"``: Per-patch BCE loss.
                - ``"contrastive_loss"``: Contrastive loss (0 if absent).
        """
        losses: dict[str, torch.Tensor] = {}

        # Per-patch anomaly detection loss (BCE)
        anomaly_loss = F.binary_cross_entropy_with_logits(
            model_output["patch_scores"],
            patch_labels.float(),
        )
        losses["anomaly_loss"] = anomaly_loss

        # Contrastive loss (from contrastive module)
        contrastive_loss = torch.tensor(0.0, device=anomaly_loss.device)
        if "contrastive" in model_output and "loss" in model_output["contrastive"]:
            contrastive_loss = model_output["contrastive"]["loss"]
        losses["contrastive_loss"] = contrastive_loss

        # Combined total
        total = (
            self.config.lambda_anomaly * anomaly_loss
            + self.config.lambda_contrastive * contrastive_loss
        )
        losses["loss"] = total

        return losses


class CombinedLoss(nn.Module):
    """Unified loss for both training phases.

    Dispatches to the appropriate loss based on the training phase.

    Attributes:
        pretrain_loss: Phase 1 loss module.
        anomaly_loss: Phase 2 loss module.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Initialise the combined loss.

        Args:
            config: Loss weight configuration.
        """
        super().__init__()
        self.pretrain_loss = PretrainLoss()
        self.anomaly_loss = AnomalyDetectionLoss(config)

    def forward(
        self,
        model_output: dict[str, Any],
        phase: str = "pretrain",
        patch_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for the given phase.

        Args:
            model_output: Model output dictionary.
            phase: ``"pretrain"`` or ``"anomaly"``.
            patch_labels: Per-patch labels (required for Phase 2).

        Returns:
            Loss dictionary with ``"loss"`` and component losses.

        Raises:
            ValueError: If phase is unknown or labels are missing.
        """
        if phase == "pretrain":
            return self.pretrain_loss(model_output)
        elif phase == "anomaly":
            if patch_labels is None:
                raise ValueError("patch_labels required for anomaly phase")
            return self.anomaly_loss(model_output, patch_labels)
        else:
            raise ValueError(f"Unknown phase: {phase!r}")
