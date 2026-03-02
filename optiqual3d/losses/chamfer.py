"""Chamfer Distance loss for point cloud reconstruction.

Used as the primary reconstruction objective during Phase 1
self-supervised pre-training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChamferDistance(nn.Module):
    """Bi-directional Chamfer Distance between two point sets.

    For point sets P₁ and P₂, the Chamfer Distance is::

        CD(P₁, P₂) = (1/|P₁|) Σ min‖p - q‖² + (1/|P₂|) Σ min‖q - p‖²

    where the first sum is over p ∈ P₁ with nearest q ∈ P₂ and vice versa.

    Attributes:
        reduction: How to aggregate across the batch (``"mean"`` or ``"sum"``).
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialise Chamfer Distance.

        Args:
            reduction: Batch reduction strategy.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Chamfer Distance.

        Args:
            pred: ``(B, N₁, 3)`` predicted point cloud.
            target: ``(B, N₂, 3)`` target point cloud.

        Returns:
            Scalar Chamfer Distance loss.
        """
        # Pairwise squared distances: (B, N1, N2)
        dist_matrix = torch.cdist(pred, target, p=2.0).pow(2)

        # Forward: for each point in pred, min distance to target
        forward_dist = dist_matrix.min(dim=2).values.mean(dim=1)
        # Backward: for each point in target, min distance to pred
        backward_dist = dist_matrix.min(dim=1).values.mean(dim=1)

        chamfer = forward_dist + backward_dist

        if self.reduction == "mean":
            return chamfer.mean()
        elif self.reduction == "sum":
            return chamfer.sum()
        return chamfer


class PatchChamferDistance(nn.Module):
    """Chamfer Distance computed per patch, then aggregated.

    Used during pre-training to measure reconstruction quality of
    individual masked patches.

    Attributes:
        reduction: Batch/patch reduction strategy.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initialise Patch Chamfer Distance.

        Args:
            reduction: Reduction strategy.
        """
        super().__init__()
        self.chamfer = ChamferDistance(reduction="none")
        self.reduction = reduction

    def forward(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-patch Chamfer Distance.

        Args:
            pred_patches: ``(B, G, P, 3)`` predicted patches.
            target_patches: ``(B, G, P, 3)`` ground-truth patches.

        Returns:
            Scalar loss (mean over patches and batch).
        """
        b, g, p, _ = pred_patches.shape

        # Reshape to treat each patch as an independent point set
        pred_flat = pred_patches.reshape(b * g, p, 3)
        target_flat = target_patches.reshape(b * g, p, 3)

        # Compute per-patch Chamfer distance
        per_patch = self.chamfer(pred_flat, target_flat)  # (B*G,)

        if self.reduction == "mean":
            return per_patch.mean()
        elif self.reduction == "sum":
            return per_patch.sum()
        return per_patch.reshape(b, g)
