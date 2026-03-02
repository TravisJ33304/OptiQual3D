"""Anomaly detection scoring head.

Takes per-patch anomaly features from the anomaly decoder branch and
produces per-point anomaly scores.  Supports multi-scale feature
aggregation for detecting anomalies at different spatial scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from optiqual3d.config.settings import AnomalyHeadConfig


class AnomalyDetectionHead(nn.Module):
    """MLP head for per-point anomaly scoring.

    Maps decoder anomaly features to scalar anomaly scores per patch
    (or per point when interpolating back to the full cloud).

    Attributes:
        config: Head configuration.
    """

    def __init__(
        self,
        in_dim: int,
        config: AnomalyHeadConfig | None = None,
    ) -> None:
        """Initialise the anomaly detection head.

        Args:
            in_dim: Input feature dimension from anomaly decoder.
            config: Head configuration.
        """
        super().__init__()
        self.config = config or AnomalyHeadConfig()

        # Build MLP layers
        layers: list[nn.Module] = []
        current_dim = in_dim

        for hidden_dim in self.config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            current_dim = hidden_dim

        # Final projection to scalar score
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-patch anomaly scores.

        Args:
            features: ``(B, G, D)`` per-patch anomaly features.

        Returns:
            ``(B, G)`` per-patch anomaly scores.
        """
        # (B, G, D) → (B, G, 1) → (B, G)
        scores = self.mlp(features).squeeze(-1)
        return scores


class MultiScaleAnomalyHead(nn.Module):
    """Anomaly head with multi-scale feature aggregation.

    Aggregates features from multiple encoder layers at different
    spatial scales before scoring.  This helps detect both fine-grained
    surface anomalies and large structural defects.

    Attributes:
        config: Head configuration.
        scale_projections: Per-scale linear projections.
    """

    def __init__(
        self,
        in_dims: list[int],
        config: AnomalyHeadConfig | None = None,
    ) -> None:
        """Initialise the multi-scale anomaly head.

        Args:
            in_dims: Feature dimensions from each scale (e.g. from
                different transformer layers).
            config: Head configuration.
        """
        super().__init__()
        self.config = config or AnomalyHeadConfig()

        # Project each scale to a common dimension
        common_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 128
        self.scale_projections = nn.ModuleList(
            [nn.Linear(d, common_dim) for d in in_dims]
        )
        self.scale_weights = nn.Parameter(torch.ones(len(in_dims)) / len(in_dims))

        # Scoring MLP on aggregated features
        self.head = AnomalyDetectionHead(
            in_dim=common_dim,
            config=config,
        )

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute anomaly scores from multi-scale features.

        Args:
            multi_scale_features: List of ``(B, G, D_i)`` feature tensors,
                one per encoder scale/layer.

        Returns:
            ``(B, G)`` per-patch anomaly scores.
        """
        weights = F.softmax(self.scale_weights, dim=0)

        projected: list[torch.Tensor] = []
        for i, (proj, feat) in enumerate(
            zip(self.scale_projections, multi_scale_features)
        ):
            projected.append(proj(feat) * weights[i])

        # Aggregate via weighted sum
        aggregated = torch.stack(projected, dim=0).sum(dim=0)

        return self.head(aggregated)


def interpolate_scores_to_points(
    patch_scores: torch.Tensor,
    centroids: torch.Tensor,
    points: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """Interpolate per-patch scores to per-point scores via KNN.

    For each point, finds the K nearest patch centroids and computes
    a distance-weighted average of their anomaly scores.

    Args:
        patch_scores: ``(B, G)`` per-patch anomaly scores.
        centroids: ``(B, G, 3)`` patch centroid coordinates.
        points: ``(B, N, 3)`` full point cloud coordinates.
        k: Number of nearest centroids to interpolate from.

    Returns:
        ``(B, N)`` per-point anomaly scores.
    """
    b, n, _ = points.shape

    # Pairwise distances: (B, N, G)
    dists = torch.cdist(points, centroids)

    # K nearest centroids per point
    topk_dists, topk_indices = torch.topk(dists, k, dim=-1, largest=False)

    # Inverse distance weights (with small epsilon for stability)
    weights = 1.0 / (topk_dists + 1e-8)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Gather scores for top-k centroids
    topk_scores = torch.gather(
        patch_scores.unsqueeze(1).expand(-1, n, -1),
        dim=2,
        index=topk_indices,
    )

    # Weighted average
    point_scores = (topk_scores * weights).sum(dim=-1)

    return point_scores
