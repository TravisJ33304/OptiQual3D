"""Low-level point cloud operations.

Provides GPU-friendly implementations of KNN search, ball-query
grouping, and utility functions operating on ``(B, N, 3)`` point
tensors.  These are used by the preprocessing and model layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def pairwise_distances(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        src: ``(B, N, C)`` source points.
        dst: ``(B, M, C)`` destination points.

    Returns:
        ``(B, N, M)`` squared-distance matrix.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    inner = torch.bmm(src, dst.transpose(1, 2))  # (B, N, M)
    src_sq = (src * src).sum(dim=-1, keepdim=True)  # (B, N, 1)
    dst_sq = (dst * dst).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, M)
    return src_sq + dst_sq - 2 * inner


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------


def knn(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-Nearest Neighbours search.

    Args:
        query: ``(B, N, 3)`` query points.
        reference: ``(B, M, 3)`` reference points.
        k: Number of neighbours.

    Returns:
        Tuple of:
        - ``(B, N, k)`` squared distances to neighbours.
        - ``(B, N, k)`` indices into *reference*.
    """
    dists = pairwise_distances(query, reference)  # (B, N, M)
    return dists.topk(k, dim=-1, largest=False)  # (values, indices)


# ---------------------------------------------------------------------------
# Ball query
# ---------------------------------------------------------------------------


def ball_query(
    query: torch.Tensor,
    reference: torch.Tensor,
    radius: float,
    max_neighbours: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ball query: find all neighbours within a radius.

    Results are padded to *max_neighbours* using the first valid index
    when fewer neighbours fall inside the radius.

    Args:
        query: ``(B, N, 3)`` query points.
        reference: ``(B, M, 3)`` reference points.
        radius: Search radius.
        max_neighbours: Maximum number of neighbours to return.

    Returns:
        Tuple of:
        - ``(B, N, max_neighbours)`` squared distances (padded with
          ``inf``).
        - ``(B, N, max_neighbours)`` indices (padded with 0).
    """
    dists = pairwise_distances(query, reference)  # (B, N, M)
    r_sq = radius * radius

    B, N, M = dists.shape
    device = dists.device

    # Mask out points outside the ball
    mask = dists > r_sq  # True for outside
    dists_masked = dists.clone()
    dists_masked[mask] = float("inf")

    # Sort and take top-k
    sorted_dists, sorted_idx = dists_masked.sort(dim=-1)  # ascending

    # Truncate to max_neighbours
    k = min(max_neighbours, M)
    out_dists = sorted_dists[:, :, :k]
    out_idx = sorted_idx[:, :, :k]

    # Pad if k < max_neighbours
    if k < max_neighbours:
        pad_size = max_neighbours - k
        out_dists = F.pad(out_dists, (0, pad_size), value=float("inf"))
        out_idx = F.pad(out_idx, (0, pad_size), value=0)

    return out_dists, out_idx


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_points(
    points: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Gather points using neighbour indices.

    Args:
        points: ``(B, N, C)`` point features.
        idx: ``(B, M, K)`` indices into dim 1 of *points*.

    Returns:
        ``(B, M, K, C)`` grouped features.
    """
    B, N, C = points.shape
    _, M, K = idx.shape

    idx_expanded = idx.unsqueeze(-1).expand(B, M, K, C)  # (B, M, K, C)
    points_expanded = points.unsqueeze(1).expand(B, M, N, C)  # (B, M, N, C)

    return torch.gather(points_expanded, dim=2, index=idx_expanded)


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------


def farthest_point_sample(
    points: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Farthest Point Sampling on a batch of point clouds.

    Iteratively selects the point farthest from the current set.

    Args:
        points: ``(B, N, 3)`` input point clouds.
        num_samples: Number of points to sample.

    Returns:
        ``(B, num_samples)`` indices of selected points.
    """
    B, N, _ = points.shape
    device = points.device

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    # Start from a random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid_point = points[
            torch.arange(B, device=device), farthest
        ].unsqueeze(1)  # (B, 1, 3)
        dist = ((points - centroid_point) ** 2).sum(dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)  # (B,)

    return centroids


def gather_points(
    points: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Gather points by index along dim 1.

    Args:
        points: ``(B, N, C)`` point features.
        idx: ``(B, M)`` indices.

    Returns:
        ``(B, M, C)`` gathered features.
    """
    B, N, C = points.shape
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)  # (B, M, C)
    return torch.gather(points, dim=1, index=idx_expanded)
