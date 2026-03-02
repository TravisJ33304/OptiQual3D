"""Point cloud preprocessing utilities.

Provides functions for point sampling, normalisation, and patch
extraction that are shared across all dataset loaders.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Point Sampling
# ---------------------------------------------------------------------------


def farthest_point_sample(
    points: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Farthest Point Sampling (FPS) on a point cloud.

    Iteratively selects the point that is farthest from the already-selected
    set, producing a well-distributed subset.

    Args:
        points: ``(N, 3)`` input point cloud.
        num_samples: Number of points to sample.

    Returns:
        ``(num_samples, 3)`` sampled point cloud.

    Raises:
        ValueError: If *num_samples* exceeds *N*.
    """
    n = points.shape[0]
    if num_samples > n:
        raise ValueError(
            f"Cannot sample {num_samples} points from cloud of size {n}"
        )
    if num_samples == n:
        return points.copy()

    selected_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float64)

    # Start from a random point
    rng = np.random.default_rng()
    current_idx = rng.integers(0, n)
    selected_indices[0] = current_idx

    for i in range(1, num_samples):
        current_point = points[current_idx]
        dist_to_current = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist_to_current)
        current_idx = int(np.argmax(distances))
        selected_indices[i] = current_idx

    return points[selected_indices]


def sample_points(
    points: np.ndarray,
    num_points: int,
    method: str = "fps",
) -> np.ndarray:
    """Sample a fixed number of points from a point cloud.

    Args:
        points: ``(N, 3)`` input point cloud.
        num_points: Desired number of output points.
        method: Sampling method — ``"fps"`` for farthest point sampling
            or ``"random"`` for uniform random sampling.

    Returns:
        ``(num_points, 3)`` sampled point cloud.
    """
    n = points.shape[0]
    if n == num_points:
        return points.copy()

    if n < num_points:
        # Pad by repeating random points
        rng = np.random.default_rng()
        pad_indices = rng.choice(n, size=num_points - n, replace=True)
        return np.concatenate([points, points[pad_indices]], axis=0)

    if method == "fps":
        return farthest_point_sample(points, num_points)
    else:
        rng = np.random.default_rng()
        indices = rng.choice(n, size=num_points, replace=False)
        return points[indices]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Normalise a point cloud to the unit sphere.

    Centres the cloud at the origin and scales so that the farthest
    point lies on the unit sphere.

    Args:
        points: ``(N, 3)`` input point cloud.

    Returns:
        ``(N, 3)`` normalised point cloud.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    max_dist = np.linalg.norm(centered, axis=1).max()
    if max_dist > 0:
        centered = centered / max_dist
    return centered


# ---------------------------------------------------------------------------
# Patch Extraction
# ---------------------------------------------------------------------------


def extract_patches(
    points: np.ndarray,
    num_patches: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide a point cloud into local patches via FPS + KNN.

    Selects *num_patches* centroids via FPS, then for each centroid
    gathers the *patch_size* nearest neighbours.

    Args:
        points: ``(N, 3)`` input point cloud.
        num_patches: Number of patches to extract.
        patch_size: Number of points per patch.

    Returns:
        Tuple of:
            - ``(num_patches, patch_size, 3)`` patch point arrays.
            - ``(num_patches, 3)`` centroid coordinates.
    """
    centroids = farthest_point_sample(points, num_patches)

    patches = np.zeros((num_patches, patch_size, 3), dtype=points.dtype)
    for i, centroid in enumerate(centroids):
        dists = np.linalg.norm(points - centroid, axis=1)
        nearest_indices = np.argsort(dists)[:patch_size]
        patch_points = points[nearest_indices]
        # Normalise patch to be centroid-relative
        patches[i] = patch_points - centroid

    return patches, centroids


def patches_to_tokens(
    patches: np.ndarray,
) -> np.ndarray:
    """Flatten patches into 1-D token vectors.

    Each ``(patch_size, 3)`` patch is reshaped to ``(patch_size * 3,)``.

    Args:
        patches: ``(num_patches, patch_size, 3)`` patch array.

    Returns:
        ``(num_patches, patch_size * 3)`` token array.
    """
    num_patches = patches.shape[0]
    return patches.reshape(num_patches, -1)
