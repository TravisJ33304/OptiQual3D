"""Point cloud data augmentation transforms.

All transforms operate on sample dictionaries containing at minimum a
``"points"`` key with a ``(N, 3)`` float tensor.  Transforms can be
composed using :class:`Compose`.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import torch

from optiqual3d.config.settings import AugmentationConfig

# ---------------------------------------------------------------------------
# Transform Protocol
# ---------------------------------------------------------------------------


class PointCloudTransform(Protocol):
    """Protocol for point cloud augmentation transforms."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply the transform to a sample dictionary.

        Args:
            sample: Dictionary containing at least a ``"points"`` tensor.

        Returns:
            Transformed sample dictionary.
        """
        ...


# ---------------------------------------------------------------------------
# Individual Transforms
# ---------------------------------------------------------------------------


class RandomRotation:
    """Apply random rotation around all three axes.

    Args:
        max_angle: Maximum rotation angle in degrees per axis.
    """

    def __init__(self, max_angle: float = 180.0) -> None:
        """Initialise the random rotation transform."""
        self.max_angle = max_angle

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply random rotation.

        Args:
            sample: Sample dict with ``"points"`` tensor.

        Returns:
            Rotated sample.
        """
        points: torch.Tensor = sample["points"]
        angles = torch.rand(3) * 2 * self.max_angle - self.max_angle
        angles = angles * (math.pi / 180.0)

        rot_matrix = _euler_to_rotation_matrix(
            angles[0].item(), angles[1].item(), angles[2].item()
        )
        sample["points"] = points @ rot_matrix.T
        return sample


class RandomScale:
    """Apply random uniform scaling.

    Args:
        scale_min: Minimum scale factor.
        scale_max: Maximum scale factor.
    """

    def __init__(self, scale_min: float = 0.8, scale_max: float = 1.2) -> None:
        """Initialise the random scale transform."""
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply random scaling.

        Args:
            sample: Sample dict with ``"points"`` tensor.

        Returns:
            Scaled sample.
        """
        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        sample["points"] = sample["points"] * scale
        return sample


class RandomJitter:
    """Add Gaussian positional noise to each point.

    Args:
        sigma: Standard deviation of Gaussian noise.
        clip: Maximum absolute noise value.
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.05) -> None:
        """Initialise the random jitter transform."""
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply jitter.

        Args:
            sample: Sample dict with ``"points"`` tensor.

        Returns:
            Jittered sample.
        """
        points: torch.Tensor = sample["points"]
        noise = torch.randn_like(points) * self.sigma
        noise = noise.clamp(-self.clip, self.clip)
        sample["points"] = points + noise
        return sample


class RandomPointDropout:
    """Randomly drop a fraction of points (replaced with duplicates).

    Args:
        dropout_ratio: Fraction of points to drop.
    """

    def __init__(self, dropout_ratio: float = 0.1) -> None:
        """Initialise the random point dropout transform."""
        self.dropout_ratio = dropout_ratio

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply point dropout.

        Args:
            sample: Sample dict with ``"points"`` tensor.

        Returns:
            Sample with dropped points replaced by random duplicates.
        """
        points: torch.Tensor = sample["points"]
        n = points.shape[0]
        num_drop = int(n * self.dropout_ratio)
        if num_drop == 0:
            return sample

        keep_mask = torch.ones(n, dtype=torch.bool)
        drop_indices = torch.randperm(n)[:num_drop]
        keep_mask[drop_indices] = False

        kept = points[keep_mask]
        # Replace dropped points with random copies of kept points
        replace_indices = torch.randint(0, kept.shape[0], (num_drop,))
        replacements = kept[replace_indices]
        sample["points"] = torch.cat([kept, replacements], dim=0)

        return sample


class RandomTranslation:
    """Apply random translation.

    Args:
        max_shift: Maximum translation per axis.
    """

    def __init__(self, max_shift: float = 0.1) -> None:
        """Initialise the random translation transform."""
        self.max_shift = max_shift

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply random translation.

        Args:
            sample: Sample dict with ``"points"`` tensor.

        Returns:
            Translated sample.
        """
        shift = torch.rand(3) * 2 * self.max_shift - self.max_shift
        sample["points"] = sample["points"] + shift
        return sample


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of transform callables.
    """

    def __init__(self, transforms: list[PointCloudTransform]) -> None:
        """Initialise the composition of transforms."""
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply all transforms in order.

        Args:
            sample: Sample dictionary.

        Returns:
            Transformed sample.
        """
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_augmentation(cfg: AugmentationConfig) -> Compose | None:
    """Build augmentation pipeline from configuration.

    Args:
        cfg: Augmentation configuration dataclass.

    Returns:
        A :class:`Compose` of transforms, or ``None`` if augmentation
        is disabled.
    """
    if not cfg.enable:
        return None

    transforms: list[PointCloudTransform] = [
        RandomRotation(max_angle=cfg.rotation_range),
        RandomScale(scale_min=cfg.scale_min, scale_max=cfg.scale_max),
        RandomJitter(sigma=cfg.jitter_sigma, clip=cfg.jitter_clip),
        RandomPointDropout(dropout_ratio=cfg.dropout_ratio),
    ]
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Convert Euler angles (XYZ) to a 3×3 rotation matrix.

    Args:
        roll: Rotation around X in radians.
        pitch: Rotation around Y in radians.
        yaw: Rotation around Z in radians.

    Returns:
        ``(3, 3)`` rotation matrix tensor.
    """
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    rx = torch.tensor(
        [[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]], dtype=torch.float32
    )
    ry = torch.tensor(
        [[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]], dtype=torch.float32
    )
    rz = torch.tensor(
        [[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]], dtype=torch.float32
    )
    return rz @ ry @ rx
