"""Programmatic synthetic anomaly generation for point clouds.

Generates realistic manufacturing defects by applying geometric
perturbations to clean 3D meshes / point clouds.  Each anomaly type
produces per-point labels indicating affected regions.

Anomaly categories:
    **Surface defects** — dents, bumps, scratches, holes.
    **Structural defects** — missing parts, extra material, cracks.
    **Measurement noise** — point dropout, Gaussian noise, outlier points.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from optiqual3d.config.settings import AnomalyGenerationConfig, AnomalyType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnomalyResult:
    """Output of an anomaly generation step.

    Attributes:
        points: ``(N, 3)`` modified point cloud.
        mask: ``(N,)`` binary mask (1 = anomalous point).
        anomaly_type: The type of anomaly applied.
        metadata: Additional info (severity, region centre, etc.).
    """

    points: np.ndarray
    mask: np.ndarray
    anomaly_type: AnomalyType
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Base class for anomaly generators
# ---------------------------------------------------------------------------


class AnomalyGenerator(ABC):
    """Abstract base for a single anomaly type generator."""

    @abstractmethod
    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Apply the anomaly to a point cloud.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Anomaly severity in ``[0, 1]``.
            rng: NumPy random generator for reproducibility.

        Returns:
            An :class:`AnomalyResult` with modified points and mask.
        """
        ...


# ---------------------------------------------------------------------------
# Surface Defect Generators
# ---------------------------------------------------------------------------


class DentGenerator(AnomalyGenerator):
    """Generate local inward dents using radial basis functions.

    Selects a random surface point as centre, then pushes nearby
    points inward along their estimated normals.
    """

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Apply a dent deformation.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls dent depth and radius.
            rng: Random generator.

        Returns:
            Anomaly result with dent applied.
        """
        n = points.shape[0]
        mask = np.zeros(n, dtype=np.float32)

        # Select random centre point
        centre_idx = rng.integers(0, n)
        centre = points[centre_idx]

        # Determine affected radius proportional to severity
        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        radius = cloud_scale * (0.05 + 0.15 * severity)
        depth = cloud_scale * (0.02 + 0.08 * severity)

        # Find points within radius
        dists = np.linalg.norm(points - centre, axis=1)
        affected = dists < radius

        # RBF-based inward deformation
        rbf_weights = np.exp(-0.5 * (dists[affected] / (radius * 0.5)) ** 2)

        # Estimate local inward direction (towards centroid)
        centroid = points.mean(axis=0)
        inward_dir = centroid - points[affected]
        norms = np.linalg.norm(inward_dir, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        inward_dir = inward_dir / norms

        modified = points.copy()
        modified[affected] += inward_dir * rbf_weights[:, None] * depth
        mask[affected] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.DENT,
            metadata={"centre": centre.tolist(), "radius": radius, "depth": depth},
        )


class BumpGenerator(AnomalyGenerator):
    """Generate local outward bumps (protrusions)."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Apply a bump protrusion.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls bump height and radius.
            rng: Random generator.

        Returns:
            Anomaly result with bump applied.
        """
        n = points.shape[0]
        mask = np.zeros(n, dtype=np.float32)

        centre_idx = rng.integers(0, n)
        centre = points[centre_idx]

        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        radius = cloud_scale * (0.05 + 0.15 * severity)
        height = cloud_scale * (0.02 + 0.08 * severity)

        dists = np.linalg.norm(points - centre, axis=1)
        affected = dists < radius

        rbf_weights = np.exp(-0.5 * (dists[affected] / (radius * 0.5)) ** 2)

        # Outward direction (away from centroid)
        centroid = points.mean(axis=0)
        outward_dir = points[affected] - centroid
        norms = np.linalg.norm(outward_dir, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        outward_dir = outward_dir / norms

        modified = points.copy()
        modified[affected] += outward_dir * rbf_weights[:, None] * height
        mask[affected] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.BUMP,
            metadata={"centre": centre.tolist(), "radius": radius, "height": height},
        )


class ScratchGenerator(AnomalyGenerator):
    """Generate linear groove scratches along the surface."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Apply a scratch deformation.

        Defines a random line segment on the surface and displaces
        nearby points inward.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls scratch width and depth.
            rng: Random generator.

        Returns:
            Anomaly result with scratch applied.
        """
        n = points.shape[0]
        mask = np.zeros(n, dtype=np.float32)

        # Define scratch line via two random surface points
        idx_a, idx_b = rng.choice(n, size=2, replace=False)
        point_a = points[idx_a]
        point_b = points[idx_b]

        line_dir = point_b - point_a
        line_len = np.linalg.norm(line_dir)
        if line_len < 1e-8:
            return AnomalyResult(
                points=points.copy(),
                mask=mask,
                anomaly_type=AnomalyType.SCRATCH,
                metadata={},
            )
        line_dir = line_dir / line_len

        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        width = cloud_scale * (0.01 + 0.03 * severity)
        depth = cloud_scale * (0.01 + 0.04 * severity)

        # Compute perpendicular distance of each point to the line
        ap = points - point_a
        proj = np.dot(ap, line_dir)
        proj_points = point_a + np.outer(proj, line_dir)
        perp_dist = np.linalg.norm(points - proj_points, axis=1)

        # Affected = close to line and between endpoints
        on_segment = (proj >= 0) & (proj <= line_len)
        close_to_line = perp_dist < width
        affected = on_segment & close_to_line

        # Displace inward
        centroid = points.mean(axis=0)
        inward = centroid - points[affected]
        norms = np.linalg.norm(inward, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        inward = inward / norms

        fall_off = 1.0 - (perp_dist[affected] / width)

        modified = points.copy()
        modified[affected] += inward * fall_off[:, None] * depth
        mask[affected] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.SCRATCH,
            metadata={"start": point_a.tolist(), "end": point_b.tolist()},
        )


class HoleGenerator(AnomalyGenerator):
    """Generate missing-geometry holes by removing local regions."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Remove a spherical region of points to simulate a hole.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls hole size.
            rng: Random generator.

        Returns:
            Anomaly result (points removed from cloud).
        """
        n = points.shape[0]

        centre_idx = rng.integers(0, n)
        centre = points[centre_idx]

        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        radius = cloud_scale * (0.03 + 0.12 * severity)

        dists = np.linalg.norm(points - centre, axis=1)
        keep_mask = dists >= radius
        removed_mask = ~keep_mask

        # Build full-length anomaly mask (1 where points were removed)
        mask = np.zeros(n, dtype=np.float32)
        mask[removed_mask] = 1.0

        modified = points[keep_mask]

        return AnomalyResult(
            points=modified,
            mask=mask[: modified.shape[0]] if modified.shape[0] < n else mask,
            anomaly_type=AnomalyType.HOLE,
            metadata={"centre": centre.tolist(), "radius": radius},
        )


# ---------------------------------------------------------------------------
# Structural Defect Generators
# ---------------------------------------------------------------------------


class MissingPartGenerator(AnomalyGenerator):
    """Remove a large connected region to simulate a missing component."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Remove a large spherical region simulating a missing part.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls region size.
            rng: Random generator.

        Returns:
            Anomaly result with missing region.
        """
        n = points.shape[0]

        # Choose a point far from centroid for more visible effect
        centroid = points.mean(axis=0)
        dists_to_centre = np.linalg.norm(points - centroid, axis=1)
        far_indices = np.argsort(dists_to_centre)[-max(1, n // 4) :]
        centre_idx = rng.choice(far_indices)
        centre = points[centre_idx]

        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        radius = cloud_scale * (0.1 + 0.2 * severity)

        dists = np.linalg.norm(points - centre, axis=1)
        keep = dists >= radius

        mask = np.zeros(n, dtype=np.float32)
        mask[~keep] = 1.0

        modified = points[keep]

        return AnomalyResult(
            points=modified,
            mask=mask[: modified.shape[0]] if modified.shape[0] < n else mask,
            anomaly_type=AnomalyType.MISSING_PART,
            metadata={"centre": centre.tolist(), "radius": radius},
        )


class ExtraMaterialGenerator(AnomalyGenerator):
    """Add spurious geometric blobs to simulate extra material."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Add random blob points near the surface.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls blob size and number of added points.
            rng: Random generator.

        Returns:
            Anomaly result with extra points added.
        """
        n = points.shape[0]

        # Select attachment point on surface
        attach_idx = rng.integers(0, n)
        attach_point = points[attach_idx]

        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        blob_radius = cloud_scale * (0.02 + 0.06 * severity)
        num_extra = int(n * (0.02 + 0.08 * severity))

        # Generate random points in a sphere around the attachment point
        extra_points = rng.normal(0, blob_radius / 3, size=(num_extra, 3))
        extra_points = extra_points + attach_point

        modified = np.concatenate([points, extra_points], axis=0)
        mask = np.zeros(modified.shape[0], dtype=np.float32)
        mask[n:] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.EXTRA_MATERIAL,
            metadata={
                "attach_point": attach_point.tolist(),
                "num_extra": num_extra,
            },
        )


class CrackGenerator(AnomalyGenerator):
    """Generate surface discontinuities (cracks) along a random plane."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Split points along a crack plane.

        Points on one side of a thin band near the crack plane are
        displaced to create a visible gap.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls crack width.
            rng: Random generator.

        Returns:
            Anomaly result with crack deformation.
        """
        n = points.shape[0]
        mask = np.zeros(n, dtype=np.float32)

        # Random crack plane through a surface point
        centre_idx = rng.integers(0, n)
        centre = points[centre_idx]
        normal = rng.standard_normal(3)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        gap_width = cloud_scale * (0.005 + 0.02 * severity)
        affect_width = cloud_scale * (0.05 + 0.1 * severity)

        # Signed distance to crack plane
        offsets = np.dot(points - centre, normal)
        near_crack = np.abs(offsets) < affect_width

        modified = points.copy()
        # Separate points on either side of the crack
        pos_side = near_crack & (offsets > 0)
        neg_side = near_crack & (offsets < 0)

        fall_off_pos = 1.0 - np.abs(offsets[pos_side]) / affect_width
        fall_off_neg = 1.0 - np.abs(offsets[neg_side]) / affect_width

        modified[pos_side] += normal * gap_width * fall_off_pos[:, None]
        modified[neg_side] -= normal * gap_width * fall_off_neg[:, None]

        mask[near_crack] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.CRACK,
            metadata={"centre": centre.tolist(), "normal": normal.tolist()},
        )


# ---------------------------------------------------------------------------
# Measurement Noise Generators
# ---------------------------------------------------------------------------


class PointDropoutGenerator(AnomalyGenerator):
    """Simulate sensor occlusion by removing random points."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Drop random points from the cloud.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Fraction of points to drop.
            rng: Random generator.

        Returns:
            Anomaly result with dropped points.
        """
        n = points.shape[0]
        drop_ratio = 0.05 + 0.25 * severity
        num_keep = max(1, int(n * (1.0 - drop_ratio)))

        indices = rng.choice(n, size=num_keep, replace=False)
        indices.sort()

        mask = np.ones(n, dtype=np.float32)
        mask[indices] = 0.0

        return AnomalyResult(
            points=points[indices],
            mask=mask[:num_keep],
            anomaly_type=AnomalyType.POINT_DROPOUT,
            metadata={"drop_ratio": drop_ratio},
        )


class GaussianNoiseGenerator(AnomalyGenerator):
    """Add positional Gaussian noise to all or a subset of points."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Add Gaussian noise.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls noise magnitude.
            rng: Random generator.

        Returns:
            Anomaly result with noise added.
        """
        n = points.shape[0]
        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        sigma = cloud_scale * (0.005 + 0.03 * severity)

        # Apply noise to a random subset
        noise_ratio = 0.2 + 0.6 * severity
        num_noisy = max(1, int(n * noise_ratio))
        noisy_indices = rng.choice(n, size=num_noisy, replace=False)

        modified = points.copy()
        noise = rng.normal(0, sigma, size=(num_noisy, 3))
        modified[noisy_indices] += noise

        mask = np.zeros(n, dtype=np.float32)
        mask[noisy_indices] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.GAUSSIAN_NOISE,
            metadata={"sigma": sigma},
        )


class OutlierPointsGenerator(AnomalyGenerator):
    """Insert spurious outlier points far from the surface."""

    def apply(
        self,
        points: np.ndarray,
        severity: float,
        rng: np.random.Generator,
    ) -> AnomalyResult:
        """Add outlier points.

        Args:
            points: ``(N, 3)`` clean point cloud.
            severity: Controls number and distance of outliers.
            rng: Random generator.

        Returns:
            Anomaly result with outliers added.
        """
        n = points.shape[0]
        cloud_scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        num_outliers = max(1, int(n * (0.01 + 0.05 * severity)))
        distance = cloud_scale * (0.3 + 0.5 * severity)

        centroid = points.mean(axis=0)
        outliers = centroid + rng.standard_normal((num_outliers, 3)) * distance

        modified = np.concatenate([points, outliers], axis=0)
        mask = np.zeros(modified.shape[0], dtype=np.float32)
        mask[n:] = 1.0

        return AnomalyResult(
            points=modified,
            mask=mask,
            anomaly_type=AnomalyType.OUTLIER_POINTS,
            metadata={"num_outliers": num_outliers},
        )


# ---------------------------------------------------------------------------
# Registry & Pipeline
# ---------------------------------------------------------------------------

GENERATOR_REGISTRY: dict[AnomalyType, type[AnomalyGenerator]] = {
    AnomalyType.DENT: DentGenerator,
    AnomalyType.BUMP: BumpGenerator,
    AnomalyType.SCRATCH: ScratchGenerator,
    AnomalyType.HOLE: HoleGenerator,
    AnomalyType.MISSING_PART: MissingPartGenerator,
    AnomalyType.EXTRA_MATERIAL: ExtraMaterialGenerator,
    AnomalyType.CRACK: CrackGenerator,
    AnomalyType.POINT_DROPOUT: PointDropoutGenerator,
    AnomalyType.GAUSSIAN_NOISE: GaussianNoiseGenerator,
    AnomalyType.OUTLIER_POINTS: OutlierPointsGenerator,
}


class AnomalyPipeline:
    """Orchestrates anomaly generation across multiple types.

    Randomly selects and applies one or more anomaly generators to a
    clean point cloud, producing labelled anomalous samples.

    Attributes:
        generators: Mapping of anomaly types to instantiated generators.
        cfg: Configuration controlling severity, types, and count.
        rng: Seeded random generator.
    """

    def __init__(self, cfg: AnomalyGenerationConfig) -> None:
        """Initialise the pipeline.

        Args:
            cfg: Anomaly generation configuration.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Instantiate selected generators
        self.generators: dict[AnomalyType, AnomalyGenerator] = {}
        for type_name in cfg.anomaly_types:
            atype = AnomalyType(type_name)
            if atype in GENERATOR_REGISTRY:
                self.generators[atype] = GENERATOR_REGISTRY[atype]()
            else:
                logger.warning("Unknown anomaly type: %s", type_name)

    def generate(
        self,
        points: np.ndarray,
        anomaly_type: AnomalyType | None = None,
    ) -> AnomalyResult:
        """Apply anomaly generation to a clean point cloud.

        If *anomaly_type* is ``None``, a random type is selected from
        the configured set.

        Args:
            points: ``(N, 3)`` clean point cloud (not modified in place).
            anomaly_type: Specific anomaly type, or ``None`` for random.

        Returns:
            :class:`AnomalyResult` with the modified cloud and mask.
        """
        if anomaly_type is None:
            available = list(self.generators.keys())
            idx = int(self.rng.integers(len(available)))
            anomaly_type = available[idx]

        generator = self.generators[anomaly_type]
        severity = self.rng.uniform(self.cfg.severity_min, self.cfg.severity_max)

        return generator.apply(points.copy(), severity, self.rng)

    def generate_multi(
        self,
        points: np.ndarray,
    ) -> AnomalyResult:
        """Apply multiple anomaly types to a single point cloud.

        The number of anomalies is drawn from
        ``cfg.num_anomalies_per_sample``.

        Args:
            points: ``(N, 3)`` clean point cloud.

        Returns:
            Combined :class:`AnomalyResult`.
        """
        lo, hi = self.cfg.num_anomalies_per_sample
        num_anomalies = self.rng.integers(lo, hi + 1)

        current_points = points.copy()
        combined_mask = np.zeros(current_points.shape[0], dtype=np.float32)
        applied_types: list[str] = []

        for _ in range(num_anomalies):
            result = self.generate(current_points)
            current_points = result.points

            # Resize combined mask if point count changed
            if result.mask.shape[0] != combined_mask.shape[0]:
                new_mask = np.zeros(current_points.shape[0], dtype=np.float32)
                overlap = min(combined_mask.shape[0], new_mask.shape[0])
                new_mask[:overlap] = combined_mask[:overlap]
                combined_mask = new_mask

            # Merge masks (union)
            min_len = min(combined_mask.shape[0], result.mask.shape[0])
            combined_mask[:min_len] = np.maximum(
                combined_mask[:min_len], result.mask[:min_len]
            )

            applied_types.append(result.anomaly_type.value)

        return AnomalyResult(
            points=current_points,
            mask=combined_mask,
            anomaly_type=AnomalyType(applied_types[-1]),
            metadata={"applied_types": applied_types},
        )
