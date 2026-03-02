"""Generated dataset that wraps a base dataset with synthetic anomalies.

Serves pre-generated or on-the-fly anomalous samples alongside their
clean counterparts for Phase 2 training.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from optiqual3d.config.settings import (
    AnomalyGenerationConfig,
    PointCloudConfig,
)
from optiqual3d.data.anomaly_generation import AnomalyPipeline
from optiqual3d.data.preprocessing import normalize_point_cloud, sample_points

logger = logging.getLogger(__name__)


class GeneratedAnomalyDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset combining normal and synthetically-anomalised samples.

    In **offline** mode the dataset reads pre-generated ``.npz`` files
    from disk.  In **online** mode it wraps a base dataset and generates
    anomalies on the fly.

    Attributes:
        mode: ``"offline"`` or ``"online"``.
        samples: List of sample metadata dicts (offline mode).
        base_dataset: Underlying clean dataset (online mode).
        pipeline: Anomaly generation pipeline.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        base_dataset: Dataset[dict[str, torch.Tensor]] | None = None,
        anomaly_cfg: AnomalyGenerationConfig | None = None,
        point_cloud_cfg: PointCloudConfig | None = None,
        anomaly_ratio: float = 0.5,
        transform: (
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None
        ) = None,
    ) -> None:
        """Initialise the generated dataset.

        Exactly one of *root* (offline) or *base_dataset* (online) must
        be provided.

        Args:
            root: Path to pre-generated ``data/generated/`` directory.
            base_dataset: A clean dataset to augment on the fly.
            anomaly_cfg: Anomaly generation configuration.
            point_cloud_cfg: Point cloud preprocessing parameters.
            anomaly_ratio: Fraction of returned samples that are anomalous
                (online mode only).
            transform: Optional per-sample transform.

        Raises:
            ValueError: If neither or both of *root* and *base_dataset*
                are provided.
        """
        if (root is None) == (base_dataset is None):
            raise ValueError(
                "Provide exactly one of 'root' (offline) or " "'base_dataset' (online)."
            )

        self.pc_cfg = point_cloud_cfg or PointCloudConfig()
        self.transform = transform
        self.anomaly_ratio = anomaly_ratio
        self.pipeline = AnomalyPipeline(anomaly_cfg or AnomalyGenerationConfig())

        if root is not None:
            self.mode = "offline"
            self.root = Path(root)
            self.samples = self._index_offline()
            self.base_dataset = None
        else:
            self.mode = "online"
            self.root = None
            self.samples = []
            self.base_dataset = base_dataset

        logger.info(
            "GeneratedAnomalyDataset mode=%s, size=%d",
            self.mode,
            len(self),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of samples."""
        if self.mode == "offline":
            return len(self.samples)
        assert self.base_dataset is not None
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load and return a sample.

        In online mode, randomly decides whether to return a clean or
        anomalous version based on :attr:`anomaly_ratio`.

        Args:
            index: Sample index.

        Returns:
            Dictionary with keys:
                - ``points``: ``(N, 3)`` point cloud tensor.
                - ``label``: 0 (normal) or 1 (anomalous).
                - ``mask``: ``(N,)`` per-point binary anomaly mask.
        """
        if self.mode == "offline":
            return self._load_offline(index)
        return self._load_online(index)

    # ------------------------------------------------------------------
    # Offline helpers
    # ------------------------------------------------------------------

    def _index_offline(self) -> list[dict[str, Any]]:
        """Index pre-generated NPZ files.

        Returns:
            List of sample metadata dicts with ``path`` and ``label``.
        """
        assert self.root is not None
        samples: list[dict[str, Any]] = []

        for subset in ["normal", "anomalous"]:
            subset_dir = self.root / subset
            if not subset_dir.exists():
                continue
            label = 0 if subset == "normal" else 1
            for npz_path in sorted(subset_dir.rglob("*.npz")):
                samples.append({"path": npz_path, "label": label})

        return samples

    def _load_offline(self, index: int) -> dict[str, torch.Tensor]:
        """Load a pre-generated sample from disk.

        Args:
            index: Sample index.

        Returns:
            Sample dictionary.
        """
        meta = self.samples[index]
        data = np.load(meta["path"])
        points = data["points"].astype(np.float32)
        mask = (
            data["mask"].astype(np.float32)
            if "mask" in data
            else np.zeros(points.shape[0], dtype=np.float32)
        )

        points = sample_points(points, self.pc_cfg.num_points)
        if self.pc_cfg.normalize:
            points = normalize_point_cloud(points)

        # Align mask length after resampling
        if mask.shape[0] != points.shape[0]:
            new_mask = np.zeros(points.shape[0], dtype=np.float32)
            overlap = min(mask.shape[0], new_mask.shape[0])
            new_mask[:overlap] = mask[:overlap]
            mask = new_mask

        sample: dict[str, Any] = {
            "points": torch.from_numpy(points).float(),
            "label": int(meta["label"]),
            "mask": torch.from_numpy(mask).float(),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Online helpers
    # ------------------------------------------------------------------

    def _load_online(self, index: int) -> dict[str, torch.Tensor]:
        """Generate a sample on the fly.

        Args:
            index: Index into the base dataset.

        Returns:
            Sample dictionary.
        """
        assert self.base_dataset is not None
        base_sample = self.base_dataset[index]
        points_np: np.ndarray = base_sample["points"].numpy()

        rng = np.random.default_rng()
        apply_anomaly = rng.random() < self.anomaly_ratio

        if apply_anomaly:
            result = self.pipeline.generate_multi(points_np)
            points_np = result.points
            mask_np = result.mask
            label = 1
        else:
            mask_np = np.zeros(points_np.shape[0], dtype=np.float32)
            label = 0

        # Re-sample to fixed count
        points_np = sample_points(points_np, self.pc_cfg.num_points)
        if self.pc_cfg.normalize:
            points_np = normalize_point_cloud(points_np)

        # Adjust mask length
        if mask_np.shape[0] != points_np.shape[0]:
            new_mask = np.zeros(points_np.shape[0], dtype=np.float32)
            overlap = min(mask_np.shape[0], new_mask.shape[0])
            new_mask[:overlap] = mask_np[:overlap]
            mask_np = new_mask

        sample: dict[str, Any] = {
            "points": torch.from_numpy(points_np).float(),
            "label": label,
            "mask": torch.from_numpy(mask_np).float(),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
