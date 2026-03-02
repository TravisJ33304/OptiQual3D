"""ShapeNet dataset loader for OptiQual3D.

Loads 3D meshes from ShapeNet, converts them to point clouds via
surface sampling, and serves them as training data for the self-supervised
pre-training phase and the anomaly generation pipeline.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from optiqual3d.config.settings import PointCloudConfig
from optiqual3d.data.preprocessing import normalize_point_cloud, sample_points

logger = logging.getLogger(__name__)


class ShapeNetDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for ShapeNet point clouds.

    Loads ShapeNet meshes, samples point clouds from their surfaces,
    and returns normalised point tensors.  Supports category filtering
    for zero-shot evaluation splits.

    Attributes:
        root: Root directory of the ShapeNet dataset.
        categories: List of synset IDs to include (``None`` = all).
        samples: List of resolved sample metadata dicts.
    """

    # Mapping from human-readable names to ShapeNet synset IDs.
    SYNSET_MAP: dict[str, str] = {
        "airplane": "02691156",
        "bag": "02773838",
        "basket": "02801938",
        "bathtub": "02808440",
        "bed": "02818832",
        "bench": "02828884",
        "bottle": "02876657",
        "bowl": "02880940",
        "bus": "02924116",
        "cabinet": "02933112",
        "camera": "02942699",
        "can": "02946921",
        "cap": "02954340",
        "car": "02958343",
        "chair": "03001627",
        "clock": "03046257",
        "dishwasher": "03207941",
        "display": "03211117",
        "earphone": "03261776",
        "faucet": "03325088",
        "file_cabinet": "03337140",
        "guitar": "03467517",
        "helmet": "03513137",
        "jar": "03593526",
        "knife": "03624134",
        "lamp": "03636649",
        "laptop": "03642806",
        "loudspeaker": "03691459",
        "mailbox": "03710193",
        "microphone": "03759954",
        "microwave": "03761084",
        "motorbike": "03790512",
        "mug": "03797390",
        "piano": "03928116",
        "pillow": "03938244",
        "pistol": "03948459",
        "pot": "03991062",
        "printer": "04004475",
        "remote": "04074963",
        "rifle": "04090263",
        "rocket": "04099429",
        "skateboard": "04225987",
        "sofa": "04256520",
        "stove": "04330267",
        "table": "04379243",
        "telephone": "04401088",
        "tower": "04460130",
        "train": "04468005",
        "vessel": "04530566",
        "washer": "04554684",
    }

    def __init__(
        self,
        root: str | Path,
        categories: list[str] | None = None,
        split: str = "train",
        point_cloud_cfg: PointCloudConfig | None = None,
        transform: Any | None = None,
    ) -> None:
        """Initialise the ShapeNet dataset.

        Args:
            root: Path to the ShapeNet root directory.
            categories: Category names or synset IDs to include.
                ``None`` loads all available categories.
            split: Dataset split (``train``, ``val``, ``test``).
            point_cloud_cfg: Point cloud sampling parameters.
            transform: Optional callable applied to each sample dict.
        """
        self.root = Path(root)
        self.split = split
        self.pc_cfg = point_cloud_cfg or PointCloudConfig()
        self.transform = transform
        self.categories = self._resolve_categories(categories)
        self.samples: list[dict[str, Any]] = []

        self._discover_samples()
        logger.info(
            "ShapeNet %s: %d samples across %d categories",
            split,
            len(self.samples),
            len(self.categories),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load and return a single sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with keys:
                - ``points``: ``(N, 3)`` float tensor of XYZ coordinates.
                - ``category``: Integer category label.
                - ``sample_id``: String identifier for the sample.
        """
        meta = self.samples[index]
        points = self._load_point_cloud(meta["zip_path"], meta["entry"])
        points = sample_points(points, self.pc_cfg.num_points)

        if self.pc_cfg.normalize:
            points = normalize_point_cloud(points)

        sample: dict[str, Any] = {
            "points": torch.from_numpy(points).float(),
            "category": meta["category_idx"],
            "sample_id": meta["sample_id"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_categories(
        self, categories: list[str] | None
    ) -> list[str]:
        """Map category names to synset IDs.

        Args:
            categories: Human-readable names or raw synset IDs.

        Returns:
            Sorted list of synset IDs.
        """
        if categories is None:
            # Discover all synset IDs from zip archives on disk
            return sorted(
                p.stem
                for p in self.root.glob("*.zip")
                if not p.stem.startswith(".")
            )

        resolved: list[str] = []
        for cat in categories:
            synset = self.SYNSET_MAP.get(cat, cat)
            resolved.append(synset)
        return sorted(resolved)

    def _discover_samples(self) -> None:
        """Walk the ShapeNet directory tree and index samples.

        Populates :attr:`samples` with metadata dicts containing
        ``path``, ``category_idx``, and ``sample_id``.
        """
        for cat_idx, synset_id in enumerate(self.categories):
            zip_path = self.root / f"{synset_id}.zip"
            if not zip_path.exists():
                logger.warning("Zip archive not found: %s", zip_path)
                continue

            with zipfile.ZipFile(zip_path) as zf:
                obj_entries = sorted(
                    e for e in zf.namelist()
                    if e.endswith("/models/model_normalized.obj")
                )

            n = len(obj_entries)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)

            if self.split == "train":
                selected = obj_entries[:train_end]
            elif self.split == "val":
                selected = obj_entries[train_end:val_end]
            else:
                selected = obj_entries[val_end:]

            for entry in selected:
                model_hash = entry.split("/")[1]
                self.samples.append(
                    {
                        "zip_path": zip_path,
                        "entry": entry,
                        "category_idx": cat_idx,
                        "sample_id": f"{synset_id}/{model_hash}",
                    }
                )

    @staticmethod
    def _find_mesh(model_dir: Path) -> Path | None:
        """Locate the mesh file inside a ShapeNet model directory.

        Args:
            model_dir: Path to the model directory.

        Returns:
            Path to the mesh file, or ``None`` if not found.
        """
        # ShapeNet stores meshes as model_normalized.obj or similar
        for suffix in (".obj", ".off", ".ply"):
            candidates = list(model_dir.rglob(f"*{suffix}"))
            if candidates:
                return candidates[0]
        return None

    def _load_point_cloud(self, zip_path: Path, entry: str) -> np.ndarray:
        """Load a mesh from a ShapeNet zip archive and sample a point cloud.

        Args:
            zip_path: Path to the category zip archive.
            entry: Internal zip path to the OBJ mesh file.

        Returns:
            ``(N, 3)`` numpy array of sampled surface points.

        Raises:
            RuntimeError: If the mesh cannot be loaded.
        """
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(entry) as f:
                mesh_bytes = f.read()

        mesh = trimesh.load(
            io.BytesIO(mesh_bytes),
            file_type="obj",
            process=False,
            force="mesh",
        )

        if not isinstance(mesh, trimesh.Trimesh):
            # Handle Scene objects (multiple sub-meshes)
            if isinstance(mesh, trimesh.Scene):
                geometries = list(mesh.geometry.values())
                mesh = (
                    trimesh.util.concatenate(geometries)
                    if geometries
                    else trimesh.Trimesh()
                )
            else:
                mesh = trimesh.Trimesh()

        if len(mesh.faces) == 0:
            logger.warning("Mesh has no faces: %s::%s", zip_path.name, entry)
            return np.asarray(mesh.vertices, dtype=np.float32)

        # Sample generously so FPS has good coverage to choose from
        n_sample = max(8192, self.pc_cfg.num_points * 4)
        pts, *_ = trimesh.sample.sample_surface(mesh, n_sample)
        return pts.astype(np.float32)
