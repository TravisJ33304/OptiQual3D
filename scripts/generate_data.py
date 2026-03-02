"""Generate synthetic anomaly data from ShapeNet meshes.

Loads ShapeNet meshes, converts them to point clouds, applies random
anomaly generation, and stores the resulting normal/anomalous pairs as
compressed NumPy archives for efficient training-time loading.

Usage::

    optiqual-generate --config configs/default.yaml \\
        --shapenet-root data/ShapeNet \\
        --output-dir data/generated
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic anomaly point cloud data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--shapenet-root",
        type=str,
        required=True,
        help="Root directory of ShapeNet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/generated",
        help="Output directory for generated data.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8192,
        help="Number of points to sample per mesh.",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.5,
        help="Fraction of samples to generate with anomalies.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="ShapeNet synset IDs or category names to process. "
        "If not provided, all available categories are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args(argv)


def generate_for_mesh(
    mesh_path: Path | tuple[Path, str],
    output_dir: Path,
    num_points: int,
    apply_anomaly: bool,
    anomaly_pipeline: object,
) -> dict[str, str]:
    """Generate a point cloud (optionally anomalous) from a single mesh.

    Args:
        mesh_path: Path to the source mesh file, or a ``(zip_path, entry)``
            tuple for zip-embedded meshes.
        output_dir: Directory to write the output ``.npz`` file.
        num_points: Number of surface points to sample.
        apply_anomaly: Whether to inject synthetic anomalies.
        anomaly_pipeline: An :class:`~optiqual3d.data.anomaly_generation.AnomalyPipeline`
            instance.

    Returns:
        Dict with keys ``path`` (output file) and ``status``.
    """
    import io as _io

    import trimesh

    from optiqual3d.data.preprocessing import normalize_point_cloud, sample_points

    # Load mesh from zip entry or from a plain file path
    if isinstance(mesh_path, tuple):
        zip_file, entry = mesh_path
        with zipfile.ZipFile(zip_file) as zf:
            with zf.open(entry) as f:
                data = f.read()
        mesh = trimesh.load(
            _io.BytesIO(data), file_type="obj", process=False, force="mesh"
        )
        stem = Path(entry).stem
    else:
        mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
        stem = Path(str(mesh_path)).stem

    if not isinstance(mesh, trimesh.Trimesh):
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
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    else:
        sample_result = trimesh.sample.sample_surface(mesh, max(num_points * 4, 8192))
        pts = np.asarray(sample_result[0], dtype=np.float32)

    pts = sample_points(pts, num_points)
    pts = normalize_point_cloud(pts)

    mask = np.zeros(pts.shape[0], dtype=np.float32)
    if apply_anomaly:
        from optiqual3d.data.anomaly_generation import AnomalyPipeline

        pipeline: AnomalyPipeline = anomaly_pipeline  # type: ignore[assignment]
        result = pipeline.generate_multi(pts)
        pts = result.points.astype(np.float32)
        mask = result.mask.astype(np.float32)
        subset = "anomalous"
    else:
        subset = "normal"

    out_subdir = output_dir / subset
    out_subdir.mkdir(parents=True, exist_ok=True)
    uid = np.random.randint(0, 10_000_000)
    out_path = out_subdir / f"{stem}_{uid:07d}.npz"

    np.savez_compressed(str(out_path), points=pts, mask=mask)
    return {"path": str(out_path), "status": "ok"}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the data generation CLI.

    Args:
        argv: Optional CLI arguments (defaults to ``sys.argv[1:]``).
    """
    from optiqual3d.utils.logging import setup_logging

    setup_logging()
    args = parse_args(argv)
    logger.info("Starting data generation with args: %s", args)

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from optiqual3d.config.settings import AnomalyGenerationConfig
    from optiqual3d.data.anomaly_generation import AnomalyPipeline
    from optiqual3d.data.datasets.shapenet import ShapeNetDataset

    np.random.seed(args.seed)

    shapenet_root = Path(args.shapenet_root)
    zip_files = sorted(shapenet_root.glob("*.zip"))
    if not zip_files:
        logger.error("No zip files found in %s", shapenet_root)
        return

    # Build the mesh entry index directly from zip manifests
    mesh_entries: list[tuple[Path, str]] = []
    cats = set(args.categories) if args.categories else None
    for zip_path in zip_files:
        if cats:
            # Accept both synset IDs and human-readable names
            synset = zip_path.stem
            readable = {v: k for k, v in ShapeNetDataset.SYNSET_MAP.items()}.get(
                synset, synset
            )
            if synset not in cats and readable not in cats:
                continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith("/models/model_normalized.obj"):
                        mesh_entries.append((zip_path, name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", zip_path.name, exc)

    if not mesh_entries:
        logger.error("No mesh entries found. Check --shapenet-root and --categories.")
        return

    logger.info("Indexed %d meshes from ShapeNetCore", len(mesh_entries))

    anomaly_pipeline = AnomalyPipeline(AnomalyGenerationConfig())

    rng = np.random.default_rng(args.seed)
    anomaly_flags = (rng.random(len(mesh_entries)) < args.anomaly_ratio).tolist()

    logger.info(
        "Generating %d samples (%d anomalous) using %d workers …",
        len(mesh_entries),
        int(sum(anomaly_flags)),
        args.num_workers,
    )

    success = failed = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                generate_for_mesh,
                mesh_path,
                output_dir,
                args.num_points,
                bool(is_anomalous),
                anomaly_pipeline,
            ): i
            for i, (mesh_path, is_anomalous) in enumerate(
                zip(mesh_entries, anomaly_flags)
            )
        }
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result["status"] == "ok":
                    success += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.debug("Sample failed: %s", exc)
            if (i + 1) % 1000 == 0:
                logger.info(
                    "Progress %d/%d  ok=%d  failed=%d",
                    i + 1, len(futures), success, failed,
                )

    logger.info(
        "Generation complete — success=%d  failed=%d  output=%s",
        success,
        failed,
        output_dir,
    )


if __name__ == "__main__":
    main()
