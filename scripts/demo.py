"""Interactive demo: load a model and visualise predictions.

Accepts a trained checkpoint and a point cloud file (PLY, PCD, or NPZ),
runs inference, and renders the result with an anomaly heatmap overlay.

Usage::

    optiqual-demo --checkpoint outputs/train/best.pt \\
        --input sample.ply \\
        --backend plotly
"""

from __future__ import annotations

import argparse
import logging
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
        description="Interactive anomaly detection demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a point cloud file (PLY, PCD, or NPZ).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8192,
        help="Number of points to sub-sample.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["plotly", "matplotlib", "open3d"],
        default="plotly",
        help="Visualization backend.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save visualization to this path instead of displaying.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly score threshold for binary classification.",
    )
    return parser.parse_args(argv)


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load a point cloud from file.

    Supported formats: ``.ply``, ``.pcd`` (via Open3D), ``.npz`` (with
    ``points`` key), and ``.npy``.

    Args:
        path: Path to the point cloud file.

    Returns:
        ``(N, 3)`` numpy array of point coordinates.

    Raises:
        ValueError: If the file format is unsupported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".ply", ".pcd"):
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float32)
    elif suffix == ".npz":
        data = np.load(str(path))
        return data["points"].astype(np.float32)
    elif suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the demo CLI.

    Args:
        argv: Optional CLI arguments.
    """
    from optiqual3d.utils.logging import setup_logging

    setup_logging()
    args = parse_args(argv)

    logger.info("Demo – loading checkpoint: %s", args.checkpoint)
    logger.info("Input: %s", args.input)

    # Load point cloud
    points = load_point_cloud(args.input)
    logger.info("Loaded point cloud: %s points", len(points))

    # Sub-sample if needed
    if len(points) > args.num_points:
        idx = np.random.choice(len(points), args.num_points, replace=False)
        points = points[idx]
    elif len(points) < args.num_points:
        logger.warning(
            "Point cloud has %d points, fewer than requested %d.",
            len(points),
            args.num_points,
        )

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    import torch

    from optiqual3d.config.settings import Config
    from optiqual3d.data.preprocessing import extract_patches, normalize_point_cloud
    from optiqual3d.models.optiqual import OptiQual3D
    from optiqual3d.utils.checkpoint import load_checkpoint

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cfg = Config()
    model = OptiQual3D(cfg.model, cfg.data.point_cloud).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    logger.info("Model loaded from %s", args.checkpoint)

    # ----------------------------------------------------------------
    # Normalise & extract patches
    # ----------------------------------------------------------------
    points_norm = normalize_point_cloud(points)
    pc_cfg = cfg.data.point_cloud
    patches_np, centroids_np = extract_patches(
        points_norm, pc_cfg.num_patches, pc_cfg.patch_size
    )

    patches_t = torch.from_numpy(patches_np).float().unsqueeze(0).to(device)
    centroids_t = torch.from_numpy(centroids_np).float().unsqueeze(0).to(device)
    points_t = torch.from_numpy(points_norm).float().unsqueeze(0).to(device)

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    with torch.no_grad():
        pred = model.predict(patches_t, centroids_t, points_t)

    scores = pred["point_scores"][0].cpu().numpy().astype(np.float32)

    global_score = float(scores.mean())
    is_anomalous = global_score > args.threshold
    logger.info(
        "Global anomaly score: %.4f (%s)",
        global_score,
        "ANOMALOUS" if is_anomalous else "NORMAL",
    )

    # Visualise
    from optiqual3d.config.settings import VisualizationConfig
    from optiqual3d.visualization.renderer import render_point_cloud

    vis_cfg = VisualizationConfig(backend=args.backend)
    render_point_cloud(
        points,
        scores=scores,
        title=f"Anomaly Score: {global_score:.4f}",
        cfg=vis_cfg,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
