"""Point cloud visualisation and anomaly heatmap rendering.

Provides functions to render 3D point clouds with overlaid anomaly
scores as colour-coded heatmaps, for both interactive viewing and
static image export.  Supports Plotly (interactive HTML), Matplotlib
(static images), and Open3D (interactive 3D viewer) backends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from optiqual3d.config.settings import VisualizationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend-agnostic API
# ---------------------------------------------------------------------------


def render_point_cloud(
    points: np.ndarray,
    scores: np.ndarray | None = None,
    title: str = "Point Cloud",
    cfg: VisualizationConfig | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Render a point cloud with optional anomaly heatmap.

    Args:
        points: ``(N, 3)`` point coordinates.
        scores: ``(N,)`` per-point anomaly scores in ``[0, 1]``.
            If ``None``, points are rendered in a uniform colour.
        title: Plot title.
        cfg: Visualization configuration.
        save_path: If provided, save the plot to this path.

    Returns:
        The figure object (type depends on backend).
    """
    cfg = cfg or VisualizationConfig()

    if cfg.backend == "plotly":
        return _render_plotly(points, scores, title, cfg, save_path)
    elif cfg.backend == "matplotlib":
        return _render_matplotlib(points, scores, title, cfg, save_path)
    elif cfg.backend == "open3d":
        return _render_open3d(points, scores, title, cfg)
    else:
        raise ValueError(f"Unknown visualization backend: {cfg.backend!r}")


def render_comparison(
    normal_points: np.ndarray,
    anomaly_points: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str = "Normal vs Anomalous",
    cfg: VisualizationConfig | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Render side-by-side comparison of normal and anomalous objects.

    Args:
        normal_points: ``(N, 3)`` normal point cloud.
        anomaly_points: ``(M, 3)`` anomalous point cloud.
        anomaly_scores: ``(M,)`` per-point anomaly scores.
        title: Plot title.
        cfg: Visualization configuration.
        save_path: If provided, save the plot.

    Returns:
        The figure object.
    """
    cfg = cfg or VisualizationConfig()

    if cfg.backend == "plotly":
        return _render_comparison_plotly(
            normal_points, anomaly_points, anomaly_scores, title, cfg, save_path
        )
    elif cfg.backend == "matplotlib":
        return _render_comparison_matplotlib(
            normal_points, anomaly_points, anomaly_scores, title, cfg, save_path
        )
    else:
        raise ValueError(
            f"Comparison rendering not supported for backend: {cfg.backend!r}"
        )


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------


def _render_plotly(
    points: np.ndarray,
    scores: np.ndarray | None,
    title: str,
    cfg: VisualizationConfig,
    save_path: str | Path | None,
) -> Any:
    """Render using Plotly (interactive HTML).

    Args:
        points: Point coordinates.
        scores: Optional anomaly scores.
        title: Plot title.
        cfg: Visualization config.
        save_path: File path to save HTML.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    if scores is not None:
        marker = dict(
            size=cfg.point_size,
            color=scores,
            colorscale=cfg.colormap,
            colorbar=dict(title="Anomaly Score"),
            cmin=0.0,
            cmax=1.0,
        )
    else:
        marker = dict(
            size=cfg.point_size,
            color="steelblue",
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        logger.info("Saved plotly visualization: %s", path)

    return fig


def _render_comparison_plotly(
    normal_points: np.ndarray,
    anomaly_points: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str,
    cfg: VisualizationConfig,
    save_path: str | Path | None,
) -> Any:
    """Render side-by-side comparison using Plotly subplots.

    Args:
        normal_points: Normal point cloud.
        anomaly_points: Anomalous point cloud.
        anomaly_scores: Anomaly scores for the anomalous cloud.
        title: Plot title.
        cfg: Visualization config.
        save_path: File path to save.

    Returns:
        Plotly Figure object.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["Normal", "Anomalous (Heatmap)"],
    )

    # Normal
    fig.add_trace(
        go.Scatter3d(
            x=normal_points[:, 0],
            y=normal_points[:, 1],
            z=normal_points[:, 2],
            mode="markers",
            marker=dict(size=cfg.point_size, color="steelblue"),
            name="Normal",
        ),
        row=1,
        col=1,
    )

    # Anomalous with heatmap
    fig.add_trace(
        go.Scatter3d(
            x=anomaly_points[:, 0],
            y=anomaly_points[:, 1],
            z=anomaly_points[:, 2],
            mode="markers",
            marker=dict(
                size=cfg.point_size,
                color=anomaly_scores,
                colorscale=cfg.colormap,
                colorbar=dict(title="Score", x=1.0),
                cmin=0.0,
                cmax=1.0,
            ),
            name="Anomalous",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=60))

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        logger.info("Saved comparison visualization: %s", path)

    return fig


# ---------------------------------------------------------------------------
# Matplotlib backend
# ---------------------------------------------------------------------------


def _render_matplotlib(
    points: np.ndarray,
    scores: np.ndarray | None,
    title: str,
    cfg: VisualizationConfig,
    save_path: str | Path | None,
) -> Any:
    """Render using Matplotlib (static image).

    Args:
        points: Point coordinates.
        scores: Optional anomaly scores.
        title: Plot title.
        cfg: Visualization config.
        save_path: File path to save image.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if scores is not None:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],  # type: ignore[arg-type]
            c=scores,
            cmap=cfg.colormap,
            s=cfg.point_size,  # type: ignore[arg-type]
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(scatter, ax=ax, label="Anomaly Score", shrink=0.6)
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],  # type: ignore[arg-type]
            c="steelblue",
            s=cfg.point_size,  # type: ignore[arg-type]
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved matplotlib visualization: %s", path)

    return fig


def _render_comparison_matplotlib(
    normal_points: np.ndarray,
    anomaly_points: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str,
    cfg: VisualizationConfig,
    save_path: str | Path | None,
) -> Any:
    """Render side-by-side comparison using Matplotlib.

    Args:
        normal_points: Normal point cloud.
        anomaly_points: Anomalous point cloud.
        anomaly_scores: Anomaly scores.
        title: Plot title.
        cfg: Visualization config.
        save_path: File path.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 8))

    # Normal
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        normal_points[:, 0],
        normal_points[:, 1],
        normal_points[:, 2],  # type: ignore[arg-type]
        c="steelblue",
        s=cfg.point_size,  # type: ignore[arg-type]
    )
    ax1.set_title("Normal")

    # Anomalous
    ax2 = fig.add_subplot(122, projection="3d")
    scatter = ax2.scatter(
        anomaly_points[:, 0],
        anomaly_points[:, 1],
        anomaly_points[:, 2],  # type: ignore[arg-type]
        c=anomaly_scores,
        cmap=cfg.colormap,
        s=cfg.point_size,  # type: ignore[arg-type]
        vmin=0.0,
        vmax=1.0,
    )
    ax2.set_title("Anomalous (Heatmap)")
    fig.colorbar(scatter, ax=ax2, label="Anomaly Score", shrink=0.6)

    fig.suptitle(title)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved comparison visualization: %s", path)

    return fig


# ---------------------------------------------------------------------------
# Open3D backend
# ---------------------------------------------------------------------------


def _render_open3d(
    points: np.ndarray,
    scores: np.ndarray | None,
    title: str,
    cfg: VisualizationConfig,
) -> Any:
    """Render using Open3D interactive viewer.

    Args:
        points: Point coordinates.
        scores: Optional anomaly scores.
        title: Window title.
        cfg: Visualization config.

    Returns:
        Open3D PointCloud object.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if scores is not None:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cfg.colormap)
        colours = cmap(scores)[:, :3]  # RGB, drop alpha
        pcd.colors = o3d.utility.Vector3dVector(colours)
    else:
        pcd.paint_uniform_color([0.27, 0.51, 0.71])  # steelblue

    o3d.visualization.draw_geometries([pcd], window_name=title)  # type: ignore[attr-defined]

    return pcd
