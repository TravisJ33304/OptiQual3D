"""Logging and metric tracking utilities.

Provides :class:`MetricTracker` for accumulating per-epoch statistics
and optional MLflow integration for experiment tracking on Rosie HPC.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------


class MetricTracker:
    """Accumulate scalar metrics across steps / epochs.

    Maintains running sums and counts so that both per-step and
    per-epoch averages can be reported.

    Example::

        tracker = MetricTracker()
        for batch in loader:
            loss = train_step(batch)
            tracker.update({"loss": loss, "lr": scheduler.get_last_lr()[0]})
        epoch_avg = tracker.average()
        tracker.reset()

    Args:
        mlflow_enabled: Whether to log metrics to MLflow.
    """

    def __init__(self, mlflow_enabled: bool = False) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._history: dict[str, list[float]] = defaultdict(list)
        self._mlflow = mlflow_enabled
        self._step: int = 0

    # -- mutators --

    def update(
        self,
        metrics: dict[str, float],
        *,
        step: int | None = None,
        log_mlflow: bool = True,
    ) -> None:
        """Record a set of metric values.

        Args:
            metrics: Mapping of metric name to scalar value.
            step: Global step (for MLflow logging). Defaults to internal
                counter.
            log_mlflow: Whether to push these values to MLflow.
        """
        global_step = step if step is not None else self._step
        for name, value in metrics.items():
            self._sums[name] += value
            self._counts[name] += 1
            self._history[name].append(value)

        if self._mlflow and log_mlflow:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=global_step)
            except Exception:
                pass

        self._step += 1

    def reset(self) -> None:
        """Clear running sums and counts (but keep history)."""
        self._sums.clear()
        self._counts.clear()

    # -- queries --

    def average(self) -> dict[str, float]:
        """Return the running average for all tracked metrics.

        Returns:
            Dict mapping metric names to their averages.
        """
        return {
            name: self._sums[name] / max(self._counts[name], 1)
            for name in self._sums
        }

    def last(self, name: str) -> float:
        """Return the last recorded value for *name*.

        Args:
            name: Metric name.

        Returns:
            Most recent value.

        Raises:
            KeyError: If *name* has not been recorded.
        """
        if name not in self._history:
            raise KeyError(f"Metric '{name}' not recorded.")
        return self._history[name][-1]

    def history(self, name: str) -> list[float]:
        """Return the full history of values for *name*.

        Args:
            name: Metric name.

        Returns:
            List of all values recorded so far.
        """
        return list(self._history.get(name, []))

    @property
    def step(self) -> int:
        """The current internal step counter."""
        return self._step


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Simple context-manager timer.

    Example::

        with Timer("epoch") as t:
            train_one_epoch()
        print(t.elapsed)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            logger.info("[%s] elapsed %.2f s", self.label, self.elapsed)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    rank: int = 0,
) -> None:
    """Configure root logger for the project.

    Sets up console (``stdout``) and optional file handlers with a
    consistent format.  When running distributed, only rank 0 logs at
    INFO level; other ranks are set to WARNING.

    Args:
        level: Base logging level (default ``INFO``).
        log_file: Optional path for a file handler.
        rank: Current process rank for distributed training.
    """
    effective_level = level if rank == 0 else logging.WARNING

    fmt = "[%(asctime)s | %(levelname)-8s | %(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None and rank == 0:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(path)))

    logging.basicConfig(
        level=effective_level,
        format=fmt,
        datefmt=date_fmt,
        handlers=handlers,
        force=True,
    )
