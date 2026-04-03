"""Checkpoint save / load utilities for distributed training.

Handles model, optimizer, scheduler, and epoch state persistence with
DDP-aware saving (only rank 0 writes) and loading (map to correct
device).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a copy of *state_dict* with optional DDP ``module.`` prefixes removed."""
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _filter_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Keep only checkpoint tensors that exist in model and have matching shape.

    Returns:
        Tuple of ``(filtered_state_dict, missing_in_model, shape_mismatch)``.
    """
    target = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    missing_in_model: list[str] = []
    shape_mismatch: list[str] = []

    for key, value in state_dict.items():
        if key not in target:
            missing_in_model.append(key)
            continue
        if target[key].shape != value.shape:
            shape_mismatch.append(key)
            continue
        filtered[key] = value

    return filtered, missing_in_model, shape_mismatch


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    path: str | Path,
    *,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
    is_distributed: bool = False,
) -> Path:
    """Save a training checkpoint.

    When running under DDP, the caller should ensure this is only
    called on the main process (rank 0).  The model state dict is
    unwrapped from :class:`~torch.nn.parallel.DistributedDataParallel`
    automatically.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Optional LR scheduler.
        epoch: Current epoch (0-indexed).
        path: Destination file path.
        metrics: Optional metric values to store alongside weights.
        extra: Arbitrary extra data to include.
        is_distributed: If ``True``, unwrap the DDP wrapper.

    Returns:
        The resolved path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP wrapper if present
    model_state = (
        model.module.state_dict()
        if is_distributed and hasattr(model, "module")
        else model.state_dict()
    )

    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        state["metrics"] = metrics

    if extra is not None:
        state.update(extra)

    torch.save(state, path)
    logger.info("Saved checkpoint to %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model whose weights will be loaded.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Device to map tensors to.
        strict: Whether to strictly enforce state-dict key matching.

    Returns:
        The full checkpoint dict (for accessing ``epoch``, ``metrics``,
        or any extra data stored at save time).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)

    # Handle DDP-wrapped models and architecture drift between checkpoints.
    state_dict = checkpoint["model_state_dict"]
    cleaned = _strip_module_prefix(state_dict)

    if strict:
        model.load_state_dict(cleaned, strict=True)
    else:
        filtered, missing_in_model, shape_mismatch = _filter_compatible_state_dict(
            model,
            cleaned,
        )
        load_result = model.load_state_dict(filtered, strict=False)

        if missing_in_model:
            logger.warning(
                "Checkpoint has %d key(s) not present in model; skipped.",
                len(missing_in_model),
            )
        if shape_mismatch:
            logger.warning(
                "Checkpoint has %d key(s) with shape mismatch; skipped.",
                len(shape_mismatch),
            )
        if load_result.missing_keys:
            logger.warning(
                "Model has %d missing key(s) after non-strict load.",
                len(load_result.missing_keys),
            )
        if load_result.unexpected_keys:
            logger.warning(
                "Checkpoint has %d unexpected key(s) after non-strict load.",
                len(load_result.unexpected_keys),
            )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    logger.info("Loaded checkpoint from %s (epoch %d)", path, epoch)
    return checkpoint


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """Find the most recent checkpoint in a directory.

    Searches for files matching ``*.pt`` or ``*.pth`` and returns the
    one with the highest modification time.

    Args:
        directory: Directory to search.

    Returns:
        Path to the latest checkpoint, or ``None`` if none found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    checkpoints = sorted(
        list(directory.glob("*.pt")) + list(directory.glob("*.pth")),
        key=lambda p: p.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None
