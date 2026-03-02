"""Distributed training utilities for multi-GPU / multi-node setups.

Provides helpers for initialising PyTorch DDP, wrapping models, and
building distributed data loaders compatible with SLURM on Rosie.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from optiqual3d.config.settings import DistributedConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def setup_distributed(cfg: DistributedConfig | None = None) -> tuple[int, int, int]:
    """Initialise the distributed process group.

    Supports both ``torchrun`` and SLURM-based launching.  Environment
    variables ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` are read
    from the launcher or SLURM equivalents.

    Args:
        cfg: Distributed configuration.

    Returns:
        Tuple of ``(rank, local_rank, world_size)``.
    """
    cfg = cfg or DistributedConfig()

    # SLURM environment variables
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("SLURM_LOCALID", "0"),
        )
    )
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", str(cfg.num_gpus * cfg.num_nodes)),
        )
    )

    if not dist.is_initialized():
        dist.init_process_group(
            backend=cfg.backend,
            rank=rank,
            world_size=world_size,
        )

    torch.cuda.set_device(local_rank)

    logger.info(
        "Distributed: rank=%d, local_rank=%d, world_size=%d",
        rank,
        local_rank,
        world_size,
    )

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Tear down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if the current process is the main (rank 0) process.

    Returns:
        ``True`` if rank is 0 or distributed is not initialised.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------


def wrap_model_ddp(
    model: nn.Module,
    local_rank: int,
    cfg: DistributedConfig | None = None,
) -> DDP:
    """Wrap a model in DistributedDataParallel.

    Args:
        model: The model to wrap.
        local_rank: Local GPU rank.
        cfg: Distributed configuration.

    Returns:
        DDP-wrapped model.
    """
    cfg = cfg or DistributedConfig()

    model = model.cuda(local_rank)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.find_unused_parameters,
    )

    return ddp_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_distributed_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader[Any], DistributedSampler[Any]]:
    """Build a DataLoader with a DistributedSampler.

    Args:
        dataset: The dataset to load.
        batch_size: Per-GPU batch size.
        num_workers: Number of data loading workers per process.
        shuffle: Whether to shuffle (via the sampler).
        drop_last: Whether to drop the last incomplete batch.
        pin_memory: Whether to pin host memory.

    Returns:
        Tuple of ``(dataloader, sampler)``.  The sampler must have
        ``set_epoch()`` called each epoch for proper shuffling.
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return loader, sampler


# ---------------------------------------------------------------------------
# Gradient synchronisation helpers
# ---------------------------------------------------------------------------


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor across processes (mean).

    Args:
        tensor: Local tensor to reduce.

    Returns:
        Reduced tensor (averaged across world).
    """
    if not dist.is_initialized():
        return tensor

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """All-gather tensors from all processes.

    Args:
        tensor: Local tensor to gather.

    Returns:
        List of tensors, one from each process.
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered
