"""Launch Phase 1: Self-supervised pre-training (Point-MAE).

Trains the masked autoencoder on ShapeNet point clouds using the
Chamfer distance reconstruction loss.  Supports multi-GPU training
via ``torchrun``.

Usage::

    # Single GPU
    optiqual-pretrain --config configs/pretrain.yaml

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 -m scripts.pretrain \\
        --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Phase 1: Point-MAE pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pretrain",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style overrides, e.g. training.pretrain.epochs=200.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for pre-training.

    Args:
        argv: Optional CLI arguments.
    """
    from optiqual3d.config.settings import load_config
    from optiqual3d.training.distributed import (
        cleanup_distributed,
        setup_distributed,
    )
    from optiqual3d.utils.logging import setup_logging

    args = parse_args(argv)

    # Distributed setup (no-op when launched without torchrun)
    rank, world_size, _ = setup_distributed()
    setup_logging(rank=rank)

    logger.info("Phase 1 - Pre-training")
    logger.info("Config: %s | Resume: %s", args.config, args.resume)

    # Load and optionally override config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(str(config_path))
    else:
        logger.warning("Config file not found: %s – using defaults.", config_path)
        from optiqual3d.config.settings import Config

        cfg = Config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from optiqual3d.data.augmentation import build_augmentation
    from optiqual3d.data.datasets.shapenet import ShapeNetDataset
    from optiqual3d.data.preprocessing import extract_patches
    from optiqual3d.models.optiqual import OptiQual3D
    from optiqual3d.training.pretrain import PreTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Point checkpoints at the requested output dir
    cfg.logging.log_dir = str(output_dir)

    # ---- Dataset ------------------------------------------------
    augment = build_augmentation(cfg.data.augmentation)
    train_ds = ShapeNetDataset(
        root=cfg.data.shapenet_dir,
        split="train",
        point_cloud_cfg=cfg.data.point_cloud,
        transform=augment,
    )
    logger.info("Training dataset: %d ShapeNet samples", len(train_ds))

    pc_cfg = cfg.data.point_cloud

    def patch_collate(batch: list[dict]) -> dict:
        """Convert raw point tensors to patch/centroid tensors."""
        patches_list, centroids_list = [], []
        for sample in batch:
            pts = sample["points"].numpy()
            p, c = extract_patches(pts, pc_cfg.num_patches, pc_cfg.patch_size)
            patches_list.append(torch.from_numpy(p).float())
            centroids_list.append(torch.from_numpy(c).float())
        return {
            "patches": torch.stack(patches_list),    # (B, P, K, 3)
            "centroids": torch.stack(centroids_list),  # (B, P, 3)
        }

    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds,
        batch_size=cfg.training.pretrain.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=patch_collate,
        drop_last=True,
    )

    # ---- Model --------------------------------------------------
    model: torch.nn.Module = OptiQual3D(
        model_cfg=cfg.model, pc_cfg=cfg.data.point_cloud
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # ---- Resume -------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw = model.module if world_size > 1 else model
        raw.load_state_dict(ckpt["model_state_dict"])
        logger.info("Resumed from %s (epoch %s)", args.resume, ckpt.get("epoch", "?"))

    # ---- Train --------------------------------------------------
    trainer = PreTrainer(model=model, train_loader=loader, cfg=cfg, device=device)
    trainer.train()

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()
