"""Launch Phase 2: Anomaly detection fine-tuning.

Trains the anomaly detection head and contrastive normality module
on top of the pre-trained encoder using generated anomalous data.
Supports multi-GPU training via ``torchrun``.

Usage::

    # Single GPU
    optiqual-train --config configs/train.yaml \\
        --pretrained outputs/pretrain/checkpoints/pretrain_latest.pt

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 -m scripts.train \\
        --config configs/train.yaml \\
        --pretrained outputs/pretrain/checkpoints/pretrain_latest.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
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
        description="Phase 2: Anomaly detection training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pre-trained Phase 1 checkpoint.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Phase 2 checkpoint to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style overrides.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for anomaly detection training.

    Args:
        argv: Optional CLI arguments.
    """
    from optiqual3d.config.settings import load_config
    from optiqual3d.training.distributed import (
        cleanup_distributed,
        is_main_process,
        setup_distributed,
    )
    from optiqual3d.utils.logging import setup_logging

    args = parse_args(argv)

    rank, local_rank, world_size = setup_distributed()
    setup_logging(rank=rank)

    logger.info("Phase 2 – Anomaly Detection Training")
    logger.info(
        "Config: %s | Pretrained: %s | Resume: %s",
        args.config,
        args.pretrained,
        args.resume,
    )

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
    from optiqual3d.data.datasets.generated import GeneratedAnomalyDataset
    from optiqual3d.data.datasets.shapenet import ShapeNetDataset
    from optiqual3d.data.preprocessing import extract_patches
    from optiqual3d.models.optiqual import OptiQual3D
    from optiqual3d.training.train_anomaly import AnomalyTrainer
    from optiqual3d.utils.checkpoint import load_checkpoint

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg.logging.log_dir = str(output_dir)

    # ---- Dataset (online anomaly generation on top of ShapeNet) ----
    augment = build_augmentation(cfg.data.augmentation)
    base_ds = ShapeNetDataset(
        root=cfg.data.shapenet_dir,
        split="train",
        point_cloud_cfg=cfg.data.point_cloud,
        transform=augment,
    )
    train_ds = GeneratedAnomalyDataset(
        base_dataset=base_ds,
        anomaly_cfg=cfg.data.anomaly_generation,
        point_cloud_cfg=cfg.data.point_cloud,
        anomaly_ratio=0.5,
    )
    logger.info("Training dataset: %d samples (online anomaly mode)", len(train_ds))

    pc_cfg = cfg.data.point_cloud

    def patch_collate(batch: list[dict]) -> dict:
        """Convert raw point/mask tensors to patch/centroid/patch_mask tensors."""
        import numpy as np

        patches_list, centroids_list, labels_list, patch_masks_list = [], [], [], []
        for sample in batch:
            pts = sample["points"].numpy()
            mask = sample["mask"].numpy()  # (N,) per-point anomaly mask
            label = sample["label"]

            # Extract patches and centroids via FPS + KNN
            p, c = extract_patches(pts, pc_cfg.num_patches, pc_cfg.patch_size)

            # Build per-patch labels by majority vote of per-point masks.
            # Replicate the KNN assignment from extract_patches.
            patch_mask = np.zeros(pc_cfg.num_patches, dtype=np.float32)
            for i, centroid in enumerate(c):
                dists = np.linalg.norm(pts - centroid, axis=1)
                nearest = np.argsort(dists)[:pc_cfg.patch_size]
                patch_mask[i] = float(mask[nearest].mean() >= 0.5)

            patches_list.append(torch.from_numpy(p).float())
            centroids_list.append(torch.from_numpy(c).float())
            labels_list.append(label)
            patch_masks_list.append(torch.from_numpy(patch_mask).float())

        return {
            "patches": torch.stack(patches_list),       # (B, G, P, 3)
            "centroids": torch.stack(centroids_list),    # (B, G, 3)
            "label": torch.tensor(labels_list, dtype=torch.long),  # (B,)
            "patch_mask": torch.stack(patch_masks_list), # (B, G)
        }

    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds,
        batch_size=cfg.training.anomaly.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=patch_collate,
    )

    # ---- Model --------------------------------------------------
    model: torch.nn.Module = OptiQual3D(
        model_cfg=cfg.model, pc_cfg=cfg.data.point_cloud
    ).to(device)

    # Load pre-trained Phase 1 weights
    load_checkpoint(args.pretrained, model, device=device, strict=False)
    logger.info("Loaded pre-trained weights from %s (non-strict)", args.pretrained)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ---- Train --------------------------------------------------
    trainer = AnomalyTrainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        cfg=cfg,
        device=device,
    )

    # ---- Resume Phase 2 -----------------------------------------
    if args.resume:
        raw_model = model.module if world_size > 1 else model
        ckpt2 = load_checkpoint(
            args.resume,
            raw_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=device,
            strict=True,
        )
        trainer.current_epoch = int(ckpt2.get("epoch", 0))
        trainer.global_step = int(ckpt2.get("global_step", 0))
        trainer.best_val_loss = float(ckpt2.get("best_val_loss", float("inf")))
        logger.info(
            "Resumed Phase 2 from %s (epoch %d, step %d)",
            args.resume,
            trainer.current_epoch,
            trainer.global_step,
        )

    trainer.train()

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()
