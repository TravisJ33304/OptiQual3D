"""Launch Phase 2: Anomaly detection fine-tuning.

Trains the anomaly detection head and contrastive normality module
on top of the pre-trained encoder using generated anomalous data.
Supports multi-GPU training via ``torchrun``.

Usage::

    # Single GPU
    optiqual-train --config configs/train.yaml \\
        --pretrained outputs/pretrain/best.pt

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 -m scripts.train \\
        --config configs/train.yaml \\
        --pretrained outputs/pretrain/best.pt
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

    rank, world_size, _ = setup_distributed()
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
    from optiqual3d.models.optiqual import OptiQual3D
    from optiqual3d.training.train_anomaly import AnomalyTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds,
        batch_size=cfg.training.anomaly.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Model --------------------------------------------------
    model: torch.nn.Module = OptiQual3D(
        model_cfg=cfg.model, pc_cfg=cfg.data.point_cloud
    ).to(device)

    # Load pre-trained Phase 1 weights
    ckpt = torch.load(args.pretrained, map_location=device)
    raw_model = model
    if "model_state_dict" in ckpt:
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("Loaded pre-trained weights from %s", args.pretrained)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # ---- Resume Phase 2 -----------------------------------------
    if args.resume:
        ckpt2 = torch.load(args.resume, map_location=device)
        (model.module if world_size > 1 else model).load_state_dict(
            ckpt2["model_state_dict"]
        )
        logger.info("Resumed Phase 2 from %s", args.resume)

    # ---- Train --------------------------------------------------
    trainer = AnomalyTrainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        cfg=cfg,
        device=device,
    )
    trainer.train()

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()
