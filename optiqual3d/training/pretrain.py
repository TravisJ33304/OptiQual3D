"""Phase 1: Self-supervised pre-training trainer.

Trains the Point-MAE encoder via masked point cloud reconstruction
on clean ShapeNet samples.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from optiqual3d.config.settings import Config, PretrainConfig
from optiqual3d.losses.combined import PretrainLoss
from optiqual3d.training.distributed import is_main_process
from optiqual3d.utils.checkpoint import save_checkpoint
from optiqual3d.utils.logging import MetricTracker

logger = logging.getLogger(__name__)


class PreTrainer:
    """Trainer for Phase 1 self-supervised pre-training.

    Handles the training loop, gradient scaling, checkpointing,
    and metric logging for masked point cloud reconstruction.

    Attributes:
        model: OptiQual3D model instance.
        config: Pre-training configuration.
        optimizer: Parameter optimizer.
        scheduler: Learning rate scheduler.
        scaler: AMP gradient scaler.
        loss_fn: Pre-training loss function.
        metric_tracker: Metric logging utility.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        cfg: Config,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the pre-trainer.

        Args:
            model: The OptiQual3D model (or DDP-wrapped).
            train_loader: Training data loader supplying clean samples.
            cfg: Full configuration object.
            device: Target device (defaults to ``cuda``).
        """
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg
        self.pretrain_cfg: PretrainConfig = cfg.training.pretrain
        self.device = device or torch.device("cuda")

        # Build optimiser
        self.optimizer: Optimizer = self._build_optimizer()
        self.scheduler: LRScheduler = self._build_scheduler()

        # AMP
        self.use_amp = cfg.training.mixed_precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # Loss
        self.loss_fn = PretrainLoss()

        # Metrics
        self.metric_tracker = MetricTracker(
            mlflow_enabled=(cfg.logging.use_mlflow and is_main_process()),
        )

        # State
        self.current_epoch = 0
        self.global_step = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full pre-training loop.

        Iterates over epochs, calling :meth:`train_epoch` for each.
        Saves checkpoints at configured intervals.
        """
        # Start MLflow run if enabled
        if self.cfg.logging.use_mlflow and is_main_process():
            import mlflow
            mlflow.set_tracking_uri(self.cfg.logging.mlflow_tracking_uri)
            mlflow.set_experiment(self.cfg.logging.mlflow_experiment_name)
            mlflow.start_run(run_name="pretrain")
            mlflow.log_params({
                "phase": "pretrain",
                "epochs": self.pretrain_cfg.epochs,
                "batch_size": self.pretrain_cfg.batch_size,
                "lr": self.pretrain_cfg.optimizer.lr,
                "weight_decay": self.pretrain_cfg.optimizer.weight_decay,
                "mask_ratio": self.cfg.model.encoder.mask_ratio,
                "embed_dim": self.cfg.model.encoder.embed_dim,
                "depth": self.cfg.model.encoder.depth,
            })

        logger.info(
            "Starting pre-training for %d epochs", self.pretrain_cfg.epochs
        )

        for epoch in range(self.current_epoch, self.pretrain_cfg.epochs):
            self.current_epoch = epoch

            # Update sampler epoch for distributed shuffling
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_metrics = self.train_epoch()

            # Logging
            if is_main_process():
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d/%d | Loss: %.6f | LR: %.2e",
                    epoch + 1,
                    self.pretrain_cfg.epochs,
                    epoch_metrics["loss"],
                    lr,
                )

                # Always save latest so we can resume after preemption
                self._save_checkpoint(epoch + 1, latest=True)

                # Interval checkpoint
                if (epoch + 1) % self.pretrain_cfg.checkpoint_interval == 0:
                    self._save_checkpoint(epoch + 1)

            self.scheduler.step()

        # Final checkpoint
        if is_main_process():
            self._save_checkpoint(self.pretrain_cfg.epochs, final=True)

            if self.cfg.logging.use_mlflow:
                import mlflow
                mlflow.end_run()

    def train_epoch(self) -> dict[str, float]:
        """Train for a single epoch.

        Returns:
            Dictionary of aggregated epoch metrics.
        """
        self.model.train()
        self.metric_tracker.reset()

        pbar = (
            tqdm(self.train_loader, desc=f"Pretrain Epoch {self.current_epoch + 1}")
            if is_main_process()
            else self.train_loader
        )

        for batch in pbar:
            step_metrics = self.train_step(batch)
            self.metric_tracker.update(step_metrics)
            self.global_step += 1

            if (
                is_main_process()
                and self.global_step % self.pretrain_cfg.log_interval == 0
            ):
                avg = self.metric_tracker.average()
                if isinstance(pbar, tqdm):
                    pbar.set_postfix(loss=f"{avg['loss']:.6f}")

        return self.metric_tracker.average()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary with ``"patches"`` and ``"centroids"``
                tensors.

        Returns:
            Dictionary of per-step metrics.
        """
        patches = batch["patches"].to(self.device, non_blocking=True)
        centroids = batch["centroids"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            model_out = self.model(patches, centroids, mode="pretrain")
            loss_dict = self.loss_fn(model_out)
            loss = loss_dict["loss"]

        self.scaler.scale(loss).backward()

        if self.cfg.training.gradient_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.gradient_clip,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> Optimizer:
        """Build the optimizer from configuration.

        Returns:
            Configured optimizer instance.
        """
        cfg = self.pretrain_cfg.optimizer
        return AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )

    def _build_scheduler(self) -> LRScheduler:
        """Build the learning rate scheduler.

        Returns:
            Configured scheduler instance.
        """
        cfg = self.pretrain_cfg.scheduler
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.pretrain_cfg.epochs - cfg.warmup_epochs,
            eta_min=cfg.min_lr,
        )

    def _save_checkpoint(self, epoch: int, final: bool = False, latest: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            final: Whether this is the final checkpoint.
            latest: Whether to overwrite the rolling ``latest.pt`` checkpoint.
        """
        tag = "latest" if latest else ("final" if final else f"epoch_{epoch:04d}")
        path = Path(self.cfg.logging.log_dir) / "checkpoints" / f"pretrain_{tag}.pt"

        raw_model = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        save_checkpoint(
            model=raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            path=path,
            extra={
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
            },
        )
        logger.info("Saved checkpoint: %s", path)
