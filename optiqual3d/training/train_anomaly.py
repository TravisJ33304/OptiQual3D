"""Phase 2: Anomaly detection training.

Trains the full model (encoder + anomaly decoder + anomaly head +
contrastive module) on mixed normal/anomalous samples with
multi-task loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from optiqual3d.config.settings import AnomalyTrainConfig, Config
from optiqual3d.losses.combined import AnomalyDetectionLoss
from optiqual3d.models.optiqual import OptiQual3D
from optiqual3d.training.distributed import is_main_process
from optiqual3d.utils.checkpoint import save_checkpoint
from optiqual3d.utils.logging import MetricTracker

logger = logging.getLogger(__name__)


class AnomalyTrainer:
    """Trainer for Phase 2 anomaly detection.

    Handles multi-task training with reconstruction, contrastive, and
    anomaly detection losses.  Supports encoder freezing during warm-up.

    Attributes:
        model: OptiQual3D model instance.
        config: Training configuration.
        optimizer: Parameter optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Multi-task loss function.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None,
        cfg: Config,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the anomaly trainer.

        Args:
            model: The OptiQual3D model (or DDP-wrapped).
            train_loader: Training data loader (mixed normal + anomalous).
            val_loader: Optional validation data loader.
            cfg: Full configuration object.
            device: Target device.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.train_cfg: AnomalyTrainConfig = cfg.training.anomaly
        self.device = device or torch.device("cuda")

        # Optionally freeze encoder
        self._raw_model.freeze_encoder() if self.train_cfg.freeze_encoder else None

        # Build optimiser (only trainable params)
        self.optimizer: Optimizer = self._build_optimizer()
        self.scheduler: LRScheduler = self._build_scheduler()

        # AMP
        self.use_amp = cfg.training.mixed_precision
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Loss
        self.loss_fn = AnomalyDetectionLoss(self.train_cfg.loss)

        # Metrics
        self.train_metrics = MetricTracker(
            mlflow_enabled=(cfg.logging.use_mlflow and is_main_process()),
        )
        self.val_metrics = MetricTracker(
            mlflow_enabled=(cfg.logging.use_mlflow and is_main_process()),
        )

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    @property
    def _raw_model(self) -> OptiQual3D:
        """Unwrap DDP if necessary."""
        if hasattr(self.model, "module"):
            return self.model.module  # type: ignore[return-value]
        return self.model  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full anomaly detection training loop."""
        # Start MLflow run if enabled
        if self.cfg.logging.use_mlflow and is_main_process():
            import mlflow
            mlflow.set_tracking_uri(self.cfg.logging.mlflow_tracking_uri)
            mlflow.set_experiment(self.cfg.logging.mlflow_experiment_name)
            mlflow.start_run(run_name="anomaly_train")
            mlflow.log_params({
                "phase": "anomaly",
                "epochs": self.train_cfg.epochs,
                "batch_size": self.train_cfg.batch_size,
                "lr": self.train_cfg.optimizer.lr,
                "freeze_encoder": self.train_cfg.freeze_encoder,
                "unfreeze_epoch": self.train_cfg.unfreeze_epoch,
                "lambda_reconstruction": self.train_cfg.loss.lambda_reconstruction,
                "lambda_contrastive": self.train_cfg.loss.lambda_contrastive,
                "lambda_anomaly": self.train_cfg.loss.lambda_anomaly,
            })

        logger.info(
            "Starting anomaly training for %d epochs", self.train_cfg.epochs
        )

        for epoch in range(self.current_epoch, self.train_cfg.epochs):
            self.current_epoch = epoch

            # Unfreeze encoder at configured epoch
            if (
                self.train_cfg.freeze_encoder
                and epoch == self.train_cfg.unfreeze_epoch
            ):
                self._raw_model.unfreeze_encoder()
                logger.info("Unfreezing encoder at epoch %d", epoch)
                # Rebuild optimizer to include encoder params
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()

            # Update sampler
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch()

            # Validation
            val_metrics: dict[str, float] = {}
            if self.val_loader is not None:
                val_metrics = self.validate()

            # Logging
            if is_main_process():
                lr = self.scheduler.get_last_lr()[0]
                val_loss_str = (
                    f" | Val Loss: {val_metrics.get('loss', 0):.6f}"
                    if val_metrics
                    else ""
                )
                logger.info(
                    "Epoch %d/%d | Train Loss: %.6f%s | LR: %.2e",
                    epoch + 1,
                    self.train_cfg.epochs,
                    train_metrics["loss"],
                    val_loss_str,
                    lr,
                )

                # Best model tracking
                val_loss = val_metrics.get("loss", train_metrics["loss"])
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch + 1, tag="best")

                # Periodic checkpoint
                if (epoch + 1) % self.train_cfg.checkpoint_interval == 0:
                    self._save_checkpoint(epoch + 1)

            self.scheduler.step()

        # Final checkpoint
        if is_main_process():
            self._save_checkpoint(self.train_cfg.epochs, tag="final")

            if self.cfg.logging.use_mlflow:
                import mlflow
                mlflow.end_run()

    def train_epoch(self) -> dict[str, float]:
        """Train for a single epoch.

        Returns:
            Aggregated training metrics.
        """
        self.model.train()
        self.train_metrics.reset()

        pbar = (
            tqdm(self.train_loader, desc=f"Anomaly Epoch {self.current_epoch + 1}")
            if is_main_process()
            else self.train_loader
        )

        for batch in pbar:
            step_metrics = self.train_step(batch)
            self.train_metrics.update(step_metrics)
            self.global_step += 1

            if (
                is_main_process()
                and self.global_step % self.train_cfg.log_interval == 0
            ):
                avg = self.train_metrics.average()
                if isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        loss=f"{avg['loss']:.4f}",
                        anom=f"{avg.get('anomaly_loss', 0):.4f}",
                    )

        return self.train_metrics.average()

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary with:
                - ``"patches"``: ``(B, G, P, 3)`` point patches.
                - ``"centroids"``: ``(B, G, 3)`` centroids.
                - ``"label"``: ``(B,)`` sample-level binary labels.
                - ``"patch_mask"``: ``(B, G)`` per-patch ground truth.

        Returns:
            Per-step metric values.
        """
        patches = batch["patches"].to(self.device, non_blocking=True)
        centroids = batch["centroids"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        patch_labels = batch["patch_mask"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=self.use_amp):
            model_out = self.model(
                patches, centroids, mode="anomaly", labels=labels
            )
            loss_dict = self.loss_fn(model_out, patch_labels)
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
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Aggregated validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()

        for batch in self.val_loader:
            patches = batch["patches"].to(self.device, non_blocking=True)
            centroids = batch["centroids"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            patch_labels = batch["patch_mask"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                model_out = self.model(
                    patches, centroids, mode="anomaly", labels=labels
                )
                loss_dict = self.loss_fn(model_out, patch_labels)

            self.val_metrics.update({k: v.item() for k, v in loss_dict.items()})

        return self.val_metrics.average()

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> Optimizer:
        """Build optimizer for trainable parameters only."""
        cfg = self.train_cfg.optimizer
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(
            trainable,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )

    def _build_scheduler(self) -> LRScheduler:
        """Build learning rate scheduler."""
        cfg = self.train_cfg.scheduler
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_cfg.epochs - cfg.warmup_epochs,
            eta_min=cfg.min_lr,
        )

    def _save_checkpoint(
        self, epoch: int, tag: str | None = None
    ) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            tag: Optional tag (e.g. ``"best"``, ``"final"``).
        """
        name = tag or f"epoch_{epoch:04d}"
        path = Path(self.cfg.logging.log_dir) / "checkpoints" / f"anomaly_{name}.pt"

        save_checkpoint(
            path=path,
            model=self._raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            extra={"best_val_loss": self.best_val_loss, "global_step": self.global_step},
        )
        logger.info("Saved checkpoint: %s", path)
