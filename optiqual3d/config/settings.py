"""Structured configuration dataclasses for OptiQual3D.

All configuration is managed through dataclasses that can be composed,
serialised to YAML, and overridden from the CLI via OmegaConf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrainingPhase(str, Enum):
    """Phases of the OptiQual3D training pipeline."""

    PRETRAIN = "pretrain"
    ANOMALY = "anomaly"


class DatasetType(str, Enum):
    """Supported dataset types."""

    SHAPENET = "shapenet"
    MVTEC3D = "mvtec3d"
    REAL3D = "real3d"
    ANOMALY_SHAPENET = "anomaly_shapenet"


class AnomalyType(str, Enum):
    """Categories of synthetic anomalies."""

    DENT = "dent"
    BUMP = "bump"
    SCRATCH = "scratch"
    HOLE = "hole"
    MISSING_PART = "missing_part"
    EXTRA_MATERIAL = "extra_material"
    CRACK = "crack"
    POINT_DROPOUT = "point_dropout"
    GAUSSIAN_NOISE = "gaussian_noise"
    OUTLIER_POINTS = "outlier_points"


# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------


@dataclass
class PointCloudConfig:
    """Parameters for point cloud preprocessing.

    Attributes:
        num_points: Number of points to sample per cloud.
        patch_size: Number of points per local patch.
        num_patches: Number of patches to divide the cloud into.
        normalize: Whether to normalize to unit sphere.
    """

    num_points: int = 1024
    patch_size: int = 32
    num_patches: int = 64
    normalize: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation hyperparameters.

    Attributes:
        enable: Whether augmentation is active.
        rotation_range: Max rotation angle in degrees per axis.
        scale_min: Minimum random scale factor.
        scale_max: Maximum random scale factor.
        jitter_sigma: Std-dev of Gaussian positional jitter.
        jitter_clip: Max absolute jitter value.
        dropout_ratio: Fraction of points randomly dropped.
    """

    enable: bool = True
    rotation_range: float = 180.0
    scale_min: float = 0.8
    scale_max: float = 1.2
    jitter_sigma: float = 0.01
    jitter_clip: float = 0.05
    dropout_ratio: float = 0.1


@dataclass
class AnomalyGenerationConfig:
    """Configuration for synthetic anomaly generation.

    Attributes:
        anomaly_types: List of anomaly types to generate.
        severity_min: Minimum anomaly severity (0-1 scale).
        severity_max: Maximum anomaly severity (0-1 scale).
        num_anomalies_per_sample: Range of anomalies applied per sample.
        seed: Random seed for reproducibility.
    """

    anomaly_types: list[str] = field(
        default_factory=lambda: [t.value for t in AnomalyType]
    )
    severity_min: float = 0.1
    severity_max: float = 0.8
    num_anomalies_per_sample: tuple[int, int] = (1, 3)
    seed: int = 42


@dataclass
class DataConfig:
    """Top-level data configuration.

    Attributes:
        root_dir: Root directory containing all datasets.
        shapenet_dir: Path to ShapeNet dataset.
        mvtec3d_dir: Path to MVTec 3D-AD dataset.
        real3d_dir: Path to Real3D-AD dataset.
        anomaly_shapenet_dir: Path to Anomaly-ShapeNet dataset.
        generated_dir: Path to store generated anomalous samples.
        num_normal_samples: Number of normal training samples to generate.
        num_anomalous_samples: Number of anomalous training samples to generate.
        num_workers: DataLoader worker count.
        point_cloud: Point cloud preprocessing config.
        augmentation: Augmentation config.
        anomaly_generation: Anomaly generation config.
    """

    root_dir: str = "datasets"
    shapenet_dir: str = "datasets/shapenet"
    mvtec3d_dir: str = "datasets/mvtec_3d_anomaly_detection"
    real3d_dir: str = "datasets/Real3D-AD"
    anomaly_shapenet_dir: str = "datasets/anomaly_shapenet"
    generated_dir: str = "datasets/generated"
    num_normal_samples: int = 200_000
    num_anomalous_samples: int = 200_000
    num_workers: int = 8
    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    anomaly_generation: AnomalyGenerationConfig = field(
        default_factory=AnomalyGenerationConfig
    )


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """Point-MAE encoder hyperparameters.

    Attributes:
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim expansion ratio.
        mask_ratio: Fraction of patches masked during pre-training.
        drop_rate: Dropout probability.
        attn_drop_rate: Attention dropout probability.
        drop_path_rate: Stochastic depth rate.
    """

    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.6
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1


@dataclass
class DecoderConfig:
    """Dual-branch decoder hyperparameters.

    Attributes:
        embed_dim: Decoder embedding dimension.
        depth: Number of decoder transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
    """

    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3
    mlp_ratio: float = 4.0


@dataclass
class AnomalyHeadConfig:
    """Anomaly detection head configuration.

    Attributes:
        hidden_dims: Hidden layer sizes for the MLP head.
        dropout: Dropout rate between layers.
        use_multi_scale: Whether to aggregate multi-scale features.
        multi_scale_layers: 0-based encoder block indices to tap for
            multi-scale feature extraction.  Defaults to layers 3, 7, 11
            (early, middle, final) for a 12-layer encoder.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    use_multi_scale: bool = True
    multi_scale_layers: list[int] = field(default_factory=lambda: [3, 7, 11])


@dataclass
class ContrastiveConfig:
    """Contrastive normality learning configuration.

    Attributes:
        prototype_dim: Dimension of the normality prototype vector.
        temperature: InfoNCE temperature parameter.
        momentum: EMA momentum for prototype updates.
    """

    prototype_dim: int = 256
    temperature: float = 0.07
    momentum: float = 0.999


@dataclass
class ModelConfig:
    """Full model configuration.

    Attributes:
        encoder: Encoder configuration.
        decoder: Decoder configuration.
        anomaly_head: Anomaly head configuration.
        contrastive: Contrastive learning configuration.
        pretrained_weights: Optional path to pre-trained checkpoint.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    anomaly_head: AnomalyHeadConfig = field(default_factory=AnomalyHeadConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    pretrained_weights: str | None = None


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters.

    Attributes:
        name: Optimizer class name (adam, adamw, sgd).
        lr: Base learning rate.
        weight_decay: L2 regularization coefficient.
        betas: Adam beta parameters.
    """

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        name: Scheduler type (cosine, step, linear_warmup_cosine).
        warmup_epochs: Number of linear warm-up epochs.
        min_lr: Minimum learning rate for cosine decay.
    """

    name: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6


@dataclass
class LossConfig:
    """Loss function weights.

    Attributes:
        lambda_reconstruction: Weight for reconstruction loss.
        lambda_contrastive: Weight for contrastive loss.
        lambda_anomaly: Weight for anomaly detection loss.
    """

    lambda_reconstruction: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_anomaly: float = 1.0


@dataclass
class PretrainConfig:
    """Phase 1: Self-supervised pre-training configuration.

    Attributes:
        epochs: Number of pre-training epochs.
        batch_size: Samples per GPU.
        optimizer: Optimizer settings.
        scheduler: Scheduler settings.
        checkpoint_interval: Save checkpoint every N epochs.
        log_interval: Log metrics every N steps.
    """

    epochs: int = 300
    batch_size: int = 128
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-3)
    )
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint_interval: int = 25
    log_interval: int = 50


@dataclass
class AnomalyTrainConfig:
    """Phase 2: Anomaly detection training configuration.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Samples per GPU.
        optimizer: Optimizer settings.
        scheduler: Scheduler settings.
        loss: Loss function weights.
        freeze_encoder: Whether to freeze the encoder initially.
        unfreeze_epoch: Epoch at which to unfreeze the encoder.
        checkpoint_interval: Save checkpoint every N epochs.
        log_interval: Log metrics every N steps.
    """

    epochs: int = 150
    batch_size: int = 64
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=5e-4)
    )
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    freeze_encoder: bool = True
    unfreeze_epoch: int = 10
    checkpoint_interval: int = 10
    log_interval: int = 50


@dataclass
class TrainingConfig:
    """Top-level training configuration.

    Attributes:
        phase: Current training phase.
        pretrain: Phase 1 configuration.
        anomaly: Phase 2 configuration.
        seed: Global random seed.
        mixed_precision: Whether to use AMP (fp16/bf16).
        compile_model: Whether to use torch.compile.
        gradient_clip: Max gradient norm (0 = disabled).
    """

    phase: str = TrainingPhase.PRETRAIN.value
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    anomaly: AnomalyTrainConfig = field(default_factory=AnomalyTrainConfig)
    seed: int = 42
    mixed_precision: bool = True
    compile_model: bool = False
    gradient_clip: float = 1.0


# ---------------------------------------------------------------------------
# Distributed Configuration
# ---------------------------------------------------------------------------


@dataclass
class DistributedConfig:
    """Distributed training configuration.

    Attributes:
        backend: NCCL backend for GPU communication.
        num_gpus: Number of GPUs per node.
        num_nodes: Number of compute nodes.
        find_unused_parameters: DDP flag for unused param detection.
    """

    backend: str = "nccl"
    num_gpus: int = 4
    num_nodes: int = 1
    find_unused_parameters: bool = False


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration.

    Attributes:
        datasets: List of evaluation dataset names.
        batch_size: Evaluation batch size.
        anomaly_threshold: Score threshold for binary classification.
        au_pro_fpr_limit: FPR integration limit for AU-PRO metric.
        save_predictions: Whether to persist per-point predictions.
        output_dir: Directory to save evaluation results.
    """

    datasets: list[str] = field(
        default_factory=lambda: ["mvtec3d", "real3d", "anomaly_shapenet"]
    )
    batch_size: int = 64
    anomaly_threshold: float = 0.5
    au_pro_fpr_limit: float = 0.3
    save_predictions: bool = True
    output_dir: str = "results"


# ---------------------------------------------------------------------------
# Visualization Configuration
# ---------------------------------------------------------------------------


@dataclass
class VisualizationConfig:
    """Visualization settings.

    Attributes:
        backend: Rendering backend (plotly, open3d, matplotlib).
        colormap: Colormap for anomaly heatmaps.
        point_size: Rendered point size.
        save_dir: Directory to save visualizations.
    """

    backend: str = "plotly"
    colormap: str = "RdYlGn_r"
    point_size: float = 2.0
    save_dir: str = "visualizations"


# ---------------------------------------------------------------------------
# Logging / Experiment Tracking
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration.

    Attributes:
        log_dir: Directory for local logs and checkpoints.
        use_mlflow: Whether to log to MLflow.
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_experiment_name: MLflow experiment name.
        log_level: Python logging level.
    """

    log_dir: str = "logs"
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "optiqual3d"
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Root Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Root configuration for the entire OptiQual3D pipeline.

    Compose from YAML files, CLI overrides, and defaults using
    :func:`load_config`.

    Attributes:
        data: Data pipeline configuration.
        model: Model architecture configuration.
        training: Training configuration.
        distributed: Distributed training configuration.
        evaluation: Evaluation configuration.
        visualization: Visualization configuration.
        logging: Logging configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Configuration Loading Helpers
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load configuration from YAML + CLI overrides.

    Merges the structured defaults defined above with an optional YAML
    file and optional dotlist CLI overrides (e.g. ``training.pretrain.epochs=500``).

    Args:
        config_path: Path to a YAML configuration file.  If ``None``, only
            the structured defaults are used.
        overrides: List of dotlist overrides, e.g.
            ``["training.pretrain.epochs=500", "data.num_workers=4"]``.

    Returns:
        A fully-resolved :class:`Config` instance.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    schema: DictConfig = OmegaConf.structured(Config)

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        file_cfg: DictConfig = OmegaConf.load(str(path))  # type: ignore[assignment]
        schema = OmegaConf.merge(schema, file_cfg)  # type: ignore[assignment]

    if overrides:
        cli_cfg: DictConfig = OmegaConf.from_dotlist(overrides)
        schema = OmegaConf.merge(schema, cli_cfg)  # type: ignore[assignment]

    cfg: Config = OmegaConf.to_object(schema)  # type: ignore[assignment]
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Persist a :class:`Config` to a YAML file.

    Args:
        cfg: Configuration object to save.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    container: DictConfig = OmegaConf.structured(cfg)
    OmegaConf.save(container, str(path))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a :class:`Config` to a plain dictionary.

    Args:
        cfg: Configuration object.

    Returns:
        Dictionary representation suitable for JSON serialisation.
    """
    container: DictConfig = OmegaConf.structured(cfg)
    return OmegaConf.to_container(container, resolve=True)  # type: ignore[return-value]
