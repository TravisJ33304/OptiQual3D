"""Run zero-shot evaluation on MVTec 3D-AD and Real3D-AD benchmarks.

Loads a trained OptiQual3D model and evaluates it on unseen categories
from the benchmark datasets, reporting AUROC, F1, and AU-PRO metrics.

Usage::

    optiqual-evaluate --config configs/evaluate.yaml \\
        --checkpoint outputs/train/best.pt \\
        --dataset mvtec3d \\
        --output-dir outputs/eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate OptiQual3D on anomaly detection benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluate.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mvtec3d", "real3d", "anomaly_shapenet", "both", "all"],
        default="both",
        help="Which benchmark dataset(s) to evaluate.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Specific categories to evaluate (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="Directory for evaluation results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style overrides.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for evaluation.

    Args:
        argv: Optional CLI arguments.
    """
    from optiqual3d.config.settings import load_config
    from optiqual3d.utils.logging import setup_logging

    setup_logging()
    args = parse_args(argv)

    logger.info("Evaluation")
    logger.info(
        "Checkpoint: %s | Dataset: %s | Categories: %s",
        args.checkpoint,
        args.dataset,
        args.categories or "all",
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
    from torch.utils.data import DataLoader

    from optiqual3d.data.datasets.mvtec3d import MVTec3DDataset
    from optiqual3d.data.datasets.real3d import Real3DDataset
    from optiqual3d.data.datasets.anomaly_shapenet import AnomalyShapeNetDataset
    from optiqual3d.evaluation.evaluator import Evaluator
    from optiqual3d.models.optiqual import OptiQual3D
    from optiqual3d.utils.checkpoint import load_checkpoint

    # ----------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Override evaluation output dir from CLI
    cfg.evaluation.output_dir = str(output_dir)

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    model = OptiQual3D(cfg.model, cfg.data.point_cloud).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    logger.info("Model loaded from %s", args.checkpoint)

    # ----------------------------------------------------------------
    # Instantiate evaluator
    # ----------------------------------------------------------------
    evaluator = Evaluator(model, cfg, device=device)

    # ----------------------------------------------------------------
    # Build dataset list: (dataset_tag, category, Dataset)
    # ----------------------------------------------------------------
    datasets_to_eval: list[tuple[str, str, Any]] = []

    if args.dataset in ("mvtec3d", "both", "all"):
        cats = args.categories or MVTec3DDataset.CATEGORIES
        for cat in cats:
            ds = MVTec3DDataset(
                root=cfg.data.mvtec3d_dir,
                category=cat,
                split="test",
                point_cloud_cfg=cfg.data.point_cloud,
            )
            datasets_to_eval.append(("mvtec3d", cat, ds))

    if args.dataset in ("real3d", "both", "all"):
        cats = args.categories or Real3DDataset.CATEGORIES
        for cat in cats:
            ds = Real3DDataset(
                root=cfg.data.real3d_dir,
                category=cat,
                split="test",
                point_cloud_cfg=cfg.data.point_cloud,
            )
            datasets_to_eval.append(("real3d", cat, ds))

    if args.dataset in ("anomaly_shapenet", "all"):
        ds = AnomalyShapeNetDataset(
            root=cfg.data.anomaly_shapenet_dir,
            categories=args.categories,
            split="test",
            point_cloud_cfg=cfg.data.point_cloud,
        )
        datasets_to_eval.append(("anomaly_shapenet", "all", ds))

    if not datasets_to_eval:
        logger.error("No datasets selected for evaluation.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Run evaluation
    # ----------------------------------------------------------------
    all_results: dict[str, Any] = {}

    for ds_tag, cat, dataset in datasets_to_eval:
        if len(dataset) == 0:
            logger.warning("Empty dataset for %s/%s – skipping.", ds_tag, cat)
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )

        run_name = f"{ds_tag}/{cat}"
        results = evaluator.evaluate_dataset(dataloader, run_name)

        # Accumulate for combined summary
        for key, metrics in results.items():
            all_results[f"{ds_tag}/{key}"] = metrics

        # Save per-category JSON
        evaluator.save_results(results, f"{ds_tag}_{cat}")

    # ----------------------------------------------------------------
    # Save combined summary JSON
    # ----------------------------------------------------------------
    summary: dict[str, Any] = {
        run_key: {
            "detection": {
                "auroc": m.detection.auroc,
                "f1": m.detection.f1,
                "precision": m.detection.precision,
                "recall": m.detection.recall,
                "threshold": m.detection.threshold,
            },
            "localisation": {
                "au_pro": m.localisation.au_pro,
                "point_auroc": m.localisation.point_auroc,
                "fpr_limit": m.localisation.fpr_limit,
            },
            "num_samples": m.num_samples,
        }
        for run_key, m in all_results.items()
    }

    if summary:
        import numpy as np

        avg_auroc = float(np.mean([v["detection"]["auroc"] for v in summary.values()]))
        avg_au_pro = float(
            np.mean([v["localisation"]["au_pro"] for v in summary.values()])
        )
        summary["_overall"] = {"avg_i_auroc": avg_auroc, "avg_au_pro": avg_au_pro}
        logger.info(
            "Overall – I-AUROC: %.2f%%  AU-PRO: %.2f%%",
            avg_auroc * 100,
            avg_au_pro * 100,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
