"""End-to-end smoke test: pretrain → train → evaluate on tiny data.

Verifies that the full pipeline executes without errors using a small
random dataset (no real data files required).  Intended for quick
iteration on a single T4 GPU before committing to a full DGX run.

Usage::

    python scripts/smoke_test.py              # default: 32 samples, 2 batches
    python scripts/smoke_test.py --samples 64 --batches 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s | %(levelname)-8s | %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ---------------------------------------------------------------------------
# Tiny synthetic dataset
# ---------------------------------------------------------------------------


class TinyPointCloudDataset(Dataset):
    """Synthetic dataset of random point clouds with anomaly masks."""

    def __init__(self, num_samples: int, num_points: int = 1024) -> None:
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        points = torch.randn(self.num_points, 3)
        label = idx % 2  # alternating normal/anomalous
        mask = torch.zeros(self.num_points)
        if label == 1:
            mask[: self.num_points // 4] = 1.0
        return {
            "points": points,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "category": "smoke",
        }


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def patch_collate(batch: list[dict], pc_cfg) -> dict:
    from optiqual3d.data.preprocessing import extract_patches

    patches_list, centroids_list = [], []
    for sample in batch:
        pts = sample["points"].numpy()
        p, c = extract_patches(pts, pc_cfg.num_patches, pc_cfg.patch_size)
        patches_list.append(torch.from_numpy(p).float())
        centroids_list.append(torch.from_numpy(c).float())
    return {
        "patches": torch.stack(patches_list),
        "centroids": torch.stack(centroids_list),
    }


def anomaly_collate(batch: list[dict], pc_cfg) -> dict:
    from optiqual3d.data.preprocessing import extract_patches

    patches_list, centroids_list, points_list = [], [], []
    labels, masks, patch_masks, categories = [], [], [], []
    for sample in batch:
        pts = sample["points"].numpy()
        p, c = extract_patches(pts, pc_cfg.num_patches, pc_cfg.patch_size)
        patches_list.append(torch.from_numpy(p).float())
        centroids_list.append(torch.from_numpy(c).float())
        points_list.append(sample["points"])
        labels.append(sample["label"])
        masks.append(sample["mask"])
        categories.append(sample["category"])
        # Per-patch label: 1 if any anomaly in the patch neighbourhood
        # Use the sample label as a proxy (coarse but sufficient for smoke test)
        pm = torch.full((pc_cfg.num_patches,), float(sample["label"].item()))
        patch_masks.append(pm)
    return {
        "patches": torch.stack(patches_list),
        "centroids": torch.stack(centroids_list),
        "points": torch.stack(points_list),
        "label": torch.stack(labels),
        "mask": torch.stack(masks),
        "patch_mask": torch.stack(patch_masks),
        "category": categories,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def run_pretrain(model, loader, cfg, device, max_batches: int) -> float:
    """Run a few pretrain batches; return final loss."""
    from optiqual3d.losses.combined import PretrainLoss

    model.train()
    loss_fn = PretrainLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        patches = batch["patches"].to(device)
        centroids = batch["centroids"].to(device)

        optimizer.zero_grad()
        out = model(patches, centroids, mode="pretrain")
        loss_dict = loss_fn(out)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        logger.info("  [pretrain] batch %d/%d  loss=%.4f", i + 1, max_batches, loss.item())

    return float(np.mean(losses))


def run_train(model, loader, cfg, device, max_batches: int) -> float:
    """Run a few anomaly-train batches; return final loss."""
    from optiqual3d.losses.combined import AnomalyDetectionLoss

    model.train()
    loss_fn = AnomalyDetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        patches = batch["patches"].to(device)
        centroids = batch["centroids"].to(device)
        points = batch["points"].to(device)
        labels = batch["label"].to(device)
        patch_labels = batch["patch_mask"].to(device)

        optimizer.zero_grad()
        out = model(patches, centroids, mode="anomaly", labels=labels)
        loss_dict = loss_fn(out, patch_labels)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        logger.info("  [train]    batch %d/%d  loss=%.4f", i + 1, max_batches, loss.item())

    return float(np.mean(losses))


def run_eval(model, loader, cfg, device, max_batches: int) -> dict:
    """Run a few eval batches via Evaluator; return metric dict."""
    from optiqual3d.evaluation.evaluator import Evaluator

    evaluator = Evaluator(model, cfg, device=device)
    model.eval()

    all_labels, all_scores = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            preds = evaluator._predict_batch(batch_gpu)
            all_labels.extend(preds["labels"])
            all_scores.extend(preds["global_scores"])
            logger.info(
                "  [eval]     batch %d/%d  mean_score=%.4f",
                i + 1, max_batches,
                float(np.mean(preds["global_scores"])),
            )

    return {"num_samples": len(all_labels), "mean_score": float(np.mean(all_scores))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OptiQual3D end-to-end smoke test.")
    p.add_argument("--samples", type=int, default=32,
                   help="Number of synthetic samples per phase.")
    p.add_argument("--batches", type=int, default=2,
                   help="Max batches to run per phase.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device (cuda or cpu).")
    p.add_argument("--output-dir", type=str, default="outputs/smoke_test",
                   help="Directory for any saved artefacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from optiqual3d.config.settings import Config
    from optiqual3d.models.optiqual import OptiQual3D

    cfg = Config()
    # Use a smaller patch config so it fits on a T4 with small batch
    cfg.data.point_cloud.num_points = 1024
    cfg.data.point_cloud.num_patches = 32
    cfg.data.point_cloud.patch_size = 32

    pc_cfg = cfg.data.point_cloud

    logger.info("=== OptiQual3D Smoke Test ===")
    logger.info("Device: %s | Samples: %d | Batches: %d | Batch size: %d",
                device, args.samples, args.batches, args.batch_size)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    results: dict[str, str] = {}
    ckpt_path = output_dir / "pretrain_smoke.pt"
    trained_path = output_dir / "train_smoke.pt"

    # ----------------------------------------------------------------
    # Phase 1: Pre-training
    # ----------------------------------------------------------------
    logger.info("\n--- Phase 1: Pre-training ---")
    t0 = time.time()
    try:
        ds = TinyPointCloudDataset(args.samples, pc_cfg.num_points)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            collate_fn=lambda b: patch_collate(b, pc_cfg),
        )
        model = OptiQual3D(cfg.model, cfg.data.point_cloud).to(device)
        loss = run_pretrain(model, loader, cfg, device, args.batches)
        logger.info("  → loss=%.4f  (%.1fs)", loss, time.time() - t0)

        torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, ckpt_path)
        results["pretrain"] = PASS
    except Exception as e:
        logger.exception("Pre-training failed: %s", e)
        results["pretrain"] = FAIL

    # ----------------------------------------------------------------
    # Phase 2: Anomaly training
    # ----------------------------------------------------------------
    logger.info("\n--- Phase 2: Anomaly Training ---")
    t0 = time.time()
    try:
        ds = TinyPointCloudDataset(args.samples, pc_cfg.num_points)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            collate_fn=lambda b: anomaly_collate(b, pc_cfg),
        )
        model2 = OptiQual3D(cfg.model, cfg.data.point_cloud).to(device)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model2.load_state_dict(ckpt["model_state_dict"])
        else:
            logger.warning("  pretrain checkpoint not found – using random weights")
        loss = run_train(model2, loader, cfg, device, args.batches)
        logger.info("  → loss=%.4f  (%.1fs)", loss, time.time() - t0)

        torch.save({"model_state_dict": model2.state_dict(), "epoch": 0}, trained_path)
        results["train"] = PASS
    except Exception as e:
        logger.exception("Anomaly training failed: %s", e)
        results["train"] = FAIL

    # ----------------------------------------------------------------
    # Phase 3: Evaluation
    # ----------------------------------------------------------------
    logger.info("\n--- Phase 3: Evaluation ---")
    t0 = time.time()
    try:
        ds = TinyPointCloudDataset(args.samples, pc_cfg.num_points)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: anomaly_collate(b, pc_cfg),
        )
        model3 = OptiQual3D(cfg.model, cfg.data.point_cloud).to(device)
        if trained_path.exists():
            ckpt = torch.load(trained_path, map_location=device, weights_only=True)
            model3.load_state_dict(ckpt["model_state_dict"])
        else:
            logger.warning("  train checkpoint not found – using random weights")
        metrics = run_eval(model3, loader, cfg, device, args.batches)
        logger.info("  → samples=%d  mean_score=%.4f  (%.1fs)",
                    metrics["num_samples"], metrics["mean_score"], time.time() - t0)
        results["evaluate"] = PASS
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        results["evaluate"] = FAIL

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    logger.info("\n=== Results ===")
    all_passed = True
    for phase, status in results.items():
        logger.info("  %-12s %s", phase, status)
        if "FAIL" in status:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
