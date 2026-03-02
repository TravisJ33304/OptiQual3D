"""InfoNCE contrastive loss for normality learning.

Provides a standalone loss function for contrastive learning that can
be used independently of the :class:`ContrastiveNormalityModule`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (Noise Contrastive Estimation) loss.

    Computes the NT-Xent / InfoNCE loss for contrastive learning.
    Positive pairs are formed between an anchor and a positive sample,
    while all other samples in the batch serve as negatives.

    Attributes:
        temperature: Softmax temperature scaling.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialise InfoNCE loss.

        Args:
            temperature: Temperature for scaling cosine similarities.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            anchors: ``(B, D)`` anchor feature vectors (L2-normalised).
            positives: ``(B, D)`` positive feature vectors.
            negatives: ``(K, D)`` optional explicit negatives. If ``None``,
                other samples in the batch serve as negatives.

        Returns:
            Scalar InfoNCE loss.
        """
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)

        # Positive similarities: (B,)
        pos_sim = torch.sum(anchors * positives, dim=-1) / self.temperature

        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            # (B, K)
            neg_sim = torch.mm(anchors, negatives.T) / self.temperature
        else:
            # All-pairs similarity matrix excluding diagonal (self)
            all_features = torch.cat([anchors, positives], dim=0)
            sim_matrix = torch.mm(anchors, all_features.T) / self.temperature
            # Mask out self-similarities
            b = anchors.shape[0]
            mask = torch.eye(b, 2 * b, device=anchors.device).bool()
            neg_sim = sim_matrix.masked_fill(mask, float("-inf"))

        # InfoNCE: -log(exp(pos) / (exp(pos) + Σ exp(neg)))
        logits = torch.cat(
            [pos_sim.unsqueeze(-1), neg_sim], dim=-1
        )
        labels = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)

        return F.cross_entropy(logits, labels)


class AnomalyContrastiveLoss(nn.Module):
    """Contrastive loss specialised for anomaly detection.

    Normal samples are pushed toward a prototype while anomalous
    samples are pushed away.  This is a simplified version of the
    loss computed inside :class:`ContrastiveNormalityModule`.

    Attributes:
        temperature: Contrastive temperature.
        margin: Minimum desired separation between normal and anomalous.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
    ) -> None:
        """Initialise anomaly contrastive loss.

        Args:
            temperature: Temperature scaling.
            margin: Contrastive margin.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the anomaly contrastive loss.

        Args:
            features: ``(B, D)`` L2-normalised feature vectors.
            labels: ``(B,)`` binary labels (0 = normal, 1 = anomalous).
            prototype: ``(D,)`` normality prototype vector.

        Returns:
            Scalar contrastive loss.
        """
        features = F.normalize(features, dim=-1)
        prototype = F.normalize(prototype, dim=0)

        similarity = torch.mm(features, prototype.unsqueeze(-1)).squeeze(-1)

        normal_mask = labels == 0
        anomaly_mask = labels == 1

        loss = torch.tensor(0.0, device=features.device)
        count = 0

        if normal_mask.any():
            # Normal: maximise similarity (minimise 1 - sim)
            normal_loss = (1.0 - similarity[normal_mask]).mean()
            loss = loss + normal_loss
            count += 1

        if anomaly_mask.any():
            # Anomalous: minimise similarity (push below margin)
            anomaly_sim = similarity[anomaly_mask]
            anomaly_loss = F.relu(anomaly_sim - self.margin + 1.0).mean()
            loss = loss + anomaly_loss
            count += 1

        return loss / max(count, 1)
