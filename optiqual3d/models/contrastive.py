"""Contrastive normality learning module.

Maintains a learned normality prototype in latent space and applies
contrastive learning to push normal sample representations toward
the prototype while pushing anomalous representations away.

Uses InfoNCE loss adapted for the anomaly detection setting where
we have explicit normal / anomalous labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from optiqual3d.config.settings import ContrastiveConfig


class NormalityPrototype(nn.Module):
    """Learnable normality prototype with EMA updates.

    The prototype is a single vector in the latent space representing
    the "centre" of normality.  During training it is updated via
    exponential moving average of the mean normal-sample representation.

    Attributes:
        prototype: ``(D,)`` normality prototype vector.
        config: Contrastive configuration.
    """

    def __init__(
        self,
        feature_dim: int,
        config: ContrastiveConfig | None = None,
    ) -> None:
        """Initialise the normality prototype.

        Args:
            feature_dim: Dimension of the feature space.
            config: Contrastive learning configuration.
        """
        super().__init__()
        self.config = config or ContrastiveConfig()

        # Projection from feature space to prototype space
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, self.config.prototype_dim),
            nn.GELU(),
            nn.Linear(self.config.prototype_dim, self.config.prototype_dim),
        )

        # Prototype vector (not a learnable parameter — updated via EMA)
        self.register_buffer(
            "prototype",
            F.normalize(torch.randn(self.config.prototype_dim), dim=0),
        )
        self.register_buffer("initialised", torch.tensor(False))

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Project features into the prototype space.

        Args:
            features: ``(B, D)`` feature vectors (global representations).

        Returns:
            ``(B, D_proto)`` L2-normalised projected features.
        """
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)

    @torch.no_grad()
    def update(self, normal_features: torch.Tensor) -> None:
        """Update the prototype via EMA on normal sample features.

        Should be called during training with a batch of features
        from normal (non-anomalous) samples only.

        Args:
            normal_features: ``(B_normal, D)`` features from normal samples.
        """
        projected = self.project(normal_features)
        batch_mean = F.normalize(projected.mean(dim=0), dim=0)

        if not self.initialised:
            self.prototype.data.copy_(batch_mean)
            self.initialised.data.fill_(True)
        else:
            m = self.config.momentum
            self.prototype.data.copy_(
                F.normalize(m * self.prototype + (1 - m) * batch_mean, dim=0)
            )

    def similarity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between features and the prototype.

        Args:
            features: ``(B, D)`` feature vectors.

        Returns:
            ``(B,)`` similarity scores in ``[-1, 1]``.
        """
        projected = self.project(features)
        return torch.sum(projected * self.prototype.detach().unsqueeze(0), dim=-1)


class ContrastiveNormalityModule(nn.Module):
    """Full contrastive normality learning module.

    Combines the normality prototype with InfoNCE-style contrastive
    loss computation.

    Attributes:
        prototype: Normality prototype with projection head.
        temperature: Contrastive temperature parameter.
    """

    def __init__(
        self,
        feature_dim: int,
        config: ContrastiveConfig | None = None,
    ) -> None:
        """Initialise the contrastive module.

        Args:
            feature_dim: Encoder/decoder feature dimension.
            config: Contrastive learning configuration.
        """
        super().__init__()
        self.config = config or ContrastiveConfig()
        self.prototype = NormalityPrototype(feature_dim, self.config)
        self.temperature = self.config.temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute contrastive loss and update prototype.

        Args:
            features: ``(B, D)`` global feature representations.
            labels: ``(B,)`` binary labels (0 = normal, 1 = anomalous).

        Returns:
            Dictionary with:
                - ``"loss"``: Contrastive loss scalar.
                - ``"similarity"``: ``(B,)`` similarity to prototype.
                - ``"normal_sim"``: Mean similarity for normal samples.
                - ``"anomaly_sim"``: Mean similarity for anomalous samples.
        """
        similarity = self.prototype.similarity(features)

        # InfoNCE-style loss:
        # Normal samples should be close to prototype (high similarity)
        # Anomalous samples should be far (low similarity)
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        loss = torch.tensor(0.0, device=features.device)

        if normal_mask.any():
            normal_sim = similarity[normal_mask]
            # Pull normal toward prototype
            loss = loss + (-normal_sim / self.temperature).mean()

            # Update prototype with normal features
            if self.training:
                self.prototype.update(features[normal_mask])

        if anomaly_mask.any():
            anomaly_sim = similarity[anomaly_mask]
            # Push anomalous away from prototype (maximise distance)
            loss = loss + (anomaly_sim / self.temperature).mean()

        # Compute metrics
        with torch.no_grad():
            normal_sim_mean = (
                similarity[normal_mask].mean()
                if normal_mask.any()
                else torch.tensor(0.0)
            )
            anomaly_sim_mean = (
                similarity[anomaly_mask].mean()
                if anomaly_mask.any()
                else torch.tensor(0.0)
            )

        return {
            "loss": loss,
            "similarity": similarity,
            "normal_sim": normal_sim_mean,
            "anomaly_sim": anomaly_sim_mean,
        }

    def compute_anomaly_score(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Convert similarity to anomaly score at inference time.

        Higher scores indicate higher anomaly likelihood.

        Args:
            features: ``(B, D)`` global feature representations.

        Returns:
            ``(B,)`` anomaly scores in ``[0, 1]``.
        """
        similarity = self.prototype.similarity(features)
        # Convert: high similarity → low anomaly, low similarity → high anomaly
        scores = (1.0 - similarity) / 2.0
        return scores.clamp(0.0, 1.0)
