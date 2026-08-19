from __future__ import annotations

import torch
import torch.nn.functional as F


def degradation_contrastive_loss(
    descriptors: torch.Tensor,
    positive_descriptors: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss for samples generated with the same synthetic degradation.

    Args:
        descriptors: Token-wise descriptors with shape ``[B, N, C]``.
        positive_descriptors: Descriptors from different HR images synthesized
            with the same degradation settings, also shaped ``[B, N, C]``.
        temperature: Contrastive temperature ``t`` from the paper.

    Synthetic degradation labels are used only to construct positive pairs
    during training. They are never passed to TGSR and are not required during
    inference. For real-world samples without known settings, omit this loss.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if descriptors.shape != positive_descriptors.shape:
        raise ValueError("descriptor tensors must have identical shapes")

    anchors = F.normalize(descriptors.mean(dim=1), dim=-1)
    positives = F.normalize(positive_descriptors.mean(dim=1), dim=-1)
    logits = anchors @ positives.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
