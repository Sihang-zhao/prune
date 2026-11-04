"""Shared group-index contracts and question-conditioned text scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SelectionResult:
    """Original merged-group IDs and role-specific selections for one view."""

    indices: torch.Tensor
    final_indices: torch.Tensor
    middle_indices: torch.Tensor
    shallow_indices: torch.Tensor
    budget: tuple[int, int, int]


def _stable_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the native text Top-K tie behavior and return IDs in spatial order."""
    if int(k) <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    return torch.topk(values, k=int(k), largest=True, sorted=False).indices.sort().values


def _text_cosine(projected: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
    """Compare visual groups with the mean native input text embedding."""
    text_vector = F.normalize(text_tokens.float().mean(dim=0), dim=0)
    return F.cosine_similarity(projected.float(), text_vector.unsqueeze(0), dim=-1)
