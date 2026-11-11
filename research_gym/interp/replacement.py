"""N1: dictionary splices into the running model (open-loop, closed-loop, replacement model) and controls."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis import AnalysisModel
from .dictionaries import TopKDictionary, rms


# A transform maps a visit context to the state that should leave that visit.
Transform = Callable[[object], Tensor]


def at_visits(transform: Transform, visits: Sequence[int] | None = None):
    """A visit hook applying ``transform`` after the given visits (every visit when ``None``).

    Visit ``v`` produces ``resid[v + 1]``, so replacing ``resid[t]`` means hooking visit ``t - 1``.
    """

    chosen = None if visits is None else set(visits)

    def hook(context) -> Tensor | None:
        if chosen is not None and context.visit not in chosen:
            return None
        return transform(context)

    return hook


def reconstruct(dictionary: TopKDictionary) -> Transform:
    return lambda context: dictionary.reconstruct_state(context.after)


def mean_state(means: Tensor) -> Transform:
    """Replace the state by its per-position mean at that site; ``means`` is (sites, positions, H)."""

    return lambda context: means[context.visit + 1].unsqueeze(0).expand_as(context.after)


def matched_noise(dictionary: TopKDictionary, *, seed: int = 0) -> Transform:
    """Add isotropic noise whose norm equals each state's reconstruction error norm."""

    generator: dict[str, torch.Generator] = {}

    def transform(context) -> Tensor:
        after = context.after
        if "g" not in generator:
            generator["g"] = torch.Generator(device=after.device).manual_seed(seed)
        error = (after - dictionary.reconstruct_state(after)).norm(dim=-1, keepdim=True)
        noise = torch.randn(after.shape, generator=generator["g"], device=after.device, dtype=after.dtype)
        return after + noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12) * error

    return transform


def pca_truncation(mean: Tensor, components: Tensor) -> Transform:
    """Project each RMS-normalized state onto the given principal components, then rescale."""

    def transform(context) -> Tensor:
        scale = rms(context.after)
        centered = context.after / scale - mean
        return scale * (mean + centered @ components @ components.T)

    return transform


def transcoder_step(dictionaries: Sequence[TopKDictionary]) -> Transform:
    """Replacement model: the next state is the previous one plus the transcoder's predicted update.

    One transcoder serves every visit of a tied loop; an untied stack passes one per layer.
    """

    def transform(context) -> Tensor:
        dictionary = dictionaries[0] if len(dictionaries) == 1 else dictionaries[context.visit]
        return context.before + dictionary.predict_update(context.before)

    return transform


@torch.no_grad()
def answer_scores(
    model: AnalysisModel, tokens: Tensor, targets: Tensor, hook=None, *, batch: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Per-example answer cross-entropy and correctness under an optional visit hook."""

    device = next(model.model.parameters()).device
    losses, correct = [], []
    for start in range(0, tokens.shape[0], batch):
        chunk = tokens[start : start + batch].to(device)
        goal = targets[start : start + batch].to(device)
        logits = (model.run(chunk, hook) if hook is not None else model.run(chunk)).logits[:, -1].float()
        losses.append(torch.nn.functional.cross_entropy(logits, goal, reduction="none").cpu())
        correct.append((logits.argmax(dim=-1) == goal).cpu())
    return torch.cat(losses).numpy(), torch.cat(correct).numpy()


def ce_recovered(clean: float, ablated: float, spliced: float) -> float:
    return (ablated - spliced) / max(ablated - clean, 1e-12)


def fit_pca(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Mean, principal directions as columns (descending variance), and per-direction variance."""

    mean = x.mean(dim=0)
    centered = x - mean
    covariance = centered.T @ centered / max(1, x.shape[0] - 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = values.argsort(descending=True)
    return mean, vectors[:, order], values[order]


def pca_components_matching(variances: Tensor, target_mse: float, total_mse_of_mean: float) -> int:
    """Fewest components whose truncation error is at most ``target_mse`` (per-vector squared error)."""

    residual = total_mse_of_mean - torch.cumsum(variances, dim=0)
    for count, value in enumerate(residual.tolist(), start=1):
        if value <= target_mse:
            return count
    return int(variances.shape[0])
