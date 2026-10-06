"""E1 arrival matrices: denoising residual patches between minimal pairs, by divergence step and visit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import Tensor

from .analysis import AnalysisModel
from .bundle import TraceExample
from .hooks import patch


MIN_SPREAD = 0.05
ARRIVAL_THRESHOLD = 0.5


@dataclass(frozen=True)
class Pairs:
    """Base (clean) and sibling (corrupt) rows whose answers differ, with the step where they diverge."""

    clean: Tensor
    corrupt: Tensor
    clean_target: Tensor
    corrupt_target: Tensor
    step: np.ndarray
    depth: np.ndarray
    group: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.clean.shape[0])

    def subset(self, index: np.ndarray) -> "Pairs":
        rows = torch.as_tensor(index, dtype=torch.long)
        return Pairs(
            self.clean[rows], self.corrupt[rows], self.clean_target[rows], self.corrupt_target[rows],
            self.step[index], self.depth[index], tuple(self.group[i] for i in index),
        )


def edge_rewire_pairs(rows: Sequence[TraceExample]) -> Pairs:
    bases = {row.example_id: row for row in rows if row.sibling is None}
    chosen = [
        (bases[row.sibling["base_id"]], row)
        for row in rows
        if row.sibling is not None
        and row.sibling["kind"] == "edge_rewire"
        and row.sibling["target_changed"]
        and row.sibling["base_id"] in bases
    ]
    return Pairs(
        torch.tensor([base.tokens for base, _ in chosen], dtype=torch.long),
        torch.tensor([sibling.tokens for _, sibling in chosen], dtype=torch.long),
        torch.tensor([base.target_token for base, _ in chosen], dtype=torch.long),
        torch.tensor([sibling.target_token for _, sibling in chosen], dtype=torch.long),
        np.array([int(sibling.sibling["step"]) for _, sibling in chosen]),
        np.array([base.difficulty for base, _ in chosen]),
        tuple(base.group_id for base, _ in chosen),
    )


def _difference(logits: Tensor, clean_target: Tensor, corrupt_target: Tensor) -> Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, clean_target] - logits[rows, corrupt_target]


@torch.no_grad()
def patch_recovery(
    model: AnalysisModel,
    pairs: Pairs,
    *,
    positions: Sequence[int] = (-1,),
    batch: int = 256,
) -> dict[str, np.ndarray]:
    """Normalized logit-difference recovery when the clean state replaces the corrupt one.

    Returns ``recovery`` of shape (pairs, visits, positions), the clean and corrupt logit differences,
    and whether the model answers each member correctly.
    """

    device = next(model.model.parameters()).device
    visits = model.config.visits
    outputs = {key: [] for key in ("recovery", "clean_ld", "corrupt_ld", "clean_ok", "corrupt_ok")}
    for start in range(0, len(pairs), batch):
        clean = pairs.clean[start : start + batch].to(device)
        corrupt = pairs.corrupt[start : start + batch].to(device)
        good = pairs.clean_target[start : start + batch].to(device)
        bad = pairs.corrupt_target[start : start + batch].to(device)
        clean_run = model.run(clean)
        corrupt_logits = model.run(corrupt).logits[:, -1].float()
        clean_ld = _difference(clean_run.logits[:, -1].float(), good, bad)
        corrupt_ld = _difference(corrupt_logits, good, bad)
        spread = clean_ld - corrupt_ld
        recovery = torch.empty(clean.shape[0], visits, len(positions))
        for visit in range(visits):
            source = clean_run.visit_outputs[visit]
            for column, position in enumerate(positions):

                def replace(after: Tensor, source=source, position=position) -> Tensor:
                    patched = after.clone()
                    patched[:, position] = source[:, position]
                    return patched

                logits = model.run(corrupt, patch(visit, replace)).logits[:, -1].float()
                recovery[:, visit, column] = ((_difference(logits, good, bad) - corrupt_ld) / spread).cpu()
        outputs["recovery"].append(recovery.numpy())
        outputs["clean_ld"].append(clean_ld.cpu().numpy())
        outputs["corrupt_ld"].append(corrupt_ld.cpu().numpy())
        outputs["clean_ok"].append((clean_run.logits[:, -1].argmax(-1) == good).cpu().numpy())
        outputs["corrupt_ok"].append((corrupt_logits.argmax(-1) == bad).cpu().numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}


def arrival_visits(recovery: np.ndarray, *, threshold: float = ARRIVAL_THRESHOLD) -> np.ndarray:
    """First visit at which recovery reaches ``threshold``; the last visit if none earlier does.

    ``recovery`` is (pairs, visits); the last visit is excluded because patching it fixes the logits.
    """

    early = recovery[:, :-1] >= threshold
    last = recovery.shape[1] - 1
    return np.where(early.any(axis=1), early.argmax(axis=1), last)


def ridge_rho(step: np.ndarray, arrival: np.ndarray, depth: np.ndarray, *, min_pairs: int = 20) -> float:
    """Pair-weighted mean over depths of the within-depth Spearman correlation of step and arrival."""

    total, weight = 0.0, 0
    for value in np.unique(depth):
        mask = depth == value
        if mask.sum() < min_pairs or len(np.unique(step[mask])) < 3 or len(np.unique(arrival[mask])) < 2:
            continue
        total += float(spearmanr(step[mask], arrival[mask]).statistic) * int(mask.sum())
        weight += int(mask.sum())
    return total / weight if weight else float("nan")


def permutation_p(
    step: np.ndarray, arrival: np.ndarray, depth: np.ndarray, *, rounds: int = 1000, seed: int = 0
) -> float:
    """One-sided p-value of ``ridge_rho`` against shuffling ``step`` within each depth."""

    observed = ridge_rho(step, arrival, depth)
    if np.isnan(observed):
        return float("nan")
    generator = np.random.default_rng(seed)
    shuffled = step.copy()
    exceed = 0
    for _ in range(rounds):
        for value in np.unique(depth):
            index = np.flatnonzero(depth == value)
            shuffled[index] = generator.permutation(step[index])
        exceed += ridge_rho(shuffled, arrival, depth) >= observed
    return (exceed + 1) / (rounds + 1)


def paired_difference_interval(
    step: np.ndarray,
    depth: np.ndarray,
    arrival: np.ndarray,
    control_arrival: np.ndarray,
    *,
    rounds: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% interval of rho(model) minus rho(control) over the same resampled pairs."""

    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(rounds):
        index = generator.integers(0, len(step), len(step))
        differences.append(
            ridge_rho(step[index], arrival[index], depth[index])
            - ridge_rho(step[index], control_arrival[index], depth[index])
        )
    values = np.array([value for value in differences if not np.isnan(value)])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def arrival_matrix(recovery: np.ndarray, step: np.ndarray) -> dict[str, list[float]]:
    """Mean recovery by divergence step (keys) and visit (list index)."""

    return {
        str(value): recovery[step == value].mean(axis=0).round(4).tolist() for value in np.unique(step)
    }
