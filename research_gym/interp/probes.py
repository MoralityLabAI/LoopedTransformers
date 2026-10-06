"""E2 trace probes: answer-position states, batched linear and MLP probes, transfer, online code length."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis import AnalysisModel
from .bundle import TraceExample


WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)
PORTIONS = tuple(2.0**-k for k in range(7, -1, -1))


@torch.no_grad()
def answer_states(model: AnalysisModel, tokens: Tensor, *, batch: int = 500) -> tuple[Tensor, Tensor]:
    """Answer-position states, (N, visits + 1, H): the state entering visit 0, then each visit's output.

    Also returns the mean over positions of the state entering visit 0, (N, H): a bag of input tokens.
    """

    device = next(model.model.parameters()).device
    sites, pooled = [], []
    for start in range(0, tokens.shape[0], batch):
        captured: list[Tensor] = []
        entering: list[Tensor] = []

        def hook(context) -> None:
            if context.visit == 0:
                entering.append(context.before)
            captured.append(context.after[:, -1])

        model.run(tokens[start : start + batch].to(device), hook)
        sites.append(torch.stack([entering[0][:, -1], *captured], dim=1).float().cpu())
        pooled.append(entering[0].mean(dim=1).float().cpu())
    return torch.cat(sites), torch.cat(pooled)


def rms_normalize(states: Tensor) -> Tensor:
    return states / states.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()


def trace_labels(family: str, rows: Sequence[TraceExample]) -> dict[str, dict[int, np.ndarray]]:
    """Per variable and algorithmic step, one label per row, or -1 where the row has no such step."""

    def by_step(values_of, steps: range) -> dict[int, np.ndarray]:
        table = {}
        for step in steps:
            labels = np.array([values_of(row, step) for row in rows])
            if (labels >= 0).sum():
                table[step] = labels
        return table

    def at(name: str, transform=int):
        return lambda row, step: transform(row.trace[name][step]) if step < len(row.trace[name]) else -1

    if family in ("pointer_chase", "gated_pointer_chase"):
        labels = {"cur_node": by_step(at("cur_node"), range(1, 9))}
        if family == "gated_pointer_chase":
            labels["latch"] = by_step(at("latch"), range(0, 9))
        return labels
    if family == "reachability":
        return {"goal_found": by_step(at("goal_found"), range(0, 9))}
    if family == "sudoku4":
        def query_candidates(row: TraceExample, step: int) -> int:
            history = row.trace["candidates"]
            return int(history[step][int(row.semantic["query"])]) if step < len(history) else -1

        return {"query_candidates": by_step(query_candidates, range(0, 4))}
    raise ValueError(f"no trace probes for {family}")


def given_values(family: str, rows: Sequence[TraceExample]) -> np.ndarray:
    """The input-given variable the control task relabels: start node, or the queried Sudoku cell."""

    key = "query" if family == "sudoku4" else "start"
    return np.array([int(row.semantic[key]) for row in rows])


def control_labels(given: np.ndarray, classes: int, seed: int) -> np.ndarray:
    mapping = np.random.default_rng(seed).integers(0, classes, int(given.max()) + 1)
    return mapping[given]


def shuffled_within(labels: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    shuffled = labels.copy()
    for value in np.unique(strata):
        index = np.flatnonzero(strata == value)
        shuffled[index] = generator.permutation(labels[index])
    return shuffled


def _standardize(train: Tensor, *others: Tensor) -> list[Tensor]:
    mean = train.mean(dim=1, keepdim=True)
    scale = train.std(dim=1, keepdim=True).clamp_min(1e-6)
    return [(value - mean) / scale for value in (train, *others)]


def fit_linear(
    features: Tensor, labels: Tensor, classes: int, *, weight_decay: float, steps: int = 400, lr: float = 0.05
) -> Tensor:
    """Multinomial logistic regression for each of S sites at once. features (S, N, H) -> weights (S, H+1, K)."""

    sites, _, width = features.shape
    inputs = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
    weights = torch.zeros(sites, width + 1, classes, device=features.device, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = torch.bmm(inputs, weights)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, classes), labels.repeat(sites))
        (loss + weight_decay * weights[:, :-1].pow(2).sum() / sites).backward()
        optimizer.step()
    return weights.detach()


def linear_logits(weights: Tensor, features: Tensor) -> Tensor:
    return torch.bmm(torch.cat([features, torch.ones_like(features[..., :1])], dim=-1), weights)


def fit_mlp(
    features: Tensor, labels: Tensor, classes: int, *, hidden: int = 256, weight_decay: float = 1e-3,
    steps: int = 600, lr: float = 0.01, seed: int = 0,
) -> tuple[Tensor, ...]:
    sites, _, width = features.shape
    generator = torch.Generator().manual_seed(seed)
    first = (torch.randn(sites, width, hidden, generator=generator) / math.sqrt(width)).to(features.device)
    second = (torch.randn(sites, hidden, classes, generator=generator) / math.sqrt(hidden)).to(features.device)
    bias = torch.zeros(sites, 1, hidden, device=features.device)
    out_bias = torch.zeros(sites, 1, classes, device=features.device)
    parameters = [first.requires_grad_(), second.requires_grad_(), bias.requires_grad_(), out_bias.requires_grad_()]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = mlp_logits(tuple(parameters), features)
        torch.nn.functional.cross_entropy(logits.reshape(-1, classes), labels.repeat(sites)).backward()
        optimizer.step()
    return tuple(value.detach() for value in parameters)


def mlp_logits(parameters: tuple[Tensor, ...], features: Tensor) -> Tensor:
    first, second, bias, out_bias = parameters
    return torch.bmm(torch.relu(torch.bmm(features, first) + bias), second) + out_bias


def accuracy(logits: Tensor, labels: Tensor) -> np.ndarray:
    return (logits.argmax(dim=-1) == labels).float().mean(dim=-1).cpu().numpy()


def probe_sites(
    train: Tensor, train_labels: np.ndarray, select: Tensor, select_labels: np.ndarray,
    report: Tensor, report_labels: np.ndarray, *, device: torch.device, transfer: bool = False,
) -> dict[str, np.ndarray]:
    """Fit a linear probe per site, pick weight decay per site on ``select``, score on ``report``.

    Inputs are (N, S, H) states and labels with -1 for rows to skip. Returns per-site accuracies and, with
    ``transfer``, the (S, S) matrix of the probe fit at site i scored at site j.
    """

    def keep(states: Tensor, labels: np.ndarray) -> tuple[Tensor, Tensor]:
        mask = labels >= 0
        return (states[torch.as_tensor(mask)].transpose(0, 1).to(device),
                torch.as_tensor(labels[mask], dtype=torch.long, device=device))

    (x_train, y_train), (x_select, y_select), (x_report, y_report) = (
        keep(train, train_labels), keep(select, select_labels), keep(report, report_labels)
    )
    x_train, x_select, x_report = _standardize(x_train, x_select, x_report)
    classes = int(max(train_labels.max(), select_labels.max(), report_labels.max())) + 1
    fits = [fit_linear(x_train, y_train, classes, weight_decay=decay) for decay in WEIGHT_DECAYS]
    chosen = np.stack([accuracy(linear_logits(weights, x_select), y_select) for weights in fits]).argmax(axis=0)
    weights = torch.stack([fits[choice][site] for site, choice in enumerate(chosen)])
    result = {"accuracy": accuracy(linear_logits(weights, x_report), y_report), "weight_decay": np.array(WEIGHT_DECAYS)[chosen]}
    if transfer:
        matrix = np.zeros((weights.shape[0], weights.shape[0]))
        for source in range(weights.shape[0]):
            shared = weights[source : source + 1].expand(weights.shape[0], -1, -1)
            matrix[source] = accuracy(linear_logits(shared, x_report), y_report)
        result["transfer"] = matrix
    return result


def mlp_sites(train: Tensor, train_labels: np.ndarray, report: Tensor, report_labels: np.ndarray, *, device) -> np.ndarray:
    mask_train, mask_report = train_labels >= 0, report_labels >= 0
    x_train = train[torch.as_tensor(mask_train)].transpose(0, 1).to(device)
    x_report = report[torch.as_tensor(mask_report)].transpose(0, 1).to(device)
    x_train, x_report = _standardize(x_train, x_report)
    y_train = torch.as_tensor(train_labels[mask_train], dtype=torch.long, device=device)
    y_report = torch.as_tensor(report_labels[mask_report], dtype=torch.long, device=device)
    classes = int(max(train_labels.max(), report_labels.max())) + 1
    return accuracy(mlp_logits(fit_mlp(x_train, y_train, classes), x_report), y_report)


def online_code_ratio(train: Tensor, labels: np.ndarray, *, device, seed: int = 0) -> np.ndarray:
    """Online code length of the labels given each site, over the code length of their entropy.

    The training rows are shuffled once; the first portion is sent uniformly, each later block with a
    probe fit on everything before it.
    """

    mask = labels >= 0
    order = np.random.default_rng(seed).permutation(int(mask.sum()))
    states = train[torch.as_tensor(mask)][torch.as_tensor(order)].transpose(0, 1).to(device)
    targets = torch.as_tensor(labels[mask][order], dtype=torch.long, device=device)
    classes = int(labels.max()) + 1
    total = targets.shape[0]
    bounds = [max(2, int(round(fraction * total))) for fraction in PORTIONS]
    bits = torch.full((states.shape[0],), bounds[0] * math.log2(classes), device=device)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        fit_states, next_states = _standardize(states[:, :start], states[:, start:stop])
        weights = fit_linear(fit_states, targets[:start], classes, weight_decay=1e-3, steps=200)
        log_probs = torch.log_softmax(linear_logits(weights, next_states), dim=-1)
        picked = log_probs.gather(-1, targets[start:stop].expand(states.shape[0], -1).unsqueeze(-1)).squeeze(-1)
        bits -= picked.sum(dim=-1) / math.log(2)
    counts = np.bincount(labels[mask], minlength=classes) / total
    entropy_bits = -sum(share * math.log2(share) for share in counts if share > 0) * total
    return (bits.cpu().numpy() / max(entropy_bits, 1e-9))
