"""E3 representation similarity: debiased linear CKA, Procrustes, monotone alignment, block Jacobian spectrum."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.func import jvp, vjp

from .analysis import AnalysisModel


def _unbiased_hsic(gram_x: Tensor, gram_y: Tensor) -> Tensor:
    """Unbiased HSIC estimator (Song et al. 2012) of two Gram matrices with their diagonals removed."""

    n = gram_x.shape[0]
    gram_x = gram_x - torch.diag(torch.diag(gram_x))
    gram_y = gram_y - torch.diag(torch.diag(gram_y))
    ones = torch.ones(n, 1, dtype=gram_x.dtype, device=gram_x.device)
    trace = (gram_x * gram_y).sum()
    sums = (ones.T @ gram_x @ ones) * (ones.T @ gram_y @ ones) / ((n - 1) * (n - 2))
    cross = 2.0 / (n - 2) * (ones.T @ gram_x @ gram_y @ ones)
    return (trace + sums - cross).squeeze() / (n * (n - 3))


def debiased_cka(x: Tensor, y: Tensor) -> float:
    """Debiased linear CKA between two (N, D) representations of the same examples."""

    x = x.double() - x.double().mean(dim=0)
    y = y.double() - y.double().mean(dim=0)
    gram_x, gram_y = x @ x.T, y @ y.T
    xy, xx, yy = _unbiased_hsic(gram_x, gram_y), _unbiased_hsic(gram_x, gram_x), _unbiased_hsic(gram_y, gram_y)
    return float(xy / torch.sqrt(xx * yy).clamp_min(1e-30))


def cka_matrix(left: Tensor, right: Tensor) -> np.ndarray:
    """CKA between every site of ``left`` and every site of ``right``; both (N, S, D)."""

    return np.array([[debiased_cka(left[:, i], right[:, j]) for j in range(right.shape[1])] for i in range(left.shape[1])])


def procrustes_similarity(x: Tensor, y: Tensor) -> float:
    """Share of ``y`` that an orthogonal rotation of ``x`` reproduces, after centering and unit scaling.

    1 means ``y`` is a rotation of ``x``; the remainder is new content.
    """

    x = x.double() - x.double().mean(dim=0)
    y = y.double() - y.double().mean(dim=0)
    x, y = x / x.norm(), y / y.norm()
    return float(torch.linalg.svdvals(x.T @ y).sum())


def monotone_path(matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Highest mean-similarity path from the first to the last cell that never steps backwards."""

    rows, cols = matrix.shape
    best = np.full((rows, cols), -np.inf)
    count = np.zeros((rows, cols), dtype=int)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    best[0, 0], count[0, 0] = matrix[0, 0], 1
    for i in range(rows):
        for j in range(cols):
            if i == j == 0:
                continue
            options = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
            options = [(a, b) for a, b in options if a >= 0 and b >= 0]
            a, b = max(options, key=lambda cell: best[cell] / max(count[cell], 1))
            best[i, j], count[i, j], back[(i, j)] = best[a, b] + matrix[i, j], count[a, b] + 1, (a, b)
    path, cell = [(rows - 1, cols - 1)], (rows - 1, cols - 1)
    while cell in back:
        cell = back[cell]
        path.append(cell)
    path.reverse()
    return path, float(best[-1, -1] / count[-1, -1])


def block_spectral_norms(
    model: AnalysisModel, tokens: Tensor, *, update: bool = False, iterations: int = 20, seed: int = 0
) -> np.ndarray:
    """Top singular value of each visit's block map, per example: (examples, visits).

    Power iteration with JVPs and VJPs on the whole-sequence state of one example at a time. With
    ``update`` the map is the block minus its residual input, the change the visit makes.
    """

    device = next(model.model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    values = np.zeros((tokens.shape[0], model.config.visits))
    for row in range(tokens.shape[0]):
        visits: list[tuple] = []

        def hook(context) -> None:
            visits.append((context.module, context.before.detach(), dict(context.kwargs)))

        with torch.no_grad():
            model.run(tokens[row : row + 1].to(device), hook)
        for visit, (module, state, kwargs) in enumerate(visits):

            def block(value: Tensor, module=module, kwargs=kwargs) -> Tensor:
                output = module(value, **kwargs)
                return output - value if update else output

            direction = torch.randn(state.shape, generator=generator).to(device)
            direction = direction / direction.norm()
            for _ in range(iterations):
                _, tangent = jvp(block, (state,), (direction,))
                _, pullback = vjp(block, state)
                direction = pullback(tangent / tangent.norm().clamp_min(1e-12))[0]
                direction = direction / direction.norm().clamp_min(1e-12)
            _, tangent = jvp(block, (state,), (direction,))
            values[row, visit] = float(tangent.detach().norm())
    return values
