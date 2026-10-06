from __future__ import annotations

import numpy as np
import torch

from research_gym.interp.analysis import AnalysisModel
from research_gym.interp.models import build_model
from research_gym.interp.similarity import (
    block_spectral_norms,
    cka_matrix,
    debiased_cka,
    monotone_path,
    procrustes_similarity,
)
from research_gym.interp.training import CellConfig


def test_cka_is_invariant_to_rotation_and_scale_and_near_zero_for_unrelated_data() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(400, 16, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
    assert abs(debiased_cka(x, 3.0 * x @ rotation) - 1.0) < 1e-6
    assert abs(debiased_cka(x, torch.randn(400, 16, generator=generator))) < 0.05
    states = torch.stack([x, x @ rotation, torch.randn(400, 16, generator=generator)], dim=1)
    matrix = cka_matrix(states, states)
    assert matrix.shape == (3, 3) and matrix[0, 1] > 0.99 and abs(matrix[0, 2]) < 0.05


def test_procrustes_reads_a_rotation_as_full_reuse() -> None:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(300, 8, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    assert abs(procrustes_similarity(x, x @ rotation) - 1.0) < 1e-6
    assert procrustes_similarity(x, torch.randn(300, 8, generator=generator)) < 0.4


def test_monotone_path_follows_the_ridge() -> None:
    matrix = np.full((4, 4), 0.1)
    for step in range(4):
        matrix[step, step] = 1.0
    path, mean = monotone_path(matrix)
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3)] and mean == 1.0
    shifted = np.full((3, 4), 0.1)
    for cell in [(0, 0), (0, 1), (1, 2), (2, 3)]:
        shifted[cell] = 1.0
    assert monotone_path(shifted)[0] == [(0, 0), (0, 1), (1, 2), (2, 3)]


def test_power_iteration_matches_the_explicit_jacobian() -> None:
    torch.manual_seed(0)
    model = build_model("LOOP-T", hidden_size=8, num_heads=2, visits=2, beta=1.0, vocab_size=32).eval()
    analysis = AnalysisModel("tiny", "tiny", "C0", 0, CellConfig(arm="LOOP-T", seed=0, families=(), visits=2), model)
    tokens = torch.randint(0, 32, (1, 5))
    for update in (False, True):
        estimated = block_spectral_norms(analysis, tokens, update=update, iterations=60)
        states = [model.embed(tokens)]
        for _ in range(2):
            states.append(model.blocks[0](states[-1]))
        for visit in range(2):
            def block(value: torch.Tensor) -> torch.Tensor:
                output = model.blocks[0](value)
                return output - value if update else output

            jacobian = torch.autograd.functional.jacobian(block, states[visit]).reshape(40, 40)
            exact = torch.linalg.svdvals(jacobian)[0].item()
            assert abs(estimated[0, visit] - exact) < 1e-3 * max(1.0, exact)
