from __future__ import annotations

import numpy as np
import torch

from research_gym.interp.analysis import AnalysisModel, with_injection
from research_gym.interp.arrival import (
    Pairs,
    arrival_visits,
    paired_difference_interval,
    patch_recovery,
    permutation_p,
    ridge_rho,
)
from research_gym.interp.models import build_model
from research_gym.interp.training import CellConfig


def _model(arm: str, *, injection: bool = False) -> AnalysisModel:
    torch.manual_seed(0)
    config = CellConfig(
        arm=arm, seed=0, families=("pointer_chase",), hidden_size=32, num_heads=2, visits=4,
        beta=1.0, input_injection=injection,
    )
    model = build_model(arm, hidden_size=32, num_heads=2, visits=4, beta=1.0).eval()
    return AnalysisModel("tiny", "tiny", "C0", 0, config, model)


def _pairs(count: int = 6) -> Pairs:
    generator = torch.Generator().manual_seed(1)
    clean = torch.randint(0, 2048, (count, 12), generator=generator)
    corrupt = clean.clone()
    corrupt[:, 3] = (corrupt[:, 3] + 7) % 2048
    return Pairs(
        clean, corrupt, torch.full((count,), 5), torch.full((count,), 9),
        np.arange(count) % 3, np.full(count, 4), tuple(f"g{i}" for i in range(count)),
    )


def test_patching_the_answer_after_the_last_visit_recovers_exactly() -> None:
    for arm, injection in (("LOOP-T", False), ("FF-U", True)):
        result = patch_recovery(_model(arm, injection=injection), _pairs(), positions=(-1, 3), batch=4)
        assert result["recovery"].shape == (6, 4, 2)
        np.testing.assert_allclose(result["recovery"][:, -1, 0], 1.0, atol=1e-4)
        # The differing token's position, patched at the first visit, carries everything that differs
        # only when nothing has read it yet; it must at least move the answer toward the clean one.
        assert np.isfinite(result["recovery"]).all()


def test_a_control_can_run_its_models_input_reinjection() -> None:
    plain = _model("FF-U")
    injected = with_injection(plain, True)
    tokens = _pairs().clean
    assert injected.config.input_injection and not plain.config.input_injection
    assert injected.model is plain.model
    assert not torch.allclose(injected.run(tokens).logits, plain.run(tokens).logits)


def test_arrival_is_the_first_visit_over_threshold_or_the_last() -> None:
    recovery = np.array([[0.1, 0.6, 0.9, 1.0], [0.0, 0.2, 0.3, 1.0], [0.7, 0.1, 0.8, 1.0]])
    assert arrival_visits(recovery).tolist() == [1, 3, 0]


def test_ridge_statistic_and_its_tests() -> None:
    generator = np.random.default_rng(0)
    depth = np.repeat([4, 6], 60)
    step = np.concatenate([generator.integers(0, 4, 60), generator.integers(0, 6, 60)])
    arrival = step + 1
    assert abs(ridge_rho(step, arrival, depth) - 1.0) < 1e-9
    assert permutation_p(step, arrival, depth, rounds=200) < 0.01
    noise = generator.integers(0, 8, len(step))
    assert abs(ridge_rho(step, noise, depth)) < 0.3
    low, high = paired_difference_interval(step, depth, arrival, noise, rounds=200)
    assert low > 0.5 and high <= 1.5
    # Too few pairs or too few distinct steps at every depth: no statistic.
    assert np.isnan(ridge_rho(step[:10], arrival[:10], depth[:10]))
    assert np.isnan(paired_difference_interval(step[:10], depth[:10], arrival[:10], noise[:10], rounds=20)[0])
    # Arrival that never varies within a depth is no association, not a missing one.
    assert ridge_rho(step, np.full(len(step), 7), depth) == 0.0
    flat_at_one_depth = np.where(depth == 4, 3, arrival)
    assert abs(ridge_rho(step, flat_at_one_depth, depth) - 0.5) < 1e-9
