from __future__ import annotations

import math
from pathlib import Path

import pytest

from research_gym.analysis.lsad_chance_level import (
    binomial_item_sd,
    family_baselines,
    prediction_profile,
    target_entropy,
)
from research_gym.scripts.audit_lsad_v1_chance_level import build_audit


ROOT = Path(__file__).resolve().parents[1]


def _row(family: str, target: int, prediction: int, difficulty: int = 5) -> dict:
    return {
        "family": family,
        "target_token": target,
        "prediction_token": prediction,
        "exact": int(target == prediction),
        "difficulty": difficulty,
    }


def test_target_entropy_is_uniform_log_for_balanced_targets() -> None:
    assert target_entropy([800, 801, 802, 803]) == pytest.approx(math.log(4))
    assert target_entropy([800, 800]) == 0.0


def test_baselines_and_profile_separate_constant_from_informed_predictions() -> None:
    rows = [
        _row("pointer_chase", 800, 800),
        _row("pointer_chase", 800, 800),
        _row("pointer_chase", 801, 800),
        _row("pointer_chase", 802, 40),
    ]
    baselines = family_baselines(rows)
    profile = prediction_profile(rows)

    assert baselines["pointer_chase"]["uniform_chance"] == pytest.approx(1 / 8)
    assert baselines["pointer_chase"]["best_constant"] == pytest.approx(0.5)
    assert profile["pointer_chase"]["correct"] == 2
    assert profile["pointer_chase"]["in_answer_range"] == 3
    assert profile["pointer_chase"]["top_prediction_share"] == pytest.approx(0.75)
    assert binomial_item_sd(baselines, rate="uniform_chance") == pytest.approx(
        math.sqrt(4 * (1 / 8) * (7 / 8))
    )


def test_sealed_lsad_v1_cells_do_not_beat_a_constant_predictor() -> None:
    audit = build_audit(ROOT / "experiments" / "loop_schedule_architecture_discovery_v1")
    summary = audit["summary"]

    assert summary["cells"] == 27
    assert summary["cells_below_best_constant"] == 27
    assert summary["cells_with_loss_below_target_entropy"] == 0
    assert summary["max_examples_seen"] == 4096
    assert all(
        cell["in_answer_range_share"] == 1.0
        for cell in audit["cells"]
        if cell["stage"] == "A2"
    )
