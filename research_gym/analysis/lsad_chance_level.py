"""Chance and best-constant baselines for sealed LSAD prediction artifacts."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Mapping, Sequence

from research_gym.architecture_discovery.tasks import (
    MODULAR_TARGET_BASE,
    POINTER_TARGET_BASE,
    REWRITE_TARGET_BASE,
)


# Answer-class counts mirror the generator locals: eight nodes, modulus 17, six symbols.
ANSWER_RANGES: Mapping[str, tuple[int, int]] = {
    "pointer_chase": (POINTER_TARGET_BASE, 8),
    "modular_recurrence": (MODULAR_TARGET_BASE, 17),
    "rewrite_normalization": (REWRITE_TARGET_BASE, 6),
}


def target_entropy(targets: Iterable[int]) -> float:
    """Entropy in nats of the marginal target distribution."""

    counts = Counter(int(target) for target in targets)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("target entropy requires at least one target")
    return -sum(count / total * math.log(count / total) for count in counts.values())


def family_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Uniform and best-constant accuracy for the scored examples of each family."""

    output: dict[str, dict[str, Any]] = {}
    for family, (_, classes) in ANSWER_RANGES.items():
        selected = [row for row in rows if row["family"] == family]
        if not selected:
            continue
        targets = Counter(int(row["target_token"]) for row in selected)
        depths = Counter(int(row["difficulty"]) for row in selected)
        output[family] = {
            "examples": len(selected),
            "uniform_chance": 1.0 / classes,
            "best_constant": max(targets.values()) / len(selected),
            "depths": {str(depth): count for depth, count in sorted(depths.items())},
        }
    return output


def prediction_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Describe how concentrated each family's predictions are."""

    output: dict[str, dict[str, Any]] = {}
    for family, (base, classes) in ANSWER_RANGES.items():
        selected = [row for row in rows if row["family"] == family]
        if not selected:
            continue
        predictions = Counter(int(row["prediction_token"]) for row in selected)
        output[family] = {
            "examples": len(selected),
            "correct": sum(int(row["exact"]) for row in selected),
            "in_answer_range": sum(
                count for token, count in predictions.items() if base <= token < base + classes
            ),
            "distinct_predictions": len(predictions),
            "top_prediction_share": max(predictions.values()) / len(selected),
        }
    return output


def binomial_item_sd(baselines: Mapping[str, Mapping[str, Any]], *, rate: str) -> float:
    """Standard deviation of total correct items for a scorer at the named rate."""

    return math.sqrt(
        sum(
            int(value["examples"]) * float(value[rate]) * (1.0 - float(value[rate]))
            for value in baselines.values()
        )
    )
