"""Regression tests for read-only NEC-04C replay of NEC-04B native outputs."""
import pytest

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04c_archived_parity_audit import (
    ARMS,
    BUDGETS,
    ROOTS,
    audit_split,
    branch_metrics,
)


def test_fixed_prior_registration():
    assert ROOTS == tuple(range(6))
    assert ARMS == ("composed", "additive")
    assert BUDGETS == (120, 240, 480, 900)


def test_balanced_accuracy_and_majority_baseline():
    score = branch_metrics(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    )
    assert score["cases"] == 4
    assert score["correct"] == 2
    assert score["accuracy"] == 0.5
    assert score["truth_positives"] == 2
    assert score["majority_baseline_accuracy"] == 0.5
    assert score["balanced_accuracy"] == 0.5
    skewed = branch_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert skewed["majority_baseline_accuracy"] == 0.75
    assert skewed["accuracy"] == 0.75
    assert skewed["balanced_accuracy"] == 0.5


def test_rejects_empty_or_nonbinary_branch():
    with pytest.raises(AssertionError):
        branch_metrics([], [])
    with pytest.raises(AssertionError):
        branch_metrics([0, 2], [0, 1])


def synthetic_split():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    rows = [
        {
            "composed_label": label,
            "additive_label": label,
            "history_bits12": [selector] + [0] * 11,
        }
        for selector, label in zip([1, 1, 1, 1, 0, 0, 0, 0], labels)
    ]
    preds = [0, 0, 1, 1, 0, 1, 0, 1]
    logits = [[1.0, -1.0] if p == 0 else [-1.0, 1.0] for p in preds]
    truth = branch_metrics(labels, preds)
    old_receipt = {
        "cases": 8,
        "correct": truth["correct"],
        "accuracy": truth["accuracy"],
        "truth_positive_cases": truth["truth_positives"],
        "balanced_accuracy": truth["balanced_accuracy"],
        "class_predictions": preds,
        "native_logits": logits,
    }
    return rows, old_receipt


def test_archive_reconstructs_full_split_from_branches():
    rows, archived = synthetic_split()
    checked = audit_split(archived, rows, "composed", selector=0)
    assert checked["whole"]["correct"] == archived["correct"]
    assert checked["parity"]["cases"] == 4
    assert checked["alternate"]["cases"] == 4
    assert checked["parity"]["correct"] + checked["alternate"]["correct"] == archived["correct"]


def test_archive_rejects_logit_tampering():
    rows, archived = synthetic_split()
    archived["class_predictions"][0] = 1
    with pytest.raises(AssertionError, match="logits/predictions"):
        audit_split(archived, rows, "composed", selector=0)


def test_archive_only_uses_preexisting_calibration_roots():
    for index in ROOTS:
        world, _ = gym.calibration_world(index)
        assert world["canonical_program"][0] == "MUX"
        assert world["canonical_program"][1] == ["VAR", world["context_positions"][0]]
