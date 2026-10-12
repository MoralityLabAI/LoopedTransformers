from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from research_gym.interp.association import auroc_columns, hash_bit, select_then_confirm


def test_auroc_matches_sklearn_with_ties() -> None:
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, 400)
    # Mostly zeros, as TopK codes are, so ties dominate.
    activations = np.where(generator.random((400, 5)) < 0.7, 0.0, generator.random((400, 5)))
    activations[:, 2] += labels * 0.5
    ours = auroc_columns(torch.tensor(activations, dtype=torch.float32), labels).numpy()
    expected = [roc_auc_score(labels, activations[:, column]) for column in range(5)]
    np.testing.assert_allclose(ours, expected, atol=1e-6)


def test_select_then_confirm_finds_a_planted_feature_and_not_noise() -> None:
    generator = np.random.default_rng(1)

    def codes(labels):
        values = torch.tensor(np.where(generator.random((len(labels), 3, 50)) < 0.8, 0.0,
                                       generator.random((len(labels), 3, 50))), dtype=torch.float32)
        values[:, 1, 7] = torch.tensor(labels * 2.0 + generator.random(len(labels)) * 0.5, dtype=torch.float32)
        return values

    select_labels = generator.integers(0, 2, 600)
    report_labels = generator.integers(0, 2, 600)
    result = select_then_confirm(codes(select_labels), select_labels, codes(report_labels), report_labels, rounds=200)
    assert result["found"] is False  # 200 rounds cannot give p < 0.001
    result = select_then_confirm(codes(select_labels), select_labels, codes(report_labels), report_labels)
    assert result["found"] and result["site"] == 2 and result["feature"] == 7 and result["report_auroc"] > 0.95

    noise_select = generator.integers(0, 2, 600)
    noise_report = generator.integers(0, 2, 600)
    null = select_then_confirm(codes(select_labels), noise_select, codes(report_labels), noise_report)
    assert not null["found"] and abs(null["report_auroc"] - 0.5) < 0.2


def test_hash_bit_is_deterministic_and_binary() -> None:
    bits = hash_bit(["a", "b", "a", "c"])
    assert set(bits.tolist()) <= {0, 1} and bits[0] == bits[2]
