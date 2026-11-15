"""Integrity, causal, and differentiable-loop tests for PEF-03."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import pytest
import torch

from research.pef03.pef03 import (
    BATCH, D, L, BitMemoryLooper, hash_weights, labels, load_frozen,
    markov_stream, ordered_jacobians, run, split_catalog, split_history,
    tensor, train_frozen,
)


def test_sealed_source_partition():
    groups = split_catalog()
    assert {k: len(v) for k, v in groups.items()} == {
        "train": 148, "probe": 56, "heldout": 52
    }
    assert set(groups["train"]).isdisjoint(groups["probe"])
    assert set(groups["train"]).isdisjoint(groups["heldout"])
    assert set(groups["probe"]).isdisjoint(groups["heldout"])
    assert split_history(groups["heldout"][0]) == "heldout"


def test_memory_token_causally_receives_transient_bit():
    torch.manual_seed(17)
    model = BitMemoryLooper().eval()
    a = tensor([[0]*L, [1]+[0]*(L-1)])
    out = model.memory(a)
    assert out.shape == (2, D)
    assert not torch.allclose(out[0], out[1])
    # Positive isolation control: equality of the two ephemeral bit
    # embeddings makes every history produce the same recurrent state.
    with torch.no_grad():
        model.bit_embedding.weight[1].copy_(model.bit_embedding.weight[0])
        out = model.memory(a)
    assert torch.allclose(out[0], out[1], atol=1e-7)


def test_checkpoint_frozen_and_seed_guard(tmp_path: Path):
    torch.set_num_threads(2)
    checkpoint = tmp_path / "seed17.frozen.pt"
    receipt = train_frozen(17, checkpoint, steps=2)
    expected_sha = receipt["checkpoint_sha256"]
    model = load_frozen(checkpoint, 17)
    assert receipt["frozen_weights_sha256"] == hash_weights(model)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_sha
    with pytest.raises(AssertionError, match="seed"):
        load_frozen(checkpoint, 29)
    assert receipt["optimizer_steps"] == 2


def test_wrong_history_and_storage_rejected():
    model = BitMemoryLooper()
    with pytest.raises(ValueError, match="eight"):
        model.memory(torch.zeros((1,L-1), dtype=torch.long))
    with pytest.raises(ValueError, match="storage"):
        model.memory(torch.zeros((1,L), dtype=torch.long), storage="int8")
    bits = torch.zeros((2,L), dtype=torch.long)
    for name in ("fp32","fp16","bf16"):
        x = model.memory(bits, storage=name)
        assert torch.isfinite(x).all()
        assert x.shape == (2,D)


def test_jacobian_chain_recomputes_direct_full_jvp():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = BitMemoryLooper().eval()
    history = split_catalog()["heldout"][0]
    report = ordered_jacobians(model, history)
    assert len(report["position_derivatives"]) == L
    assert max(r["relative_jvp_chain_error"]
               for r in report["position_derivatives"]) < 0.05
    for row in report["position_derivatives"]:
        assert row["product_of_step_operator_norms_upper_bound"] + 1e-5 >= (
            row["ordered_suffix_top_singular"]
        )


def test_markov_source_mechanism_matches_registered_prior():
    a = markov_stream(4096, 9101)
    # Oldest history bit correlated with newest without any state model.
    empirical = float((a[:,0] == a[:,6]).float().mean())
    expected = (1 + 0.9**6) / 2
    assert abs(empirical - expected) < 0.04


def test_short_training_scratch_smoke_without_registered_test_changes(tmp_path):
    torch.set_num_threads(2)
    result = run(17, tmp_path, steps=3, jacobian=False)
    assert result["training"]["optimizer_steps"] == 3
    assert not result["train_steps_follow_preregistration"]
    assert result["split_sizes"] == {"train":148,"probe":56,"heldout":52}
    assert result["frozen_checkpoint_integrity_passed"]
    assert len(result["causal_discrete_bit_interventions"]["fp32"]) == L
    assert result["ordered_local_jacobians"] is None
    assert result["markov_prior_diagnostic"]["source_shift_is_not_history_disjoint"]
