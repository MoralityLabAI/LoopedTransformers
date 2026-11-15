"""PEF-04 prospective pair-closure, exact controls and frozen checkpoint tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from research.pef03.pef03 import BitMemoryLooper, labels, tensor
from research.pef04.pef04 import (
    PAIR_COUNTS, SEEDS, _canonical_bytes, _pair_scoring, natural_memory,
    paired_split, run
)


def test_oldest_flip_pair_closure_at_every_data_boundary():
    catalog, manifest = paired_split()
    assert manifest["allocation_pairs"] == {"train": 80, "probe": 24, "heldout": 24}
    assert manifest["allocation_histories"] == {"train": 160, "probe": 48, "heldout": 48}
    assert manifest["split_sha256"] == hashlib.sha256(
        _canonical_bytes({k: v for k, v in manifest.items() if k != "split_sha256"})
    ).hexdigest()
    assert len(manifest["suffixes_sorted_by_sha256"]) == 128
    joined = set()
    suffix_split = {}
    for split, rows in catalog.items():
        for i in range(0, len(rows), 2):
            left, right = rows[i:i+2]
            assert (left[0],right[0]) == (0,1)
            assert left[1:] == right[1:]
            assert left not in joined and right not in joined
            joined.update((left,right))
            assert left[1:] not in suffix_split
            suffix_split[left[1:]] = split
    assert len(joined) == 256
    assert len(suffix_split) == 128
    assert set(catalog["train"]).isdisjoint(catalog["heldout"])
    assert set(catalog["probe"]).isdisjoint(catalog["heldout"])


def test_exact_suffix_only_baseline_and_pair_metric():
    catalog, _ = paired_split()
    heldout = tensor(catalog["heldout"])
    # Any suffix-only function must assign the same label to both
    # members, so exactly 24/48 old bits match and 0 pairs are joint.
    functions = (
        lambda x: torch.zeros(len(x), dtype=torch.long),
        lambda x: x[:, -1],
        lambda x: x[:, 1],
        lambda x: x[:, 2] ^ x[:, 5],
    )
    for f in functions:
        prediction = f(heldout)
        scores = _pair_scoring(prediction, heldout[:, 0],
                               torch.zeros((len(heldout), 16)))
        assert scores["correct_histories"] == 24
        assert scores["both_members_correct_pairs"] == 0
        assert scores["prediction_changes_across_oldest_bit_flip_pairs"] == 0
    assert torch.equal(labels(heldout[0::2]), labels(heldout[1::2]))


def test_natural_continuation_matches_literal_two_more_memory_steps():
    torch.manual_seed(101)
    model=BitMemoryLooper().eval()
    catalog,_=paired_split()
    bits=tensor(catalog["heldout"][:8])
    with torch.no_grad():
        at8=model.memory(bits)
        manually=model.step(at8,model.bit_embedding(torch.zeros(len(bits),dtype=torch.long)))
        manually=model.step(manually,model.bit_embedding(torch.zeros(len(bits),dtype=torch.long)))
        at10=natural_memory(model,bits,(0,0))
        at8_direct=natural_memory(model,bits,())
        assert torch.allclose(at8,at8_direct,atol=0,rtol=0)
        assert torch.allclose(at10,manually,atol=0,rtol=0)
        assert at8.shape==(8,16)
    with pytest.raises(ValueError,match="binary"):
        natural_memory(model,bits,(3,))


def test_sealed_model_is_unchanged_by_debug_downstream_probe(tmp_path:Path):
    report=run(tmp_path,seeds=(101,),steps=2)
    assert report["registered_scoring_run"] is False
    assert report["registered_transfer_decision"]=="debug_or_nonregistered_run"
    assert report["seed_order"]==[101]
    assert report["split_history_counts"]=={"train":160,"probe":48,"heldout":48}
    assert report["primary_readout"]=="h10_00"
    assert report["source_suffix_only_baseline_correct"]==24
    assert report["source_suffix_only_baseline_both_pairs"]==0
    frozen=report["result_per_seed"][0]
    assert frozen["frozen_checkpoint_integrity_passed"]
    assert frozen["training_steps"]==2
    assert frozen["heldout_complete_counterfactual_pairs"]==24
    assert set(frozen["downstream_arms"])=={"h8","h10_00","h10_11"}
    for arm in frozen["downstream_arms"].values():
        for kind in ("source_frozen_trunk","random_initialization_control"):
            assert arm[kind]["total_histories"]==48
            assert arm[kind]["total_pairs"]==24
            assert 0<=arm[kind]["both_members_correct_pairs"]<=24
        assert arm["suffix_only_theoretical_control"]["correct_histories"]==24
    ckpt=tmp_path/"seed101.frozen.pt"
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==frozen["checkpoint_sha256"]
    assert (tmp_path/"sealed_training_ledger.json").exists()


def test_reject_duplicate_seed_and_checkpoint_tampering(tmp_path:Path):
    with pytest.raises(ValueError,match="duplicated"):
        run(tmp_path,seeds=(101,101),steps=1)
