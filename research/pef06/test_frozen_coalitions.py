"""PEF-06 finite theorem and neural pair-synergy regression tests."""
from __future__ import annotations

import torch
import pytest

from research.pef03.pef03 import BitMemoryLooper
from research.pef06.frozen_coalitions import (
    ALL_MASKS, latch, reference_latch_certificate, source_histories,
    finite_difference_audit, forward_natural_01
)


def test_exact_local_coalition_latch_contrast():
    evidence=reference_latch_certificate()
    assert evidence["zero_local_single_sensitivity"]==0
    assert evidence["zero_local_block_sensitivity"]==3
    assert evidence["uniform_source_sensitivity_total_over_64_states"]==192
    assert evidence["bsc_p_one_fourth_zero_activation_numerator_over_4096"]==676
    assert evidence["bsc_p_one_fourth_zero_activation_exact"]=="169/1024"
    assert len(evidence["zero_double_flip_outputs_by_pair"])==15
    for i in range(6):
        assert latch(1<<i)==0
    for i,j in ((0,1),(2,3),(4,5)):
        assert latch((1<<i)|(1<<j))==1
    for i,j in ((0,2),(1,3),(2,4)):
        assert latch((1<<i)|(1<<j))==0


def test_independent_tensor_oracle_detects_pair_only_task_activation():
    torch.set_num_threads(1)
    vals=torch.tensor([latch(w) for w in range(256)],dtype=torch.float32)
    states=torch.zeros((256,16))
    states[:,0]=vals
    logits=torch.zeros((256,4))
    logits[:,0]=1-vals
    logits[:,1]=vals
    metrics=finite_difference_audit(states,logits)
    for pair in ((0,1),(2,3),(4,5)):
        row=next(r for r in metrics["two_bit_interventions"]
                 if r["pair_positions"]==list(pair))
        assert row["pair_only_task_class_changes"]>0
        assert row["mean_second_difference_hidden_l2"]>0
    assert metrics["number_of_pair_only_class_changes_across_all_15_pairs"]>0
    assert len(metrics["single_bit_interventions"])==6
    assert metrics["all_supervised_source_task_bits_held_constant"]


def test_every_history_index_is_unique_and_old_input_flips_preserve_task():
    source=source_histories()
    assert source.shape==(ALL_MASKS,8)
    assert len({tuple(row) for row in source.tolist()})==256
    for word in range(256):
        assert source[word].tolist()==[(word>>j)&1 for j in range(8)]
    for i in range(6):
        for word in range(256):
            assert torch.equal(source[word,6:],source[word^(1<<i),6:])


def test_random_initialized_attention_frozen_intervention_smoke():
    torch.set_num_threads(1)
    torch.manual_seed(157)
    model=BitMemoryLooper().eval()
    from research.pef03.pef03 import hash_weights
    before=hash_weights(model)
    with torch.no_grad():
        hidden,logits=forward_natural_01(model,source_histories())
        results=finite_difference_audit(hidden,logits)
    assert hidden.shape==(256,16)
    assert logits.shape==(256,4)
    assert results["full_finite_cube_states"]==256
    assert len(results["two_bit_interventions"])==15
    assert all(v["mean_hidden_l2"]>=0
               for v in results["single_bit_interventions"].values())
    assert hash_weights(model)==before


def test_guard_incomplete_or_invalid_inputs():
    with pytest.raises(ValueError):
        latch(-1)
    with pytest.raises(ValueError):
        latch(256)
    with pytest.raises(ValueError):
        finite_difference_audit(torch.zeros(1,16),torch.zeros(1,4))
