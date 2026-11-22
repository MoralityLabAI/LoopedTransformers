"""NEC-04E source-only native recurrent memory/readout preflight tests."""
from __future__ import annotations

import torch
import pytest

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04e_preflight import lock
from mechinterp.neuron_communities.nec04e_hidden_probe import (
    REPRESENTATIONS, VISITS, ALPHA, STDMIN,
    eligible_targets, target_labels, class_score, shuffle_seed,
    fit_source_only_ridge, recurrent_states,
)
from mechinterp.neuron_communities.nec04e_independent_probe_audit import (
    own_targets, source_shuffle, independent_fit, independent_score,
)
from mechinterp.neuron_communities.nec04e_aggregate import (
    summary, root_snapshot,
)
from research.pef07.pef07 import Looper, as_tensor


def test_preinference_fixed_original_world_lock():
    c, digest = lock()
    assert len(digest) == 64
    assert c["pilot_calibration_world_root_indices"] == list(range(6, 12))
    assert c["existing_optimizer_steps"] == [240, 900]
    assert c["sequential_recurrent_visits"] == 12
    assert c["original_reserved_context_groups_scored"] == 0
    assert c["new_task_model_optimizer_updates"] == 0
    assert c["new_development_WORLD_roots_instantiated"] == 0
    assert c["new_confirmation_WORLD_roots_instantiated"] == 0
    assert c["FSI_01_gate"] == "REDESIGN_OR_PILOT_ONLY"
    assert ALPHA == 1. and STDMIN == 1e-4
    assert len(REPRESENTATIONS) == 4 and VISITS == tuple(range(1, 13))


def test_target_visits_only_observed_and_future_focal_inputs():
    focal = [0, 2, 4, 6, 8]
    assert eligible_targets(1, focal) == (
        "prefix1", "future_unseen", "full_parity_teacher"
    )
    assert eligible_targets(3, focal) == (
        "prefix1", "prefix2", "prefix_current", "future_unseen",
        "full_parity_teacher",
    )
    assert eligible_targets(9, focal) == (
        "prefix1", "prefix2", "prefix_current", "full_parity_teacher",
    )
    assert "future_unseen" not in eligible_targets(12, focal)
    with pytest.raises(ValueError):
        eligible_targets(13, focal)


def test_parity_targets_balance_per_complete_original_context():
    focal = [0, 2, 4, 6, 8]
    rows=[]
    for mask in range(32):
        bits=[0]*12
        for j,p in enumerate(focal):
            bits[p]=(mask>>j)&1
        rows.append(bits)
    x=as_tensor(rows)
    final=torch.tensor([mask.bit_count()%2 for mask in range(32)])
    for visit in (1,3,5,7,9,12):
        generated=own_targets([
            {"history_bits12":rows[mask],"composed_label":final[mask].item()}
            for mask in range(32)
        ],focal,visit)
        for task in eligible_targets(visit,focal):
            label=target_labels(x,final,focal,visit,task)
            assert label.tolist()==generated[task]
            assert int(label.sum())==16
    with pytest.raises(ValueError):
        target_labels(x,final,focal,1,"prefix2")


def test_source_only_ridge_positive_onebit_control():
    source_bits=torch.tensor([[0.], [1.]]*64)
    source_labels=source_bits[:,0].long()
    val_bits=torch.tensor([[1.], [0.]]*20)
    val_labels=val_bits[:,0].long()
    metric=fit_source_only_ridge(
        source_bits,source_labels,val_bits,val_labels,source_labels
    )
    assert metric["owned_validation_metrics"]["accuracy"] == 1.
    assert metric["source_true_label_metrics"]["accuracy"] == 1.
    assert metric["owned_validation_metrics"]["majority_class_baseline_accuracy"] == .5


def test_ridge_does_not_fit_validation_labels():
    source=torch.tensor([[0.], [1.]]*64)
    labels=source[:,0].long()
    validation=torch.tensor([[1.], [0.]]*32)
    yval=validation[:,0].long()
    a=fit_source_only_ridge(source,labels,validation,yval,labels)
    b=fit_source_only_ridge(source,labels,validation,1-yval,labels)
    assert a["source_only_ridge_coefficient_sha256"]==b["source_only_ridge_coefficient_sha256"]
    assert a["owned_validation_predictions_sha256"]==b["owned_validation_predictions_sha256"]
    assert a["owned_validation_metrics"]["accuracy"]==1.
    assert b["owned_validation_metrics"]["accuracy"]==0.


def test_source_label_shuffle_fixed_and_no_new_task_training():
    first=shuffle_seed(6,8,"prefix_current")
    assert first==shuffle_seed(6,8,"prefix_current")
    assert first!=shuffle_seed(7,8,"prefix_current")
    indices=source_shuffle(6,8,"prefix_current",64)
    assert torch.equal(indices,source_shuffle(6,8,"prefix_current",64))
    assert set(indices.tolist())==set(range(64))


def test_independent_ridge_replays_original_source_only_scores():
    source=torch.tensor([[float((i>>0)&1), float((i>>1)&1)] for i in range(256)])
    y=torch.tensor([(i>>0)&1 for i in range(256)],dtype=torch.long)
    val=source[50:178]
    vy=y[50:178]
    a=fit_source_only_ridge(source,y,val,vy,y)
    b=independent_fit(source,y,val,vy,y)
    assert a["owned_validation_metrics"]["accuracy"] == b["owned_validation_metrics"]["accuracy"]
    assert a["source_fit_label_metrics"]["accuracy"] == b["source_fit_label_metrics"]["accuracy"]
    assert a["owned_validation_predictions_sha256"] == b["owned_validation_predictions_sha256"]
    assert independent_score(y,y)["accuracy"] == 1.


def test_native_hidden_visit_twelve_reconstructs_original_head():
    torch.manual_seed(93571)
    model=Looper().eval()
    bits=as_tensor([[0]*12,[1]*12,[0,1]*6,[1,0]*6])
    states, logits, diff=recurrent_states(model,bits)
    assert len(states)==12
    assert all(x.shape==(4,32) for x in states)
    assert diff < 3e-4
    with torch.no_grad():
        assert torch.allclose(logits,model.logits(bits),atol=3e-4,rtol=0.)


def test_future_bit_cannot_be_reconstructed_from_raw_visible_bits():
    # For every observed-bit prefix there are equally many full original
    # focal masks carrying opposite unseen-focal XOR labels.
    focal=[0,2,4,6,8]
    bits=[]
    for mask in range(32):
        row=[0]*12
        for i,p in enumerate(focal):
            row[p]=(mask>>i)&1
        bits.append(row)
    x=as_tensor(bits)
    original=torch.tensor([r[0]^r[2]^r[4]^r[6]^r[8] for r in bits])
    future=target_labels(x,original,focal,3,"future_unseen")
    for observed in range(4):
        selected=(x[:,0]==(observed&1))&(x[:,2]==((observed>>1)&1))
        assert int(future[selected].sum()) * 2 == int(selected.sum())


def test_root_unit_descriptive_statistics():
    record=summary([.50,.75,1.])
    assert record["root_count"]==3
    assert record["mean"]==pytest.approx(.75)
    assert record["root_standard_error_descriptive"]>0
