"""Prospective NEC-04F source-only original-world auxiliary XOR gym tests."""
from __future__ import annotations

import torch
import pytest

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04f_oracles import (
    ARMS,ROOTS,STAGES,STEPS,SOURCE,VALIDATION,
    lock,seed,xor_prefix,shuffle_map,input_history,
)
from mechinterp.neuron_communities.nec04f_independent_oracles import (
    independently_shuffled_labels,independently_prefix,
)
from mechinterp.neuron_communities.nec04f_native import (
    AuxLooper,metrics,branch_metrics,
)
from mechinterp.neuron_communities.nec04f_audit import independent_native_sampler
from mechinterp.neuron_communities.nec04f_aggregate import (
    DELTA_MIN,POSITIVE_MIN,PREFIX_ACC_MIN,ALTERNATE_FLOOR,
)
from research.pef07.pef07 import Looper,as_tensor,model_digest


def test_immutable_worlds_and_native_auxiliary_arms():
    cfg,receipt=lock()
    assert len(receipt)==64
    assert ROOTS==(12,13,14,15,16,17)
    assert ARMS==("aux_shuffled","aux_prefix","vanilla")
    assert STAGES==(2,3,4,5)
    assert STEPS==(240,900)
    assert (SOURCE,VALIDATION)==(2560,768)
    assert cfg["continuous_optimizer_steps"]==900
    assert cfg["batch_size"]==64
    assert cfg["auxiliary_weight"]==.5
    assert cfg["new_development_world_roots_materialized"]==0
    assert cfg["new_confirmation_world_roots_materialized"]==0
    assert cfg["original_reserved_contexts_scored"]==0
    assert cfg["FSI_01_gate"]=="REDESIGN_OR_PILOT_ONLY"
    assert (DELTA_MIN,POSITIVE_MIN,PREFIX_ACC_MIN,ALTERNATE_FLOOR)==(.1,4,.6,-.05)


def test_root_specific_pair_seeds_and_untouched_range():
    assert len({seed(i) for i in ROOTS})==6
    with pytest.raises(PermissionError):
        seed(11)
    with pytest.raises(PermissionError):
        seed(18)


def test_original_circuit_targets_and_stages():
    for i in ROOTS:
        world,_=gym.calibration_world(i)
        assert world["canonical_program"][:2]==[
            "MUX",["VAR",world["context_positions"][0]]
        ]
        focal=world["focal_positions"]
        context=world["context_positions"]
        visits=[sorted(focal)[k-1]+1 for k in STAGES]
        assert visits==sorted(set(visits))
        source,_=gym.contexts_for_world(i)
        for mask in (0,1,11,31):
            bits=input_history(focal,context,source[0],mask)
            result=xor_prefix(bits,focal)
            assert set(result)=={"2","3","4","5"}
            for k in STAGES:
                assert result[str(k)]==independently_prefix(bits,focal,k)
            assert oracle(world["canonical_program"],bits) in (True,False)


def test_balanced_sha_source_label_controls_independent():
    for i in ROOTS:
        for group in (0,21,79):
            for k in STAGES:
                expected=shuffle_map(i,group,k)
                independently=independently_shuffled_labels(i,group,k)
                assert expected==independently
                assert sum(expected)==16
                assert len(expected)==32
    assert shuffle_map(12,0,4)!=shuffle_map(12,1,4)
    with pytest.raises(PermissionError):
        shuffle_map(12,80,4)


def test_all_true_prefix_contexts_balance_16_of_32():
    for i in ROOTS:
        w,_=gym.calibration_world(i)
        contexts,_=gym.contexts_for_world(i)
        for context in contexts[:2]:
            for k in STAGES:
                target=[
                    independently_prefix(
                        input_history(w["focal_positions"],w["context_positions"],
                                      context,mask),
                        w["focal_positions"],k)
                    for mask in range(32)
                ]
                assert sum(target)==16


def test_auxiliary_net_is_original_looper_plus_shared_single_2logit_head():
    torch.manual_seed(1065)
    model=AuxLooper().eval()
    assert isinstance(model.core,Looper)
    assert model.aux_head.in_features==32
    assert model.aux_head.out_features==2
    bits=as_tensor([[0]*12,[1]*12,[1,0]*6,[0,1]*6])
    with torch.no_grad():
        logits,extra=model(bits,(2,4,6,8))
        original=model.core.logits(bits)
    assert logits.shape==(4,2)
    assert extra.shape==(4,4,2)
    assert torch.allclose(logits,original,atol=3e-4,rtol=0.)


def test_equal_initial_weights_and_matched_native_sampler_sha():
    index=12
    torch.manual_seed(seed(index))
    first=AuxLooper()
    torch.manual_seed(seed(index))
    second=AuxLooper()
    assert model_digest(first)==model_digest(second)
    order=independent_native_sampler(seed(index),STEPS)
    assert set(order)=={240,900}
    assert order[240]!=order[900]
    assert order==independent_native_sampler(seed(index),STEPS)


def test_gradients_of_prefix_auxiliary_reach_original_recurrent_core():
    torch.manual_seed(7547)
    model=AuxLooper()
    x=as_tensor([[int((j>>bit)&1) for bit in range(12)]
                  for j in range(32)])
    visits=(2,4,6,8)
    final,aux=model(x,visits)
    targets=(x[:,0]+x[:,1])%2
    loss=(torch.nn.functional.cross_entropy(final,targets)
          +.5*torch.nn.functional.cross_entropy(aux[:,0,:],targets))
    loss.backward()
    assert model.aux_head.weight.grad is not None
    assert model.core.bit_embedding.weight.grad is not None
    assert float(model.core.bit_embedding.weight.grad.abs().sum())>0.


def test_predictions_scored_separately_per_original_mux_branch():
    logits=[[1.,-1.],[-1.,1.],[1.,-1.],[-1.,1.]]
    labels=[0,1,1,0]
    rows=[{"mux_selector":v} for v in (1,1,0,0)]
    scored=metrics(logits,labels)
    assert scored["cases"]==4 and scored["correct"]==2
    branches=branch_metrics(logits,labels,rows)
    assert branches["parity"]["correct"]==2
    assert branches["alternate"]["correct"]==0
    assert branches["parity"]["cases"]==branches["alternate"]["cases"]==2


def test_missing_original_context_selector_fails_closed():
    with pytest.raises(AssertionError,match="missing"):
        branch_metrics([[1.,-1.],[0.,1.]],[0,1],
                       [{"mux_selector":1},{"mux_selector":1}])


def test_original_contexts_do_not_overlap_reserved_contexts():
    for index in ROOTS:
        source,parts=gym.contexts_for_world(index)
        assert len(source)==80
        assert len(parts["development"])==len(parts["reserved"])==24
        assert not set(source)&set(parts["development"])
        assert not set(source)&set(parts["reserved"])
        assert not set(parts["development"])&set(parts["reserved"])
