"""GG-02 independent oracle, locked panel and causal-subspace controls."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
import math

import pytest

torch=pytest.importorskip("torch")
from research_gym.interp.goal_guarding_causal import (
    PRIORS,RATIOS,EvaluationBeliefLooper,exact_action,
)
from research_gym.interp.goal_guarding_subspace import (
    CUTOFF,RANKS,generate_disjoint_panels,cache_states,
    orthogonal_basis,projected_patch,principal_components,
    train_subspace,score_basis,measure_simple_patch,
    score_selected_rank,
)


def _parameters(context):
    choices=tuple(product(PRIORS,RATIOS))
    return min(choices,key=lambda x:
        abs(float(context[0])-math.log(float(x[0]/(1-x[0]))))
        +abs(float(context[1])-math.log(float(x[1]))))


def test_frozen_disjoint_panels_and_independent_exact_action_oracle():
    sizes=(("calibration",64,31001),("validation",32,31002),("locked",64,31003))
    a=generate_disjoint_panels(sizes=sizes)
    b=generate_disjoint_panels(sizes=sizes)
    assert {k:len(v) for k,v in a.items()}=={
        "calibration":64,"validation":32,"locked":64}
    all_ids=[]
    for name,panel in a.items():
        assert panel.ids==b[name].ids
        assert panel.receiver.shape==(len(panel),6)
        assert panel.donor.shape==(len(panel),6)
        assert torch.equal(panel.hybrid[:,:2],panel.donor[:,:2])
        assert torch.equal(panel.hybrid[:,2:],panel.receiver[:,2:])
        all_ids.extend(panel.ids)
        for index in range(len(panel)):
            p,ratio=_parameters(panel.context[index])
            receiver=tuple(int(b) for b in panel.receiver[index])
            donor=tuple(int(b) for b in panel.donor[index])
            hybrid=tuple(int(b) for b in panel.hybrid[index])
            assert int(panel.oracle[index])==exact_action(p,ratio,hybrid)
            assert int(panel.receiver_oracle[index])==exact_action(p,ratio,receiver)
            assert int(panel.donor_oracle[index])==exact_action(p,ratio,donor)
            assert int(panel.oracle[index])!=int(panel.receiver_oracle[index])
            assert int(panel.oracle[index])!=int(panel.donor_oracle[index])
    assert len(all_ids)==len(set(all_ids))
    with pytest.raises(ValueError,match="distinct"):
        generate_disjoint_panels(sizes=(("x",1,1),("x",1,2)))
    with pytest.raises(ValueError,match="cutoff"):
        generate_disjoint_panels(visits=2,cutoff=2)


def test_exact_orthogonal_projector_and_donor_equals_receiver_sham():
    torch.manual_seed(41)
    receiver=torch.randn(9,32)
    donor=torch.randn(9,32)
    for rank in RANKS:
        basis=orthogonal_basis(torch.randn(32,rank))
        assert basis.shape==(32,rank)
        assert torch.allclose(basis.T@basis,torch.eye(rank),atol=1e-5)
        patch=projected_patch(receiver,donor,basis)
        assert torch.allclose((patch-receiver)@basis,
                              (donor-receiver)@basis,atol=1e-5)
        assert torch.allclose(
            projected_patch(receiver,receiver,basis),receiver)
    full=orthogonal_basis(torch.randn(32,32))
    assert torch.allclose(projected_patch(receiver,donor,full),donor,atol=1e-5)
    with pytest.raises(ValueError,match="dimensions"):
        projected_patch(receiver,donor,torch.randn(31,4))
    with pytest.raises(ValueError,match="width"):
        orthogonal_basis(torch.randn(3,4))


def test_native_tied_decoder_state_identity_and_pca():
    torch.manual_seed(71)
    torch.set_num_threads(1)
    model=EvaluationBeliefLooper().eval()
    panel=generate_disjoint_panels(sizes=(("calibration",40,317),))["calibration"]
    cache=cache_states(model,panel)
    assert cache.receiver_state.shape==(40,32)
    assert cache.donor_state.shape==(40,32)
    assert cache.receiver_logits.shape==(40,6,2)
    assert cache.hybrid_logits.shape==(40,6,2)
    for rank in RANKS:
        pca=principal_components(cache,rank)
        assert pca.shape==(32,rank)
        assert torch.allclose(pca.T@pca,torch.eye(rank),atol=1e-5)
    assert 0<=cache.eligible_count<=40


def test_calibration_only_causal_fit_freezes_native_model_weights():
    torch.manual_seed(84)
    torch.set_num_threads(1)
    model=EvaluationBeliefLooper().eval()
    panel=generate_disjoint_panels(sizes=(("calibration",32,318),))["calibration"]
    cache=cache_states(model,panel)
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    for rank in (2,4,8):
        fitted=train_subspace(model,cache,rank,seed=11,steps=4,batch=8)
        assert fitted.rank==rank and fitted.seed==11
        assert [t for t,_ in fitted.loss_trace]==[1,4]
        assert torch.allclose(fitted.basis.T@fitted.basis,
                              torch.eye(rank),atol=1e-5)
    for key,value in before.items():
        assert torch.equal(value,model.state_dict()[key])
    with pytest.raises(ValueError,match="registered"):
        train_subspace(model,cache,1,seed=11,steps=0)


def test_locked_score_matched_random_and_sham_controls():
    torch.manual_seed(45)
    torch.set_num_threads(1)
    model=EvaluationBeliefLooper().eval()
    panel=generate_disjoint_panels(sizes=(("locked",32,350),))["locked"]
    cache=cache_states(model,panel)
    # Exercise the scorer independently of an untrained model's accuracy;
    # the actual pilot requires correctness on both receiver and hybrid.
    cache=replace(cache,eligible=torch.ones(len(panel),dtype=torch.bool))
    basis=orthogonal_basis(torch.randn(32,2))
    ceiling=float(torch.quantile(torch.cat((
        cache.receiver_state.norm(dim=-1),
        cache.donor_state.norm(dim=-1))),.99))
    result=score_basis(model,cache,basis,ceiling,random_seed=99,
                        neighbors=torch.cat((cache.receiver_state,
                                             cache.donor_state)))
    assert result["status"]=="MEASURED" and result["eligible"]==32
    assert len(result["random_controls"])==8
    assert 0<=result["hit"]<=1
    assert 0<=result["inside_clean_norm_ceiling"]<=1
    assert -1<=result["nearest_clean_cosine"]<=1
    assert result["mean_patch_norm"]>=0
    assert 0<=measure_simple_patch(model,cache)["hit"]<=1
    with pytest.raises(ValueError,match="exactly eight"):
        score_basis(model,cache,basis,ceiling,random_seed=99,
                    random_controls=7)


def test_validation_only_rank_selection_and_fail_closed():
    records={rank:{"status":"MEASURED","hit":Q(1,2),
                   "margin_lift":float(rank)} for rank in RANKS}
    assert score_selected_rank(records)==2
    records[4]["hit"]=Q(3,4)
    assert score_selected_rank(records)==4
    del records[8]
    with pytest.raises(ValueError,match="all ranks"):
        score_selected_rank(records)
    records[8]={"status":"NO_ELIGIBLE_PAIRS"}
    with pytest.raises(ValueError,match="eligible"):
        score_selected_rank(records)
