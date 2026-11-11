"""GG-03: independent exact Bayes controls and causal-specificity assay guards."""
from __future__ import annotations

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
    CUTOFF,generate_disjoint_panels,cache_states,
    orthogonal_basis,projected_patch,
)
from research_gym.interp.goal_guarding_specificity import (
    GG03_SIZES,GG03_INVARIANT_SIZE,GG03_INVARIANT_SEED,
    invariant_order_panel,legal_prefix_support,support_metrics,
    output_gradient_patch,patched_metrics,run_gg03,
)


def _parameters(context):
    candidates=tuple(product(PRIORS,RATIOS))
    return min(candidates,key=lambda x:
        abs(float(context[0])-math.log(float(x[0]/(1-x[0]))))
        +abs(float(context[1])-math.log(float(x[1]))))


def test_new_heldout_counterfactual_ids_remain_disjoint_from_all_gg02():
    a=generate_disjoint_panels(sizes=GG03_SIZES)
    b=generate_disjoint_panels(sizes=GG03_SIZES)
    assert tuple(a)==("calibration","validation","locked","gg03_reporting")
    assert [len(a[key]) for key in a]==[256,128,256,256]
    observed=[]
    for key in a:
        p=a[key]
        assert p.ids==b[key].ids
        observed.extend(p.ids)
        if key=="gg03_reporting":
            for i in range(len(p)):
                prior,ratio=_parameters(p.context[i])
                receiver=tuple(map(int,p.receiver[i]))
                donor=tuple(map(int,p.donor[i]))
                hybrid=tuple(map(int,p.hybrid[i]))
                assert hybrid==donor[:CUTOFF]+receiver[CUTOFF:]
                assert int(p.oracle[i])==exact_action(prior,ratio,hybrid)
                assert int(p.oracle[i])!=exact_action(prior,ratio,receiver)
                assert int(p.oracle[i])!=exact_action(prior,ratio,donor)
    assert len(observed)==len(set(observed))


def test_equal_statistic_order_swaps_keep_exact_bayes_label():
    panel=invariant_order_panel()
    replay=invariant_order_panel(GG03_INVARIANT_SIZE,GG03_INVARIANT_SEED)
    assert len(panel)==len(replay)==256
    assert panel.ids==replay.ids
    assert len(set(panel.ids))==256
    for i in range(len(panel)):
        p,ratio=_parameters(panel.context[i])
        receiver=tuple(map(int,panel.receiver[i]))
        donor=tuple(map(int,panel.donor[i]))
        assert receiver[:CUTOFF] in ((0,1),(1,0))
        assert donor[:CUTOFF]==tuple(reversed(receiver[:CUTOFF]))
        assert donor[CUTOFF:]==receiver[CUTOFF:]
        assert int(panel.oracle[i])==exact_action(p,ratio,receiver)
        assert int(panel.oracle[i])==exact_action(p,ratio,donor)
        assert int(panel.receiver_oracle[i])==int(panel.donor_oracle[i])
    with pytest.raises(ValueError,match="1..288"):
        invariant_order_panel(289)
    with pytest.raises(ValueError,match="1..288"):
        invariant_order_panel(True)


def _model(seed=71):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return EvaluationBeliefLooper().eval()


def test_exact_legal_two_cue_state_support_and_full_state_identity():
    model=_model()
    legal=legal_prefix_support(model)
    assert legal.states.shape==(9,4,32)
    assert legal.contexts.shape==(9,2)
    assert not legal.degenerate
    assert (legal.minimum_positive_separation>0).all()
    panel=generate_disjoint_panels(
        sizes=(("small",24,34701),))["small"]
    cache=cache_states(model,panel)
    assert cache.receiver_state.shape==(24,32)
    assert cache.donor_state.shape==(24,32)
    sham=support_metrics(cache.receiver_state,panel.context,legal)
    full=support_metrics(cache.donor_state,panel.context,legal)
    assert sham["fraction_exact_state"]==1.0
    assert full["fraction_exact_state"]==1.0
    assert sham["max_raw_distance"]<=1e-4
    assert full["max_raw_distance"]<=1e-4
    halfway=(cache.receiver_state+cache.donor_state)/2
    intermediate=support_metrics(halfway,panel.context,legal)
    assert intermediate["fraction_within_tenth_gap"]<=1
    assert intermediate["p95_scaled_distance"]>=0
    with pytest.raises(ValueError,match="unregistered context"):
        support_metrics(halfway,panel.context+2,legal)
    with pytest.raises(ValueError,match="legal-state width"):
        support_metrics(halfway[:,:31],panel.context,legal)


def test_output_gradient_is_label_blind_and_locally_sensitive():
    model=_model(83)
    panel=generate_disjoint_panels(
        sizes=(("small",16,34702),))["small"]
    cache=cache_states(model,panel)
    patched=output_gradient_patch(model,cache)
    assert patched.shape==cache.receiver_state.shape
    assert torch.isfinite(patched).all()
    # The comparator is determined by the frozen receiver suffix + weights.
    # Change exact oracle labels and hybrid outputs; its patch must be identical.
    altered_panel=replace(panel,
        oracle=1-panel.oracle,
        receiver_oracle=1-panel.receiver_oracle,
        donor_oracle=1-panel.donor_oracle)
    altered=replace(cache,panel=altered_panel,
                    hybrid_logits=-cache.hybrid_logits)
    assert torch.equal(patched,output_gradient_patch(model,altered))
    # A target-free readout margin derivative predicts a local perturbation.
    h=cache.receiver_state.detach().clone().requires_grad_(True)
    logits=model(panel.receiver,panel.context,patch_visit=CUTOFF,
                 patch_state=h)[0][:,-1,:]
    margin=(logits[:,1]-logits[:,0])
    derivative=torch.autograd.grad(margin.sum(),h)[0]
    direction=derivative/derivative.norm(dim=-1,keepdim=True).clamp_min(1e-10)
    with torch.no_grad():
        epsilon=1e-3
        h_perturbed=h.detach()+epsilon*direction
        new=model(panel.receiver,panel.context,patch_visit=CUTOFF,
                  patch_state=h_perturbed)[0][:,-1,:]
        finite=((new[:,1]-new[:,0])-margin.detach())/epsilon
        expected=(derivative*direction).sum(-1)
    assert torch.allclose(finite,expected,atol=5e-2,rtol=5e-2)


def test_legal_geometry_distinguishes_support_from_arbitrary_perturbation():
    model=_model(121)
    legal=legal_prefix_support(model)
    states=legal.states.reshape(36,32)
    contexts=legal.contexts.repeat_interleave(4,dim=0)
    assert support_metrics(states,contexts,legal)["fraction_exact_state"]==1.0
    noisy=states+20
    geometry=support_metrics(noisy,contexts,legal)
    assert geometry["fraction_exact_state"]==0
    assert geometry["fraction_within_tenth_gap"]==0
    assert geometry["median_scaled_distance"]>1
    with pytest.raises(ValueError,match="patched states"):
        support_metrics(states,contexts[:,:1],legal)


def test_patch_metric_endpoints_and_invariant_controls():
    model=_model(123)
    legal=legal_prefix_support(model)
    for panel in (
        generate_disjoint_panels(sizes=(("test",24,34703),))["test"],
        invariant_order_panel(24,34704),
    ):
        cache=cache_states(model,panel)
        basis=orthogonal_basis(torch.randn(32,2))
        candidates=(
            cache.receiver_state,
            cache.donor_state,
            projected_patch(cache.receiver_state,cache.donor_state,basis),
            output_gradient_patch(model,cache),
        )
        for candidate in candidates:
            outcome=patched_metrics(model,cache,candidate,legal)
            if cache.eligible_count==0:
                assert outcome["status"]=="NO_ELIGIBLE"
            else:
                assert outcome["status"]=="MEASURED"
                assert outcome["eligible"]==cache.eligible_count
                assert 0<=outcome["hit"]<=1
                assert 0<=outcome["support"]["fraction_within_tenth_gap"]<=1
        with pytest.raises(ValueError,match="one hidden state"):
            patched_metrics(model,cache,cache.receiver_state[:1],legal)


def test_full_run_refuses_unregistered_parameter_changes(tmp_path):
    with pytest.raises(ValueError,match="model seeds"):
        run_gg03(output=tmp_path/"x.json",artifacts=tmp_path,
                 seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg03(output=tmp_path/"x.json",artifacts=tmp_path,
                 train_steps=1)
    with pytest.raises(ValueError,match="frozen"):
        run_gg03(output=tmp_path/"x.json",artifacts=tmp_path,
                 reporting_size=64)
