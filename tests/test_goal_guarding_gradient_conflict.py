"""GG-05: independent native physical SGD and gradient-conflict controls."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
import math

import pytest

torch=pytest.importorskip("torch")
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, make_panel, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    query_splits, _tensor_rows,
)
from research_gym.interp.goal_guarding_gradient_conflict import (
    BASE_SHA, LR_GRID, SIGN_EPS, GOLD_SPECS,
    sign, mixture_threshold, dot64, cosine64, norm64,
    subtract, gradient, panel_ce, panel_gradients, query_gradients,
    physical_step_scores, physical_report, classify_pair,
    summarize_rows, key_eta, run_gg05,
)


def native_model(seed=83):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return EvaluationBeliefLooper().eval()


def test_frozen_source_panels_and_query_roster():
    assert BASE_SHA=="15512fb3b85c988442372c242317e82fe726372e"
    assert LR_GRID==(0.0125,0.025,0.05)
    assert [key_eta(r) for r in LR_GRID]==[
        "0.0125","0.0250","0.0500"]
    assert GOLD_SPECS==(
        ("fresh4",4,69002),("fresh6",6,68002),
        ("old4",4,69001),("old6",6,68001),
    )
    rowsets=query_splits()
    assert [len(rowsets[k]) for k in (
        "actor_train","calibration","heldout")]==[72,24,48]
    assert len(set(q.uid for items in rowsets.values()
                   for q in items))==144
    for row in rowsets["heldout"]:
        assert len(row.bits)==4
        assert row.label in (0,1)


def test_sign_and_mixture_threshold_exact_controls():
    assert [sign(x) for x in (-1.0,-SIGN_EPS,0,SIGN_EPS,1)]==[-1,0,0,0,1]
    assert mixture_threshold(3,-1)==pytest.approx(.75)
    assert mixture_threshold(-3,1)==pytest.approx(.75)
    assert mixture_threshold(3,1) is None
    assert mixture_threshold(1e-8,-1) is None
    with pytest.raises(ValueError,match="finite"):
        sign(float("nan"))
    with pytest.raises(ValueError,match="finite"):
        sign(1.0,epsilon=-1)


def test_independent_first_order_decomposition_and_finite_classification():
    # A direct conflict predicts a real reversal, when its residual is small.
    f={"gap":.30,"linear_gap":.40,"curvature_residual":-.10}
    s={"gap":-.20,"linear_gap":-.25,"curvature_residual":.05}
    x=classify_pair(f,s)
    assert x["measured_reversal"]
    assert x["linear_reversal"]
    assert x["reversal_first_order_explained"]
    assert not x["reversal_curvature_implicated"]
    assert x["critical_finite_gold_mix"]==pytest.approx(.6)
    assert x["critical_linear_gold_mix"]==pytest.approx(.4/.65)
    # Residual dominates six-cue linear prediction and reverses its sign.
    changed=classify_pair(f,{"gap":-.20,"linear_gap":.06,
                             "curvature_residual":-.26})
    assert changed["measured_reversal"] and changed[
        "reversal_curvature_implicated"]
    assert not changed["linear_reversal"]
    assert changed["critical_linear_gold_mix"] is None
    # Do not count finite differences indistinguishable from zero.
    ambiguous=classify_pair({"gap":1e-8,"linear_gap":3e-7},
                            {"gap":-1,"linear_gap":-1})
    assert not ambiguous["measured_reversal"]
    assert ambiguous["measured_signs"]==[0,-1]


def test_summary_partitions_reversals_and_preserves_rate_ambiguity():
    rows=[]
    for i in range(10):
        measured4=.25
        measured6=-.2
        linear4=.20
        linear6=-.20 if i<5 else .10
        block={}
        for eta in LR_GRID:
            ratio=eta/.05
            block[key_eta(eta)]={
                "fresh4":{"gap":measured4*ratio,"linear_gap":linear4*ratio,
                          "curvature_residual":(measured4-linear4)*ratio},
                "fresh6":{"gap":measured6*ratio,"linear_gap":linear6*ratio,
                          "curvature_residual":(measured6-linear6)*ratio},
            }
        rows.append({"panel_readouts":block})
    summary=summarize_rows(rows,"fresh4","fresh6")
    assert summary["measured_reversal_count"]==10
    assert summary["first_order_explained_measured_reversal_count"]==5
    assert summary["finite_residual_implicated_measured_reversal_count"]==5
    assert summary["first_order_supported_at_least_half_if_reversals_ge_8"]
    assert not summary["curvature_sensitive_at_least_8"]
    assert summary["per_panel"]["fresh4"]["sign_stable_three_rates"]==10


def test_native_clipped_direction_and_no_update_state_mutation():
    model=native_model()
    digest=_weights_digest(model)
    row=query_splits()["heldout"][0]
    grads,norms=query_gradients(model,row)
    assert set(grads)=={"truth","flip"}
    assert all(x>0 and math.isfinite(x) for x in norms.values())
    assert all(norm64(g)<=1.00001 for g in grads.values())
    assert _weights_digest(model)==digest
    diff=subtract(grads["flip"],grads["truth"])
    assert math.isfinite(norm64(diff))
    assert -1<=cosine64(grads["flip"],grads["truth"])<=1
    assert dot64(diff,diff)>=0
    with pytest.raises(ValueError,match="shape"):
        dot64(grads["flip"],(torch.zeros(3),))


def test_physical_SGD_matches_independent_backward_optimizer_reference():
    model=native_model(45)
    origin=_weights_digest(model)
    gold=make_panel(8,6,68101)
    panels={"six":gold}
    params,baseline=panel_gradients(model,panels)
    assert _weights_digest(model)==origin
    row=query_splits()["heldout"][1]
    grads,norms=query_gradients(model,row)
    score=physical_step_scores(model,grads["flip"],.05,
                               panels,origin)["six"]
    # Independent reference recomputes the report gradient on a clone
    # directly from its own native forward/backward graph.
    trial=deepcopy(model)
    bits,ctx,_=_tensor_rows((row,))
    opt=torch.optim.SGD(trial.parameters(),lr=.05,
                        momentum=0.0,weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(trial(bits,ctx)[0][:,-1,:],
                    torch.tensor([1-row.label])).backward()
    torch.nn.utils.clip_grad_norm_(trial.parameters(),1.0)
    opt.step()
    with torch.no_grad():
        actual=float(panel_ce(trial,gold))
    assert score==pytest.approx(actual,abs=1e-7)
    # First-order variation can be verified independently by a short
    # central finite-difference directional perturbation at source.
    goldgrad=params["six"]
    truthgrad=grads["flip"]
    predicted=-.05*dot64(goldgrad,truthgrad)
    assert math.isfinite(predicted)
    plus=deepcopy(model)
    minus=deepcopy(model)
    epsilon=1e-3
    with torch.no_grad():
        for p,g in zip(plus.parameters(),truthgrad):
            p.add_(g,alpha=epsilon)
        for p,g in zip(minus.parameters(),truthgrad):
            p.add_(g,alpha=-epsilon)
        slope=(float(panel_ce(plus,gold))-float(panel_ce(minus,gold)))/(2*epsilon)
    assert slope==pytest.approx(dot64(goldgrad,truthgrad),abs=3e-3,rel=.08)
    assert _weights_digest(model)==origin


def test_real_report_contains_physical_nonlinear_residual_and_rate_grid():
    model=native_model(46)
    origin=_weights_digest(model)
    panels={
        "four":make_panel(12,4,69102),
        "six":make_panel(12,6,68102),
    }
    gold_grads,baselines=panel_gradients(model,panels)
    row=query_splits()["heldout"][0]
    receipt=physical_report(model,row,panels,gold_grads,baselines,origin)
    assert receipt["uid"]==row.uid
    assert len(receipt["panel_readouts"])==3
    assert set(receipt["clipped_report_gradient_norms"])=={"truth","flip"}
    for eta in LR_GRID:
        for gold in panels:
            v=receipt["panel_readouts"][key_eta(eta)][gold]
            assert v["gap"]==pytest.approx(
                v["flip_post_sgd_ce"]-v["truth_post_sgd_ce"])
            assert v["curvature_residual"]==pytest.approx(
                v["gap"]-v["linear_gap"])
            assert v["truth_delta"]==pytest.approx(
                v["truth_post_sgd_ce"]-baselines[gold])
    assert _weights_digest(model)==origin


def test_full_protocol_rejects_changed_source_parameters(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg05(output=tmp_path/"x.json",artifacts=tmp_path,
                 seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg05(output=tmp_path/"x.json",artifacts=tmp_path,
                 train_steps=1)
