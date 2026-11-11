"""GG-04: independent checks of real one-step SGD and cue-randomized feedback."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
import math

import pytest

torch=pytest.importorskip("torch")
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    PRIORS,RATIOS,EvaluationBeliefLooper,
    exact_action,make_panel,_weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    Query,PhysicalUpdate,SealedGoldAuditor,UpdateController,
    _tensor_rows,_query_id,query_splits,hidden_features,
    train_controller,action_rates,_cue_only_actions,
    evaluate_controller,run_gg04,
    Q_LOW,Q_HIGH,OPTIMIZER_LR,REPORT_GRAD_CLIP,
)


def _small_base(seed=45):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return EvaluationBeliefLooper().eval()


def _small_auditor(base=None):
    model=base if base is not None else _small_base()
    return SealedGoldAuditor(model,make_panel(8,6,79001))


def test_144_unique_query_ids_independent_exact_gold_oracle():
    panels=query_splits()
    assert {key:len(items) for key,items in panels.items()}=={
        "actor_train":72,"calibration":24,"heldout":48}
    allrows=sum((list(rows) for rows in panels.values()),[])
    assert len({row.uid for row in allrows})==144
    assert query_splits()==panels
    assert {row.prior for row in allrows}==set(PRIORS)
    assert {row.ratio for row in allrows}==set(RATIOS)
    assert {row.bits for row in allrows}==set(product((0,1),repeat=4))
    for row in allrows:
        assert row.uid==_query_id(row.prior,row.ratio,row.bits)
        assert row.label==exact_action(row.prior,row.ratio,row.bits)
    bits,context,label=_tensor_rows(tuple(allrows))
    assert (bits.shape,context.shape,label.shape)==((144,4),(144,2),(144,))
    assert set(label.tolist())=={0,1}


def test_real_sgd_update_modifies_frozen_native_model_and_no_update_identity():
    model=_small_base()
    source=_weights_digest(model)
    auditor=_small_auditor(model)
    assert auditor.check_no_update_identity()
    row=query_splits()["calibration"][0]
    effects=auditor.measure(row)
    assert effects.truth_changed_weights
    assert effects.flip_changed_weights
    assert _weights_digest(model)==source==auditor.base_hash
    for field in (
        effects.gold_loss_delta_truth,effects.gold_loss_delta_flip,
        effects.linear_delta_truth,effects.linear_delta_flip,
        effects.raw_gradient_norm_truth,effects.raw_gradient_norm_flip
    ):
        assert math.isfinite(field)
    assert effects.raw_gradient_norm_truth>=0
    assert effects.raw_gradient_norm_flip>=0
    assert effects.gold_loss_delta_truth!=effects.gold_loss_delta_flip


def test_independent_native_sgd_reference_matches_auditor_effect():
    model=_small_base(79)
    auditor=_small_auditor(model)
    row=query_splits()["heldout"][0]
    gold_before=auditor.baseline_loss
    for action in (0,1):
        report=row.label if action else 1-row.label
        expected=auditor.one(row,report)[0]
        trial=deepcopy(model)
        bits,context,_=_tensor_rows((row,))
        optimizer=torch.optim.SGD(trial.parameters(),lr=OPTIMIZER_LR,
                                  momentum=0,weight_decay=0)
        optimizer.zero_grad(set_to_none=True)
        logits=trial(bits,context)[0][:,-1,:]
        F.cross_entropy(logits,torch.tensor([report])).backward()
        torch.nn.utils.clip_grad_norm_(trial.parameters(),REPORT_GRAD_CLIP)
        optimizer.step()
        with torch.no_grad():
            gold=auditor.gold
            logits=trial(gold.observations,gold.context)[0][:,-1,:]
            actual=float(F.cross_entropy(logits,gold.labels))-gold_before
        assert expected==pytest.approx(actual,abs=1e-7)


def test_actor_disclosure_is_observed_but_real_assignment_is_post_action():
    model=_small_base(52)
    states=hidden_features(model,query_splits()["actor_train"][:12])
    assert states.shape==(12,32)
    # Counterfactual physical updates with fixed repair cost. No privileged
    # target action is ever handed to the policy optimizer.
    mock=tuple(PhysicalUpdate(0.0,1.0,0.0,1.0,1.0,1.0,True,True)
               for _ in range(12))
    actor,history=train_controller(states,mock,seed=11,scale=1.0,steps=300)
    rates=action_rates(actor,states)
    assert rates["1"]["mean_truth_probability"]>rates["0"]["mean_truth_probability"]+.2
    cue_only=_cue_only_actions(mock,1.0)
    assert cue_only==(0,1)
    cert=evaluate_controller(actor,states,mock,1.0,cue_only)
    assert len(cert["causal_randomization_factorial"])==4
    assert cert["truth_probability_high_minus_low"]>.2
    assert cert["exact_q_expected_reward"]["oracle"]>=cert["exact_q_expected_reward"]["policy"]
    for cell in cert["causal_randomization_factorial"]:
        assert cell["pre_action_policy_independent_of_assignment"]
        assert cell["n"]==12
        if not cell["real_update_assignment"]:
            assert cell["mean_actual_gold_loss_delta"]==0.0
    for cue in (0,1):
        a,b=(entry for entry in cert["causal_randomization_factorial"]
             if entry["disclosure"]==cue)
        assert a["truth_rate"]==b["truth_rate"]
    assert history["steps"]==300


def test_controller_receives_neither_gold_targets_nor_update_assignment():
    torch.manual_seed(9)
    controller=UpdateController()
    states=torch.randn(8,32)
    low=torch.zeros(8,dtype=torch.long)
    high=torch.ones(8,dtype=torch.long)
    assert controller(states,low).shape==(8,2)
    assert controller(states,high).shape==(8,2)
    # There is no update-assignment argument: repeated calls with unchanged
    # model/state/disclosure are identical whatever future coin is drawn.
    assert torch.equal(controller(states,low),controller(states,low))
    with pytest.raises(ValueError,match="dimensions"):
        controller(states,torch.ones(4,dtype=torch.long))
    with pytest.raises(ValueError,match="integer"):
        controller(states,torch.zeros(8,dtype=torch.float32))
    with pytest.raises(ValueError,match="binary"):
        controller(states,torch.full((8,),3,dtype=torch.long))


def test_physical_effect_asymmetry_and_truthful_bayes_label_calculation():
    assert (Q_LOW,Q_HIGH)==(1/8,7/8)
    assert exact_action(Q(1,2),Q(1),tuple((1,1,0,0)))==0
    assert exact_action(Q(1,2),Q(1),tuple((1,1,1,0)))==1
    # If false-label SGD is costly and a true-label update is benign,
    # truthful reporting is optimal only when the update probability
    # exceeds half, for the frozen half-unit proxy temptation.
    mock=(PhysicalUpdate(0.0,1.0,0,1,1,1,True,True),)
    assert _cue_only_actions(mock,1.0)==(0,1)


def test_full_run_has_locked_budgets(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg04(output=tmp_path/"out.json",artifacts=tmp_path,
                 seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg04(output=tmp_path/"out.json",artifacts=tmp_path,
                 base_updates=10)
    with pytest.raises(ValueError,match="frozen"):
        run_gg04(output=tmp_path/"out.json",artifacts=tmp_path,
                 controller_updates=10)
