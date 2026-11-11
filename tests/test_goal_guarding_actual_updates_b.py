"""GG-04B: bounded-feedback and independently paired cross-depth controls."""
from dataclasses import replace
from fractions import Fraction as Q
import math

import pytest
torch=pytest.importorskip("torch")

from research_gym.interp.goal_guarding_causal import (
    make_panel,EvaluationBeliefLooper,
)
from research_gym.interp.goal_guarding_actual_updates import (
    UpdateController,SealedGoldAuditor,
    PhysicalUpdate,query_splits,hidden_features,
    evaluate_controller,
)
from research_gym.interp.goal_guarding_actual_updates_b import (
    clip_penalty,bounded_effects,_effect_ledger,
    _cross_depth_summary,_panel_hash,_actual_raw_factorial,run_gg04b
)


def physical(truth,flip):
    return PhysicalUpdate(truth,flip,truth,flip,1.0,1.0,True,True)


def test_bounded_rewards_only_and_raw_ledger_unchanged():
    originals=tuple(physical(x,y) for x,y in (
        (0.0,0.1),(-0.05,10.0),(-10.0,-0.01),(0.0,1.0)
    ))
    bounded=bounded_effects(originals,0.1)
    assert [x.gold_loss_delta_flip for x in bounded]==[
        1.0,4.0,-0.1,4.0]
    assert [x.gold_loss_delta_truth for x in bounded]==[
        0.0,-0.5,-4.0,0.0]
    assert [x.gold_loss_delta_flip for x in originals]==[
        .1,10.0,-.01,1.0]
    rows=query_splits()["calibration"][:4]
    output=_effect_ledger(rows,originals,0.1)
    assert [v["flip_delta"] for v in output]==[
        .1,10.0,-.01,1.0]
    assert [v["bounded_controller_flip_penalty"] for v in output]==[
        1.0,4.0,-0.1,4.0]
    with pytest.raises(ValueError,match="positive finite"):
        clip_penalty(1,0.0)
    with pytest.raises(ValueError,match="positive finite"):
        clip_penalty(float("nan"),.1)


def test_opposing_signs_and_paired_cross_depth_report():
    four=(physical(0,.01),physical(0,.02),physical(0,.03))
    six=(physical(0,-.01),physical(0,.04),physical(0,-.05))
    summary=_cross_depth_summary(four,six)
    assert summary["queries"]==3
    assert summary["aligned_four_cue_median_gap"]==.02
    assert summary["cross_depth_six_cue_median_gap"]==-.01
    assert summary["fraction_aligned_positive"]==1
    assert summary["fraction_six_cue_positive"]==1/3
    assert summary["fraction_opposing_nonzero_signs"]==2/3
    with pytest.raises(ValueError,match="paired"):
        _cross_depth_summary(four,six[:2])


def test_stable_distinct_gold_panel_source_hashes():
    p4=make_panel(16,4,69001)
    p6=make_panel(16,6,68001)
    assert _panel_hash(p4)==_panel_hash(make_panel(16,4,69001))
    assert _panel_hash(p4)!=_panel_hash(p6)


def test_factorial_receipt_preserves_raw_actual_gold_units():
    torch.manual_seed(1)
    actor=UpdateController().eval()
    features=torch.randn(8,32)
    physicals=tuple(physical(.01,.09) for _ in range(8))
    normalized=bounded_effects(physicals,.02)
    score=evaluate_controller(actor,features,normalized,1.0,(0,1))
    _actual_raw_factorial(score,actor,features,physicals)
    assert len(score["causal_randomization_factorial"])==4
    for row in score["causal_randomization_factorial"]:
        assert "mean_actual_gold_loss_delta" not in row
        assert row["unit_type_for_mean_realized_reward"]=="bounded controller reward"
        assert row["mean_bounded_controller_penalty_units"]>=0
        if not row["real_update_assignment"]:
            assert row["mean_raw_gold_ce_delta"]==0.0
        else:
            assert .009<=row["mean_raw_gold_ce_delta"]<=.091


def test_physical_four_cue_step_reports_change_from_same_base():
    torch.set_num_threads(1)
    torch.manual_seed(94)
    model=EvaluationBeliefLooper()
    panel=make_panel(8,4,69001)
    audit=SealedGoldAuditor(model,panel)
    row=query_splits()["calibration"][0]
    effect=audit.measure(row)
    assert audit.check_no_update_identity()
    assert effect.truth_changed_weights and effect.flip_changed_weights
    assert math.isfinite(effect.damage_gap)
    assert effect.raw_gradient_norm_truth>=0
    assert effect.raw_gradient_norm_flip>=0


def test_full_protocol_rejects_changed_budgets(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg04b(output=tmp_path/"x.json",artifacts=tmp_path,seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg04b(output=tmp_path/"x.json",artifacts=tmp_path,base_updates=32)
    with pytest.raises(ValueError,match="frozen"):
        run_gg04b(output=tmp_path/"x.json",artifacts=tmp_path,controller_updates=32)
