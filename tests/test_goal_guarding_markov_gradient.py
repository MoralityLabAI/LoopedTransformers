"""GG-10: independent gradients, native SGD and protected-score identity tests."""
from __future__ import annotations

from fractions import Fraction as Q
from copy import deepcopy
import math

import pytest
torch=pytest.importorskip("torch")
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    CuePanel,EvaluationBeliefLooper,_weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import SealedGoldAuditor
from research_gym.interp.goal_guarding_markov_transfer import (
    DEPTHS,COHORTS,REPLICATES,BLOCK_N,SWITCH,task_split,
    markov_gold_bank,_hmm_report_queries,_id,
)
from research_gym.interp.goal_guarding_protected_tail import (
    cohort_effect_bounds,score_record,
)
from research_gym.interp.goal_guarding_gradient_conflict import (
    gradient,query_gradients,dot64,
)
from research_gym.interp.goal_guarding_markov_gradient import (
    SOURCE_MAIN,ARMS,FAMILIES,EXPECTED_SHA,ORIGINAL_CENSUS,
    NEW_NATURAL_SEED_BASE,NEW_CHALLENGE_SEED_BASE,LR,
    new_gold_bank,panel_hashes,banks_all,gold_cohort_gradients,
    depth_gradient_cosines,nominal_impact,ambiguity_width,
    unresolved_reason,_sgd_outcomes,compare_decomposition_rows,
    _summary_one,summarize_arm,run_gg10,
)


def _model(seed=78):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return EvaluationBeliefLooper().eval()


def test_source_known_checkpoint_hashes_and_original_registered_outcomes():
    assert SOURCE_MAIN=="8299198f925dcbf5e875d8a241be317bb42e35cf"
    assert ARMS==("static_control","markov_transfer")
    assert FAMILIES==("historical","fresh")
    assert NEW_NATURAL_SEED_BASE==105000
    assert NEW_CHALLENGE_SEED_BASE==106000
    assert LR==.05
    assert EXPECTED_SHA[11]["markov_transfer"]=="31e31149171ba6452acc27abd552c83624922f5d8b051651aed3142835ad1ad7"
    assert EXPECTED_SHA[17]["markov_transfer"]=="0c11e22c64e0f50bf5ee732e004adeb7200584bf8fee1cdd4dabe1d38427011b"
    assert ORIGINAL_CENSUS[11]["markov_transfer"]==(10,1,37)
    assert ORIGINAL_CENSUS[17]["markov_transfer"]==(19,1,28)


def test_independent_fresh_gold_panels_respect_heldout_markov_oracle():
    for depth in DEPTHS:
        old=markov_gold_bank(depth)
        new=new_gold_bank(depth)
        mirror=new_gold_bank(depth)
        assert new.observations.shape==(REPLICATES*BLOCK_N,depth)
        assert [x.hashes for x in new.reps]==[x.hashes for x in mirror.reps]
        assert new.reps[0].hashes["natural"]!=old.reps[0].hashes["natural"]
        permitted={x.uid for x in task_split(depth,SWITCH)["heldout"]}
        training={x.uid for x in task_split(depth,SWITCH)["training"]}
        assert not permitted&training
        from research_gym.interp.goal_guarding_markov_transfer import markov_action
        for rep in new.reps:
            assert set(rep.cohorts)==set(COHORTS)
            for name,cohort in rep.cohorts.items():
                assert len(cohort)==(64 if name=="natural" else 32)
                for i,(p,r) in enumerate(cohort.parameters):
                    bits=tuple(int(v) for v in cohort.observations[i])
                    assert _id(p,r,bits) in permitted
                    assert int(cohort.labels[i])==markov_action(p,r,bits)
    h=panel_hashes(banks_all())
    assert len(h["fresh"]["4"])==4
    assert all(len(hashstr)==64 for group in h["fresh"]["4"]
               for hashstr in group.values())


def test_nominal_first_order_plus_finite_residual_is_exact():
    template={
        depth:{rep:{
            name:{"physical":2*depth+rep*.02,
                  "first_order":depth+rep*.02,
                  "residual":depth}
            for name in COHORTS
        } for rep in range(REPLICATES)}
        for depth in DEPTHS
    }
    result=nominal_impact(template)
    assert len(result)==REPLICATES
    assert all(float(x["physical"])==pytest.approx(
        float(x["first_order"])+float(x["residual"])) for x in result)
    assert all(x["physical_sign"]==x["first_order_sign"]==1 for x in result)
    good=[{"nominal":x["physical"]} for x in result]
    compare_decomposition_rows(template,good)
    bad=[{"nominal":x["physical"]+1} for x in result]
    with pytest.raises(AssertionError,match="differs"):
        compare_decomposition_rows(template,bad)


def test_finite_support_TV_and_depth_width_identity_is_exact():
    bank={d:new_gold_bank(d) for d in DEPTHS}
    records={}
    for d,b in bank.items():
        # Known spread of finite report effects across cohorts and depths.
        gold=torch.arange(REPLICATES*BLOCK_N,dtype=torch.float64).reshape(
            REPLICATES,BLOCK_N)
        effects=(gold%19-9)*.001+(d-6)*.0003
        records[d]=cohort_effect_bounds(effects,b)
    protected=score_record(records)["1/2"]["1/64"]["replicates"]
    for rep in range(REPLICATES):
        width=ambiguity_width(records,protected[rep],rep)
        assert width["full_protected_interval_width"]==pytest.approx(
            width["cohort_TV_width_component"]+
            width["depth_selection_width_component"],abs=1e-10)
        assert width["cohort_TV_width_component"]>=0
        assert width["depth_selection_width_component"]>=0
        assert set(width["weighted_per_cohort_TV_width"])==set(COHORTS)
        assert width["lower_worst_depth"] in DEPTHS
        assert width["upper_worst_depth"] in DEPTHS


def test_unresolved_reasons_partition_without_ignoring_numerical_zeros():
    robust={"status":"unresolved"}
    assert unresolved_reason(
        {"status":"unresolved","nominal_signs":[1,-1,1,1]},
        robust)=="panel_sign_heterogeneity"
    assert unresolved_reason(
        {"status":"unresolved","nominal_signs":[1]*4},
        robust)=="depth_mixture_ambiguity"
    assert unresolved_reason(
        {"status":"harm","nominal_signs":[1]*4},
        robust)=="cohort_TV_ambiguity"
    assert unresolved_reason(
        {"status":"harm","nominal_signs":[1]*4},
        {"status":"harm"}) is None
    assert unresolved_reason(
        {"status":"benefit","nominal_signs":[0]*4},
        robust)=="panel_sign_heterogeneity"


def test_native_gold_first_order_gradient_matches_symmetric_finite_difference():
    model=_model(91)
    digest=_weights_digest(model)
    bank={("fresh",4):new_gold_bank(4)}
    g=gold_cohort_gradients(model,bank)
    assert len(g)==REPLICATES*len(COHORTS)
    gold=g[("fresh",4,0,"boundary")]
    gold_panel=bank[("fresh",4)].reps[0].cohorts["boundary"]
    source=_hmm_report_queries()[0]
    report_grad,_=query_gradients(model,source)
    direction=tuple(a-b for a,b in zip(report_grad["flip"],
                                       report_grad["truth"]))
    predicted=dot64(gold,direction)
    plus=deepcopy(model)
    minus=deepcopy(model)
    eps=1e-3
    with torch.no_grad():
        for param,v in zip(plus.parameters(),direction):
            param.add_(v,alpha=eps)
        for param,v in zip(minus.parameters(),direction):
            param.add_(v,alpha=-eps)
        def ce(model):
            return float(F.cross_entropy(
                model(gold_panel.observations,gold_panel.context)[0][:,-1,:],
                gold_panel.labels))
        observed=(ce(plus)-ce(minus))/(2*eps)
    assert observed==pytest.approx(predicted,abs=.005,rel=.09)
    assert _weights_digest(model)==digest


def test_real_native_sgd_replay_matches_independent_GG04_step():
    model=_model(62)
    original=_weights_digest(model)
    bank=new_gold_bank(4)
    row=_hmm_report_queries()[1]
    grads,_=query_gradients(model,row)
    physical=_sgd_outcomes(model,grads,{("fresh",4):bank},original)
    assert set(physical)=={("fresh",4)}
    scored=physical[("fresh",4)]
    assert scored["gap"].shape==(4,160)
    assert torch.isfinite(scored["gap"]).all()
    assert torch.equal(scored["flip"]-scored["truth"],scored["gap"])
    # Independent original GG-04 reference performs backward and SGD
    # anew on a different cloned task-model instance.
    panel=CuePanel(bank.observations[:160],bank.contexts[:160],
                   bank.labels[:160],torch.zeros(160,4,dtype=torch.long),
                   tuple((Q(1,2),Q(1)) for _ in range(160)))
    independent=SealedGoldAuditor(model,panel).measure(row)
    assert float(scored["gap"][0].mean())==pytest.approx(
        independent.damage_gap,abs=3e-6)
    assert _weights_digest(model)==original


def test_summary_agreement_uses_nonzero_physical_denominator():
    fake=[]
    for i in range(48):
        fs={}
        for family in FAMILIES:
            fs[family]={
                "protected_composite":{
                    "1/2":{
                        "0":{"status":"harm","nominal_signs":[1]*4},
                        "1/64":{"status":"unresolved" if i<12 else "harm"}
                    }},
                "nominal_protected":[{
                    "physical":.05,"first_order":.04,"residual":.01,
                    "physical_sign":1,"first_order_sign":1,
                } for _ in range(4)],
                "unresolved_reason":"cohort_TV_ambiguity" if i<12 else None,
                "largest_weighted_abs_first_order_cohort":"rare",
            }
        fake.append({"uid":f"query{i}","family":fs})
    result=summarize_arm(fake,{"fresh":{},"historical":{}})
    assert result["fresh"]["robust_status_counts"]=={
        "harm":36,"benefit":0,"unresolved":12}
    assert result["fresh"]["nominal_first_order_supported_gate"]
    assert result["fresh"]["unambiguous_sign_agreement_fraction"]==1
    assert result["fresh"]["unresolved_reason_counts"]=={
        "cohort_TV_ambiguity":12}
    assert result["unresolved_overlap_count"]==12
    assert not result["unresolved_replication_event"]
    with pytest.raises(ValueError,match="48"):
        summarize_arm(fake[:4],{"fresh":{},"historical":{}})


def test_run_refuses_other_seed_or_optimizer_budget(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg10(output=tmp_path/"gg10.json",artifacts=tmp_path,
                 seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg10(output=tmp_path/"gg10.json",artifacts=tmp_path,
                 transfer_steps=20)
