"""GG-06: independent TV-LP, tail weighting, and native SGD controls."""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product
import math

import pytest

torch=pytest.importorskip("torch")

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, exact_action, make_panel, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import query_splits
from research_gym.interp.goal_guarding_robust_gold import (
    BASE_SHA,FOUR_SEEDS,SIX_SEEDS,PANELS_PER_DEPTH,PANEL_EXAMPLES,
    TV_RADII,TAIL_MULTIPLIERS, _finite_values,
    sharp_tv_interval,robust_sign,tail_indices,weighted_tail_gap,
    _bank,sample_cross_entropies,native_sgd_gold_outcomes,
    _bounds_summary,_summary,_cross_depth_summary,run_gg06,
)


def _brute_grid_values(vals,denominator,radius):
    """Independent rational-grid LP oracle, searching every simplex vector.

    With n=4 and nominal 1/4, radius=1/16, all vertices have weights
    whose denominator divides 16. A brute 17^3 grid thus captures the
    continuous LP optimum.
    """
    n=len(vals)
    outcomes=[]
    for first in product(range(denominator+1),repeat=n-1):
        last=denominator-sum(first)
        if last<0:continue
        fractions=tuple(Q(k,denominator) for k in (*first,last))
        tv=sum((abs(w-Q(1,n)) for w in fractions),Q(0))/2
        if tv<=radius:
            outcomes.append(float(sum(float(w)*v for w,v in zip(fractions,vals))))
    assert outcomes
    return min(outcomes),max(outcomes)


def test_preregistered_sampling_and_query_identity():
    assert BASE_SHA=="7c437c4b1e66574bfe453e12992737fc59104fe1"
    assert FOUR_SEEDS==tuple(range(69020,69028))
    assert SIX_SEEDS==tuple(range(68020,68028))
    assert TV_RADII==(Q(0),Q(1,64),Q(1,16))
    assert TAIL_MULTIPLIERS==(1,4,16)
    assert (PANELS_PER_DEPTH,PANEL_EXAMPLES)==(8,128)
    queries=query_splits()["heldout"]
    assert len(queries)==48 and len(set(r.uid for r in queries))==48
    for row in queries:
        assert row.label==exact_action(row.prior,row.ratio,row.bits)


def test_sharp_tv_linear_program_matches_independent_complete_grid():
    scenarios=(
        (1.,2.,3.,4.),
        (-1.5,.25,1.75,-.5),
        (3.,3.,-9.,1.),
        (4.,4.,4.,4.),
        (-1.,1.,-1.,1.),
    )
    for values in scenarios:
        for rho in (Q(0),Q(1,16)):
            certificate=sharp_tv_interval(values,rho)
            brute=_brute_grid_values(values,16,rho)
            assert certificate.lower.objective==pytest.approx(brute[0],abs=1e-12)
            assert certificate.upper.objective==pytest.approx(brute[1],abs=1e-12)
            assert certificate.lower.moved_mass<=rho
            assert certificate.upper.moved_mass<=rho
            certificate.validate(values)
    flat=sharp_tv_interval((5.,)*128,Q(1,16))
    assert flat.lower.objective==flat.upper.objective==5
    assert flat.lower.moved_mass==flat.upper.moved_mass==0


def test_tv_support_shift_witness_and_sign_categories():
    values=(1.,)*127+(-15.,)
    nominal=sum(values)/len(values)
    interval=sharp_tv_interval(values,Q(1,16))
    assert interval.nominal==pytest.approx(nominal)
    assert interval.lower.objective<0<interval.upper.objective
    assert robust_sign(interval)=="unresolved"
    assert interval.lower.moved_mass==Q(1,16)
    # For the upper envelope, all improving donor mass can be exhausted
    # before the TV budget: the single negative example has mass 1/128.
    assert interval.upper.moved_mass==Q(1,128)
    for ext in (interval.lower,interval.upper):
        witness=ext.witness(128)
        assert sum(witness,Q(0))==1
        assert all(w>=0 for w in witness)
        assert sum((abs(w-Q(1,128)) for w in witness),Q(0))/2==ext.moved_mass
    assert robust_sign(sharp_tv_interval((1.,)*128,Q(1,16)))=="harm"
    assert robust_sign(sharp_tv_interval((-1.,)*128,Q(1,16)))=="benefit"
    with pytest.raises(ValueError,match="exact rational"):
        sharp_tv_interval(values,0.0625)
    with pytest.raises(ValueError,match="at least 4"):
        sharp_tv_interval((1.,2.),Q(0))
    with pytest.raises(ValueError,match="exact rational"):
        sharp_tv_interval(values,Q(1,8))


def test_tail_is_selected_only_on_baseline_loss_and_is_independent_of_effects():
    baseline=[0.1]*121+[20.+i for i in range(7)]
    indices=tail_indices(baseline)
    assert indices==tuple(range(127,120,-1))
    gap=[0.]*121+[1.]*7
    assert weighted_tail_gap(gap,indices,1)==pytest.approx(7/128)
    assert weighted_tail_gap(gap,indices,4)==pytest.approx(28/149)
    assert weighted_tail_gap(gap,indices,16)==pytest.approx(112/233)
    assert tail_indices(baseline)==tail_indices(baseline)
    assert tail_indices([0.]*128)==tuple(range(7))
    with pytest.raises(ValueError,match="unregistered"):
        weighted_tail_gap(gap,indices,3)
    with pytest.raises(ValueError,match="distinct"):
        weighted_tail_gap(gap,(1,1),4)
    with pytest.raises(ValueError,match="preregistration"):
        tail_indices([0.]*127)


def test_native_sgd_actual_weight_changes_and_untouched_source():
    torch.set_num_threads(1)
    torch.manual_seed(56)
    model=EvaluationBeliefLooper().eval()
    original=_weights_digest(model)
    banks={}
    for depth,seed in ((4,69020),(6,68020)):
        panels=tuple(make_panel(128,depth,seed+i) for i in range(8))
        banks[depth]=_bank(depth,panels)
        assert banks[depth].observations.shape==(1024,depth)
        assert sample_cross_entropies(model,banks[depth]).shape==(8,128)
    row=query_splits()["heldout"][3]
    score=native_sgd_gold_outcomes(model,row,banks,original)
    assert set(score)=={4,6}
    for depth in (4,6):
        assert set(score[depth])=={"truth","flip","gap"}
        assert score[depth]["truth"].shape==(8,128)
        assert torch.allclose(score[depth]["flip"]-score[depth]["truth"],
                              score[depth]["gap"])
        assert torch.isfinite(score[depth]["gap"]).all()
    assert _weights_digest(model)==original


def test_panel_summaries_keep_independent_witnesses_and_denominators():
    effects=torch.full((8,128),1e-3,dtype=torch.float64)
    losses=torch.ones_like(effects)
    outcome=_bounds_summary(effects,losses,depth=4)
    assert len(outcome["hardest_seven_indices"])==8
    assert all(index==list(range(7)) for index in outcome["hardest_seven_indices"])
    assert outcome["panel_bounds_by_radius"]["1/16"]["status"]=="harm"
    assert outcome["pooled_bounds_by_radius"]["1/16"]["sign"]=="harm"
    assert outcome["tail_probability_masses"]["4"]==pytest.approx(28/149)
    assert all(abs(x-1e-3)<1e-8 for x in outcome["tail_gaps"]["16"])
    fake=[{"depths":{"4":outcome,"6":outcome}} for _ in range(48)]
    for depth in (4,6):
        result=_summary(fake,depth)
        assert result["nominal_same_nonzero_sign_all_eight_queries"]==48
        assert result["TV_robust_counts"]["1/16"]=={
            "harm":48,"benefit":0,"unresolved":0}
        assert result["strong_nominal_sampling_stability"]
        assert result["strong_TV_robustness_at_1_16"]
        assert result["hard_case_weight_sensitivity"]["4"][
            "different_sign_query_panel_cells"]==0
    opposite={**outcome,"nominal_panel_signs":[-1]*8}
    bad=[{"depths":{"4":outcome,"6":opposite}} for _ in range(48)]
    cross=_cross_depth_summary(bad)
    assert cross["paired_panel_index_reversal_counts_out_of_48"]==[48]*8
    assert cross["queries_with_all_eight_reversals"]==48
    with pytest.raises(ValueError,match="48"):
        _summary(fake[:-1],4)


def test_registered_runner_rejects_changes_without_starting_training(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg06(output=tmp_path/"x.json",artifacts=tmp_path,seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg06(output=tmp_path/"x.json",artifacts=tmp_path,train_steps=1)
