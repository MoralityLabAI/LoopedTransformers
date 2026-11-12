"""GG-07 independent Bayes cohort, robust LP and native physical SGD tests."""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product
import math

import pytest
torch=pytest.importorskip("torch")

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, exact_action, make_panel, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    query_splits,SealedGoldAuditor,
)
from research_gym.interp.goal_guarding_protected_tail import (
    SOURCE_MAIN,SEEDS,DEPTHS,REPLICATES,COHORTS,BLOCK_N,
    NATURAL_N,CHALLENGE_N,LAMBDA_GRID,RHO_GRID,
    PRIMARY_LAMBDA,PRIMARY_RHO,SPLIT,
    full_exact_grammar,challenge_pools,generate_replicate,
    build_gold_bank,evaluate_bank,native_report_successor_losses,
    depth_simplex_extreme,cohort_effect_bounds,protected_depth_bounds,
    composite_bounds,robust_status,score_record,aggregate,run_gg07,
)
from research_gym.interp.goal_guarding_robust_gold import sharp_tv_interval


def model(seed=47):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return EvaluationBeliefLooper().eval()


def test_registered_source_shapes_and_exact_cohort_classifiers():
    assert SOURCE_MAIN=="0bdfaee3af9474d9df5cf736a915db0a44998d05"
    assert SEEDS==(11,17)
    assert DEPTHS==(4,6,8)
    assert (REPLICATES,NATURAL_N,CHALLENGE_N,BLOCK_N)==(4,64,32,160)
    assert SPLIT=={"natural":(0,64),"boundary":(64,96),
                   "rare":(96,128),"late":(128,160)}
    assert LAMBDA_GRID==(Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1))
    assert RHO_GRID==(Q(0),Q(1,64))
    assert (PRIMARY_LAMBDA,PRIMARY_RHO)==(Q(1,2),Q(1,64))
    for depth in DEPTHS:
        rows=full_exact_grammar(depth)
        assert len(rows)==9*(2**depth)
        assert len(set(row.identity() for row in rows))==len(rows)
        assert all(row.label==exact_action(row.prior,row.ratio,row.bits)
                   for row in rows)
        pool=challenge_pools(depth)
        assert set(pool)=={"boundary","rare","late"}
        assert len(pool["boundary"])==len(pool["rare"])==len(rows)//4
        margin_boundary=max(x.margin for x in pool["boundary"])
        rare_cutoff=max(x.likelihood for x in pool["rare"])
        assert all(x.margin+1e-12>=margin_boundary
                   for x in rows if x not in pool["boundary"])
        assert all(x.likelihood>=rare_cutoff
                   for x in rows if x not in pool["rare"])
        assert all(exact_action(
            x.prior,x.ratio,x.bits[:depth//2])!=x.label
                   for x in pool["late"])


def test_independent_replicate_sampling_finite_pool_membership_and_hash():
    for depth in DEPTHS:
        pools=challenge_pools(depth)
        p=generate_replicate(depth,0,pools=pools)
        q=generate_replicate(depth,0,pools=pools)
        assert p.hashes==q.hashes
        assert p.unique_counts==q.unique_counts
        assert set(p.cohorts)==set(COHORTS)
        for name,count in (("natural",64),("boundary",32),
                           ("rare",32),("late",32)):
            gold=p.cohorts[name]
            assert gold.observations.shape==(count,depth)
            assert gold.labels.shape==(count,)
            for i,(prior,ratio) in enumerate(gold.parameters):
                bits=tuple(map(int,gold.observations[i].tolist()))
                assert int(gold.labels[i])==exact_action(prior,ratio,bits)
                if name!="natural":
                    key=(str(prior),str(ratio),bits)
                    assert key in {x.identity() for x in pools[name]}
        bank=build_gold_bank(depth)
        assert bank.observations.shape==(640,depth)
        assert bank.contexts.shape==(640,2)
        assert bank.labels.shape==(640,)
        assert len(bank.reps)==REPLICATES
        assert len(set(x.hashes["natural"] for x in bank.reps))==4
        with pytest.raises(ValueError,match="registered"):
            generate_replicate(depth,4,pools=pools)
    with pytest.raises(ValueError,match="registered depth"):
        full_exact_grammar(10)


def test_independent_three_simplex_vertex_enumeration():
    vals=(-2.0,1.0,4.0)
    candidates=[]
    for index in range(3):
        w=[Q(1,6)]*3
        w[index]+=Q(1,2)
        assert sum(w,Q(0))==1
        candidates.append(sum(float(wi)*v for wi,v in zip(w,vals)))
    assert depth_simplex_extreme(vals,lower=True)==pytest.approx(min(candidates))
    assert depth_simplex_extreme(vals,lower=False)==pytest.approx(max(candidates))
    with pytest.raises(ValueError,match="3 finite"):
        depth_simplex_extreme((1.,2.),lower=True)


def test_product_TV_certificate_matches_independent_extreme_weights():
    # Cohorts have disjoint admissible weight sets; the objective separates
    # into a weighted sum of four independent finite TV-LP problems.
    cohorts={}
    sample={
        "natural":(2.,2.,2.,2.,2.,2.,2.,2.),
        "boundary":(-1.,-2.,0.,1.,-3.,2.,0.,1.),
        "rare":(3.,-2.,0.,-1.,1.,-4.,2.,0.),
        "late":(-2.,0.,3.,1.,1.,0.,-1.,2.),
    }
    for name,values in sample.items():
        cohorts[name]={}
        for rho in RHO_GRID:
            c=sharp_tv_interval(values,rho)
            cohorts[name][str(rho)]={
                "nominal":c.nominal,
                "lower":c.lower.objective,
                "upper":c.upper.objective,
            }
    primary=protected_depth_bounds(cohorts,Q(1,2),Q(1,64))
    intervals={name:sharp_tv_interval(values,Q(1,64))
               for name,values in sample.items()}
    for direction in ("lower","upper"):
        actual=(
            .5*sum(float(x)*v for x,v in zip(
                getattr(intervals["natural"],direction).witness(8),
                sample["natural"]))
            +sum((1/6)*sum(float(x)*v for x,v in zip(
                getattr(intervals[name],direction).witness(8),
                sample[name])) for name in ("boundary","rare","late"))
        )
        assert primary[direction]==pytest.approx(actual)
    assert primary["lower"]<=primary["nominal"]<=primary["upper"]
    with pytest.raises(ValueError,match="unregistered"):
        protected_depth_bounds(cohorts,Q(1,3),Q(0))


def test_actual_native_SGD_and_independent_GG04_auditor_agree():
    torch.set_num_threads(1)
    base=model(55)
    sha=_weights_digest(base)
    banks={4:build_gold_bank(4)}
    row=query_splits()["heldout"][0]
    gold,raw_norms=native_report_successor_losses(
        base,row,banks,sha)
    assert set(gold)=={4}
    assert set(raw_norms)=={"truth","flip"}
    assert gold[4]["gap"].shape==(4,160)
    assert torch.allclose(gold[4]["gap"],
                          gold[4]["flip"]-gold[4]["truth"])
    assert bool(torch.isfinite(gold[4]["gap"]).all())
    # Independent ordinary backward + SGD; uses just one complete gold
    # replicate, with the exact cohort weights of this test's baseline.
    # Independent GG-04 assessor receives a synthetic CuePanel stitched
    # from four cohorts for one replicate. Its raw mean equals their mean.
    gold_panel=banks[4]
    from research_gym.interp.goal_guarding_causal import CuePanel
    from research_gym.interp.goal_guarding_causal import exact_action
    subset=CuePanel(
        gold_panel.observations[:160],
        gold_panel.contexts[:160],
        gold_panel.labels[:160],
        torch.zeros(160,4,dtype=torch.long),
        tuple((Q(1,2),Q(1)) for _ in range(160)),
    )
    auditor=SealedGoldAuditor(base,subset)
    oracle=auditor.measure(row)
    assert float(gold[4]["gap"][0].mean())==pytest.approx(
        oracle.damage_gap,abs=2e-6)
    assert auditor.check_no_update_identity()
    assert _weights_digest(base)==sha


def test_cohort_and_composite_certificates_sign_and_nominal_perturbation():
    banks={d:build_gold_bank(d) for d in DEPTHS}
    # Synthetic exact arrays give each natural cohort positive and every
    # challenge cohort negative, so lambda changes the sign of the
    # depth-mixed objective. No neural model outcome assumed.
    depths={}
    for d,bank in banks.items():
        effect=torch.cat((
            torch.ones(4,64,dtype=torch.float64),
            -3*torch.ones(4,96,dtype=torch.float64),
        ),dim=1)
        depths[d]=cohort_effect_bounds(effect,bank)
        assert len(depths[d]["replicates"])==4
    zero=composite_bounds(depths,Q(0),Q(0))
    protected=composite_bounds(depths,Q(1,2),Q(1,64))
    assert robust_status(zero)=="harm"
    assert robust_status(protected)=="benefit"
    assert [v["nominal"] for v in zero]==pytest.approx([1.]*4)
    assert [v["nominal"] for v in protected]==pytest.approx([-1.]*4)
    outcomes=score_record(depths)
    fake=[{"composite":outcomes} for _ in range(48)]
    summary=aggregate(fake)
    assert summary["primary_protected_summary"]=={
        "harm":0,"benefit":48,"unresolved":0,
        "nominal_same_nonzero_sign_in_all_four":48}
    assert summary["natural_to_protected_any_sign_flip_queries"]==48
    assert summary["natural_to_protected_nominal_sign_flip_query_replicate_cells"]==192
    assert summary["secondary_tail_induced_sign_change_event"]
    with pytest.raises(ValueError,match="all three"):
        composite_bounds({4:depths[4]},Q(1,2),Q(1,64))
    with pytest.raises(ValueError,match="all 48"):
        aggregate(fake[:-1])


def test_full_registered_run_rejects_budget_or_seed_changes(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg07(output=tmp_path/"x.json",artifacts=tmp_path,seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg07(output=tmp_path/"x.json",artifacts=tmp_path,train_steps=16)
