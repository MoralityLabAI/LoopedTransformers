"""GG-09: independent exact HMM, heldout grammar and native SGD controls."""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product
import math
import torch
import pytest

from research_gym.interp.goal_guarding_causal import (
    PRIORS,RATIOS, CuePanel, EvaluationBeliefLooper,
    exact_action, bayes_masses, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    SealedGoldAuditor,
)
from research_gym.interp.goal_guarding_protected_tail import (
    DEPTHS,COHORTS,REPLICATES,BLOCK_N,
    cohort_effect_bounds, score_record,aggregate,
    native_report_successor_losses,
)
from research_gym.interp.goal_guarding_markov_transfer import (
    SOURCE_MAIN,MODEL_SEEDS,SWITCH,STATIC,
    BASE_STEPS,GG08_MIXED_STEPS,TRANSFER_STEPS,TRANSFER_BATCH,
    POOL_SIZE,TOTAL_SAMPLE_VISITS, EXPECTED_DIFFERENCES,
    forward_masses,markov_action,task_grammar,task_split,
    mismatch_census,as_panel,pool,task_pools,
    continue_task,_heldout_pools,markov_gold_bank,
    _static_labels_for_bank,ability,_hmm_report_queries,run_gg09,
)


def _path_oracle(prior,bits,switch):
    """Independent sum over every hidden state path, using rational math."""
    joint=[Q(0),Q(0)]
    for path in product((0,1),repeat=len(bits)):
        mass=prior if path[0]==0 else (1-prior)
        for i,(state,bit) in enumerate(zip(path,bits)):
            if i:mass*=switch if path[i-1]!=state else (1-switch)
            if state==0:
                mass*=Q(3,4) if bit else Q(1,4)
            else:
                mass*=Q(1,4) if bit else Q(3,4)
        joint[path[-1]]+=mass
    return tuple(joint)


def _model(seed=81):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return EvaluationBeliefLooper().eval()


def test_exact_markov_forward_matches_independent_path_enumerator():
    assert (SWITCH,STATIC)==(Q(1,4),Q(0))
    for n in range(1,5):
        for prior in PRIORS:
            for bits in product((0,1),repeat=n):
                expected=_path_oracle(prior,bits,SWITCH)
                observed=forward_masses(prior,bits,SWITCH)
                assert expected==observed
                a,b=observed
                assert a+b>0
                for ratio in RATIOS:
                    assert markov_action(prior,ratio,bits)==int(ratio*a>b)
    with pytest.raises(ValueError,match="transition"):
        forward_masses(Q(1,2),(0,1),Q(3,4))
    with pytest.raises(ValueError,match="binary"):
        forward_masses(Q(1,2),(2,),SWITCH)
    with pytest.raises(ValueError,match="interior"):
        forward_masses(Q(0),(1,),SWITCH)


def test_q_zero_reduces_to_original_gg01_and_disagreement_counts_exact():
    census=mismatch_census()
    assert census=={
        "4":{"total":144,"disagreements":35},
        "6":{"total":576,"disagreements":164},
        "8":{"total":2304,"disagreements":727},
    }
    for depth in DEPTHS:
        static=task_grammar(depth,STATIC)
        moving=task_grammar(depth,SWITCH)
        assert tuple(x.uid for x in static)==tuple(x.uid for x in moving)
        assert sum(a.label!=b.label for a,b in zip(static,moving))==(
            EXPECTED_DIFFERENCES[depth])
        for row in static:
            a,b=forward_masses(row.prior,row.bits,STATIC)
            assert (a,b)==bayes_masses(row.prior,Q(1),row.bits)
            assert row.label==exact_action(row.prior,row.ratio,row.bits)
    with pytest.raises(ValueError,match="registered"):
        task_grammar(10,SWITCH)


def test_hash_split_is_disjoint_and_new_gold_uses_only_heldout_ids():
    expected={4:(96,48),6:(384,192),8:(1536,768)}
    for depth in DEPTHS:
        mark=task_split(depth,SWITCH)
        fixed=task_split(depth,STATIC)
        assert (len(mark["training"]),len(mark["heldout"]))==expected[depth]
        assert tuple(x.uid for x in mark["training"])==tuple(
            x.uid for x in fixed["training"])
        assert tuple(x.uid for x in mark["heldout"])==tuple(
            x.uid for x in fixed["heldout"])
        train_ids={x.uid for x in mark["training"]}
        held_ids={x.uid for x in mark["heldout"]}
        assert not train_ids&held_ids
        pools=_heldout_pools(depth)
        assert len(pools["natural"])==len(held_ids)
        assert len(pools["boundary"])==len(held_ids)//4
        assert len(pools["rare"])==len(held_ids)//4
        assert all(x.uid in held_ids for xs in pools.values() for x in xs)
        assert all(markov_action(x.prior,x.ratio,
                   x.bits[:depth//2])!=x.label for x in pools["late"])
        bank=markov_gold_bank(depth)
        mirror=markov_gold_bank(depth)
        assert bank.observations.shape==(4*160,depth)
        assert bank.labels.shape==(4*160,)
        assert len(bank.reps)==REPLICATES
        assert [rep.hashes for rep in bank.reps]==[
            rep.hashes for rep in mirror.reps]
        assert all(len(set(rep.hashes.values()))==4 for rep in bank.reps)
        for rep in bank.reps:
            for name in COHORTS:
                panel=rep.cohorts[name]
                assert len(panel)==(64 if name=="natural" else 32)
                assert all(x==markov_action(
                    p,r,tuple(map(int,panel.observations[i])))
                    for i,((p,r),x) in enumerate(zip(
                        panel.parameters,panel.labels.tolist())))
                for i,(p,r) in enumerate(panel.parameters):
                    bits=tuple(map(int,panel.observations[i]))
                    from research_gym.interp.goal_guarding_markov_transfer import _id
                    assert _id(p,r,bits) in held_ids
        static_labels=_static_labels_for_bank(bank)
        assert static_labels.shape==bank.labels.shape
        assert (static_labels!=bank.labels).any()


def test_markov_and_static_train_pools_use_same_allowed_input_support():
    for task in (STATIC,SWITCH):
        small=pool(11,4,task)
        other=pool(11,4,task)
        assert len(small)==POOL_SIZE
        assert torch.equal(small.observations,other.observations)
        assert torch.equal(small.labels,other.labels)
        from research_gym.interp.goal_guarding_markov_transfer import _id
        source={x.uid for x in task_split(4,task)["training"]}
        assert all(_id(p,r,tuple(map(int,small.observations[i]))) in source
                   for i,(p,r) in enumerate(small.parameters))
        for i in (0,1,30,550):
            p,r=small.parameters[i]
            bits=tuple(int(x) for x in small.observations[i])
            assert int(small.labels[i])==markov_action(p,r,bits,task)
            assert tuple(int(x) for x in small.prefix_labels[i])==tuple(
                markov_action(p,r,bits[:j],task) for j in range(1,5))
    with pytest.raises(ValueError,match="registered"):
        pool(99,4,SWITCH)


def test_task_accuracy_and_gg07_gold_certificate_reuse():
    torch.set_num_threads(1)
    model=_model(76)
    bank={d:markov_gold_bank(d) for d in DEPTHS}
    mark,static,delta,baseline=ability(model,bank)
    assert set(mark)==set(static)==set(delta)=={"4","6","8"}
    for depth in DEPTHS:
        assert baseline[depth].shape==(4,160)
        assert set(mark[str(depth)])==set(COHORTS)
        for name in COHORTS:
            assert 0<=mark[str(depth)][name]<=1
            assert 0<=static[str(depth)][name]<=1
            assert 0<=delta[str(depth)][name]<=1
    # A report harm vector positive everywhere must be certified
    # by the unchanged GG-07 product TV and depth ambiguity math.
    synthetic={
        d:cohort_effect_bounds(torch.ones(4,160,dtype=torch.float64),
                               bank[d])
        for d in DEPTHS
    }
    outputs=score_record(synthetic)
    aggregate_result=aggregate([{"composite":outputs} for _ in range(48)])
    assert aggregate_result["primary_protected_summary"]=={
        "harm":48,"benefit":0,"unresolved":0,
        "nominal_same_nonzero_sign_in_all_four":48}


def test_new_markov_report_sgd_matches_independent_gg04_physical_reference():
    torch.set_num_threads(1)
    base=_model(64)
    digest=_weights_digest(base)
    bank4=markov_gold_bank(4)
    row=_hmm_report_queries()[0]
    measured,norms=native_report_successor_losses(
        base,row,{4:bank4},digest)
    assert len(_hmm_report_queries())==48
    assert set(norms)=={"truth","flip"}
    assert torch.isfinite(measured[4]["gap"]).all()
    assert measured[4]["gap"].shape==(4,160)
    # GG-04 auditor runs its own fresh report forward, backward and SGD
    # instead of the GG-07 shared gradient routine.
    gold=CuePanel(bank4.observations[:160],bank4.contexts[:160],
                  bank4.labels[:160],torch.zeros(160,4,dtype=torch.long),
                  tuple((Q(1,2),Q(1)) for _ in range(160)))
    independent=SealedGoldAuditor(base,gold)
    reference=independent.measure(row)
    assert float(measured[4]["gap"][0].mean())==pytest.approx(
        reference.damage_gap,abs=3e-6)
    assert independent.check_no_update_identity()
    assert _weights_digest(base)==digest


def test_training_guard_refuses_budget_or_incompatible_task():
    model=_model(57)
    sha=_weights_digest(model)
    pools=task_pools(11,SWITCH)
    with pytest.raises(ValueError,match="frozen"):
        continue_task(model,11,SWITCH,pools,steps=1)
    with pytest.raises(ValueError,match="registered"):
        continue_task(model,11,STATIC,pools,steps=1)
    assert _weights_digest(model)==sha


def test_full_gg09_refuses_protocol_changes_before_any_training(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg09(output=tmp_path/"x.json",artifacts=tmp_path,seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg09(output=tmp_path/"x.json",artifacts=tmp_path,transfer_steps=4)
    with pytest.raises(ValueError,match="frozen"):
        run_gg09(output=tmp_path/"x.json",artifacts=tmp_path,gg08_steps=20)
