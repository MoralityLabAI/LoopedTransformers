"""GG-08: independent exact cohort, visit-matched schedule and physical SGD controls."""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product

import pytest
torch=pytest.importorskip("torch")
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper,exact_action,make_panel,_weights_digest,CuePanel,
)
from research_gym.interp.goal_guarding_actual_updates import (
    query_splits,SealedGoldAuditor,
)
from research_gym.interp.goal_guarding_protected_tail import (
    COHORTS,DEPTHS,REPLICATES,BLOCK_N,
    challenge_pools,cohort_effect_bounds,score_record,aggregate,
)
from research_gym.interp.goal_guarding_competency_recovery import (
    SOURCE_MAIN,MODEL_SEEDS,BASE_UPDATES,CONTINUATION_UPDATES,
    POOL_SIZE,FOUR_BATCH,MIXED_BATCH,SAMPLE_VISITS,
    training_pools,continue_training,fresh_bank,accuracy_cells,
    sign_and_robustness,run_gg08,
)


def model(seed=76):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return EvaluationBeliefLooper().eval()


def test_registered_exact_training_budget_and_source():
    assert SOURCE_MAIN=="97b3eba35a5e0c037fc4f5f65b962e68b07bbdbb"
    assert MODEL_SEEDS==(11,17)
    assert (BASE_UPDATES,CONTINUATION_UPDATES,POOL_SIZE)==(400,300,8192)
    assert FOUR_BATCH*4*CONTINUATION_UPDATES==SAMPLE_VISITS
    assert 100*MIXED_BATCH*sum(DEPTHS)==SAMPLE_VISITS
    assert (FOUR_BATCH,MIXED_BATCH,SAMPLE_VISITS)==(192,128,230400)
    assert len(query_splits()["heldout"])==48
    for row in query_splits()["heldout"]:
        assert row.label==exact_action(row.prior,row.ratio,row.bits)


def test_independent_train_pools_are_fixed_and_have_full_prefix_labels():
    pools=training_pools(11)
    again=training_pools(11)
    assert set(pools)==set(DEPTHS)
    for depth,pool in pools.items():
        assert len(pool)==POOL_SIZE
        assert pool.observations.shape==(POOL_SIZE,depth)
        assert pool.prefix_labels.shape==(POOL_SIZE,depth)
        assert torch.equal(pool.observations,again[depth].observations)
        assert torch.equal(pool.labels,again[depth].labels)
        for i in (0,1,50,190):
            p,ratio=pool.parameters[i]
            bits=tuple(map(int,pool.observations[i]))
            assert pool.labels[i]==exact_action(p,ratio,bits)
            assert tuple(map(int,pool.prefix_labels[i]))==tuple(
                exact_action(p,ratio,bits[:t])
                for t in range(1,depth+1))
    different=training_pools(17)
    assert not torch.equal(pools[8].observations,different[8].observations)
    with pytest.raises(ValueError,match="frozen"):
        training_pools(99)


def test_new_gold_panels_retain_original_GG07_oracle_cohorts():
    from research_gym.interp.goal_guarding_protected_tail import (
        build_gold_bank,
    )
    for depth in DEPTHS:
        new=fresh_bank(depth)
        reloaded=fresh_bank(depth)
        old=build_gold_bank(depth)
        assert new.observations.shape==(640,depth)
        assert new.contexts.shape==(640,2)
        assert new.labels.shape==(640,)
        assert len(new.reps)==REPLICATES
        assert [r.hashes for r in new.reps]==[
            r.hashes for r in reloaded.reps]
        assert any(new.reps[i].hashes["natural"]!=
                   old.reps[i].hashes["natural"] for i in range(REPLICATES))
        assert set(new.cohort_slices)==set(COHORTS)
        pool=challenge_pools(depth)
        keys={name:{x.identity() for x in pool[name]}
              for name in ("boundary","rare","late")}
        for rep in new.reps:
            assert set(rep.cohorts)==set(COHORTS)
            for name,gold in rep.cohorts.items():
                assert len(gold)==(64 if name=="natural" else 32)
                for i,(prior,ratio) in enumerate(gold.parameters):
                    bits=tuple(map(int,gold.observations[i]))
                    assert int(gold.labels[i])==exact_action(prior,ratio,bits)
                    if name!="natural":
                        assert (str(prior),str(ratio),bits) in keys[name]
    with pytest.raises(ValueError,match="registered"):
        fresh_bank(10)


def test_native_gold_model_accuracy_and_protected_score_preserve_original_math():
    native=model(81)
    banks={d:fresh_bank(d) for d in DEPTHS}
    scores,ce=accuracy_cells(native,banks)
    assert set(scores)=={"4","6","8"}
    assert set(ce)==set(DEPTHS)
    for depth in scores:
        assert set(scores[depth])==set(COHORTS)
        assert all(0<=v<=1 for v in scores[depth].values())
    for depth in DEPTHS:
        assert ce[depth].shape==(4,160)
        # Synthetic uniform positive report effects should be positively
        # robust to all GG-07 cohort-TV and depth-simplex perturbations.
    rec={d:cohort_effect_bounds(
        torch.ones(4,160,dtype=torch.float64),
        banks[d]) for d in DEPTHS}
    scoring=score_record(rec)
    summary=aggregate([{"composite":scoring} for _ in range(48)])
    assert summary["primary_protected_summary"]["harm"]==48
    assert summary["primary_protected_summary"]["unresolved"]==0


def test_unchanged_native_report_SGD_and_independent_gold_auditor():
    model_0=model(57)
    sha=_weights_digest(model_0)
    bank={4:fresh_bank(4)}
    row=query_splits()["heldout"][0]
    from research_gym.interp.goal_guarding_protected_tail import (
        native_report_successor_losses,
    )
    effects,norms=native_report_successor_losses(model_0,row,bank,sha)
    assert set(norms)=={"truth","flip"}
    assert effects[4]["gap"].shape==(4,160)
    assert torch.allclose(effects[4]["gap"],
                          effects[4]["flip"]-effects[4]["truth"])
    scoring=CuePanel(
        bank[4].observations[:160],bank[4].contexts[:160],
        bank[4].labels[:160],torch.zeros(160,4,dtype=torch.long),
        tuple((Q(1,2),Q(1)) for _ in range(160)),
    )
    independent=SealedGoldAuditor(model_0,scoring).measure(row)
    assert float(effects[4]["gap"][0].mean())==pytest.approx(
        independent.damage_gap,abs=3e-6)
    assert _weights_digest(model_0)==sha


def test_continuation_guard_catches_invalid_pools_before_model_updates():
    native=model(55)
    pools=training_pools(11)
    digest=_weights_digest(native)
    with pytest.raises(ValueError,match="continuation arm"):
        continue_training(native,11,"invalid",pools)
    with pytest.raises(ValueError,match="frozen"):
        continue_training(native,11,"mixed",pools,steps=16)
    with pytest.raises(ValueError,match="frozen training pools"):
        continue_training(native,11,"mixed",{4:pools[4]})
    assert _weights_digest(native)==digest


def test_full_runner_refuses_any_change_to_registered_training_budget(tmp_path):
    with pytest.raises(ValueError,match="frozen"):
        run_gg08(output=tmp_path/"out.json",artifacts=tmp_path,
                 seeds=(11,))
    with pytest.raises(ValueError,match="frozen"):
        run_gg08(output=tmp_path/"out.json",artifacts=tmp_path,
                 base_updates=10)
    with pytest.raises(ValueError,match="frozen"):
        run_gg08(output=tmp_path/"out.json",artifacts=tmp_path,
                 continuation_updates=32)
