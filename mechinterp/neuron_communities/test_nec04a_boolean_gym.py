"""NEC-04A root commitments, generator and independent oracle tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import (
    oracle,verify_lock,EXPECTED_ROOT_REGISTRY,EXPECTED_SEED_HASH_ROOT,
    EXPECTED_LAYOUT_HASH_ROOT,
)


def test_64_root_registry_committed_before_any_program():
    cfg,digest=verify_lock()
    assert len(digest)==64
    r=gym.commitments()
    assert r["registration_sha256"]==EXPECTED_ROOT_REGISTRY
    assert cfg["seed_registry_sha256"]==EXPECTED_ROOT_REGISTRY
    assert r["split_counts"]=={"calibration":24,"development":24,"confirmation":16}
    roots=r["registration_worlds"]
    assert len(roots)==64
    assert len({x["root_id"] for x in roots})==64
    assert len({x["seed_commitment_sha256"] for x in roots})==64
    assert len({x["original_focal_layout_commitment_sha256"] for x in roots})==64
    assert roots[0]["root_id"]=="nec04a-cal-000"
    assert roots[24]["root_id"]=="nec04a-dev-000"
    assert roots[48]["root_id"]=="nec04a-con-000"
    seed_hash=hashlib.sha256(
        "".join(x["seed_commitment_sha256"] for x in roots).encode()
    ).hexdigest()
    layout_hash=hashlib.sha256(
        "".join(x["original_focal_layout_commitment_sha256"] for x in roots).encode()
    ).hexdigest()
    assert seed_hash==EXPECTED_SEED_HASH_ROOT
    assert layout_hash==EXPECTED_LAYOUT_HASH_ROOT
    assert r["development_programs_generated"]==0
    assert r["confirmation_programs_generated"]==0
    assert r["old_NEC03P_recruited_models_scored"]==0
    assert r["cryptographically_hidden_world_specifications"] is False
    assert r["FSI_01_gate"]=="REDESIGN_OR_PILOT_ONLY"
    assert all(not e["program_materialized"] and not e["outcome_scored"] for e in roots)


def test_root_classification_and_strict_calibration_only_materialization():
    assert gym.root_tier("calibration",0)=="easy"
    assert gym.root_tier("calibration",1)=="medium"
    assert gym.root_tier("calibration",2)=="hard"
    assert gym.root_tier("development",0)=="easy"
    assert gym.root_tier("confirmation",2)=="hard"
    assert gym.global_root_number("calibration",23)==23
    assert gym.global_root_number("development",0)==24
    assert gym.global_root_number("confirmation",0)==48
    for idx in (-1,24,64):
        with pytest.raises(PermissionError):
            gym.calibration_world(idx)
    with pytest.raises(PermissionError):
        gym.root_id("old_NEC03P",0)
    with pytest.raises(PermissionError):
        gym.root_id("confirmation",16)
    with pytest.raises(PermissionError):
        gym.root_id("development",24)


def test_boolean_gate_semantics_independent_of_bit_parallel_oracle():
    f0=["VAR",0]
    f1=["VAR",1]
    f2=["VAR",2]
    examples=[
        ["XOR",f0,f1],
        ["AND",f0,["NOT",f1]],
        ["MUX",f2,["XOR",f0,f1],["NOT",f0]],
    ]
    for expr in examples:
        table=gym.evaluate_bit_parallel(expr)
        for history in range(1<<gym.TOTAL_BITS):
            bits=[(history>>j)&1 for j in range(12)]
            assert bool((table>>history)&1)==oracle(expr,bits)
    with pytest.raises(AssertionError):
        oracle(["OR",f0,f1],[0]*12)


def test_calibration_root_program_is_balanced_essential_and_budgeted():
    for index in (0,1,2):
        world,truth=gym.calibration_world(index)
        assert world["cohort"]=="calibration"
        assert world["root_index"]==index
        tier=gym.TIERS[index%3]
        assert world["difficulty_tier"]==tier
        assert world["program_gates"]==gym.TIER_GATES[tier]
        assert world["program_depth"]<=gym.TIER_MAX_DEPTH[tier]
        assert world["candidate_attempt_index"]<gym.MAX_PROGRAM_ATTEMPTS
        assert world["program_sha256"]==gym.digest(world["canonical_program"])
        assert len(world["program_metrics"]["essential_focal_bits"])==5
        assert len(world["program_metrics"]["essential_context_bits"])>=3
        assert world["program_metrics"]["balanced_focal_context_groups"]>=64
        assert gym.BALANCE_BAND[0]<=world["program_metrics"]["class_one_rate"]<=gym.BALANCE_BAND[1]
        assert not world["model_baseline_accuracy_measured"]
        assert world["measured_model_headroom_band"] is None
        assert gym.evaluate_bit_parallel(world["canonical_program"])==truth


def test_original_full_32_mask_source_context_ownership():
    world,truth=gym.calibration_world(0)
    source,parts=gym.contexts_for_world(0)
    assert len(source)==80
    assert len(parts["development"])==24 and len(parts["reserved"])==24
    assert len(set(source)|set(parts["development"])|set(parts["reserved"]))==128
    assert not set(source)&set(parts["development"])
    assert not set(source)&set(parts["reserved"])
    rows=list(gym.source_rows(0,world,truth))
    assert len(rows)==80*32==2560
    assert len({tuple(row["history_bits12"]) for row in rows})==2560
    for group in range(80):
        block=rows[group*32:(group+1)*32]
        assert [row["focal_mask"] for row in block]==list(range(32))
        assert len({tuple(row["history_bits12"][i] for i in world["context_positions"])
                    for row in block})==1
    for row in rows:
        assert row["split"]=="source_train"
        assert row["cohort"]=="calibration"
        assert row["label"]==int(oracle(world["canonical_program"],row["history_bits12"]))


def test_deterministic_rerun_same_source_without_tuning_model_accuracy():
    a,ta=gym.calibration_world(2)
    b,tb=gym.calibration_world(2)
    assert a==b and ta==tb
    assert a["world_sha256"]==b["world_sha256"]
    assert a["program_sha256"]==b["program_sha256"]
    assert a["native_models_trained"]==0 and b["source_SAEs_fitted"]==0
    assert gym.commitments()["registration_sha256"]==EXPECTED_ROOT_REGISTRY


def test_lock_preserves_previous_protected_cohorts():
    cfg,_=verify_lock()
    assert cfg["old_NEC03Z_sealed_analysis_groups"]==list(range(56,64))
    assert cfg["original_NEC03P_recruited_roots_scored"]==0
    assert cfg["old_NEC03_embargo_or_confirmation_scored"]==0
    assert cfg["development_or_confirmation_labels_scored"]==0
    assert cfg["native_or_SAE_training_updates"]==0
    assert cfg["FSI_01_gate"]=="REDESIGN_OR_PILOT_ONLY"
