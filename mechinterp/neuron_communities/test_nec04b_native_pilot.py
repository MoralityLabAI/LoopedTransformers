"""NEC-04B six calibration root oracle, split, model and gate tests."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04b_oracles import (
    SCHEMA,PILOT,ARMS,BUDGETS,SOURCE,VALIDATION,
    EXPECTED_CORPUS_SHA,EXPECTED_REGISTRY,
    additive_spec,additive_weights,additive_label,
    root_checkpoint_seed,rows_for_owned_validation,verify_lock,
)
from mechinterp.neuron_communities.nec04b_independent_oracles import independently_linear_spec
from mechinterp.neuron_communities.nec04b_native_pilot import BATCH,LR,WEIGHT_DECAY,CLIP,run
from mechinterp.neuron_communities.nec04b_aggregate import BANDS,band_for_accuracy
from research.pef07.pef07 import Looper,model_digest


def test_root_teacher_and_checkpoint_preregistration():
    cfg,sha=verify_lock()
    assert len(sha)==64
    assert cfg["pilot_calibration_root_indices"]==list(PILOT)==list(range(6))
    assert cfg["pilot_model_teacher_arms"]==list(ARMS)
    assert cfg["native_training_steps"]==list(BUDGETS)==[120,240,480,900]
    assert cfg["original_64_root_registry_sha256"]==EXPECTED_REGISTRY
    assert cfg["original_source_calibration_corpus_sha256"]==EXPECTED_CORPUS_SHA
    assert cfg["native_batch_size"]==BATCH==64
    assert cfg["native_steps_max"]==900
    assert cfg["model_architecture"].startswith("research.pef07.pef07.Looper")
    assert len({root_checkpoint_seed(i,arm) for i in PILOT for arm in ARMS})==12
    assert cfg["development_world_specs_materialized"]==0
    assert cfg["confirmation_world_specs_materialized"]==0
    assert cfg["old_NEC03Z_source_group_56_63_scored"]==0
    assert cfg["old_NEC03P_recruited_worlds_scored"]==0
    assert cfg["FSI_01_gate"]=="REDESIGN_OR_PILOT_ONLY"


def test_pilot_roots_two_per_syntactic_tier_and_no_world_recruitment():
    assert [gym.root_tier("calibration",i) for i in PILOT]==[
        "easy","medium","hard","easy","medium","hard"
    ]
    for i in (-1,6,23,24,96):
        with pytest.raises(PermissionError):
            additive_spec(i)
        with pytest.raises(PermissionError):
            root_checkpoint_seed(i,"composed")
    with pytest.raises(ValueError):
        root_checkpoint_seed(0,"unknown")
    with pytest.raises(PermissionError):
        gym.calibration_world(24)


def test_additive_oracle_root_heldout_independent_truth():
    for i in PILOT:
        world,truth=gym.calibration_world(i)
        spec,truth_add=additive_spec(i)
        weights,threshold,num,table_hash,independent_truth=independently_linear_spec(i,world)
        assert weights==additive_weights(i)==spec["signed_weights_12"]
        assert threshold==spec["threshold_twice"]
        assert num==spec["positive_count_full4096"]
        assert table_hash==spec["truth_table_4096_sha256"]
        assert independent_truth==truth_add
        assert .35<=num/4096<=.65
        assert len(spec["essential_focal_bits"])==5
        assert len(spec["essential_context_bits"])>=3
        for sample in (0,1,3,31,127,511,1023,2048,4095):
            bits=[(sample>>k)&1 for k in range(12)]
            assert additive_label(bits,spec)==((truth_add>>sample)&1)
            assert int(oracle(world["canonical_program"],bits))==((truth>>sample)&1)


def test_calibration_world_owned_context_validation_is_disjoint():
    for index in PILOT:
        world,_=gym.calibration_world(index)
        spec,_=additive_spec(index)
        source,parts=gym.contexts_for_world(index)
        assert len(source)==80
        assert len(parts["development"])==24
        assert len(parts["reserved"])==24
        assert len(set(source)|set(parts["development"])|set(parts["reserved"]))==128
        assert not (set(source)&set(parts["development"]))
        assert not (set(parts["development"])&set(parts["reserved"]))
        val=list(rows_for_owned_validation(index,world,spec))
        assert len(val)==VALIDATION
        assert [val[i]["focal_mask"] for i in range(32)]==list(range(32))
        assert set(row["split"] for row in val)=={"calibration_world_validation"}
        assert set(row["world_id"] for row in val)=={f"nec04a-cal-{index:03d}"}
        assert set(row["context_group"] for row in val)==set(range(80,104))
        assert len({tuple(row["history_bits12"]) for row in val})==VALIDATION
        assert set(row["composed_label"] for row in val)=={0,1}
        assert set(row["additive_label"] for row in val)=={0,1}


def test_headroom_bands_are_disjoint_and_upper_bound():
    assert band_for_accuracy(.59999)=="outside_headroom"
    assert band_for_accuracy(.60)=="moderate"
    assert band_for_accuracy(.75)=="intermediate"
    assert band_for_accuracy(.90)=="high"
    assert band_for_accuracy(.98)=="high"
    assert band_for_accuracy(.981)=="outside_headroom"
    assert [BANDS[x] for x in ("moderate","intermediate","high")]==[
        (.60,.75),(.75,.90),(.90,.98)
    ]


def test_pilot_trainer_rejects_unregistered_world_before_checkpoint(tmp_path):
    for bad in (-1,6,23,24,96):
        with pytest.raises(PermissionError):
            run(bad,tmp_path/"does-not-exist",tmp_path/"root")
    assert not (tmp_path/"root").exists()


def test_two_step_native_torch_train_and_frozen_model_digest_smoke():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(701)
    model=Looper().train()
    start=model_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    bits=torch.randint(2,(4,12),dtype=torch.long)
    labels=torch.tensor([0,1,1,0],dtype=torch.long)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(bits),labels)
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        assert torch.isfinite(grad)
        optimizer.step()
    assert model_digest(model)!=start
    assert all(int(s["step"].item())==2
               for s in optimizer.state_dict()["state"].values() if "step" in s)
