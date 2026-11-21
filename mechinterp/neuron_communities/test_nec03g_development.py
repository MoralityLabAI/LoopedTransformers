"""NEC-03G: root-level train and fail-closed development controls."""
import hashlib
import json

import pytest
import torch

from research.pef07.pef07 import model_digest
from mechinterp.neuron_communities.nec03g_worlds import (
    worlds, context_catalog, target_bits, additive_control_truth,
)
from mechinterp.neuron_communities.nec03g_development import (
    train_one, clean_score, root_effects, run, sha_file,
)


def test_short_native_trunk_training_freezes_source_only_weights(tmp_path):
    torch.set_num_threads(1)
    world=worlds()[0]
    cat,source=context_catalog(world)
    model,receipt=train_one(world,cat,tmp_path,arm="composed",steps=2)
    assert receipt["steps"]==2
    assert receipt["trained_on_source_train_only"]
    assert receipt["frozen_before_discovery"]
    assert sha_file(tmp_path/receipt["checkpoint_name"])==receipt["checkpoint_sha256"]
    assert model_digest(model)==receipt["model_weights_sha256"]
    score=clean_score(model,cat["development"],world,arm="composed")
    assert score["total"]==768
    with pytest.raises(ValueError):
        train_one(world,cat,tmp_path,arm="invalid",steps=2)


def test_matched_additive_controls_same_teacher_balance():
    w=worlds()[2]
    cat,_=context_catalog(w)
    bits=torch.tensor(cat["train"][:16],dtype=torch.long)
    y=target_bits(bits,w).tolist()
    a=target_bits(bits,w,control=True).tolist()
    assert sum(y)==sum(a)==w["truth_table_16"].bit_count()
    assert y != a


def test_intention_to_treat_zero_for_no_candidate_and_incompetent_model():
    world=worlds()[0]
    report={
        "composed":{"clean":{"accuracy":0.88},"candidate_status":"NO_SOURCE_COMMUNITY"},
        "additive":{"clean":{"accuracy":1.0},"candidate_status":"SOURCE_SELECTED"}}
    e=root_effects(world,report)
    assert e["root_primary_effect"]=="0"
    assert e["sham_effects"]==["0"]
    assert "composed_incompetent_clean_head" in e["failure_reasons"]
    assert not e["planned_world_valid"]


def test_primary_accuracy_and_development_controls_are_signed_and_bounded():
    world=worlds()[0]
    def arm(a,b,c):
        return {"clean":{"accuracy":1.0},"candidate_status":"SOURCE_SELECTED",
                "pairs":[
                    {"pair":["topk","topk_alt"],
                     "left_minus_right_accuracy":a,
                     "physical_dose":{"candidate_physically_comparable_and_signal_sufficient":True}},
                    {"pair":["native","sham"],
                     "left_minus_right_accuracy":b,
                     "physical_dose":{"candidate_physically_comparable_and_signal_sufficient":True}},
                    {"pair":["native","topk"],
                     "left_minus_right_accuracy":c,
                     "physical_dose":{"candidate_physically_comparable_and_signal_sufficient":True}},
                ]}
    effect=root_effects(world,{"composed":arm(.03,-.02,.01),
                               "additive":arm(.02,.01,0.)})
    assert effect["planned_world_valid"]
    assert float(effect["root_primary_effect"])==pytest.approx(.03)
    assert list(map(float,effect["sham_effects"]))==pytest.approx([.02,-.02,.01])
    assert effect["observed_specificity_difference_in_differences"]==pytest.approx(.01)


def test_development_runner_cannot_open_confirmation_checkpoint(tmp_path):
    with pytest.raises(PermissionError,match="confirmation"):
        run(64,tmp_path,trunk_steps=2,sae_steps=2)
    with pytest.raises(PermissionError,match="confirmation"):
        run(127,tmp_path,trunk_steps=2,sae_steps=2)
