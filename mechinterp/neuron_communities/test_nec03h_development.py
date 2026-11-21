"""NEC-03H v2 tests: independently gated primary and untouched cohort."""
from __future__ import annotations

import pytest

from mechinterp.neuron_communities.nec03h_development import (
    root_effects,run,MODEL_STEPS,SAE_STEPS,
)
from mechinterp.neuron_communities.nec03h_engine import training_world


def _arm(*,clean=1.,site="fallback",primary=0.03,primary_dose=True,
         other=True,sham=0.01):
    return {
        "clean":{"accuracy":clean,"correct":round(768*clean),"total":768},
        "candidate_status":"SOURCE_SELECTED" if site!="none" else "NO_SOURCE_COMMUNITY",
        "source_site_kind":site,
        "pairs":[] if site=="none" else [
            {"pair":["topk","topk_alt"],"left_minus_right_accuracy":primary,
             "physical_dose":{"candidate_physically_comparable_and_signal_sufficient":primary_dose}},
            {"pair":["native","sham"],"left_minus_right_accuracy":sham,
             "physical_dose":{"candidate_physically_comparable_and_signal_sufficient":other}},
            {"pair":["native","topk"],"left_minus_right_accuracy":-.005,
             "physical_dose":{"candidate_physically_comparable_and_signal_sufficient":other}},
        ]}


def test_v2_primary_is_preserved_if_additive_sham_site_absent():
    root=training_world(8,calib=False)
    r=root_effects(root,{
        "composed":_arm(primary=.03,site="fallback"),
        "additive":_arm(site="none"),
    })
    assert r["planned_world_valid"]
    assert r["root_primary_effect"]=="0.03"
    assert not r["additive_sham_valid"]
    assert r["sham_effects"]==["1.0","0.01"]
    assert not r["null_floor_calibration_eligible"]
    assert r["control_failures"]==["additive_sham_unavailable"]


def test_invalid_composed_primary_is_intention_to_treat_zero():
    world=training_world(8,calib=False)
    for model in (_arm(clean=.88), _arm(site="none"),
                  _arm(primary_dose=False)):
        r=root_effects(world,{"composed":model,
                              "additive":_arm(primary=.04)})
        assert not r["planned_world_valid"]
        assert r["root_primary_effect"]=="0"
        assert r["failure_reasons"]


def test_auxiliary_native_sham_is_falsification_not_primary_gate():
    world=training_world(9,calib=False)
    result=root_effects(world,{
        "composed":_arm(primary=.022,other=False),
        "additive":_arm(primary=-.002),
    })
    assert result["root_primary_effect"]=="0.022"
    assert result["planned_world_valid"]
    assert result["sham_effects"]==["-0.002","1.0"]
    assert result["control_failures"]==["native_sham_unavailable"]


def test_additive_control_held_separate_and_registered_when_valid():
    world=training_world(10,calib=False)
    result=root_effects(world,{
        "composed":_arm(primary=.04,sham=-.01),
        "additive":_arm(primary=.005,sham=.02),
    })
    assert result["null_floor_calibration_eligible"]
    assert result["sham_effects"]==["0.005","-0.01"]
    assert result["root_primary_effect"]=="0.04"


def test_new_runner_cannot_touch_old_development_or_confirmation(tmp_path):
    for index in (0,7,64,127):
        with pytest.raises(PermissionError):
            run(index,tmp_path/"forbidden",trunk_steps=2,sae_steps=2)
    assert MODEL_STEPS==900
    assert SAE_STEPS==480
