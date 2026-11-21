"""NEC-03P source-only recruitment contract and finite-sample tests."""
from __future__ import annotations

import math
from pathlib import Path
import pytest

from mechinterp.neuron_communities.nec03p_power import (
    plan, binomial_tail, exact_cp_interval, fsi_effect_margin,
    minimum_tolerance_roots,
)
from mechinterp.neuron_communities.nec03p_source_recruit import (
    ROOTS, verify_lock, run, NATIVE_STEPS, RESET_STEPS, SAE_STEPS,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, context_catalog,
)


def test_source_root_and_embargo_lock():
    lock=verify_lock()
    assert ROOTS==tuple(range(8,96))
    assert len(ROOTS)==88
    assert len({worlds()[i]["world_sha256"] for i in ROOTS})==88
    assert lock["development_source_recruitment_world_indices"]==list(ROOTS)
    assert lock["preceding_source_B_verified_run_id"]==44900628360
    assert lock["source_B_feasible_model_arms"]==8
    assert lock["source_B_complete_feasible_worlds"]==3
    assert (NATIVE_STEPS,RESET_STEPS,SAE_STEPS)==(900,900,480)
    assert lock["source_training_groups"]==80
    assert lock["source_fit_groups"]==lock["source_rank_groups"]==40
    assert lock["source_geometry_groups"]==8
    assert lock["max_direction_cosine_gap"]==.25
    assert lock["native_competence_floor"]==.9
    for key in (
        "new_development_evaluation_groups_scored",
        "reserved_context_groups_scored",
        "embargoed_worlds_scored",
        "four_input_confirmation_worlds_scored",
    ):
        assert lock[key]==0


def test_root_membership_stops_before_work(tmp_path):
    for i in (-1,0,1,4,7,96,127):
        with pytest.raises(PermissionError):
            run(i,tmp_path/"outputs")
    assert not (tmp_path/"outputs").exists()


def test_sources_40_40_and_all_focal_masks_preserved():
    for i in (8,15,31,63,95):
        w=worlds()[i]
        catalog, manifest=context_catalog(w)
        assert w["world_id"]==f"dev5-{i:03d}"
        assert {k:len(v) for k,v in catalog.items()}=={
            "train":2560,"development":768,"reserved":768}
        assert len(set(catalog["train"])&set(catalog["development"]))==0
        assert len(set(catalog["train"])&set(catalog["reserved"]))==0
        assert manifest["source_fit_context_count"]==40
        assert manifest["source_rank_context_count"]==40
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_59_root_95_95_tolerance_is_exactly_registered():
    assert minimum_tolerance_roots(.95,.95)==59
    assert .95**58>.05 and .95**59<=.05
    assert minimum_tolerance_roots(.9,.9)==22


def test_exact_recruitment_probability_is_only_sensitivity():
    assert math.isclose(binomial_tail(88,59,.75),.96450173278901,abs_tol=1e-12)
    assert math.isclose(binomial_tail(88,59,.65),.38975424022546656,abs_tol=1e-12)
    lo,hi=exact_cp_interval(6,8)
    assert .34<lo<.36 and .96<hi<.98


def test_fsi_root_macro_effect_resolution_requires_pilot():
    assert math.isclose(fsi_effect_margin(32),.43270459565057134,abs_tol=1e-12)
    assert math.isclose(fsi_effect_margin(88),.2609306888791626,abs_tol=1e-12)
    z=plan()
    assert z["source_recruitment_root_count"]==88
    assert z["sample_maximum_95quantile_95confidence_minimum_independent_eligible_roots"]==59
    assert z["causal_results_measured"] is False
    assert z["scientific_gate"]=="REDESIGN_OR_PILOT_ONLY"
    assert z["five_input_development_evaluation_groups_scored"]==0
    assert z["original_four_input_confirmation_worlds_scored"]==0
