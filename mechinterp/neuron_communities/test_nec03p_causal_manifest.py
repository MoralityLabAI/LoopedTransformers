"""NEC-03P developmental causal pilot manifest, with sealed readout."""
from hashlib import sha256
import json
import math

from mechinterp.neuron_communities.nec03p_causal_manifest import (
    ROOTS,PREVIOUS_SOURCE_ELIGIBLE,PAIRED_SOURCE_FAILURES,
    development_pilot_manifest,write_manifest,
)


def test_eight_worlds_fixed_with_itt_failures():
    z=development_pilot_manifest()
    assert ROOTS==tuple(range(8))
    assert z["pilot_world_root_indices"]==list(range(8))
    assert z["root_count_ITT"]==8
    assert z["primary_root_macro_mean_denominator"]==8
    assert z["source_eligible_roots"]==[0,1,2,4,6,7]
    assert z["source_ineligible_roots_ITT_zero"]==[3,5]
    assert set(PREVIOUS_SOURCE_ELIGIBLE)|set(PAIRED_SOURCE_FAILURES)==set(ROOTS)
    assert len({a["world_sha256"] for a in z["roots"]})==8
    assert len({a["world_id"] for a in z["roots"]})==8
    assert all(a["source_preflight_eligible"]==(a["world_index"] in PREVIOUS_SOURCE_ELIGIBLE)
               for a in z["roots"])


def test_primary_effect_is_a_root_level_one_comparison_pilot():
    z=development_pilot_manifest()
    assert z["primary_comparisons"]==1
    assert z["minimum_scientifically_meaningful_effect_fraction"]==.02
    assert z["primary_root_effect_direction"]=="positive"
    assert "/768" in z["primary_root_effect_definition"]
    assert z["development_focal_histories_per_native_arm"]==768
    assert z["source_focal_histories_per_native_arm"]==2560
    assert z["source_rank_split_first40_fit_last40_rank"]
    assert z["site_angle_absolute_cosine_gap_max"]==.25
    assert "attenuation-only" in z["physical_dose_policy"]
    assert z["primary_ITT_missing_root_imputation"].startswith("0")
    assert math.isclose(z["fsi_01_finite_effect_bound_margin_at_8_roots"],.8654091913011427,rel_tol=1e-9)


def test_manifest_creation_does_not_open_any_prospective_readout(tmp_path):
    p=tmp_path/"manifest.json"
    identity=write_manifest(p)
    assert identity==sha256(p.read_bytes()).hexdigest()
    z=json.loads(p.read_text())
    assert z["pilot_readout_status"]=="SEALED_UNOPENED"
    assert z["source_training_models_retrained_for_pilot"]==0
    assert z["development_evaluation_contexts_scored"]==0
    assert z["reserved_contexts_scored"]==0
    assert z["embargoed_five_input_worlds_scored"]==0
    assert z["original_four_input_confirmation_worlds_scored"]==0
    assert z["statistical_verdict"]=="REDESIGN_OR_PILOT_ONLY"
    assert z["source_registry_sha256"]=="6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217"
