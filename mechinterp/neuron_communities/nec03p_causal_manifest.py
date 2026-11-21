"""NEC-03P future development causal pilot contract, without reading outputs.

This file writes a signed-style immutable JSON manifest of the fixed
root/arm/dose/comparison policy. It never loads a native checkpoint,
generates development inputs, or computes a development-evaluation label.
Execution requires a later separately approved causal-run implementation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

SCHEMA="NEC-03P/development-only-causal-pilot-manifest/v1"
ROOTS=tuple(range(8))
PREVIOUS_SOURCE_ELIGIBLE=(0,1,2,4,6,7)
PAIRED_SOURCE_FAILURES={
    3:"reset1800 additive native source candidate missing (visit-8 hub saturation)",
    5:"reset1800 composed native head source competence 2237/2560 below 90%",
}
REGISTRY_SHA="6777d478f6e988e3f57f8882dfef2e84898f765b7adf7e142a914b4cbdcd1217"
ORIGINAL_CONFIRMATION_SHA="670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39"


def canonical(payload):
    return (json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n").encode()


def development_pilot_manifest():
    from mechinterp.neuron_communities.nec03l_five_input_worlds import (
        worlds,registration,
    )
    from mechinterp.neuron_communities.nec03g_worlds import registry as original
    w=worlds()
    if (registration()["registry_sha256"]!=REGISTRY_SHA
        or original()["confirmation_full_spec_sha256"]!=ORIGINAL_CONFIRMATION_SHA
        or set(PREVIOUS_SOURCE_ELIGIBLE)|set(PAIRED_SOURCE_FAILURES)!=set(ROOTS)):
        raise AssertionError("pilot source eligibility or registered world seals changed")
    roots=[]
    for i in ROOTS:
        world=w[i]
        if world["world_id"]!=f"dev5-{i:03d}" or world["cohort"]!="development":
            raise AssertionError("invalid pilot world membership")
        roots.append({
            "world_id":world["world_id"],
            "world_index":i,
            "world_sha256":world["world_sha256"],
            "source_model_run":44898361742 if i<4 else 44900628360,
            "source_SAE_run":44899750508 if i<4 else 44900628360,
            "source_preflight_eligible":i in PREVIOUS_SOURCE_ELIGIBLE,
            "source_ITT_failure_reason":PAIRED_SOURCE_FAILURES.get(i),
            "development_context_groups_prospective":24,
            "focal_masks_per_context":32,
            "source_eligibility_known_before_dev_evaluation":True,
        })
    return {
        "schema":SCHEMA,
        "purpose":"development-only causal-resolution pilot; never confirmation",
        "pilot_world_root_indices":list(ROOTS),
        "root_unit":"one independently seeded five-input Boolean world; composed/additive arms paired within root",
        "root_count_ITT":8,
        "source_eligible_roots":list(PREVIOUS_SOURCE_ELIGIBLE),
        "source_ineligible_roots_ITT_zero":sorted(PAIRED_SOURCE_FAILURES),
        "primary_endpoint":"per-root signed difference-in-differences of source and angle-matched sham task correctness",
        "primary_root_effect_definition":"((composed_source_correct-composed_sham_correct)-(additive_source_correct-additive_sham_correct))/768, else zero if paired source-ineligible",
        "primary_root_effect_direction":"positive",
        "primary_comparisons":1,
        "minimum_scientifically_meaningful_effect_fraction":.02,
        "primary_root_macro_mean_denominator":8,
        "primary_ITT_missing_root_imputation":"0; all eight registered roots in denominator",
        "primary_outcome_binary_truth":"original five-focal-bit composed truth table and independent same-prior additive teacher",
        "development_focal_histories_per_native_arm":768,
        "development_context_groups_per_root":24,
        "source_focal_histories_per_native_arm":2560,
        "source_rank_split_first40_fit_last40_rank":True,
        "recurrent_visits":12,
        "native_checkpoint_training_recipe":"frozen source900, byte-reload, fresh AdamW reset +900 source-only updates",
        "source_candidate_policy":"frozen source graph or original deterministic two-per-visit nonhub fallback; no evaluation-based sites",
        "sae_policy":"new checkpoint-specific TopK16/128, source-only 480-step, fixed 40/40 rank",
        "sham_policy":"every disjoint equal-n subset from remaining active TopK16 codes; minimum 32D gradient cosine gap",
        "orientation_policy":"clean-chosen logit margin, independent of teacher correctness labels",
        "physical_dose_policy":"attenuation-only min(source32 L2,sham32 L2) at post-mlp_down, matched within 1pct",
        "site_angle_absolute_cosine_gap_max":.25,
        "missing_per_input_control_policy":"entire input uses hubs-only donor for both arms of the source-versus-sham comparison; zero signed difference",
        "source_admission_thresholds":{"native_accuracy":.9,"sae_mse_max":.03,
            "source_angle_coverage_min":.8,"source_positive_dose_min":.8,
            "median_retained_dose_min":.25},
        "secondary_endpoints":"per-arm effects, memory trajectory and linear/nonlinear margin decomposition, descriptive only",
        "root_sign_flip_test":"exploratory exact null mean 0 under explicitly checked root-sign exchangeability",
        "fsi_01_finite_effect_bound_margin_at_8_roots":math.sqrt(2*math.log(1/.05)/8),
        "pilot_readout_status":"SEALED_UNOPENED",
        "readout_selection_history":"all eight root source outcomes, feature choices and instruments were inspected before this pilot registration",
        "source_registry_sha256":REGISTRY_SHA,
        "four_input_confirmation_sha256":ORIGINAL_CONFIRMATION_SHA,
        "source_training_models_retrained_for_pilot":0,
        "development_evaluation_contexts_scored":0,
        "reserved_contexts_scored":0,
        "embargoed_five_input_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "statistical_verdict":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }


def write_manifest(out:Path):
    item=development_pilot_manifest()
    body=canonical(item)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(body)
    return sha256(body).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    print("NEC03P_SEALED_CAUSAL_PILOT_MANIFEST",write_manifest(a.out))


if __name__=="__main__":
    main()
