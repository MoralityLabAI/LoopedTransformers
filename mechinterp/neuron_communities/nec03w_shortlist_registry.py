"""NEC03W original source-only calibration shortlist pre-execution SHA registry.

This source-selection stage is purely label blind and produces zero model
forwards and zero task outcome reads. Later functional behavioral inference
requires a distinct pre-execution lock and independent native 12-pass replay.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path


def registered_shortlist():
    here=Path(__file__).parent
    entry=json.loads((here/"NEC03W_CALIBRATION_SHORTLIST_LOCK.json").read_text("utf-8"))
    expected={
        "schema":"NEC-03W/preregistered-source-calibration-physical-shortlist/v1",
        "allowed_original_roots":list(range(8)),
        "original_discovery_groups":list(range(16)),
        "original_calibration_groups":list(range(16,24)),
        "previously_behavior_scored_groups_excluded":list(range(24,32)),
        "source_query_groups_never_scored":list(range(32,40)),
        "candidate_menu_max":12,
        "minimum_total_source_calibration_positive_quartet_support":64,
        "max_candidates_physically_tested_per_source_visit":1,
        "three_two_pair_partitions":3,
        "physical_masks_per_calibration_group":[0,8,16,24],
        "physical_sample_cases_per_site":32,
        "sham_proposals_per_case":96,
        "select_root_arm_visit_using_task_outcomes":False,
        "new_original_model_forwards":0,
        "new_task_label_or_logit_reads":0,
        "original_source_query_behavioral_scoring_permitted":False,
        "recruited_embargoed_or_confirmation_reads":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(entry.get(key)!=value for key,value in expected.items()):
        raise ValueError("source original calibration shortlist method or protected scope changed")
    originals=json.loads((here/"NEC03W_CANDIDATE_LOCK.json").read_text("utf-8"))
    calibration=json.loads((here/"NEC03W_CALIBRATION_CODE_LOCK.json").read_text("utf-8"))
    if (entry["original_candidate_lock_blob_sha1"]!=_blob(here/"NEC03W_CANDIDATE_LOCK.json")
            or entry["calibration_coefficient_lock_blob_sha1"]!=_blob(here/"NEC03W_CALIBRATION_CODE_LOCK.json")
            or originals["allowed_world_roots"]!=list(range(8))
            or calibration["calibration_group_ids"]!=list(range(16,24))
            or entry["original_source_parent_manifests"]!=
               originals["parent_source_manifest_sha256_by_root"]):
        raise ValueError("NEC03W parent original source artifact identities changed")
    root=here.parents[1]
    pins=entry.get("method_git_blob_sha1",{})
    required={
        "mechinterp/neuron_communities/nec03w_calibration_shortlist.py",
        "mechinterp/neuron_communities/nec03w_calibration_shortlist_audit.py",
        "mechinterp/neuron_communities/nec03w_shortlist_registry.py",
        "mechinterp/neuron_communities/NEC03W_CALIBRATION_SHORTLIST_PROTOCOL.md",
        "mechinterp/neuron_communities/nec03w_factorial.py",
        "mechinterp/neuron_communities/nec03w_independent.py",
        "mechinterp/neuron_communities/nec03w_candidate_registry.py",
        "mechinterp/neuron_communities/nec03w_calibration_registry.py",
        "mechinterp/neuron_communities/nec03v_geometry.py",
    }
    if set(pins)!=required:
        raise ValueError("source shortlist frozen candidate/physical method pin set incomplete")
    for path,pin in pins.items():
        f=root/path
        if pin!=_blob(f):
            raise ValueError("source-only calibration quartet shortlist code changed from registered bytes: "+path)
    return entry


def _blob(path:Path):
    content=path.read_bytes()
    return sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
