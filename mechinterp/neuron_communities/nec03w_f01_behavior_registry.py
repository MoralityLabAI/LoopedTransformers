"""Pre-execution root0-only NEC-03W/F01 source factorial behavioral lock.

The exact SOURCE physical 512-case roster SHA, original source checkpoint,
12-pass CPU scorer, separate native-score auditor and root0-only source-case
membership are immutable before any new source CALIBRATION label scoring.
Original query32..39 and all recruited/confirmation roots remain sealed.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path


def registered_behavior():
    base=Path(__file__).parent
    lock=json.loads((base/"NEC03W_F01_BEHAVIOR_LOCK.json").read_text("utf-8"))
    fixed={
        "schema":"NEC-03W/F01/frozen-source-fit-factorial-calibration-behavior/v1",
        "allowed_initial_source_calibration_behavior_roots":[0],
        "original_source_groups":[16,17,18,19,20,21,22,23],
        "source_cases_per_arm":256,
        "original_root3_and_5_ITT_fixed_zero":[3,5],
        "native_recurrent_visits_per_inference":12,
        "factorial_modes":[
            "baseline","remove_A","remove_B","remove_AB",
            "sham_C","sham_D","sham_CD",
            "late_target_rescue","late_matched_sham_rescue",
            "same_visit_exact_addback",
        ],
        "source_only_new_behavioral_scoring":True,
        "source_query_groups32to39_authorized":False,
        "previously_scored_source_groups24to31_read":False,
        "protected_recruited_reserved_embargoed_confirmation_access":False,
        "training_or_optimizer_updates":0,
        "origin_model_requires_CPU_only":True,
        "original_source_calibration_is_developmental":True,
        "root0_registration_requires_independent_512_case_physical_audit":True,
        "meaningful_positive_root_macro_floor_for_later_8root_study":.02,
        "original_eight_root_result_requires_distinct_root_level_preregistration":True,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(lock.get(k)!=v for k,v in fixed.items()):
        raise ValueError("NEC03W F01 original root0 task score method or scientific gate changed")
    parent=base/"NEC03W_F01_PHYSICAL_LOCK.json"
    if lock.get("registered_original_physical_method_blob_sha1")!=_blob(parent):
        raise ValueError("original F01 source physical method was revised after behavior registration")
    original=lock.get("original_source_input_roots")
    if not isinstance(original,dict) or set(original)!={"0"}:
        raise ValueError("original F01 single-root initial outcome authorization expanded")
    root0=original["0"]
    if (root0.get("source_pack_manifest_sha256")!="ef79f9816a0143bc65aaa11d03626f5380dacd76e8b5c3f11437739b928e8c61"
            or root0.get("source_calibration_codes_sha256")!="688f7c92beb7015b575d1addff606a357dbf2af360c81b048cc5bd271af9191d"
            or root0.get("shortlist_roster_sha256")!="9cce6566bd0e019bca14374a4c8213ff80fde0e4267045d67aca6c94a3140030"
            or root0.get("physical_roster_sha256")!="b3f13ab106b0d60f5c7d97e8bd521fa5890195fe8f025cf843126169e0f58513"
            or root0.get("physical_roster_file_sha256")!="2e7f12838b265c05d825233dfd0387c69cb1e1f3e4cb7968b69a434f27128455"
            or root0.get("admitted_cases_by_arm")!={"composed":256,"additive":152}
            or root0.get("original_source_groups")!=list(range(16,24))):
        raise ValueError("original root0 frozen source, quartet or full physical input no longer registered")
    methods=lock.get("method_git_blob_sha1")
    expected={
        "mechinterp/neuron_communities/nec03w_f01_behavior.py",
        "mechinterp/neuron_communities/nec03w_f01_behavior_independent.py",
        "mechinterp/neuron_communities/nec03w_f01_behavior_registry.py",
        "mechinterp/neuron_communities/NEC03W_F01_BEHAVIOR_PROTOCOL.md",
        "mechinterp/neuron_communities/nec03w_f01_physical.py",
        "mechinterp/neuron_communities/nec03w_f01_physical_audit.py",
        "mechinterp/neuron_communities/nec03w_f01_registry.py",
        "mechinterp/neuron_communities/nec03w_native.py",
        "mechinterp/neuron_communities/nec03w_factorial.py",
        "mechinterp/neuron_communities/nec03w_independent.py",
        "mechinterp/neuron_communities/nec03w_calibration_shortlist.py",
        "mechinterp/neuron_communities/nec03w_shortlist_registry.py",
        "mechinterp/neuron_communities/nec03v_source_attest.py",
        "mechinterp/neuron_communities/nec03r_source_recoverability.py",
        "mechinterp/neuron_communities/nec03l_five_input_worlds.py",
        "research/pef07/pef07.py",
    }
    if not isinstance(methods,dict) or set(methods)!=expected:
        raise ValueError("original F01 behavior scorer, independent audit and source dependency byte pins incomplete")
    root=base.parents[1]
    for filename,pin in methods.items():
        if _blob(root/filename)!=pin:
            raise ValueError("F01 registered root0 original source behavior method changed: "+filename)
    return lock


def _blob(path:Path):
    data=path.read_bytes()
    return sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
