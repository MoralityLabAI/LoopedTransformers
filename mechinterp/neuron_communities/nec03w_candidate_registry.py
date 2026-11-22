"""NEC-03W original-source candidate pre-execution method registry.

This is an engineering registration of label-blind source ID discovery and
physical sham matching, never authorization for new native model forwards.
Later functional selection/behavioral scoring requires a *different* frozen
registration, with independent correct-label custody and scorer audits.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path


def registered():
    here=Path(__file__).parent
    p=here/"NEC03W_CANDIDATE_LOCK.json"
    entry=json.loads(p.read_text(encoding="utf-8"))
    if (entry.get("schema")!="NEC-03W/registered-original-source-candidate-geometry/v1"
            or entry.get("allowed_world_roots")!=list(range(8))
            or entry.get("original_source_fit_discovery")!=list(range(16))
            or entry.get("functional_calibration_reserved")!=list(range(16,24))
            or entry.get("previous_nec03v_outcome_exposed_excluded")!=list(range(24,32))
            or entry.get("source_only_query_reserved")!=list(range(32,40))
            or entry.get("candidate_quartet_budget")!=12
            or entry.get("candidate_overlap_cap")!=2
            or entry.get("physical_sham_trials")!=96
            or entry.get("mechanical_source_context_group")!=24
            or entry.get("mechanical_focal_masks")!=32
            or entry.get("new_behavioral_model_forwards")!=0
            or entry.get("new_task_label_or_logit_reads")!=0
            or entry.get("confirmation_or_recruited_roots_authorized") is not False
            or entry.get("source_query_was_used_in_original_SAE_fit") is not True
            or entry.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("NEC03W candidate method or original-source custody registration changed")
    original=json.loads((here/"NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json").read_text())
    if (set(entry.get("parent_source_manifest_sha256_by_root",{}))
            !=set(str(r) for r in range(8))):
        raise ValueError("NEC03W original eight source archive parent SHA roster incomplete")
    for root in range(8):
        if (entry["parent_source_manifest_sha256_by_root"][str(root)]
                !=original["roots"][str(root)]["source_pack_manifest_sha256"]):
            raise ValueError("NEC03W original source pack SHA does not match prior immutable archive")
    allowed=set(entry.get("method_git_blob_sha1",{}))
    if allowed!={
        "mechinterp/neuron_communities/nec03w_factorial.py",
        "mechinterp/neuron_communities/nec03w_independent.py",
        "mechinterp/neuron_communities/nec03w_source_geometry_pilot.py",
        "mechinterp/neuron_communities/nec03w_native.py",
        "mechinterp/neuron_communities/NEC03W_FACTORIAL_RESCUE_PROTOCOL.md",
        "mechinterp/neuron_communities/nec03w_candidate_registry.py",
        "mechinterp/neuron_communities/nec03v_geometry.py",
        "mechinterp/neuron_communities/NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json",
        "mechinterp/neuron_communities/NEC03T_SOURCE_LOCK.json",
    }:
        raise ValueError("original source candidate code signatures missing")
    rootdir=here.parents[1]
    for filename,expected_sha in entry["method_git_blob_sha1"].items():
        data=(rootdir/filename).read_bytes()
        actual=sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if actual!=expected_sha:
            raise ValueError("NEC03W source candidate method changed after frozen registration: "+filename)
    return entry
