"""NEC-03W/F01 zero-inference full physical-roster pre-execution lock.

The physical F01 method is SHA frozen before use on authenticated original
training-source code geometry. It does not authorize neural scoring.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path


def physical_method_lock():
    base=Path(__file__).parent
    data=json.loads((base/"NEC03W_F01_PHYSICAL_LOCK.json").read_text("utf-8"))
    checks={
        "schema":"NEC-03W/F01/pre-execution-source-only-full-256-physical-roster/v1",
        "original_roots":list(range(8)),
        "source_calibration_groups":list(range(16,24)),
        "denominator_per_arm":256,
        "min_admitted_cases_each_paired_source_eligible_arm":64,
        "source_sham_proposals":96,
        "prior_behavior_scored_groups_excluded":list(range(24,32)),
        "developmental_query_groups_excluded":list(range(32,40)),
        "new_native_model_forwards":0,
        "new_task_labels_or_logits_read":0,
        "new_factorial_or_delayed_rescue_behavior_scored":False,
        "recruited_embargoed_or_confirmation_access":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(data.get(k)!=v for k,v in checks.items()):
        raise ValueError("NEC03W F01 original source physical method or protection gate changed")
    original=base/"NEC03W_CALIBRATION_SHORTLIST_LOCK.json"
    if data.get("original_shortlist_lock_git_sha1")!=_blob(original):
        raise ValueError("NEC03W original source label-blind candidate lock changed")
    names={
        "mechinterp/neuron_communities/nec03w_f01_physical.py",
        "mechinterp/neuron_communities/nec03w_f01_physical_audit.py",
        "mechinterp/neuron_communities/nec03w_f01_registry.py",
        "mechinterp/neuron_communities/NEC03W_F01_PHYSICAL_PROTOCOL.md",
        "mechinterp/neuron_communities/nec03w_factorial.py",
        "mechinterp/neuron_communities/nec03w_independent.py",
        "mechinterp/neuron_communities/nec03w_calibration_shortlist.py",
        "mechinterp/neuron_communities/nec03w_calibration_shortlist_audit.py",
        "mechinterp/neuron_communities/nec03w_shortlist_registry.py",
        "mechinterp/neuron_communities/nec03v_geometry.py",
    }
    stored=data.get("method_git_blob_sha1")
    if not isinstance(stored,dict) or set(stored)!=names:
        raise ValueError("NEC03W F01 physical producer/auditor and parent method must all be frozen")
    root=base.parents[1]
    for name,expected in stored.items():
        if _blob(root/name)!=expected:
            raise ValueError("NEC03W F01 original source method SHA differs: "+name)
    return data


def _blob(path:Path):
    value=path.read_bytes()
    return sha1(b"blob "+str(len(value)).encode()+b"\0"+value).hexdigest()
