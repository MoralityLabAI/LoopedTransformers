"""Pre-execution immutable NEC03W/F02 original-source paired-case mask registry.

This is a LABEL-BLIND geometric/custody tool and was registered AFTER F01
root0 task outcomes were already seen. It cannot retroactively modify the
accepted root0 result or authorize a new scored source/confirmation run.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path


def registered_paired():
    here=Path(__file__).parent
    lock=json.loads((here/"NEC03W_F02_PAIRED_LOCK.json").read_text("utf-8"))
    checks={
        "schema":"NEC-03W/F02/post-root0-source-physical-pair-mask-preexecution/v1",
        "original_world_roots":list(range(8)),
        "source_fit_calibration_group_ids":list(range(16,24)),
        "original_source_eligible_roots":[0,1,2,4,6,7],
        "original_ITT_zero_roots":[3,5],
        "original_cases_per_arm":256,
        "new_source_paired_coverage_threshold":.95,
        "mask_rule":"both original teacher arms must have admissible F01 2x2 source/sham physical geometry on same original case mask",
        "original_root0_behavior_was_seen_before_mask_registration":True,
        "allow_reinterpret_prior_source_behavior":False,
        "allow_new_model_forwards":False,
        "allow_task_labels_logits_or_gradients":False,
        "allow_new_model_training":False,
        "allow_new_source_query_or_protected_worlds":False,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(lock.get(k)!=value for k,value in checks.items()):
        raise ValueError("NEC03W/F02 original paired source mask or post-root0 custody changed")
    root=here.parents[1]
    original_file=here/"NEC03W_F01_EIGHT_ROOT_PHYSICAL_RECEIPTS.json"
    if lock.get("original_eight_source_physical_receipts_git_blob")!=_blob(original_file):
        raise ValueError("original eight source-case physical SHA manifest was modified")
    expected={
        "mechinterp/neuron_communities/nec03w_f02_paired.py",
        "mechinterp/neuron_communities/nec03w_f02_paired_audit.py",
        "mechinterp/neuron_communities/nec03w_f02_registry.py",
        "mechinterp/neuron_communities/NEC03W_F02_PAIRED_CASE_PROTOCOL.md",
        "mechinterp/neuron_communities/NEC03W_F01_EIGHT_ROOT_PHYSICAL_RECEIPTS.json",
        "mechinterp/neuron_communities/nec03w_factorial.py",
        "mechinterp/neuron_communities/nec03v_geometry.py",
    }
    pins=lock.get("method_git_blob_sha1")
    if not isinstance(pins,dict) or set(pins)!=expected:
        raise ValueError("F02 original physical mask methods or source identity pins missing")
    for path,sha in pins.items():
        if _blob(root/path)!=sha:
            raise ValueError("F02 paired source case algorithm SHA differs: "+path)
    return lock


def _blob(path:Path):
    bits=path.read_bytes()
    return sha1(b"blob "+str(len(bits)).encode()+b"\0"+bits).hexdigest()
