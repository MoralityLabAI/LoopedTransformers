"""NEC-03W source-only calibration native-forward registration verifier.

Byte-pins the original eight native/SAE sources, original quartet method,
calibration coefficient extractor, independent CPU replay and query exclusion
BEFORE allowing any newly recorded source-fit model forward.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path


def calibration_lock():
    base=Path(__file__).parent
    entry=json.loads((base/"NEC03W_CALIBRATION_CODE_LOCK.json").read_text("utf-8"))
    if (entry.get("schema")!="NEC-03W/preregistered-source-fit-calibration-topk-export/v1"
            or entry.get("allowed_original_roots")!=list(range(8))
            or entry.get("discovery_group_ids")!=list(range(16))
            or entry.get("calibration_group_ids")!=list(range(16,24))
            or entry.get("previously_scored_source_group_ids_excluded")!=list(range(24,32))
            or entry.get("query_source_group_ids_excluded")!=list(range(32,40))
            or entry.get("original_source_rank_groups_excluded")!=list(range(40,80))
            or entry.get("source_calibration_cases_per_arm")!=256
            or entry.get("native_forward_batch_size")!=32
            or entry.get("maximum_new_native_forward_batches_per_eligible_arm")!=8
            or entry.get("allow_new_original_source_calibration_only_native_forwards") is not True
            or entry.get("allow_task_labels_or_model_logits_saved") is not False
            or entry.get("allow_optimizer_or_SAE_training") is not False
            or entry.get("recruited_embargoed_reserved_or_confirmation_access") is not False
            or entry.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("NEC03W source-only calibration input or forward budget changed")
    required={
        "mechinterp/neuron_communities/nec03w_calibration_codes.py",
        "mechinterp/neuron_communities/nec03w_calibration_independent.py",
        "mechinterp/neuron_communities/nec03w_calibration_registry.py",
        "mechinterp/neuron_communities/NEC03W_CALIBRATION_CODE_PROTOCOL.md",
        "mechinterp/neuron_communities/nec03w_candidate_registry.py",
        "mechinterp/neuron_communities/NEC03W_CANDIDATE_LOCK.json",
        "mechinterp/neuron_communities/nec03v_source_attest.py",
        "mechinterp/neuron_communities/nec03r_source_recoverability.py",
        "mechinterp/neuron_communities/nec03v_geometry.py",
        "mechinterp/neuron_communities/nec03v_activation_extract.py",
        "mechinterp/neuron_communities/nec03l_five_input_worlds.py",
        "mechinterp/neuron_communities/source_sae.py",
        "research/pef07/pef07.py",
    }
    if set(entry.get("method_git_blob_sha1",{}))!=required:
        raise ValueError("NEC03W calibration-native method or source dependency not registered")
    root=base.parents[1]
    for path,previous in entry["method_git_blob_sha1"].items():
        binary=(root/path).read_bytes()
        actual=sha1(b"blob "+str(len(binary)).encode()+b"\0"+binary).hexdigest()
        if previous!=actual:
            raise ValueError("NEC03W calibration method changed after pre-forward lock: "+path)
    return entry
