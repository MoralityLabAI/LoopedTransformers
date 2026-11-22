"""NEC-03V: strictly source-training-only frozen CPU activation extraction.

This is NEW NATIVE INFERENCE on PREVIOUSLY OPENED source-fit histories 0..39.
It is blocked until the code and exact protocol are frozen by
NEC03V_ACTIVATION_EXTRACT_LOCK.json. No task labels, development/held-out
contexts, recruited roots, gradients, training or behavioral interventions.

A world-root extraction requires original archived NEC03T report, source SAE,
reset-native checkpoint, and separately audited projected SAE decoder columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .nec03v_source_attest import attest, canonical, sha256_file

SCHEMA = "NEC-03V/frozen-source-positive-topk-extraction/v1"
GEO_SCHEMA = "NEC-03V/label-blind-source-geometry/v1"
ARM_SCHEMA = "NEC-03V/source-geometry-arm/v1"
EVIDENCE = "registered_source_only_activation_extraction_unscored"
SOURCE_FIT_GROUPS = tuple(range(40))
PANEL_GROUPS = tuple(range(24, 32))
FRAMES_PER_GROUP = 32
SOURCE_ROWS = 1280
BATCH_SIZE = 32
MAX_ACTIVE = 16


def sha_json(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def positive_codes(row):
    """Source positive TopK codes, 0..16 entries (TopK can include zeros)."""
    if len(row)!=128:
        raise ValueError("original TopK16 source encoder must have 128 codes")
    items = []
    for i, value in enumerate(row):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError("nonfinite source code")
        if float(value) > 1e-7:
            items.append([i, float(value)])
    if len(items) > MAX_ACTIVE or any(not 0 <= pair[0] < 128 for pair in items):
        raise ValueError("source SAE code capacity changed")
    return items


def _source_only_catalog(world: int, frozen: dict):
    from mechinterp.neuron_communities.nec03l_five_input_worlds import (
        worlds, context_catalog,
    )
    if type(world) is not int or world not in range(8):
        raise PermissionError("NEC03V activation extraction is original dev5-000..007 only")
    original = worlds()[world]
    if original["world_sha256"] != frozen["world_sha256"]:
        raise ValueError("original five-input world identity changed")
    catalog, source_manifest = context_catalog(original)
    if source_manifest["source_sha256"] != frozen["source_manifest_sha256"]:
        raise ValueError("original 80/24/24 source split manifest changed")
    # Read only the FIRST forty source-training groups. The 24..31 intervention
    # cases are a subset of these already fitted original source histories.
    fit_rows = catalog["train"][:SOURCE_ROWS]
    if len(fit_rows) != SOURCE_ROWS or len(set(map(tuple, fit_rows))) != SOURCE_ROWS:
        raise ValueError("incomplete original source-fit context cohort")
    return fit_rows, source_manifest


def registered_lock():
    from hashlib import sha1
    here = Path(__file__).parent
    lock = json.loads((here/"NEC03V_ACTIVATION_EXTRACT_LOCK.json").read_text("utf-8"))
    expected = {
        "schema": "NEC-03V/frozen-source-only-activation-extraction-registration/v1",
        "allowed_worlds": list(range(8)),
        "source_fit_groups": list(SOURCE_FIT_GROUPS),
        "source_case_groups": list(PANEL_GROUPS),
        "complete_focal_masks": FRAMES_PER_GROUP,
        "frozen_source_fit_cases": SOURCE_ROWS,
        "batch_size": BATCH_SIZE,
        "source_code_cutoff": 1e-7,
        "all_original_selected_visits_only": True,
        "forbid_heldout_or_recruited_inputs": True,
        "allow_new_native_inference": True,
        "allow_optimizer_updates": False,
        "model_logits_stored": False,
        "new_nec03v_behavioral_scoring": False,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(lock.get(k) != v for k, v in expected.items()):
        raise ValueError("NEC03V source-only activation lock changed")
    methods = lock.get("git_blob_sha1", {})
    if set(methods) != {
        "mechinterp/neuron_communities/nec03v_activation_extract.py",
        "mechinterp/neuron_communities/nec03v_independent_activation_audit.py",
        "mechinterp/neuron_communities/NEC03V_ACTIVATION_EXTRACT_PROTOCOL.md",
    }:
        raise ValueError("NEC03V registered methods missing")
    root = here.parents[1]
    extra=lock.get("dependency_git_blob_sha1",{})
    required_deps={
        "mechinterp/neuron_communities/nec03l_five_input_worlds.py",
        "mechinterp/neuron_communities/nec03r_source_recoverability.py",
        "mechinterp/neuron_communities/source_sae.py",
        "mechinterp/neuron_communities/topk_sae.py",
        "mechinterp/neuron_communities/nec03v_source_attest.py",
        "mechinterp/neuron_communities/nec03v_decoder_columns.py",
        "mechinterp/neuron_communities/nec03v_geometry.py",
        "research/pef07/pef07.py",
    }
    if set(extra)!=required_deps:
        raise ValueError("NEC03V original source/SAE dependency pins missing")
    for path,pin in {**methods,**extra}.items():
        original=(root/path).read_bytes()
        actual=sha1(b"blob "+str(len(original)).encode()+b"\0"+original).hexdigest()
        if actual!=pin:
            raise ValueError(f"NEC03V registered source or dependency changed: {path}")
    return lock


def plan(root: int, source: Path, t: Path, reset: Path | None,
         decoder: Path, decoder_audit: Path):
    if type(root) is not int or root not in range(8):
        raise PermissionError("protected world root blocked")
    lock = registered_lock()
    att = attest(root, source, t, reset)
    dec = json.loads(decoder.read_text(encoding="utf-8"))
    audit = json.loads(decoder_audit.read_text(encoding="utf-8"))
    if (dec.get("schema") != "NEC-03V/source-frozen-projected-decoder-columns/v1"
            or dec.get("original_source_attestation") != att
            or dec.get("world_index") != root
            or audit.get("decoder_columns_independently_verified") is not True
            or audit.get("original_decoder_export_sha256") != sha256_file(decoder)
            or audit.get("world_id") != att["world_id"]
            or audit.get("new_neural_forward_calls") != 0
            or audit.get("protected_cohort_reads") != 0):
        raise ValueError("missing authentic independently audited original decoder geometry")
    # Verify the original source catalog provenance before any new forwards.
    import json as _json
    report_name = ("nec03o_source_refit_receipt.json" if root < 4
                   else "nec03ob_source_receipt.json")
    source_report = _json.loads((source/report_name).read_text(encoding="utf-8"))
    fit_rows, original_manifest = _source_only_catalog(root, att|{
        "source_manifest_sha256": source_report["source_manifest_sha256"]})
    return lock, att, dec, audit, fit_rows, original_manifest


def extract(root: int, source: Path, t: Path, reset: Path | None,
            decoder: Path, decoder_audit: Path, out: Path,
            *, allow_registered_forwards: bool = False):
    lock, att, dec, aud, fit_rows, source_manifest = plan(
        root, source, t, reset, decoder, decoder_audit)
    if not allow_registered_forwards:
        return {"schema": SCHEMA, "world_id": att["world_id"],
                "dry_run": True, "source_rows_to_forward":len(fit_rows),
                "selected_visits": {
                    arm:dec["arms"][arm]["selected_visits"] for arm in ("composed","additive")},
                "new_model_forwards":0, "source_lock_registered":True,
                "protected_queries":0}
    if out.exists():
        raise FileExistsError("NEC03V extraction output directory must be fresh")
    import torch
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("CPU-only build required")
    from research.pef07.pef07 import as_tensor, model_digest
    from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
    from mechinterp.neuron_communities.source_sae import clean_activations

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = {}
    forwards = {}
    for arm in ("composed", "additive"):
        frozen = load_source_arm(root, arm, source, reset=reset)
        original_model_sha = model_digest(frozen["model"])
        if (frozen["model_sha256"] != att["source_checkpoint_receipts"][arm]["native_sha256"]
                or frozen["sae_sha256"] != att["source_checkpoint_receipts"][arm]["sae_sha256"]
                or original_model_sha != frozen["model_sha256"]):
            raise ValueError("native model or SAE differs from original source checkpoint")
        selected = {str(v) for v in frozen["source_sites"]}
        if selected != set(dec["arms"][arm]["decoder32"]):
            raise ValueError("source-selected visits differ from independently audited decoder")
        visit_rows = {str(v): {"decoder32": dec["arms"][arm]["decoder32"][str(v)],
                               "source_fit_active": []}
                      for v in frozen["source_sites"]}
        sparse_cases = {str(v): [] for v in frozen["source_sites"]}
        forwards[arm] = 0
        if visit_rows:
            sae,mean,scale = frozen["sae"],frozen["mean"],frozen["scale"]
            if sae is None:
                raise ValueError("registered source site has no frozen source SAE")
            model = frozen["model"].eval()
            with torch.inference_mode():
                for offset in range(0, SOURCE_ROWS, BATCH_SIZE):
                    bits = as_tensor(fit_rows[offset:offset+BATCH_SIZE])
                    act = clean_activations(model, bits)
                    forwards[arm] += 1
                    if tuple(act.shape) != (len(bits), 12, 64):
                        raise ValueError("native source clean activation shape changed")
                    for visit in frozen["source_sites"]:
                        z = sae.codes((act[:, visit, :] - mean)/scale)
                        if z.shape != (len(bits),128) or not bool(torch.isfinite(z).all()):
                            raise ValueError("source SAE TopK coefficients invalid")
                        for i in range(len(bits)):
                            row = positive_codes(z[i].tolist())
                            visit_rows[str(visit)]["source_fit_active"].append(
                                [pair[0] for pair in row])
                            if 24*32 <= offset+i < 32*32:
                                sparse_cases[str(visit)].append(row)
            if any(len(v["source_fit_active"])!=SOURCE_ROWS for v in visit_rows.values()):
                raise ValueError("source-fit TopK code cohort incomplete")
            if any(len(rows)!=256 for rows in sparse_cases.values()):
                raise ValueError("source intervention coefficient panel incomplete")
        if original_model_sha != model_digest(frozen["model"]):
            raise ValueError("source-only extraction mutated original native weights")
        cases = [{"group":group,"mask":mask,
                  "visits":{str(v):sparse_cases[str(v)][(group-24)*32+mask]
                            for v in frozen["source_sites"]}}
                 for group in PANEL_GROUPS for mask in range(32)]
        data[arm] = {"schema": ARM_SCHEMA, "arm":arm,"world_index":root,
                     "visits":visit_rows,"cases":cases}
    out.mkdir(parents=True, exist_ok=False)
    files={}
    for arm in ("composed","additive"):
        path=out/f"{arm}.json"
        path.write_text(canonical(data[arm])+"\n",encoding="utf-8")
        files[arm]={"file":path.name,"sha256":sha256_file(path)}
    receipt={
        "schema":SCHEMA,"world_id":att["world_id"],"world_index":root,
        "original_source_attestation_sha256":sha_json(att),
        "source_checkpoint_receipts":att["source_checkpoint_receipts"],
        "source_fit_context_group_indices":list(SOURCE_FIT_GROUPS),
        "intervention_source_groups":list(PANEL_GROUPS),
        "original_source_catalog_sha256":source_manifest["source_sha256"],
        "source_fit_input_sha256":sha_json([list(row) for row in fit_rows]),
        "source_only_forward_batches":forwards,
        "native_model_forward_calls":sum(forwards.values()),
        "decoder_columns_sha256":sha256_file(decoder),
        "independent_decoder_audit_sha256":sha256_file(decoder_audit),
        "method_lock_sha256":sha256_file(
            Path(__file__).with_name("NEC03V_ACTIVATION_EXTRACT_LOCK.json")),
        "native_checkpoint_unchanged":True,
        "new_optimizer_updates":0,"task_labels_accessed":0,"model_logits_stored":False,
        "native_logits_computed_transiently_by_forward":True,
        "recruited_or_reserved_queries":0,"embargoed_or_confirmation_queries":0,
        "positive_topk_support_may_be_less_than_16":True,
        "source_case_records_per_arm":256,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "behavioral_necessity_scoring":False,
    }
    receipt_path=out/"extraction_receipt.json"
    receipt_path.write_text(canonical(receipt)+"\n",encoding="utf-8")
    manifest={
        "schema":GEO_SCHEMA,"world_index":root,"world_id":att["world_id"],
        "source_panel":list(PANEL_GROUPS),
        "original_lock_sha256":att["original_nec03t_source_lock_sha256"],
        "evidence":EVIDENCE,
        "source_checkpoint_receipts":att["source_checkpoint_receipts"],
        "arm_shards":files,
        "extraction_receipt_sha256":sha256_file(receipt_path),
    }
    (out/"manifest.json").write_text(canonical(manifest)+"\n",encoding="utf-8")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--decoder-columns",type=Path,required=True)
    p.add_argument("--decoder-audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-registered-source-forwards",action="store_true")
    a=p.parse_args(argv)
    receipt=extract(a.world_index,a.source_root,a.nec03t_root,a.reset_root,
                    a.decoder_columns,a.decoder_audit,a.out,
                    allow_registered_forwards=a.allow_registered_source_forwards)
    print(canonical(receipt))
    return receipt


if __name__=="__main__":
    main()
