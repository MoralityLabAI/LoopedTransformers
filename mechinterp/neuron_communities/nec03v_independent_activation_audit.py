"""NEC-03V independent recomputation of newly extracted source-fit SAE codes.

Separate from the extractor's clean_activations and TopKSAE.codes methods.
Loads the same byte-authenticated ORIGINAL models, independently hooks GELU
and computes ReLU+TopK from frozen SAE encoder weights, with absolutely no
task labels, accuracy scoring, optimizer updates or protected source queries.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .nec03v_source_attest import attest, sha256_file, canonical
from .nec03v_geometry import read_pack


def _digest_json(value):
    from hashlib import sha256
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def independent_codes(model, state, bits, visit: int):
    """Reimplement the SAE source code path without calling SAE.codes."""
    import torch
    captured = []
    def hook(module, args):
        captured.append(args[0][:,1,:].detach().clone())
    handle = model.block.mlp_down.register_forward_pre_hook(hook)
    try:
        # Outputs are discarded; this forward computes but does not expose
        # model predictions or task truth labels.
        with torch.inference_mode():
            model.logits(bits)
    finally:
        handle.remove()
    if len(captured) != 12:
        raise ValueError("original 12-visit recurrent GELU capture changed")
    activated = captured[visit].float().cpu()
    mean = state["mean"].float().cpu()
    scale = state["scale"].float().cpu()
    W = state["state"]["encoder.weight"].float().cpu()
    b = state["state"]["encoder.bias"].float().cpu()
    x = (activated-mean)/scale
    raw = (x@W.T+b).clamp_min(0)
    values, selected = torch.topk(raw,k=16,dim=-1)
    kept = torch.zeros_like(raw).scatter(1,selected,values)
    result = []
    for row in kept.tolist():
        result.append([[i,float(value)] for i,value in enumerate(row) if value>1e-7])
    return result


def audit(root:int, source:Path, t:Path, reset:Path|None, decoder:Path,
          decoder_audit:Path, pack:Path):
    from .nec03v_activation_extract import (
        BATCH_SIZE, SOURCE_ROWS, _source_only_catalog,
        registered_lock, sha_json,
    )
    lock=registered_lock()
    original=attest(root,source,t,reset)
    manifest,shards,manifest_sha=read_pack(pack)
    dec=json.loads(decoder.read_text("utf-8"))
    audit_ref=json.loads(decoder_audit.read_text("utf-8"))
    receipt=json.loads((pack/"extraction_receipt.json").read_text("utf-8"))
    if (manifest["evidence"]!="registered_source_only_activation_extraction_unscored"
            or manifest["world_index"]!=root
            or manifest["source_checkpoint_receipts"]!=original["source_checkpoint_receipts"]
            or receipt.get("schema")!="NEC-03V/frozen-source-positive-topk-extraction/v1"
            or receipt["original_source_attestation_sha256"]!=sha_json(original)
            or receipt["source_checkpoint_receipts"]!=original["source_checkpoint_receipts"]
            or receipt["method_lock_sha256"]!=sha256_file(
                Path(__file__).with_name("NEC03V_ACTIVATION_EXTRACT_LOCK.json"))
            or receipt["decoder_columns_sha256"]!=sha256_file(decoder)
            or receipt["independent_decoder_audit_sha256"]!=sha256_file(decoder_audit)
            or dec["original_source_attestation"]!=original
            or audit_ref["original_decoder_export_sha256"]!=sha256_file(decoder)
            or audit_ref["decoder_columns_independently_verified"] is not True
            or receipt["source_fit_context_group_indices"]!=list(range(40))
            or receipt["intervention_source_groups"]!=list(range(24,32))
            or receipt["source_case_records_per_arm"]!=256
            or receipt["new_optimizer_updates"]!=0
            or receipt["task_labels_accessed"]!=0
            or receipt["model_logits_stored"] is not False
            or receipt["native_logits_computed_transiently_by_forward"] is not True
            or receipt["recruited_or_reserved_queries"]!=0
            or receipt["embargoed_or_confirmation_queries"]!=0
            or receipt["behavioral_necessity_scoring"] is not False
            or receipt["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("source activation provenance, split or scope changed")
    src_report_name=("nec03o_source_refit_receipt.json" if root<4
                     else "nec03ob_source_receipt.json")
    src_report=json.loads((source/src_report_name).read_text("utf-8"))
    fit,fit_manifest=_source_only_catalog(root,original|{
        "source_manifest_sha256":src_report["source_manifest_sha256"]})
    if (receipt["source_fit_input_sha256"]!=sha_json([list(row) for row in fit])
            or receipt["original_source_catalog_sha256"]!=fit_manifest["source_sha256"]):
        raise ValueError("source input catalog changed from frozen original fit membership")
    def check_positive_codes(code):
        # Independently filter the ReLU+TopK output using the registered
        # numerical positivity floor, rather than calling the extractor.
        if len(code)!=128:
            raise ValueError("independent SAE code width must be 128")
        if any(not math.isfinite(float(v)) for v in code):
            raise ValueError("nonfinite independent source code")
        return [[i,float(v)] for i,v in enumerate(code) if v>1e-7]
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("NEC03V independent activation audit requires CPU-only Torch")
    from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
    from research.pef07.pef07 import as_tensor,model_digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    scalar_values=0
    total_forwards=0
    counters={}
    for arm in ("composed","additive"):
        original_arm=load_source_arm(root,arm,source,reset=reset)
        original_model_digest=model_digest(original_arm["model"])
        selected=sorted(original_arm["source_sites"])
        case_coefs={str(v):[] for v in selected}
        records=shards[arm]
        if (set(records["visits"])!={str(v) for v in selected}
                or records["world_index"]!=root):
            raise ValueError("source-reported original site selection changed")
        for v in selected:
            if records["visits"][str(v)]["decoder32"]!=dec["arms"][arm]["decoder32"][str(v)]:
                raise ValueError("source-geometry decoder columns differ from original SHA-verified archive")
        if (original_model_digest!=original["source_checkpoint_receipts"][arm]["native_sha256"]
                or original_arm["sae_sha256"]!=original["source_checkpoint_receipts"][arm]["sae_sha256"]
                or original["source_checkpoint_detail"][arm]["source_roster_sha256"]!=
                      original["source_checkpoint_receipts"][arm]["source_sites_sha256"]):
            raise ValueError("independent source-checkpoint identity mismatch")
        n_forwards=0
        if selected:
            sae_file=source/src_report["arms"][arm]["SAE_source"]["source_dictionary_file"]
            sae_checkpoint=torch.load(sae_file,weights_only=True,map_location="cpu")
            batch_lists={str(v):[] for v in selected}
            model=original_arm["model"].eval()
            with torch.inference_mode():
                for offset in range(0,SOURCE_ROWS,BATCH_SIZE):
                    bits=as_tensor(fit[offset:offset+BATCH_SIZE])
                    # One fresh model forward per batch, with all selected
                    # visits hooked at once. This is a genuinely independent
                    # native activation implementation from the extractor.
                    seen=[]
                    def capture(_module, args):
                        seen.append(args[0][:,1,:].detach().clone())
                    handle=model.block.mlp_down.register_forward_pre_hook(capture)
                    try:
                        model.logits(bits)
                    finally:
                        handle.remove()
                    n_forwards+=1
                    if len(seen)!=12:
                        raise ValueError("independent original recurrent visit count wrong")
                    state=sae_checkpoint["state"]
                    W=state["encoder.weight"].float().cpu()
                    b=state["encoder.bias"].float().cpu()
                    mean=sae_checkpoint["mean"].float().cpu()
                    scale=sae_checkpoint["scale"].float().cpu()
                    for visit in selected:
                        raw=((seen[visit].float().cpu()-mean)/scale)@W.T+b
                        values,chosen=torch.topk(raw.clamp_min(0),k=16,dim=-1)
                        sparse=torch.zeros_like(raw).scatter(1,chosen,values)
                        for i,row in enumerate(sparse.tolist()):
                            positive=check_positive_codes(row)
                            ids=[p[0] for p in positive]
                            expected=records["visits"][str(visit)]["source_fit_active"][offset+i]
                            if ids!=expected:
                                raise ValueError(f"frozen source-fit positive ID mismatch {arm} v{visit} row{offset+i}")
                            if 768<=offset+i<1024:
                                case=records["cases"][offset+i-768]
                                observed=case["visits"][str(visit)]
                                if len(positive)!=len(observed):
                                    raise ValueError("frozen original case TopK positive count changed")
                                for actual,saved in zip(positive,observed):
                                    if actual[0]!=saved[0] or not math.isclose(
                                        actual[1], saved[1], rel_tol=1e-6,abs_tol=2e-6):
                                        raise ValueError("source-positive TopK coefficient differs")
                                    scalar_values+=1
            if model_digest(model)!=original_model_digest:
                raise ValueError("independent extraction modified source native weights")
        if receipt["source_only_forward_batches"][arm]!=n_forwards:
            raise ValueError("registered source-fit native forward cost differs")
        total_forwards+=n_forwards
        counters[arm]={"selected_visits":len(selected),
                       "source_fit_rows":SOURCE_ROWS if selected else 0,
                       "source_case_roots":256,
                       "native_forward_calls":n_forwards}
    if receipt["native_model_forward_calls"]!=total_forwards:
        raise ValueError("source-fit total native inference calls mismatched")
    return {
        "schema":"NEC-03V/independent-source-positive-topk-audit/v1",
        "world_id":original["world_id"],"world_index":root,
        "original_archival_lock_verified":True,
        "source_geometry_manifest_sha256":manifest_sha,
        "source_only_native_forward_calls":total_forwards,
        "independent_source_only_native_forward_calls":total_forwards,
        "source_case_positive_coefficients_compared":scalar_values,
        "arms":counters,
        "new_optimizer_updates":0,"task_labels_read":0,
        "model_logits_stored":False,"protected_cohort_queries":0,
        "world_source_fit_subset_only":True,
        "registration_sha256":receipt["method_lock_sha256"],
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "behavioral_effect_measured":False,
        "independent_activation_audit_passed":True,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--decoder-columns",type=Path,required=True)
    p.add_argument("--decoder-audit",type=Path,required=True)
    p.add_argument("--pack",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("audit output must be fresh")
    result=audit(args.world_index,args.source_root,args.nec03t_root,args.reset_root,
                 args.decoder_columns,args.decoder_audit,args.pack)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(canonical(result)+"\n",encoding="utf-8")
    print(canonical(result))
    return result


if __name__=="__main__":
    main()
