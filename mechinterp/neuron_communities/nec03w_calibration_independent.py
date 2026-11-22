"""NEC-03W independently audit original-source calibration coefficients on CPU.

This auditor does not import the extractor or call TopKSAE.codes or
clean_activations. It independently hooks native mlp_down, recomputes
frozen source SAE linear/ReLU/TopK16 features, checks all 256 original
source-fit calibration rows (groups16..23) against previously stored
positive feature IDs, and validates every saved positive coefficient.
Original query groups32..39 remain unqueried.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03v_source_attest import attest,canonical,sha256_file
from .nec03w_calibration_registry import calibration_lock
from .nec03w_candidate_registry import registered

SCHEMA="NEC-03W/independently-replayed-source-calibration-topk16/v1"
BATCH=32


def audit(root:int,source:Path,t:Path,reset:Path|None,pack:Path,codes:Path):
    if type(root) is not int or root not in range(8):
        raise PermissionError("only original source historical roots are permitted")
    lock=calibration_lock()
    parent=registered()
    att=attest(root,source,t,reset)
    manifest,shards,pack_sha=read_pack(pack)
    receipt=json.loads(codes.read_text(encoding="utf-8"))
    if (pack_sha!=parent["parent_source_manifest_sha256_by_root"][str(root)]
            or manifest["source_checkpoint_receipts"]!=att["source_checkpoint_receipts"]
            or receipt.get("schema")!="NEC-03W/original-source-functional-calibration-positive-codes/v1"
            or receipt.get("world_index")!=root
            or receipt.get("original_source_pack_manifest_sha256")!=pack_sha
            or receipt.get("original_calibration_group_indices")!=list(range(16,24))
            or receipt.get("source_fit_group_indices_used_by_discovery")!=list(range(16))
            or receipt.get("previously_scored_source_groups_excluded")!=list(range(24,32))
            or receipt.get("source_only_query_groups_excluded")!=list(range(32,40))
            or receipt.get("source_query_groups_32_to_39_read") is not False
            or receipt.get("task_labels_read")!=0
            or receipt.get("model_logits_stored") is not False
            or receipt.get("model_or_SAE_optimizer_updates")!=0
            or receipt.get("behavioral_factorial_interventions_executed")!=0
            or receipt.get("recruited_or_confirmation_world_reads")!=0
            or receipt.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("original source calibration coefficient provenance changed")
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("independent calibration code audit requires CPU-only Torch")
    from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
    from research.pef07.pef07 import as_tensor,model_digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    counts={}
    forwards={}
    for arm in ("composed","additive"):
        origin=load_source_arm(root,arm,source,reset=reset)
        model=origin["model"].eval()
        model_before=model_digest(model)
        if (model_before!=att["source_checkpoint_receipts"][arm]["native_sha256"]
                or origin["sae_sha256"]!=att["source_checkpoint_receipts"][arm]["sae_sha256"]):
            raise ValueError("independent original source checkpoint changed")
        rows=origin["catalog"]["train"][512:768]
        if len(rows)!=256:
            raise ValueError("source-only functional calibration group count wrong")
        expected=receipt["arms"][arm]
        sites=sorted(origin["source_sites"])
        if (expected["selected_original_source_visits"]!=sites
                or set(expected["positive_codes_by_visit"])!={str(v) for v in sites}
                or expected["source_fit_ID_indices_start_end"]!=[512,768]
                or expected["original_native_model_sha256"]!=model_before
                or expected["original_source_sae_sha256"]!=origin["sae_sha256"]):
            raise ValueError("source-only original SAE visits or weights changed")
        digest=sha256(canonical([list(r) for r in rows]).encode("utf-8")).hexdigest()
        if expected["original_original_source_input_sha256"]!=digest:
            raise ValueError("source calibration input catalog changed")
        forward_count=0
        verified_coeffs=0
        if sites:
            checkpoint=source / json.loads((source/(
                "nec03o_source_refit_receipt.json" if root<4 else
                "nec03ob_source_receipt.json")).read_text("utf-8"))["arms"][arm]["SAE_source"]["source_dictionary_file"]
            sae=torch.load(checkpoint,weights_only=True,map_location="cpu")
            W=sae["state"]["encoder.weight"].float().cpu()
            bias=sae["state"]["encoder.bias"].float().cpu()
            mean=sae["mean"].float().cpu()
            scale=sae["scale"].float().cpu()
            if tuple(W.shape)!=(128,64) or tuple(bias.shape)!=(128,):
                raise ValueError("source-frozen positive code encoder dimensions changed")
            with torch.no_grad():
                for offset in range(0,256,BATCH):
                    bits=as_tensor(rows[offset:offset+BATCH])
                    captured=[]
                    hook=model.block.mlp_down.register_forward_pre_hook(
                        lambda _module,args:captured.append(args[0][:,1,:].detach().clone()))
                    try:
                        model.logits(bits)
                    finally:
                        hook.remove()
                    forward_count+=1
                    if len(captured)!=12:
                        raise ValueError("original 12 tied recurrent site capture incomplete")
                    for site in sites:
                        activation=captured[site].float().cpu()
                        raw=torch.relu(((activation-mean)/scale)@W.T+bias)
                        top,idx=torch.topk(raw,k=16,dim=-1)
                        sparse=torch.zeros_like(raw).scatter_(dim=-1,index=idx,src=top)
                        for j,row in enumerate(sparse.tolist()):
                            positive=[[i,float(value)] for i,value in enumerate(row) if value>1e-7]
                            row_index=offset+j
                            parent_ids=shards[arm]["visits"][str(site)]["source_fit_active"][512+row_index]
                            saved=expected["positive_codes_by_visit"][str(site)][row_index]
                            if [i for i,_ in positive]!=parent_ids or len(saved)!=len(positive):
                                raise ValueError("source-only positive feature IDs mismatch original SAE fit")
                            for (expected_i,expected_value),(actual_i,actual_value) in zip(saved,positive):
                                if (type(expected_i) is not int or actual_i!=expected_i
                                        or not math.isclose(float(expected_value),actual_value,
                                                            abs_tol=2e-6,rel_tol=1e-6)):
                                    raise ValueError("independent original positive SAE coefficient replay mismatch")
                                verified_coeffs+=1
        if (receipt["native_model_forward_calls_by_arm"][arm]!=forward_count
                or model_digest(model)!=model_before
                or any(len(expected["positive_codes_by_visit"][str(v)])!=256 for v in sites)):
            raise ValueError("source-only code training, case denominator or forward count changed")
        counts[arm]=verified_coeffs
        forwards[arm]=forward_count
    if receipt["native_model_forward_calls"]!=sum(forwards.values()):
        raise ValueError("new original-source code forward accounting mismatch")
    return {
        "schema":SCHEMA,"world_index":root,"world_id":att["world_id"],
        "source_calibration_groups":list(range(16,24)),
        "previously_exposed_source_groups_interventions":0,
        "original_source_query_groups_32_to_39_read":False,
        "independent_positive_code_coefficient_counts_by_arm":counts,
        "independent_source_only_native_forward_calls_by_arm":forwards,
        "new_source_only_native_forward_calls":sum(forwards.values()),
        "original_source_calibration_codes_sha256":sha256_file(codes),
        "original_model_weights_unchanged":True,
        "native_and_SAE_optimizer_updates":0,
        "task_labels_read":0,
        "model_logits_saved":False,
        "factorial_behavior_or_rescue_scored":False,
        "recruited_embargoed_or_confirmation_reads":0,
        "independent_code_audit_passed":True,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--calibration-codes",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError("independent original calibration-code audit must use new output")
    receipt=audit(a.world_index,a.source_root,a.nec03t_root,a.reset_root,
                  a.original_pack,a.calibration_codes)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(canonical(receipt)+"\n",encoding="utf-8")
    print(canonical(receipt))
    return receipt


if __name__=="__main__":
    main()
