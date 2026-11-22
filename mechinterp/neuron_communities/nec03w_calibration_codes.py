"""NEC-03W original source-fit CALIBRATION coefficient recovery, CPU-only.

Original sources: dev5-000..007, each teacher's source-frozen native/SAE
checkpoints and original source-fit positive-code ID manifest.

STRICT NEW READ: original source FIT context groups 16..23 ONLY, 256 inputs
per model arm. Earlier source candidate DISCOVERY groups0..15 remain
label-free; groups24..31 already scored by NEC03V are excluded; original
source query groups32..39 are NEVER forwarded or accessed by this tool.

The method must be SHA-registered in NEC03W_CALIBRATION_CODE_LOCK.json before
any new native inference. Only source SAE code coefficients are saved;
original native logits compute transiently for GELU hooks and are discarded.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .nec03v_geometry import read_pack,file_hash
from .nec03v_source_attest import attest,canonical,sha256_file
from .nec03w_candidate_registry import registered

SCHEMA="NEC-03W/original-source-functional-calibration-positive-codes/v1"
ORIGINAL_ROOTS=tuple(range(8))
CALIBRATION_GROUPS=tuple(range(16,24))
SOURCE_CALIBRATION_ROWS=256
BATCH=32


def preflight(root:int,source:Path,t:Path,reset:Path|None,pack:Path):
    if type(root) is not int or root not in ORIGINAL_ROOTS:
        raise PermissionError("only historical original eight source-fit roots allowed")
    from .nec03w_calibration_registry import calibration_lock
    frozen=calibration_lock()
    parent=registered()
    original=attest(root,source,t,reset)
    source_manifest,source_shards,sha=read_pack(pack)
    if (sha!=parent["parent_source_manifest_sha256_by_root"][str(root)]
            or source_manifest["world_index"]!=root
            or source_manifest["source_checkpoint_receipts"]!=original["source_checkpoint_receipts"]
            or source_manifest["evidence"]!="registered_source_only_activation_extraction_unscored"):
        raise ValueError("source calibration requires byte-authenticated original SAE source pack")
    return frozen,original,source_shards,sha


def extract(root:int,source:Path,t:Path,reset:Path|None,
            original_pack:Path,out:Path,*,allow_new_source_calibration_forwards=False):
    frozen,original,shards,original_pack_sha=preflight(root,source,t,reset,original_pack)
    if not allow_new_source_calibration_forwards:
        return {"schema":SCHEMA,"dry_run":True,"root":root,
                "calibration_group_ids":list(CALIBRATION_GROUPS),
                "original_source_fit_rows":SOURCE_CALIBRATION_ROWS,
                "source_query_groups_32_to_39_read":False,
                "native_model_forwards":0,
                "scientific_gate":"REDESIGN_OR_PILOT_ONLY"}
    if out.exists():
        raise FileExistsError("NEC03W calibration coefficients require a new output directory")
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("original-source calibration must run on CPU-only PyTorch")
    from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
    from mechinterp.neuron_communities.source_sae import clean_activations
    from mechinterp.neuron_communities.nec03v_activation_extract import positive_codes
    from research.pef07.pef07 import as_tensor,model_digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arms={}
    forward_count={}
    for arm in ("composed","additive"):
        prior=load_source_arm(root,arm,source,reset=reset)
        model=prior["model"].eval()
        model_hash=model_digest(model)
        if (model_hash!=original["source_checkpoint_receipts"][arm]["native_sha256"]
                or prior["sae_sha256"]!=original["source_checkpoint_receipts"][arm]["sae_sha256"]
                or set(str(v) for v in prior["source_sites"])!=
                   set(shards[arm]["visits"])):
            raise ValueError("old frozen native/SAE source visit or hash changed")
        all_rows=prior["catalog"]["train"]
        if len(all_rows)!=80*32:
            raise ValueError("original training source catalog size differs")
        calibration=all_rows[16*32:24*32]
        if len(calibration)!=SOURCE_CALIBRATION_ROWS:
            raise ValueError("calibration requires 8 full source-fit contexts x32 focal masks")
        site_codes={str(v):[] for v in prior["source_sites"]}
        forward_count[arm]=0
        if site_codes:
            with torch.inference_mode():
                for start in range(0,SOURCE_CALIBRATION_ROWS,BATCH):
                    bit_batch=as_tensor(calibration[start:start+BATCH])
                    neural=clean_activations(model,bit_batch)
                    forward_count[arm]+=1
                    if tuple(neural.shape)!=(len(bit_batch),12,64):
                        raise ValueError("source-native GELU recurrent activations changed")
                    for site in prior["source_sites"]:
                        z=prior["sae"].codes(
                            (neural[:,site,:]-prior["mean"])/prior["scale"])
                        if not bool(torch.isfinite(z).all()) or tuple(z.shape)!=(len(bit_batch),128):
                            raise ValueError("original source SAE code geometry changed")
                        for j,row in enumerate(z.tolist()):
                            positive=positive_codes(row)
                            old=shards[arm]["visits"][str(site)]["source_fit_active"][512+start+j]
                            if [id for id,_ in positive]!=old:
                                raise ValueError("registered original source-fit positive-code IDs do not replay")
                            site_codes[str(site)].append(positive)
        if (model_digest(model)!=model_hash
                or any(len(values)!=SOURCE_CALIBRATION_ROWS for values in site_codes.values())):
            raise ValueError("source calibration model weights changed or cohorts incomplete")
        arms[arm]={
            "selected_original_source_visits":sorted(prior["source_sites"]),
            "original_calibration_group_ids":list(CALIBRATION_GROUPS),
            "positive_codes_by_visit":site_codes,
            "source_fit_ID_indices_start_end":[512,768],
            "original_native_model_sha256":model_hash,
            "original_source_sae_sha256":prior["sae_sha256"],
            "original_original_source_input_sha256":sha256(canonical(
                [list(row) for row in calibration]).encode("utf-8")).hexdigest(),
        }
    receipt={
        "schema":SCHEMA,"world_index":root,
        "world_id":original["world_id"],
        "original_source_pack_manifest_sha256":original_pack_sha,
        "original_source_attestation_sha256":sha256(canonical(original).encode("utf-8")).hexdigest(),
        "original_calibration_group_indices":list(CALIBRATION_GROUPS),
        "source_fit_group_indices_used_by_discovery":list(range(16)),
        "previously_scored_source_groups_excluded":list(range(24,32)),
        "source_only_query_groups_excluded":list(range(32,40)),
        "native_model_forward_calls_by_arm":forward_count,
        "native_model_forward_calls":sum(forward_count.values()),
        "original_source_fit_cases_per_arm":SOURCE_CALIBRATION_ROWS,
        "source_query_groups_32_to_39_read":False,
        "task_labels_read":0,"model_logits_stored":False,
        "transient_native_logits_computed_for_GELU_hook":True,
        "model_or_SAE_optimizer_updates":0,
        "recruited_or_confirmation_world_reads":0,
        "behavioral_factorial_interventions_executed":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":arms,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"calibration_codes.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--original-pack",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-registered-source-calibration-forwards",action="store_true")
    args=p.parse_args(argv)
    result=extract(args.world_index,args.source_root,args.nec03t_root,
                   args.reset_root,args.original_pack,args.out,
                   allow_new_source_calibration_forwards=args.allow_registered_source_calibration_forwards)
    print(canonical({name:result[name] for name in (
        "schema","world_id" if not result.get("dry_run") else "root",
        "dry_run" if result.get("dry_run") else "native_model_forward_calls",
        "scientific_gate")}))
    return result


if __name__=="__main__":
    main()
