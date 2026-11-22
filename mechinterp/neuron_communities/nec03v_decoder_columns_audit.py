"""Independent original NEC-03V 32D decoder projection audit, CPU-only.

This does not import the scorer implementation. It independently multiplies
the original frozen native MLP-down tensor with each of 128 physical SAE
decoder columns, omitting original source hubs at registered visits.

Only immutable historical checkpoint bytes are loaded via torch.load with
weights_only=True; no model objects, logits, forward passes or task inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .nec03v_source_attest import attest, canonical, sha256_file, _existing_name


def parse_node(key):
    parts=key.split(":") if isinstance(key,str) else []
    if (len(parts)!=2 or any(not q.isdecimal() for q in parts)
            or int(parts[0]) not in range(12) or int(parts[1]) not in range(64)):
        raise ValueError("out-of-range original recurrent visit/native neuron")
    return int(parts[0]),int(parts[1])


def independently_review(root, source:Path, t:Path, reset:Path|None, result:Path):
    att=attest(root,source,t,reset)
    report=json.loads(result.read_text(encoding="utf-8"))
    if (report.get("schema")!="NEC-03V/source-frozen-projected-decoder-columns/v1"
            or report["world_index"]!=root
            or report["world_id"]!=att["world_id"]
            or report["original_source_attestation"]!=att
            or report["native_forward_calls"]!=0
            or report["SAE_forward_calls"]!=0
            or report["model_or_SAE_optimizer_steps"]!=0
            or report["labels_accessed"]!=0
            or report["logits_accessed"]!=0
            or report["new_recruited_root_reads"]!=0
            or report["embargoed_confirmation_reads"]!=0
            or report["source_fit_activations_recovered"] is not False
            or report["per_case_active_code_coefficients_recovered"] is not False
            or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("decoder export claims inconsistent scope or inference")
    receipt_name=("nec03o_source_refit_receipt.json" if root<4 else
                  "nec03ob_source_receipt.json")
    metadata=json.loads((source/receipt_name).read_text(encoding="utf-8"))
    if set(report["arms"])!={"composed","additive"}:
        raise ValueError("decoder export must contain both original model arms")
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("independent decoder audit requires CPU-only PyTorch")
    counted=0
    for arm in ("composed","additive"):
        original=metadata["arms"][arm]
        a=report["arms"][arm]
        refs=att["source_checkpoint_detail"][arm]
        record=json.loads((source/f"{arm}.source_roster.json").read_text(encoding="utf-8"))
        source_graph=record.get("roster",record.get("source_roster"))
        if not isinstance(source_graph,dict):
            raise ValueError("missing original source graph")
        selected=original["source_candidate"]
        nodes=selected["nodes"] if selected is not None else []
        visits=sorted(set(parse_node(node)[0] for node in nodes))
        if (a["selected_visits"]!=visits
                or a["source_sites_sha256"]!=refs["source_roster_sha256"]
                or set(a["decoder32"])!={str(v) for v in visits}):
            raise ValueError("original source recurrent visits or roster changed")
        if not visits:
            if root not in (3,5) or a["original_SAE_absent"] is not True:
                raise ValueError("ineligible original source arm altered")
            continue
        hubs=source_graph["source_graph"]["hubs"]
        hubs_by_visit={}
        for node in hubs:
            v,n=parse_node(node)
            hubs_by_visit.setdefault(v,set()).add(n)
        native_dir=reset if root<4 else source
        if native_dir is None:
            raise ValueError("old original source requires reset-native files")
        native=torch.load(native_dir/_existing_name(refs["native_checkpoint_file"]),
                          map_location="cpu",weights_only=True)
        sae=torch.load(source/_existing_name(original["SAE_source"]["source_dictionary_file"]),
                       map_location="cpu",weights_only=True)
        down=native["model"]["block.mlp_down.weight"].detach().float().cpu().tolist()
        decoder=sae["state"]["decoder.weight"].detach().float().cpu().tolist()
        scale=sae["scale"].detach().float().cpu().tolist()
        if len(down)!=32 or any(len(row)!=64 for row in down):
            raise ValueError("original MLP-down dimensions differ")
        if len(decoder)!=64 or any(len(row)!=128 for row in decoder):
            raise ValueError("original SAE feature dimensions differ")
        if len(scale)!=64 or not all(math.isfinite(v) and v>0 for v in scale):
            raise ValueError("invalid original SAE normalization scale")
        for v in visits:
            reported=a["decoder32"][str(v)]
            if len(reported)!=128 or any(len(row)!=32 for row in reported):
                raise ValueError("reported projected columns have wrong dimensions")
            hub_set=hubs_by_visit.get(v,set())
            for f in range(128):
                for i in range(32):
                    expected=math.fsum(
                        down[i][j]*decoder[j][f]*scale[j]
                        for j in range(64) if j not in hub_set)
                    value=reported[f][i]
                    if (type(value) not in (float,int) or not math.isfinite(value)
                            or not math.isclose(expected,float(value),
                                                abs_tol=2e-5,rel_tol=2e-5)):
                        raise ValueError(
                            f"projected 32D frozen decoder differs: {arm}/v{v}/f{f}/d{i}")
                    counted+=1
    return {
        "schema":"NEC-03V/independent-frozen-decoder-32D-audit/v1",
        "world_index":root,"world_id":att["world_id"],
        "original_archival_lock_verified":True,
        "source_checkpoint_bytes_verified":True,
        "decoder_values_audited":counted,
        "decoder_columns_independently_verified":True,
        "per_case_active_code_coefficients_verified":False,
        "original_decoder_export_sha256":sha256_file(result),
        "new_neural_forward_calls":0,
        "new_model_training_updates":0,
        "labels_or_logits_read":0,
        "protected_cohort_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--columns",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise ValueError("independent decoder report must be a fresh file")
    result=independently_review(
        a.world_index,a.source_root,a.nec03t_root,a.reset_root,a.columns)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(canonical(result)+"\n",encoding="utf-8")
    print(canonical(result))
    return result


if __name__=="__main__":
    main()
