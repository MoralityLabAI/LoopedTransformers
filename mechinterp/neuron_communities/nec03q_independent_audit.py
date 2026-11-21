"""NEC-03Q independent causal replay from immutable models and logged routes.

This auditor does not call the runner's direction matcher or dose matcher.
It independently enumerates every disjoint active-code combination from
the checkpoint-specific decoder, reconstructs each physically attenuated
donor, reexecutes the recurrent forward trajectory, and checks all truth
labels and paired predictions on 24 complete developmental context groups.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import torch

from mechinterp.neuron_communities.nec03q_causal_run import (
    ROOTS,SCHEMA,DEV_CASES,DEVELOPMENT_GROUPS,read,verify_lock,load_frozen_arm,
    EXPECTED_MANIFEST_SHA256,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.matched_dose import by_visit
from research.pef07.pef07 import as_tensor,model_digest

AUDIT_SCHEMA="NEC-03Q/independent-development-causal-exhaustive-audit/v1"


def close(actual,expected,*,tol=1e-4):
    if abs(float(actual)-float(expected))>tol+abs(float(expected))*1e-5:
        raise AssertionError(f"independent physical source/sham route changed {actual} vs {expected}")


def single_cos(a,b):
    n=a.norm()*b.norm()
    return float((a*b).sum()/n.clamp_min(1e-10))


def independent_replay(loaded:dict,record:dict,world:dict,arm:str):
    """Prove source and sham vectors, exhaustive direction, full recurrent logits."""
    group=record["development_group"]
    if (record["schema"]!=SCHEMA or record["arm"]!=arm
        or record["world_id"]!=world["world_id"]
        or record["original_split_group_index"]!=80+group
        or record["focal_masks"]!=list(range(32))
        or record["true_labels_used_in_donor_selection"] is not False):
        raise AssertionError("development input membership or label-blind control policy changed")
    catalog,_=context_catalog(world)
    expected=catalog["development"][group*32:(group+1)*32]
    if [tuple(z) for z in record["input_histories"]]!=list(expected):
        raise AssertionError("development context not the original 32-state focal block")
    x=as_tensor(expected)
    model=loaded["model"]
    baseline=traced_forward(model,x,mode="clean")
    orientation=baseline["logits"].argmax(1)
    if orientation.tolist()!=record["clean_chosen_class"]:
        raise AssertionError("clean-chosen margin orientation changed")
    sites=loaded["chosen"]
    hubs=loaded["hubs"]
    gates={}
    donors={}
    for visit in sites:
        mask=torch.zeros(64)
        if visit in hubs:mask[list(sorted(hubs[visit]))]=1.
        gates[visit]=mask
        donors[visit]={"hubs":torch.zeros(32,64)}
    reference=traced_forward(
        model,x,mode="hubs",orientation=orientation,donors=donors,gates=gates,derivative=True)
    acts=clean_activations(model,x)
    down=model.block.mlp_down.weight.detach()
    decoder64=loaded["sae"].decoder.weight.detach().T*loaded["scale"][None,:]
    decoder32=decoder64@down.T
    valid=torch.ones(32,dtype=torch.bool)
    if set(record["site_geometry"])!={str(v) for v in sites}:
        raise AssertionError("causal site set not original source-selected recurrent visits")
    vectors={}
    for visit,units in sorted(sites.items()):
        row=record["site_geometry"][str(visit)]
        n=len(units)
        if row["n_components"]!=n or any(
            len(row[name])!=32 for name in (
                "source_feature_ids","disjoint_sham_feature_ids","all_active_feature_ids",
                "cosine_gap","angle_gate","candidate_count","shared_direct_output_dose",
                "source_gain","sham_gain","dose_match_1pct","shared_nonzero","code_support_exact16")):
            raise AssertionError("missing a full 32-mask source or sham route")
        codes=loaded["sae"].codes((acts[:,visit,:]-loaded["mean"])/loaded["scale"])
        rank={feature:k for k,feature in enumerate(loaded["ranks"][visit])}
        src=torch.zeros(32,64)
        sham=torch.zeros(32,64)
        for i in range(32):
            z=codes[i]
            active=[int(j) for j in torch.where(z>1e-7)[0].tolist()]
            supported=len(active)==16
            if supported!=row["code_support_exact16"][i]:
                raise AssertionError("TopK positive-code support changed")
            if not supported:
                if (row["candidate_count"][i]!=0 or row["source_feature_ids"][i]
                    or row["disjoint_sham_feature_ids"][i] or row["angle_gate"][i]
                    or row["shared_nonzero"][i]):
                    raise AssertionError("unsupported TopK input erroneously admitted")
                valid[i]=False
                continue
            selected=sorted(active,key=lambda j:rank[j])[:n]
            if selected!=row["source_feature_ids"][i]:
                raise AssertionError("source rank does not match frozen 40-group feature ordering")
            other=sorted(set(active)-set(selected))
            allowed=list(combinations(other,n))
            chosen=tuple(row["disjoint_sham_feature_ids"][i])
            if (len(allowed)!=row["candidate_count"][i]
                or chosen not in allowed or active!=row["all_active_feature_ids"][i]):
                raise AssertionError("incorrect or overlapping active sham composition")
            source64=(z[selected,None]*decoder64[selected]).sum(dim=0)
            control64=(z[list(chosen),None]*decoder64[list(chosen)]).sum(dim=0)
            source32=source64@down.T
            control32=control64@down.T
            grad=reference["grad_down"][visit][i]
            cos_src=single_cos(source32,grad)
            gap=abs(cos_src-single_cos(control32,grad))
            # Independent exhaustive finite search; ties accepted within
            # physical floating-point tolerance from the original scorer.
            distances=[]
            for subset in allowed:
                v=(z[list(subset),None]*decoder32[list(subset)]).sum(dim=0)
                distances.append(abs(cos_src-single_cos(v,grad)))
            if gap>min(distances)+5e-5:
                raise AssertionError("development control is not closest angular sham")
            close(gap,row["cosine_gap"][i],tol=5e-5)
            a=float(source32.norm())
            b=float(control32.norm())
            shared=min(a,b)
            close(shared,row["shared_direct_output_dose"][i])
            gain_src=shared/a if a>1e-10 else 0.
            gain_sham=shared/b if b>1e-10 else 0.
            close(gain_src,row["source_gain"][i])
            close(gain_sham,row["sham_gain"][i])
            if gain_src>1.000001 or gain_sham>1.000001:
                raise AssertionError("source or control dose was amplified")
            matched=(abs(a*gain_src-b*gain_sham)/max(shared,1e-6)<=.01)
            angle_good=(gap<=.25 and float(grad.norm())>1e-7 and a>1e-6 and b>1e-6)
            here=bool(angle_good and matched and shared>1e-6)
            if (row["angle_gate"][i]!=angle_good
                or row["shared_nonzero"][i]!=(shared>1e-6)
                or row["dose_match_1pct"][i]!=matched):
                raise AssertionError("physical null source gates changed")
            if not here:
                valid[i]=False
            src[i]=source64*gain_src
            sham[i]=control64*gain_sham
        vectors[visit]=(src,sham)
    if valid.tolist()!=record["valid_input_control"]:
        raise AssertionError("missing per-input active controls were post-selected")
    for visit,(src,sham) in vectors.items():
        keep=valid.to(src.dtype).unsqueeze(1)
        donors[visit]["native"]=src*keep
        donors[visit]["sham"]=sham*keep
    with torch.no_grad():
        source=traced_forward(model,x,mode="native",orientation=orientation,donors=donors,gates=gates)
        sham=traced_forward(model,x,mode="sham",orientation=orientation,donors=donors,gates=gates)
    for name,logits in (("clean",baseline["logits"]),("hubs",reference["logits"]),
                        ("source",source["logits"]),("sham",sham["logits"])):
        if not torch.allclose(logits,torch.tensor(record[f"{name}_logits"]),rtol=1e-5,atol=3e-4):
            raise AssertionError(f"independently reconstructed recurrent {name} logits differ")
    y=labels(x,world,control=(arm=="additive")).tolist()
    if record["true_labels"]!=y:
        raise AssertionError("true task labels differ from explicit five-bit teacher")
    counts={}
    for name,logits in (
        ("source",source["logits"]),("sham",sham["logits"]),
        ("clean",baseline["logits"]),("hubs",reference["logits"])):
        pred=logits.argmax(1).tolist()
        if record[f"{name}_class_prediction"]!=pred:
            raise AssertionError("source/sham paired correctness labels not reproducible")
        counts[name]=sum(int(a==b) for a,b in zip(pred,y))
    if any(source["logits"][~valid].flatten()!=sham["logits"][~valid].flatten()):
        if not torch.allclose(source["logits"][~valid],sham["logits"][~valid],atol=1e-6):
            raise AssertionError("invalid-input hubs-only fallback differs across arms")
    counts["valid_input_controls"]=sum(valid.tolist())
    return counts


def audit(index:int,source:Path,output:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("cannot audit a source-ineligible or embargoed root")
    report_path=output/"nec03q_development_causal_report.json"
    report=read(report_path)
    world=worlds()[index]
    catalog,source_manifest=context_catalog(world)
    if (report["schema"]!=SCHEMA or report["world_index"]!=index
        or report["world_sha256"]!=world["world_sha256"]
        or report["frozen_execution_lock_sha256"]!=lock_sha
        or report["frozen_prior_manifest_sha256"]!=EXPECTED_MANIFEST_SHA256
        or report["old_source_registry_sha256"]!=registration()["registry_sha256"]
        or report["four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or report["original_source_manifest_sha256"]!=source_manifest["source_sha256"]
        or report["development_groups_evaluated"]!=DEVELOPMENT_GROUPS
        or report["native_and_sae_weights_updated"] is not False
        or report["true_labels_used_for_control_selection"] is not False
        or report["primary_comparison_family_size"]!=1
        or report["reserved_contexts_scored"]!=0
        or report["embargoed_worlds_scored"]!=0
        or report["original_four_input_confirmation_worlds_scored"]!=0
        or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("locked causal development experiment changed")
    scores={}
    for arm in ("composed","additive"):
        model=load_frozen_arm(index,arm,source,reset=reset)
        a=report["arms"][arm]
        if (model["native_sha256"]!=a["native_sha256"]
            or model["sae_checkpoint_sha256"]!=a["sae_checkpoint_sha256"]
            or model["sae_rank_sha256"]!=a["source_rank_sha256"]
            or model["source_receipt_sha256"]!=a["source_receipt_sha256"]
            or a["total_input_cases"]!=DEV_CASES):
            raise AssertionError("causal report uses another frozen native or SAE")
        path=output/a["raw_file"]
        if sha_file(path)!=a["raw_sha256"]:
            raise AssertionError("raw case-level causal evidence SHA256 mismatch")
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if len(records)!=DEVELOPMENT_GROUPS:
            raise AssertionError("missing full development context groups")
        counts={key:0 for key in ("source","sham","clean","hubs","valid_input_controls")}
        for j,rec in enumerate(records):
            if rec["development_group"]!=j:
                raise AssertionError("duplicate or missing group index")
            rows=independent_replay(model,rec,world,arm)
            for key,value in rows.items():
                counts[key]+=value
        for key in ("source","sham","clean","hubs"):
            if counts[key]!=a[f"{key}_correct"]:
                raise AssertionError("reported paired correctness numerator changed")
        if (counts["valid_input_controls"]!=a["valid_input_controls"]
            or counts["source"]-counts["sham"]!=a["signed_source_minus_sham_correct"]):
            raise AssertionError("causal source minus sham or invalid-case coverage changed")
        scores[arm]={
            "source_correct":counts["source"],"sham_correct":counts["sham"],
            "clean_correct":counts["clean"],"hubs_correct":counts["hubs"],
            "valid_input_controls":counts["valid_input_controls"],
            "case_count":DEV_CASES,
            "source_minus_sham_correct":counts["source"]-counts["sham"],
            "raw_sha256":a["raw_sha256"],
            "native_sha256":a["native_sha256"],"sae_sha256":a["sae_checkpoint_sha256"],
        }
    effect=(scores["composed"]["source_minus_sham_correct"]
            -scores["additive"]["source_minus_sham_correct"])/DEV_CASES
    close(effect,report["root_ITT_effect"],tol=1e-10)
    result={
        "schema":AUDIT_SCHEMA,"world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "reported_root_sha256":sha_file(report_path),
        "frozen_execution_lock_sha256":lock_sha,
        "all_dev_group_memberships_and_true_task_labels_independently_recomputed":True,
        "all_32D_disjoint_active_controls_independently_exhaustively_reenumerated":True,
        "all_paired_recurrent_logits_and_correctness_independently_replayed":True,
        "source_checkpoints_and_SAEs_immutable":True,
        "root_ITT_effect":effect,"arms":scores,
        "reserved_contexts_scored":0,"embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (output/"independent_nec03q_causal_audit.json").write_bytes(encoded(result))
    print("NEC03Q_DEVELOPMENT_CAUSAL_ROOT_INDEPENDENTLY_REPLAYED",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
