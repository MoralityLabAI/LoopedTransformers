"""NEC-03R source-only recoverability of clean recurrent state.

Frozen NEC03O A/B reset1800 native checkpoints, source graph and TopK16
dictionary. All eight old development roots, both teacher arms, original
training-source context groups 8..15 only. No native/SAE optimization.
The 24 development-evaluation / 24 reserved groups are never scored.
"""
from __future__ import annotations

import argparse
from hashlib import sha1,sha256
from itertools import combinations
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03o_source_refit import restore_model
from mechinterp.neuron_communities.nec03ob_source_audit import read_model
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations,seal_array
from mechinterp.neuron_communities.topk_sae import TopKSAE
from mechinterp.neuron_communities.nec03e_pairs import attenuate_pair
from mechinterp.neuron_communities.matched_dose import by_visit
from research.pef07.pef07 import as_tensor,model_digest

SCHEMA="NEC-03R/source-only-sae-clean-gap-recovery/v1"
LOCK="NEC03R_SOURCE_LOCK.json"
ROOTS=tuple(range(8))
SOURCE_PANEL=tuple(range(8,16))
FOCAL_STATES=32
CASES=256
SAE_FEATURES=128
SAE_K=16
SOURCE_ELIGIBLE_ROOTS=(0,1,2,4,6,7)
SOURCE_ITT_ZERO=(3,5)
MODELS=(
    "clean","hubs","native_nonhub","sae_affine_nonhub",
    "sae16_contrib_nonhub","ranked_n","ranked_angle",
    "angle_sham","ranked_random","random_active",
)
MEANINGFUL_EFFECT=.02
ANGLE_LIMIT=.25
RANDOM_SALT=b"NEC03R-frozen-disjoint-hash-active-source-v1"


def read(p:Path):
    return json.loads(p.read_text())


def verify_lock():
    path=Path(__file__).with_name(LOCK)
    cfg=read(path)
    expected={
        "schema":"NEC-03R/source-only-recoverability-registration/v1",
        "five_input_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "original_eight_development_root_indices":list(ROOTS),
        "source_context_group_indices":list(SOURCE_PANEL),
        "source_focal_masks_per_group":FOCAL_STATES,
        "source_training_histories_per_arm":CASES,
        "source_native_a_run_id":44898361742,
        "source_sae_a_run_id":44899750508,
        "source_native_and_sae_b_run_id":44900628360,
        "primary_root_ITT_eligible_indices":list(SOURCE_ELIGIBLE_ROOTS),
        "primary_root_ITT_zero_indices":list(SOURCE_ITT_ZERO),
        "primary_comparisons":1,
        "primary_meaningful_effect":MEANINGFUL_EFFECT,
        "source_sham_angle_gap_max":ANGLE_LIMIT,
        "random_active_salt":RANDOM_SALT.decode(),
        "sae_features":SAE_FEATURES,
        "sae_topk":SAE_K,
        "measured_intervention_modes":list(MODELS),
        "native_or_sae_training_updates":0,
        "new_development_eval_contexts_scored":0,
        "reserved_contexts_scored":0,
        "embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise AssertionError("NEC03R source-only study has been reconfigured")
    root=Path(__file__).resolve().parents[2]
    for name,pin in cfg["git_blob_sha1"].items():
        data=(root/name).read_bytes()
        actual=sha1(b"blob "+str(len(data)).encode()+b"\x00"+data).hexdigest()
        if actual!=pin:
            raise AssertionError("NEC03R source protocol or source code changed: "+name)
    return cfg,sha_file(path)


def hashed_disjoint_subset(remaining:list[int],n:int,world_sha:str,arm:str,
                           group:int,mask:int,visit:int)->list[int]:
    """Uniform hash-indexed disjoint equal-n TopK control; label blind."""
    options=list(combinations(sorted(remaining),n))
    if not options:
        raise AssertionError("no disjoint active TopK controls")
    token=(f"{world_sha}|{arm}|{group}|{mask}|{visit}").encode()
    digest=sha256(RANDOM_SALT+b"|"+token).digest()
    index=int.from_bytes(digest,"big")%len(options)
    return list(options[index])


def cosine(a:torch.Tensor,b:torch.Tensor)->float:
    return float(((a*b).sum()/
                  (a.norm()*b.norm()).clamp_min(1e-10)).detach())


def load_source_arm(index:int,arm:str,source:Path,*,reset:Path|None=None):
    """Restore every original arm including known admission failures."""
    if index not in ROOTS or arm not in ("composed","additive"):
        raise PermissionError("source-only frozen population changed")
    world=worlds()[index]
    catalog,manifest=context_catalog(world)
    if index<4:
        if reset is None:
            raise ValueError("NEC03R old source root needs exact reset1800 native archive")
        report=read(source/"nec03o_source_refit_receipt.json")
        audit=read(source/"independent_nec03o_source_audit.json")
        native=read(reset/"checkpoint_reset_source_report.json")["arms"][arm]
        if (audit["source_refit_receipt_sha256"]!=sha_file(source/"nec03o_source_refit_receipt.json")
            or audit["world_sha256"]!=world["world_sha256"]
            or audit["arms"][arm]["new_model_sha256"]!=native["new_native_model_sha256"]
            or not audit["fresh_1800_specific_source_SAE_and_rank_audited"]):
            raise AssertionError("NEC03O-A source model was not independently verified")
        model=restore_model(reset/native["new_checkpoint_file"],world,arm,native)
    else:
        report=read(source/"nec03ob_source_receipt.json")
        audit=read(source/"independent_nec03ob_source_audit.json")
        native=report["arms"][arm]["native"]
        if (audit["source_receipt_sha256"]!=sha_file(source/"nec03ob_source_receipt.json")
            or audit["world_sha256"]!=world["world_sha256"]
            or not audit["both_stage_checkpoint_bytes_and_training_source_labels_audited"]
            or not audit["both_arm_source_SAEs_and_geometry_independently_checked"]):
            raise AssertionError("NEC03O-B source model has not passed its independent audit")
        model,_=read_model(
            source/native["reset1800_checkpoint_file"],
            native["reset1800_checkpoint_sha256"],world,arm,
            native["reset1800_model_sha256"],old=False)
    if (report["world_index"]!=index
        or report["world_sha256"]!=world["world_sha256"]
        or report["source_manifest_sha256"]!=manifest["source_sha256"]
        or report["five_input_registry_sha256"]!=registration()["registry_sha256"]):
        # The old and new receipt schemas store eligibility differently.
        raise AssertionError("wrong source-only world or source eligibility")
    if index<4:
        if report["arms"][arm]["source_model_eligible"]!=audit["arms"][arm]["source_model_eligible"]:
            raise AssertionError("NEC03O-A original source eligibility not preserved")
    else:
        if report["arms"][arm]["source_model_eligible"]!=audit["arms"][arm]["source_eligible"]:
            raise AssertionError("NEC03O-B original source eligibility not preserved")
    rosterfile=source/f"{arm}.source_roster.json"
    roster=read(rosterfile)
    if sha_file(rosterfile)!=report["arms"][arm]["source_roster_sha256"]:
        raise AssertionError("source-selected native graph changed after source preflight")
    detail=roster["roster"] if index<4 else roster["source_roster"]
    candidate=report["arms"][arm]["source_candidate"]
    if (detail["source_candidate"]!=candidate
        or roster["world_sha256"]!=world["world_sha256"]
        or (roster.get("source_sha256") or roster.get("source_manifest_sha256"))!=manifest["source_sha256"]):
        raise AssertionError("native source candidate changed after source registration")
    chosen=by_visit(candidate["nodes"]) if candidate else {}
    hubs=by_visit(detail["source_graph"]["hubs"])
    ae_info=report["arms"][arm]["SAE_source"]
    sae=mean=scale=ranks=None
    sae_hash=rank_hash=None
    if candidate is None:
        if ae_info is not None:
            raise AssertionError("source missing candidate but invented SAE")
    else:
        if ae_info is None:
            raise AssertionError("source candidate exists without its fitted SAE")
        checkpoint=source/ae_info["source_dictionary_file"]
        if sha_file(checkpoint)!=ae_info["source_dictionary_sha256"]:
            raise AssertionError("source TopK16 checkpoint bytes changed")
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if (payload["world_sha256"]!=world["world_sha256"]
            or payload["arm"]!=arm
            or payload["model_sha256"]!=model_digest(model)
            or payload["source_fit_context_groups"]!=40
            or payload["source_rank_context_groups"]!=40):
            raise AssertionError("SAE tied to wrong model or original source group split")
        sae=TopKSAE().eval()
        sae.load_state_dict(payload["state"],strict=True)
        mean,scale,ranks=payload["mean"],payload["scale"],payload["rank"]
        if (seal_array(mean)!=ae_info["normalization_mean_sha256"]
            or seal_array(scale)!=ae_info["normalization_scale_sha256"]
            or payload["rank_sha256"]!=ae_info["source_rank_sha256"]):
            raise AssertionError("source SAE or attribution rank not byte-anchored")
        sae_hash=ae_info["source_dictionary_sha256"]
        rank_hash=ae_info["source_rank_sha256"]
    return {
        "world":world,"catalog":catalog,"manifest":manifest,
        "model":model,"sae":sae,"mean":mean,"scale":scale,"ranks":ranks,
        "source_sites":chosen,"source_hubs":hubs,"source_candidate":candidate,
        "model_sha256":model_digest(model),
        "sae_sha256":sae_hash,"rank_sha256":rank_hash,
        "source_report_sha256":sha_file(source/(
            "nec03o_source_refit_receipt.json" if index<4 else
            "nec03ob_source_receipt.json")),
        "original_arm_source_eligible":report["arms"][arm]["source_model_eligible"],
        "original_paired_root_source_eligible":report["both_arms_source_eligible"],
    }


def group_geometry(prepared:dict,group:int,rows):
    """Source contexts only. No task labels enter active-code selection."""
    if group not in SOURCE_PANEL or len(rows)!=FOCAL_STATES:
        raise PermissionError("NEC03R requires original complete 32-state source group 8..15")
    world=prepared["world"]
    model=prepared["model"]
    x=as_tensor(rows)
    clean=traced_forward(model,x,mode="clean")
    chosen_class=clean["logits"].argmax(1)
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    gates={}
    donors={}
    for visit in sites:
        gate=torch.zeros(64)
        if visit in hubs:
            gate[sorted(hubs[visit])]=1.
        gates[visit]=gate
        donors[visit]={"hubs":torch.zeros(FOCAL_STATES,64)}
    hubtrace=traced_forward(model,x,mode="hubs",orientation=chosen_class,donors=donors,gates=gates,derivative=bool(sites))
    # All variants are exact clean/hubs when original source candidates
    # were missing; never invent a replacement search.
    variants={k:{} for k in MODELS if k not in ("clean","hubs")}
    site_geometry={}
    # An arm with no source-selected recurrent site has no available
    # active sham even though its unmodified logits remain scoreable.
    angle_ok=torch.full((FOCAL_STATES,),bool(sites),dtype=torch.bool)
    random_ok=torch.full((FOCAL_STATES,),bool(sites),dtype=torch.bool)
    ranked_ok=torch.full((FOCAL_STATES,),bool(sites),dtype=torch.bool)
    down=model.block.mlp_down.weight.detach()
    acts=clean_activations(model,x) if sites else None
    for visit,units in sorted(sites.items()):
        states=acts[:,visit,:]
        hub_mask=gates[visit]
        nonhub=1.-hub_mask
        sae,mean,scale=prepared["sae"],prepared["mean"],prepared["scale"]
        normalized=(states-mean)/scale
        with torch.no_grad():
            decoded,z=sae(normalized)
            full_affine=decoded*scale+mean
            decoder64=sae.decoder.weight.detach().T*scale[None,:]
            code_sum=z@decoder64
            recon_offset=mean+sae.decoder.bias.detach()*scale
        if not torch.allclose(full_affine,code_sum+recon_offset,atol=2e-5,rtol=1e-5):
            raise AssertionError("TopK16 physical affine reconstruction does not close")
        n=len(units)
        rankpos={key:i for i,key in enumerate(prepared["ranks"][visit])}
        source=torch.zeros_like(states)
        angular=torch.zeros_like(states)
        random=torch.zeros_like(states)
        angle_valid=torch.zeros(FOCAL_STATES,dtype=torch.bool)
        rnd_valid=torch.zeros(FOCAL_STATES,dtype=torch.bool)
        supported=torch.zeros(FOCAL_STATES,dtype=torch.bool)
        active_ids=[];source_ids=[];angle_ids=[];random_ids=[]
        gaps=[];total_options=[]
        for mask in range(FOCAL_STATES):
            active=[int(i) for i in torch.where(z[mask]>1e-7)[0].tolist()]
            active_ids.append(active)
            sufficient=len(active)==16
            supported[mask]=sufficient
            if not sufficient:
                source_ids.append([]);angle_ids.append([]);random_ids.append([])
                gaps.append(None);total_options.append(0)
                continue
            ranked=sorted(active,key=lambda f:rankpos[f])[:n]
            others=sorted(set(active)-set(ranked))
            combos=list(combinations(others,n))
            source[mask]=z[mask,ranked]@decoder64[ranked]
            target=source[mask]@down.T
            grad=hubtrace["grad_down"][visit][mask]
            source_cos=cosine(target,grad)
            # Match the original NEC03K exhaustive implementation:
            # vectorized candidate 32D contributions, then deterministic
            # lowest angular error over the entire disjoint n-subset family.
            idx=torch.tensor(combos,dtype=torch.long)
            decoder32=decoder64@down.T
            control32=(z[mask,idx].unsqueeze(-1)*decoder32[idx]).sum(dim=1)
            cs=(control32@grad)/(
                control32.norm(dim=1)*grad.norm()).clamp_min(1e-10)
            gaps_tensor=(cs-source_cos).abs()
            selected=int(torch.argmin(gaps_tensor))
            best=list(combos[selected])
            gap=float(gaps_tensor[selected])
            rand=hashed_disjoint_subset(others,n,world["world_sha256"],
                                        "composed" if prepared["arm"]=="composed" else "additive",
                                        group,mask,visit)
            angular[mask]=z[mask,best]@decoder64[best]
            random[mask]=z[mask,rand]@decoder64[rand]
            source_ids.append(ranked);angle_ids.append(best);random_ids.append(rand)
            gaps.append(gap);total_options.append(len(combos))
            angle_valid[mask]=bool(gap<=ANGLE_LIMIT and grad.norm()>1e-7 and
                                   target.norm()>1e-6 and control32[selected].norm()>1e-6)
            rnd_valid[mask]=bool(target.norm()>1e-6 and
                                 (random[mask]@down.T).norm()>1e-6)
        ranked_ok &= supported
        source_angle,sham_angle,angle_dose=attenuate_pair(source,angular,down)
        source_random,sham_random,random_dose=attenuate_pair(source,random,down)
        angle_valid &= torch.tensor(
            [bool(angle_dose["shared_nonzero"][i] and
                  angle_dose["matched_within_1pct"][i])
             for i in range(FOCAL_STATES)])
        rnd_valid &= torch.tensor(
            [bool(random_dose["shared_nonzero"][i] and
                  random_dose["matched_within_1pct"][i])
             for i in range(FOCAL_STATES)])
        angle_ok &= angle_valid
        random_ok &= rnd_valid
        variants["native_nonhub"][visit]=states*nonhub
        variants["sae_affine_nonhub"][visit]=full_affine*nonhub
        variants["sae16_contrib_nonhub"][visit]=code_sum*nonhub
        variants["ranked_n"][visit]=source
        variants["ranked_angle"][visit]=source_angle
        variants["angle_sham"][visit]=sham_angle
        variants["ranked_random"][visit]=source_random
        variants["random_active"][visit]=sham_random
        native_nonhub=states*nonhub
        native32=native_nonhub@down.T
        affine32=(full_affine*nonhub)@down.T
        full16_32=(code_sum*nonhub)@down.T
        ranked32=source@down.T
        site_geometry[str(visit)]={
            "units":sorted(units),
            "hubs":sorted(hubs.get(visit,[])),
            "n_active_per_source":n,
            "all_active_ids":active_ids,
            "source_active_ids":source_ids,
            "exhaustive_angle_sham_ids":angle_ids,
            "hashed_disjoint_sham_ids":random_ids,
            "candidate_search_count":total_options,
            "absolute_cosine_gap":gaps,
            "code_support_exact16":supported.tolist(),
            "angle_control_valid_site":angle_valid.tolist(),
            "random_control_valid_site":rnd_valid.tolist(),
            "natural_output_dose":{
                "native_nonhub":native32.norm(dim=1).tolist(),
                "sae_affine_nonhub":affine32.norm(dim=1).tolist(),
                "sae16_contrib_nonhub":full16_32.norm(dim=1).tolist(),
                "source_ranked_n":ranked32.norm(dim=1).tolist(),
            },
            "source_vs_angle_shared_output_dose":angle_dose["shared_achievable_dose"],
            "source_vs_angle_source_gain":angle_dose["left_gain"],
            "source_vs_angle_sham_gain":angle_dose["right_gain"],
            "source_vs_random_shared_output_dose":random_dose["shared_achievable_dose"],
            "source_vs_random_source_gain":random_dose["left_gain"],
            "source_vs_random_sham_gain":random_dose["right_gain"],
            "full_64d_reconstruction_relative_error":(
                (full_affine-states).norm(dim=1)/
                states.norm(dim=1).clamp_min(1e-6)).tolist(),
            "full_nonhub_32d_reconstruction_relative_error":(
                (affine32-native32).norm(dim=1)/
                native32.norm(dim=1).clamp_min(1e-6)).tolist(),
            "topk16_contrib_32d_reconstruction_relative_error":(
                (full16_32-native32).norm(dim=1)/
                native32.norm(dim=1).clamp_min(1e-6)).tolist(),
        }
    for visit in sites:
        variants["ranked_n"][visit]*=ranked_ok[:,None]
        for key in ("ranked_angle","angle_sham"):
            variants[key][visit]*=angle_ok[:,None]
        for key in ("ranked_random","random_active"):
            variants[key][visit]*=random_ok[:,None]
    traces={"clean":clean,"hubs":hubtrace}
    for mode in variants:
        vdonors={visit:{"native":vec} for visit,vec in variants[mode].items()}
        with torch.no_grad():
            traces[mode]=traced_forward(
                model,x,mode="native",orientation=chosen_class,
                donors=vdonors,gates=gates)
    if not torch.allclose(traces["ranked_angle"]["logits"][~angle_ok],
                          traces["angle_sham"]["logits"][~angle_ok],atol=1e-6):
        raise AssertionError("invalid-angle control emitted a nonzero causal signal")
    if not torch.allclose(traces["ranked_random"]["logits"][~random_ok],
                          traces["random_active"]["logits"][~random_ok],atol=1e-6):
        raise AssertionError("invalid-random control emitted a nonzero causal signal")
    return {
        "source_group":group,
        "original_source_input_histories":rows,
        "focal_masks":list(range(32)),
        "clean_chosen_class":chosen_class.tolist(),
        "logits":{key:val["logits"].tolist() for key,val in traces.items()},
        "memory_distance_from_hubs_by_visit":{
            key:(val["memory"]-hubtrace["memory"]).norm(dim=-1).mean(dim=0).tolist()
            for key,val in traces.items()},
        "site_geometry":site_geometry,
        "valid_angle_pair_on_all_source_visits":angle_ok.tolist(),
        "valid_random_pair_on_all_source_visits":random_ok.tolist(),
        "valid_ranked_active_on_all_source_visits":ranked_ok.tolist(),
        "true_teacher_labels_used_in_routing":False,
    }


def run(index:int,source:Path,out:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("source recovery only on original eight model worlds")
    world=worlds()[index]
    out.mkdir(parents=True,exist_ok=True)
    arm_results={}
    for arm in ("composed","additive"):
        model=load_source_arm(index,arm,source,reset=reset)
        model["arm"]=arm
        if model["world"]["world_sha256"]!=world["world_sha256"]:
            raise AssertionError("world native source checkpoint changed")
        counts={key:0 for key in MODELS}
        raw=[]
        for g in SOURCE_PANEL:
            # Must only use original 80 training source groups. No model
            # evaluation on 24 developmental or 24 reserved groups.
            bits=model["catalog"]["train"][g*32:(g+1)*32]
            record=group_geometry(model,g,bits)
            y=labels(as_tensor(bits),world,control=(arm=="additive")).tolist()
            predicted={}
            for key in MODELS:
                pred=torch.tensor(record["logits"][key]).argmax(dim=1).tolist()
                predicted[key]=pred
                counts[key]+=sum(int(a==b) for a,b in zip(pred,y))
            record["true_teacher_labels"]=y
            record["class_predictions"]=predicted
            record["arm"]=arm
            raw.append(record)
        blob=b"".join(encoded(record) for record in raw)
        output=out/f"{arm}.source_recoverability.jsonl"
        output.write_bytes(blob)
        gap=counts["clean"]-counts["hubs"]
        fraction={
            key:((counts[key]-counts["hubs"])/gap if gap>0 else None)
            for key in MODELS if key not in ("clean","hubs")
        }
        arm_results[arm]={
            "arm":arm,"native_sha256":model["model_sha256"],
            "sae_checkpoint_sha256":model["sae_sha256"],
            "source_rank_sha256":model["rank_sha256"],
            "source_receipt_sha256":model["source_report_sha256"],
            "source_candidate_present":bool(model["source_candidate"]),
            "source_eligible_under_original_90pct_gate":model["original_arm_source_eligible"],
            "total_source_training_cases":CASES,
            "source_context_groups_scored":list(SOURCE_PANEL),
            "per_intervention_correct_counts":counts,
            "clean_minus_hubs_correct_gap":gap,
            "correctness_gap_fraction_recovered_if_positive":fraction,
            "ranked_source_minus_angle_sham_correct":counts["ranked_angle"]-counts["angle_sham"],
            "ranked_source_minus_random_sham_correct":counts["ranked_random"]-counts["random_active"],
            "full_affine_minus_ranked_n_correct":counts["sae_affine_nonhub"]-counts["ranked_n"],
            "source_root_raw_file":output.name,
            "source_root_raw_sha256":sha256(blob).hexdigest(),
        }
        if model_digest(model["model"])!=model["model_sha256"]:
            raise AssertionError("source-only causal recovery mutated the native model")
    observed=(arm_results["composed"]["ranked_source_minus_angle_sham_correct"]-
              arm_results["additive"]["ranked_source_minus_angle_sham_correct"])/CASES
    admitted=index in SOURCE_ELIGIBLE_ROOTS
    report={
        "schema":SCHEMA,"world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "source_manifest_sha256":context_catalog(world)[1]["source_sha256"],
        "original_five_input_registry_sha256":registration()["registry_sha256"],
        "original_four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "source_lock_sha256":lock_sha,
        "original_paired_root_source_eligible":admitted,
        "original_ITT_source_failure":index in SOURCE_ITT_ZERO,
        "raw_root_angle_sham_DID":observed,
        "primary_ITT_angle_sham_DID":observed if admitted else 0.,
        "source_groups_scored":list(SOURCE_PANEL),
        "source_training_histories_per_arm":CASES,
        "models_or_SAEs_retrained":0,
        "new_development_eval_contexts_scored":0,
        "reserved_contexts_scored":0,
        "embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "arms":arm_results,
        "scientific_evidence":"source-only in-sample mechanism resolution",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec03r_source_recovery_report.json").write_bytes(encoded(report))
    print("NEC03R_SOURCE_RECOVERABILITY_ROOT_WRITTEN",world["world_id"])
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.world_index,a.source_root,a.out,reset=a.reset_root)
