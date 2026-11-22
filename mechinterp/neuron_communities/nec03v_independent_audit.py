"""NEC-03V independent native/SAE physical coherent-group replay.

Rebuild all original source groups, decoder-coherent quartets and sextets,
every disjoint active-code control, matched physical 32D removal, frozen
hash shams, 11 full recurrent trajectories and true Boolean task labels.
Never call the NEC03V scorer's group_measure/group selection/dose helper.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03v_coherent_source import (
    ROOTS,PANEL,QUALIFIED,ITT_ZERO,SCHEMA,FOCAL,CASES,
    GROUPS,ANGLE_LIMIT,COHESION_LIMIT,HASH_SALT,MODES,read,verify_lock,
)
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor,model_digest

AUDIT_SCHEMA="NEC-03V/independent-coherent-group-physical-null-replay/v1"


def near(actual,expected,*,atol=2e-4,rtol=1e-5,what="native source"):
    a=torch.as_tensor(actual,dtype=torch.float64)
    b=torch.as_tensor(expected,dtype=torch.float64)
    if a.shape!=b.shape or not torch.allclose(a,b,atol=atol,rtol=rtol):
        err=float((a-b).abs().max()) if a.shape==b.shape else -1
        raise AssertionError(f"independent {what} disagrees: {err}")


def cos(a,b):
    return (a*b).sum(dim=-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-10)


def independent_pair(a,b,weight):
    d0=(a@weight.T).norm(dim=1)
    d1=(b@weight.T).norm(dim=1)
    target=torch.minimum(d0,d1)
    g0=torch.where(d0>1e-10,target/d0.clamp_min(1e-10),torch.zeros_like(d0)).clamp(0,1)
    g1=torch.where(d1>1e-10,target/d1.clamp_min(1e-10),torch.zeros_like(d1)).clamp(0,1)
    src=a*g0[:,None];sham=b*g1[:,None]
    matched=((src@weight.T).norm(dim=1)-(sham@weight.T).norm(dim=1)).abs()/target.clamp_min(1e-6)<=.01
    return src,sham,d0,d1,target,g0,g1,matched


def replay_group(prepared,record,arm):
    group=record["source_group"]
    if (record["schema"]!=SCHEMA or group not in PANEL
        or record["arm"]!=arm
        or record["focal_masks"]!=list(range(FOCAL))
        or record["teacher_labels_used_for_group_selection"] is not False):
        raise AssertionError("NEC03V original-source group or label-blind route altered")
    original=prepared["catalog"]["train"][group*FOCAL:(group+1)*FOCAL]
    if [tuple(v) for v in record["source_input_histories"]]!=original:
        raise AssertionError("source group 32..39 changed, missing, or incomplete")
    model=prepared["model"]
    x=as_tensor(original)
    clean=traced_forward(model,x,mode="clean")
    direction=clean["logits"].argmax(1)
    if direction.tolist()!=record["clean_chosen_orientation"]:
        raise AssertionError("clean-chosen margin orientation changed")
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    if set(record["site_geometry"])!={str(v) for v in sites}:
        raise AssertionError("source-frozen graph visits differ from original")
    gate={}
    for visit in sites:
        m=torch.zeros(64)
        if visit in hubs: m[sorted(hubs[visit])]=1.
        gate[visit]=m
    empty={visit:{"hubs":torch.zeros(FOCAL,64)} for visit in sites}
    hubs_trace=traced_forward(model,x,mode="hubs",
        orientation=direction,donors=empty,gates=gate)
    acts=clean_activations(model,x) if sites else None
    variants={name:{} for name in MODES if name not in ("clean","hubs")}
    full={}
    codes={}
    physical_decoder={}
    down=model.block.mlp_down.weight.detach()
    for visit in sorted(sites):
        native=acts[:,visit,:]
        nonhub=1.-gate[visit]
        sae=prepared["sae"]
        with torch.no_grad():
            z=sae.codes((native-prepared["mean"])/prepared["scale"])
            cols=sae.decoder.weight.detach().T*prepared["scale"][None,:]
            aff=sae.decoder(z)*prepared["scale"]+prepared["mean"]
            feature=z@cols
            off=prepared["mean"]+sae.decoder.bias.detach()*prepared["scale"]
            if not torch.allclose(aff,feature+off,rtol=1e-5,atol=2e-5):
                raise AssertionError("immutable full SAE physical decoder affine identity failed")
        full[visit]=aff*nonhub
        codes[visit]=z
        physical_decoder[visit]=cols
        variants["native_nonhub"][visit]=native*nonhub
        variants["full_affine"][visit]=full[visit]
        variants["contrib16_only"][visit]=feature*nonhub
    if sites:
        base=traced_forward(model,x,mode="native",
            orientation=direction,donors={v:{"native":x} for v,x in full.items()},
            gates=gate,derivative=True)
    else:
        base=clean
    availability={b:torch.full((FOCAL,),bool(sites),dtype=torch.bool) for b in GROUPS}
    hash_avail=torch.full((FOCAL,),bool(sites),dtype=torch.bool)
    for visit,nodes in sorted(sites.items()):
        saved=record["site_geometry"][str(visit)]
        if (saved["native_selected_units"]!=sorted(nodes)
            or saved["original_graph_hubs"]!=sorted(hubs.get(visit,[]))
            or set(saved["groups"])!=set(GROUPS)):
            raise AssertionError("registered original source graph, group family missing")
        nonhub=1.-gate[visit]
        cols=physical_decoder[visit]
        phys=(cols*nonhub[None,:])@down.T
        lengths=phys.norm(dim=1)
        matrix=(phys@phys.T)/(lengths[:,None]*lengths[None,:]).clamp_min(1e-10)
        z=codes[visit]
        rank={f:i for i,f in enumerate(prepared["ranks"][visit])}
        near((full[visit]@down.T).norm(dim=1).tolist(),
            saved["full_affine_32d_output_dose"],what="full affine output dose")
        near(((full[visit]-variants["native_nonhub"][visit])@down.T).norm(dim=1).div(
             (variants["native_nonhub"][visit]@down.T).norm(dim=1).clamp_min(1e-6)).tolist(),
             saved["full_affine_32d_native_relative_error"],
             what="physical full affine native reconstruction error")
        for budget,k in GROUPS.items():
            summary=saved["groups"][budget]
            if summary["k"]!=k or not 2<=k<=8:
                raise AssertionError("pre-registered group width changed")
            src=torch.zeros(FOCAL,64)
            sham=torch.zeros(FOCAL,64)
            rnd=torch.zeros(FOCAL,64)
            good=torch.ones(FOCAL,dtype=torch.bool)
            rand_good=torch.ones(FOCAL,dtype=torch.bool)
            for mask in range(FOCAL):
                active=torch.where(z[mask]>1e-7)[0].tolist()
                if (summary["active_ids"][mask]!=active
                    or summary["exactly_16_positive_active_codes"][mask]!=(len(active)==16)):
                    raise AssertionError("original TopK16 active source support changed")
                if len(active)!=16:
                    good[mask]=False
                    rand_good[mask]=False
                    if any(summary[key][mask] for key in ("source_ids","sham_ids","hash_sham_ids")):
                        raise AssertionError("unavailable SAE support invented a code group")
                    if summary["exhaustive_control_count"][mask]!=0:
                        raise AssertionError("unavailable SAE support invented sham candidates")
                    if any(summary[key][mask] is not None for key in (
                        "angular_gap","cohesion_gap","source_physical_cohesion",
                        "sham_physical_cohesion","sham_joint_score")):
                        raise AssertionError("unsupported source group has physical measure")
                    continue
                seed=min(active,key=lambda f:(rank[f],f))
                partners=sorted((v for v in active if v!=seed),
                    key=lambda f:(-float(matrix[seed,f]),rank[f],f))
                selected=[seed]+partners[:k-1]
                possibilities=list(combinations(sorted(set(active)-set(selected)),k))
                if (selected!=summary["source_ids"][mask]
                    or len(possibilities)!=summary["exhaustive_control_count"][mask]
                    or len(possibilities)!=__import__("math").comb(16-k,k)):
                    raise AssertionError("source cohesive group or full disjoint sham roster changed")
                idx=torch.tensor(possibilities,dtype=torch.long)
                sx=(z[mask,selected]@cols[selected])*nonhub
                controls=(z[mask,idx].unsqueeze(-1)*phys[idx]).sum(1)
                grad=base["grad_down"][visit][mask]
                src_angle=cos(sx@down.T,grad)
                gaps=(cos(controls,grad)-src_angle).abs()
                coh_source=sum(matrix[selected[i],selected[j]]
                    for i in range(k) for j in range(i+1,k))/(k*(k-1)//2)
                candidate_coh=torch.zeros(len(possibilities))
                for i in range(k):
                    for j in range(i+1,k):
                        candidate_coh+=matrix[idx[:,i],idx[:,j]]
                candidate_coh/=k*(k-1)//2
                diffs=(candidate_coh-coh_source).abs()
                scores=gaps+.5*diffs
                optimum=float(scores.min())
                chosen=tuple(summary["sham_ids"][mask])
                if chosen not in possibilities:
                    raise AssertionError("archived sham overlaps source or omits active feature")
                loc=possibilities.index(chosen)
                if float(scores[loc])>optimum+5e-5:
                    raise AssertionError("archived sham does not minimize frozen joint physical criterion")
                near(float(coh_source),summary["source_physical_cohesion"][mask],what="group source coherence")
                near(float(candidate_coh[loc]),summary["sham_physical_cohesion"][mask],what="sham group coherence")
                near(float(diffs[loc]),summary["cohesion_gap"][mask],what="coherence gap")
                near(float(gaps[loc]),summary["angular_gap"][mask],what="angular gap")
                near(float(scores[loc]),summary["sham_joint_score"][mask],what="joint physical objective")
                src[mask]=sx
                sham[mask]=(z[mask,list(chosen)]@cols[list(chosen)])*nonhub
                if budget=="four":
                    token=f"{prepared['world']['world_sha256']}|{arm}|{group}|{mask}|{visit}|four".encode()
                    h=int.from_bytes(sha256(HASH_SALT+b"|"+token).digest(),"big")
                    random_ids=list(possibilities[h%len(possibilities)])
                    if summary["hash_sham_ids"][mask]!=random_ids:
                        raise AssertionError("hash-selected disjoint quartet changed")
                    rnd[mask]=(z[mask,random_ids]@cols[random_ids])*nonhub
                elif summary["hash_sham_ids"][mask]:
                    raise AssertionError("unregistered hash sham on secondary sextet")
                if (float(grad.norm())<=1e-7 or
                    float(gaps[loc])>ANGLE_LIMIT or
                    float(diffs[loc])>COHESION_LIMIT):
                    good[mask]=False
            matched_source,matched_sham,d0,d1,target,g0,g1,matched=independent_pair(src,sham,down)
            dose=summary["angular_physical_dose"]
            for name,actual in (
                ("raw_source_32d_dose",d0),("raw_sham_32d_dose",d1),
                ("shared_output_32d_dose",target),("source_gain",g0),
                ("sham_gain",g1)):
                near(actual.tolist(),dose[name],what="sham and cohesive source dose "+name)
            if matched.tolist()!=dose["matched_1pct"] or (target>1e-6).tolist()!=dose["nonzero_dose"]:
                raise AssertionError("physical matched 32D source/sham availability changed")
            good &= matched & (target>1e-6)
            availability[budget]&=good
            variants[f"coherent{k}_source"][visit]=full[visit]-matched_source
            variants[f"coherent{k}_sham"][visit]=full[visit]-matched_sham
            if budget=="four":
                hash_source,hash_sham,h0,h1,hmin,hg0,hg1,hmatched=independent_pair(src,rnd,down)
                hashdose=summary["hash_physical_dose"]
                for name,actual in (
                    ("raw_source_32d_dose",h0),("raw_sham_32d_dose",h1),
                    ("shared_output_32d_dose",hmin),("source_gain",hg0),
                    ("sham_gain",hg1)):
                    near(actual.tolist(),hashdose[name],what="hash sham dose "+name)
                if (hmatched.tolist()!=hashdose["matched_1pct"]
                    or (hmin>1e-6).tolist()!=hashdose["nonzero_dose"]):
                    raise AssertionError("hash quartet sham physical dose changed")
                rand_good &= hmatched & (hmin>1e-6)
                hash_avail &= rand_good
                variants["hash4_source"][visit]=full[visit]-hash_source
                variants["hash4_sham"][visit]=full[visit]-hash_sham
    if (set(record["valid_group_source_sham_on_all_visits"])!=set(GROUPS)
        or any(availability[key].tolist()!=record["valid_group_source_sham_on_all_visits"][key]
               for key in GROUPS)
        or hash_avail.tolist()!=record["valid_hash_on_all_visits"]):
        raise AssertionError("registered source group entire-trajectory availability changed")
    for visit in sites:
        for budget,k in GROUPS.items():
            for side in ("source","sham"):
                name=f"coherent{k}_{side}"
                variants[name][visit]=full[visit]+(
                    variants[name][visit]-full[visit])*availability[budget][:,None]
        for side in ("source","sham"):
            name=f"hash4_{side}"
            variants[name][visit]=full[visit]+(
                variants[name][visit]-full[visit])*hash_avail[:,None]
    traces={"clean":clean,"hubs":hubs_trace}
    for name,vecs in variants.items():
        donors={v:{"native":value} for v,value in vecs.items()}
        with torch.no_grad():
            traces[name]=traced_forward(model,x,mode="native",
                orientation=direction,donors=donors,gates=gate)
    ground_truth=labels(x,prepared["world"],control=(arm=="additive")).tolist()
    if ground_truth!=record["original_source_true_labels"]:
        raise AssertionError("original source Boolean teacher label changed")
    correct={}
    signed_margin={}
    for name in MODES:
        near(traces[name]["logits"].tolist(),record["logits"][name],
             atol=3e-4,rtol=1e-5,what="full recurrent source grouped causal logits")
        preds=traces[name]["logits"].argmax(1).tolist()
        if preds!=record["source_predictions"][name]:
            raise AssertionError("replayed source coherent group predictions changed")
        correct[name]=sum(int(a==b) for a,b in zip(preds,ground_truth))
        signed_margin[name]=[
            float(logits[y]-logits[1-y])
            for logits,y in zip(traces[name]["logits"],ground_truth)]
        drift=(traces[name]["memory"]-base["memory"]).norm(dim=-1).mean(dim=0)
        near(drift.tolist(),record["memory_drift_from_full_affine_by_visit"][name],
             what="group nonlinear recurrent drift")
    for budget,k in GROUPS.items():
        invalid=~availability[budget]
        if not torch.allclose(traces[f"coherent{k}_source"]["logits"][invalid],
                              traces[f"coherent{k}_sham"]["logits"][invalid],atol=1e-6):
            raise AssertionError("control failure leaked into correct-count contrast")
    if not torch.allclose(traces["hash4_source"]["logits"][~hash_avail],
                          traces["hash4_sham"]["logits"][~hash_avail],atol=1e-6):
        raise AssertionError("hash source-control failure leaked")
    return correct,{k:sum(availability[k].tolist()) for k in GROUPS},sum(hash_avail.tolist()),signed_margin


def audit(index,source,out,*,reset=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("NEC03V independent audit original eight model roots only")
    report_path=out/"nec03v_source_group_root_report.json"
    report=read(report_path)
    world=worlds()[index]
    if (report["schema"]!=SCHEMA or report["world_index"]!=index
        or report["world_id"]!=world["world_id"]
        or report["world_sha256"]!=world["world_sha256"]
        or report["source_lock_sha256"]!=lock_sha
        or report["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or report["four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or report["source_manifest_sha256"]!=context_catalog(world)[1]["source_sha256"]
        or report["source_groups_scored"]!=list(PANEL)
        or report["source_cases_per_model_arm"]!=CASES
        or report["paired_qualified_in_original_source_roster"]!=(index in QUALIFIED)
        or report["original_ITT_source_failure"]!=(index in ITT_ZERO)
        or report["primary_comparison_family_size"]!=1
        or report["minimum_meaningful_primary"]!=.02
        or any(report[k]!=0 for k in (
            "new_native_or_SAE_training_updates","new_development_eval_groups_scored",
            "reserved_groups_scored","new_recruited_worlds_scored",
            "five_input_embargoed_worlds_scored","old_four_input_confirmation_worlds_scored"))
        or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or set(report["arms"])!={"composed","additive"}):
        raise AssertionError("source registry, parent model ancestry or original site experiment changed")
    arms={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        prepared["arm"]=arm
        original=report["arms"][arm]
        path=out/original["raw_filename"]
        if (original["source_model_sha256"]!=prepared["model_sha256"]
            or original["source_sae_checkpoint_sha256"]!=prepared["sae_sha256"]
            or original["source_sae_rank_sha256"]!=prepared["rank_sha256"]
            or original["source_parent_receipt_sha256"]!=prepared["source_report_sha256"]
            or original["original_source_candidate_present"]!=bool(prepared["source_candidate"])
            or original["original_arm_source_eligible"]!=prepared["original_arm_source_eligible"]
            or original["source_groups_scored"]!=list(PANEL)
            or original["cases_per_arm"]!=CASES
            or sha_file(path)!=original["raw_sha256"]):
            raise AssertionError("frozen native/SAE code group ancestry or source raw receipt changed")
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        if len(rows)!=len(PANEL):
            raise AssertionError("one original source group missing")
        total={name:0 for name in MODES}
        valid={name:0 for name in GROUPS}
        hash_valid=0
        m={name:[] for name in MODES}
        for j,row in enumerate(rows):
            if row["source_group"]!=PANEL[j] or row["arm"]!=arm:
                raise AssertionError("source case group moved after scorer")
            counted,available,h_available,margins=replay_group(prepared,row,arm)
            for name in MODES:
                total[name]+=counted[name]
                m[name].extend(margins[name])
            for group in GROUPS:valid[group]+=available[group]
            hash_valid+=h_available
        scale=max(1.,median(abs(v) for v in m["full_affine"]))
        delta=[(b-a)/scale for a,b in zip(m["coherent4_source"],m["coherent4_sham"])]
        norm_mean=float(sum(delta)/len(delta))
        necessity=total["coherent4_sham"]-total["coherent4_source"]
        wide=total["coherent6_sham"]-total["coherent6_source"]
        if (original["correct_counts"]!=total
            or original["matched_group_valid_cases"]!=valid
            or original["hash_valid_cases"]!=hash_valid
            or original["four_source_ranked_necessity_correct_cases"]!=necessity
            or original["six_source_ranked_necessity_correct_cases"]!=wide):
            raise AssertionError("new independent source coherent-group necessity counts differ")
        near(scale,original["within_arm_full_affine_median_margin_scale"],
             what="within-arm frozen full-affine normalized logit scale")
        near(norm_mean,original["secondary_normalized_source_vs_sham_margin_mean"],
             what="secondary source-group normalized teacher margin")
        if original["secondary_normalized_margin_not_confirmatory"] is not True:
            raise AssertionError("posthoc secondary margin was promoted")
        arms[arm]={
            "correct_counts":total,"valid_physical_null_cases":valid,
            "hash_valid_cases":hash_valid,
            "four_necessity_correct_case_difference":necessity,
            "six_necessity_correct_case_difference":wide,
            "full_affine_baseline_margin_scale":scale,
            "secondary_normalized_sham_minus_source_margin_mean":norm_mean,
            "model_sha256":prepared["model_sha256"],
            "SAE_sha256":prepared["sae_sha256"],
            "source_rank_sha256":prepared["rank_sha256"],
            "raw_sha256":sha_file(path),
            "original_source_eligible":prepared["original_arm_source_eligible"],
            "source_candidate_present":bool(prepared["source_candidate"]),
        }
    did=(arms["composed"]["four_necessity_correct_case_difference"]-
         arms["additive"]["four_necessity_correct_case_difference"])/CASES
    if (abs(did-report["raw_root_primary_coherent4_necessity_DID"])>1e-12
        or abs((did if index in QUALIFIED else 0.)-report["registered_ITT_primary_root_DID"])>1e-12):
        raise AssertionError("root-level original cohort ITT computation changed")
    result={
        "schema":AUDIT_SCHEMA,
        "world_index":index,"world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "source_lock_sha256":lock_sha,
        "original_root_report_sha256":sha_file(report_path),
        "all_original_source_inputs_and_Boolean_labels_independently_replayed":True,
        "frozen_model_and_SAE_sha256_verified":True,
        "all_coherent_group_source_ids_and_disjoint_shams_reconstructed":True,
        "all_original_sham_corners_and_physical_32d_output_doses_recomputed":True,
        "all_eleven_full_recurrent_neural_outcomes_independently_replayed":True,
        "all_source_case_dropout_and_hash_controls_verified":True,
        "both_original_teacher_arms_replayed":True,
        "original_source_eligible_world":index in QUALIFIED,
        "primary_root_ITT_DID":did if index in QUALIFIED else 0.,
        "raw_primary_root_DID":did,
        "arms":arms,
        "source_training_cases_per_arm":CASES,
        "development_eval_groups_scored":0,
        "reserved_groups_scored":0,"new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"independent_nec03v_source_group_audit.json").write_bytes(encoded(result))
    print("NEC03V_INDEPENDENT_PHYSICAL_COHERENT_GROUP_SOURCE_AUDIT_PASS",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
