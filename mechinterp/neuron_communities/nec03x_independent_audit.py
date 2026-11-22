"""NEC-03X independent checkpoint-based continuous null and recurrent replay.

Deliberately reimplements the 495-support 32-step nonnegative box
solver and physical quality search without importing/calling the scorer's
select_continuous_sham or its dose helper. All selected support IDs,
four optimized coefficients, box constraints, output-32D null quality,
native/SAE original hashes, full logit trajectories and teacher labels
are reconstructed from original source-only archives.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03x_source import (
    ROOTS,PANEL,ELIGIBLE,ITT_ZERO,FOCAL,CASES,MODES,SCHEMA,read,verify_lock,
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

AUDIT_SCHEMA="NEC-03X/independent-box-solver-and-continuous-physical-sham-replay/v1"


def near(a,b,*,atol=2e-4,rtol=1e-5,what="original native physical"):
    x=torch.as_tensor(a,dtype=torch.float64)
    y=torch.as_tensor(b,dtype=torch.float64)
    if x.shape!=y.shape or not torch.allclose(x,y,atol=atol,rtol=rtol):
        magnitude=float((x-y).abs().max()) if x.shape==y.shape else -1.
        raise AssertionError(f"independent NEC03X {what} error {magnitude}")


def cos(a,b):
    return (a*b).sum(dim=-1) / (a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-10)


def replay_optimal_sham(active,source_ids,code,phys,physical_cosines,grad):
    """Separate deterministic 495-support implementation, one restart, 32 steps."""
    supports=list(combinations(sorted(set(active)-set(source_ids)),4))
    if len(supports)!=495:raise AssertionError("original support universe changed")
    ix=torch.as_tensor(supports,dtype=torch.long)
    b=phys[ix]
    start=code[ix]
    limit=3.*start
    c=start.clone()
    h=torch.bmm(b,b.transpose(1,2))
    y=(b*sum((code[f]*phys[f] for f in source_ids))[None,None,:]).sum(-1)
    lipschitz=b.square().sum((1,2)).clamp_min(1e-10)
    for iteration in range(32):
        gradient=torch.bmm(h,c.unsqueeze(-1)).squeeze(-1)-y
        c=torch.minimum((c-gradient/lipschitz[:,None]).clamp_min(0),limit)
    src32=sum((code[f]*phys[f] for f in source_ids))
    control32=(b*c[:,:,None]).sum(1)
    gradient_angle=(cos(control32,grad)-cos(src32,grad)).abs()
    src_coh=sum(physical_cosines[source_ids[i],source_ids[j]]
                for i in range(4) for j in range(i+1,4))/6
    coh=torch.zeros(495,dtype=phys.dtype)
    for i in range(4):
        for j in range(i+1,4):
            coh+=physical_cosines[ix[:,i],ix[:,j]]
    coh=coh/6
    cgap=(coh-src_coh).abs()
    cross=(physical_cosines[torch.as_tensor(source_ids,dtype=torch.long)[:,None,None],
                            ix[None,:,:]]).abs().mean(dim=(0,2))
    samevec=cos(control32,src32)
    score=gradient_angle+.25*cgap+.25*(1-samevec)
    feasible=((gradient_angle<=.25)&(cgap<=.75)&(cross>=.12)&
              (samevec>=.25)&(control32.norm(dim=1)>1e-6)&
              (src32.norm()>1e-6)&(grad.norm()>1e-7))
    allowed=torch.where(feasible,score,torch.full_like(score,float("inf")))
    idx=int(torch.argmin(allowed)) if bool(feasible.any()) else int(torch.argmin(score))
    return list(supports[idx]),c[idx],{
        "source_ids":list(source_ids),
        "selected_sham_ids":list(supports[idx]),
        "optimized_coefficients":c[idx].tolist(),
        "original_coefficients":start[idx].tolist(),
        "coefficient_upper_bounds":limit[idx].tolist(),
        "angle_gap":float(gradient_angle[idx]),
        "cohesion_gap":float(cgap[idx]),
        "cross_group_absolute_gram_overlap":float(cross[idx]),
        "physical_source_control_output_cosine":float(samevec[idx]),
        "source_physical_cohesion":float(src_coh),
        "sham_physical_cohesion":float(coh[idx]),
        "joint_score":float(score[idx]),
        "source_physical_output_norm":float(src32.norm()),
        "control_physical_output_norm":float(control32[idx].norm()),
        "qualified_candidate_count":int(feasible.sum()),
        "selected_candidate_index":idx,
        "candidate_count":495,
        "selected_candidate_quality_valid":bool(feasible[idx]),
        "solver_iterations":32,"solver_restarts":1,
        "solver_max_relative_coefficient_multiplier":3.,
        "solver_optimal_objective":float(score[idx]),
    }


def independent_equal_dose(s,t,down):
    a=(s@down.T).norm(dim=1)
    b=(t@down.T).norm(dim=1)
    d=torch.minimum(a,b)
    ga=torch.where(a>1e-10,d/a.clamp_min(1e-10),torch.zeros_like(a)).clamp(0,1)
    gb=torch.where(b>1e-10,d/b.clamp_min(1e-10),torch.zeros_like(b)).clamp(0,1)
    ss=s*ga[:,None]
    tt=t*gb[:,None]
    condition=((ss@down.T).norm(dim=1)-(tt@down.T).norm(dim=1)).abs()/d.clamp_min(1e-6)<=.01
    return {"source":ss,"sham":tt},{
        "raw_source_output_L2":a.tolist(),
        "raw_sham_output_L2":b.tolist(),
        "shared_output_L2":d.tolist(),
        "source_gain":ga.tolist(),
        "sham_gain":gb.tolist(),
        "dose_match_within_1pct":condition.tolist(),
        "positive_matched_dose":(d>1e-6).tolist(),
    }


def check_dose(a,b):
    if set(a)!=set(b):raise AssertionError("dose receipts keys changed")
    for key in a:
        near(a[key],b[key],atol=1e-4,what="matched physical dose "+key)


def replay_group(prepared,rec,arm):
    group=rec["group"]
    if (rec["schema"]!=SCHEMA or group not in PANEL
        or rec["arm"]!=arm or rec["focal_masks"]!=list(range(32))
        or rec["teacher_labels_used_for_sham_selection"] is not False):
        raise AssertionError("NEC03X registered source window or blinded sham changed")
    original=prepared["catalog"]["train"][group*32:(group+1)*32]
    if [tuple(a) for a in rec["source_input_histories"]]!=original:
        raise AssertionError("original 32-mask source-training context moved")
    model=prepared["model"]
    x=as_tensor(original)
    clean=traced_forward(model,x,mode="clean")
    orient=clean["logits"].argmax(1)
    if orient.tolist()!=rec["clean_orientation"]:
        raise AssertionError("source candidate clean predicted orientation differs")
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    if set(rec["site_geometry"])!={str(v) for v in sites}:
        raise AssertionError("model's original selected recurrent graph sites changed")
    gates={}
    for visit in sites:
        gates[visit]=torch.zeros(64)
        if visit in hubs:gates[visit][sorted(hubs[visit])]=1.
    lesion=traced_forward(model,x,mode="hubs",orientation=orient,
            donors={v:{"hubs":torch.zeros(32,64)} for v in sites},gates=gates)
    acts=clean_activations(model,x) if sites else None
    variants={m:{} for m in MODES if m not in ("clean","hubs")}
    full={}
    code={}
    cols={}
    down=model.block.mlp_down.weight.detach()
    for visit in sorted(sites):
        state=acts[:,visit,:]
        nonhub=1.-gates[visit]
        sae=prepared["sae"]
        with torch.no_grad():
            z=sae.codes((state-prepared["mean"])/prepared["scale"])
            dec=sae.decoder.weight.detach().T*prepared["scale"][None,:]
            aff=sae.decoder(z)*prepared["scale"]+prepared["mean"]
            f=z@dec
            o=prepared["mean"]+sae.decoder.bias.detach()*prepared["scale"]
        if not torch.allclose(aff,f+o,atol=2e-5,rtol=1e-5):
            raise AssertionError("independent source SAE reconstruction changed")
        full[visit]=aff*nonhub
        code[visit]=z
        cols[visit]=dec
        variants["native_nonhub"][visit]=state*nonhub
        variants["full_affine"][visit]=full[visit]
        variants["contrib16_only"][visit]=f*nonhub
    if sites:
        base=traced_forward(model,x,mode="native",orientation=orient,
            donors={v:{"native":a} for v,a in full.items()},gates=gates,derivative=True)
    else:base=clean
    all_cont=torch.full((32,),bool(sites),dtype=torch.bool)
    all_discrete=torch.full((32,),bool(sites),dtype=torch.bool)
    for visit,nodes in sorted(sites.items()):
        saved=rec["site_geometry"][str(visit)]
        if (saved["native_selected_units"]!=sorted(nodes)
            or saved["graph_hubs"]!=sorted(hubs.get(visit,[]))
            or len(saved["case_geometry"])!=32):
            raise AssertionError("frozen physical native graph site changed")
        nonhub=1.-gates[visit]
        phys=(cols[visit]*nonhub[None,:])@down.T
        norm=phys.norm(dim=1)
        dot=phys@phys.T/(norm[:,None]*norm[None,:]).clamp_min(1e-10)
        z=code[visit]
        rank={i:p for p,i in enumerate(prepared["ranks"][visit])}
        grad=base["grad_down"][visit]
        near((full[visit]@down.T).norm(dim=1).tolist(),
             saved["full_affine_32d_output_dose"],what="restored 32D output dose")
        near((((full[visit]-variants["native_nonhub"][visit])@down.T).norm(dim=1)/
             (variants["native_nonhub"][visit]@down.T).norm(dim=1).clamp_min(1e-6)).tolist(),
             saved["full_affine_32d_reconstruction_error"],what="full SAE affine error")
        src=torch.zeros(32,64)
        sham=torch.zeros(32,64)
        dsrc=torch.zeros(32,64)
        dsham=torch.zeros(32,64)
        for mask in range(32):
            item=saved["case_geometry"][mask]
            active=torch.where(z[mask]>1e-7)[0].tolist()
            if (item["active_ids"]!=active or
                item["full_topk_support"]!=(len(active)==16)):
                raise AssertionError("source positive TopK16 support changed")
            if len(active)!=16:
                if item["source_ids"] or item["continuous"] is not None or item["discrete"] is not None:
                    raise AssertionError("source lacking positive active quartet invented a sham")
                all_cont[mask]=False;all_discrete[mask]=False
                continue
            seed=min(active,key=lambda f:(rank[f],f))
            neighbors=sorted((f for f in active if f!=seed),
                key=lambda f:(-float(dot[seed,f]),rank[f],f))
            source_ids=[seed]+neighbors[:3]
            if item["source_ids"]!=source_ids:
                raise AssertionError("source quartet physical coherent definition altered")
            chosen,weights,geometry=replay_optimal_sham(
                active,source_ids,z[mask],phys,dot,grad[mask])
            old=item["continuous"]
            if set(old)!=set(geometry):
                raise AssertionError("continuous sham solver receipt fields changed")
            for key,value in geometry.items():
                if isinstance(value,list):
                    if key in ("selected_sham_ids","source_ids"):
                        if value!=old[key]:raise AssertionError("continuous disjoint support changed")
                    else:near(value,old[key],atol=5e-5,what="continuous "+key)
                elif isinstance(value,bool) or isinstance(value,int):
                    if value!=old[key]:raise AssertionError("continuous shim quality flag changed "+key)
                else:near(value,old[key],atol=5e-5,what="continuous "+key)
            original_src=(z[mask,source_ids]@cols[visit][source_ids])*nonhub
            original_sham=(weights@cols[visit][chosen])*nonhub
            src[mask]=original_src;sham[mask]=original_sham
            if not geometry["selected_candidate_quality_valid"]:
                all_cont[mask]=False
            # Independently reproduce NEC03V's discrete physical null.
            options=list(combinations(sorted(set(active)-set(source_ids)),4))
            idx=torch.as_tensor(options,dtype=torch.long)
            controls=(z[mask,idx].unsqueeze(-1)*phys[idx]).sum(dim=1)
            gap=(cos(controls,grad[mask])-cos(original_src@down.T,grad[mask])).abs()
            original_cohesion=sum(dot[source_ids[a],source_ids[b]]
                                  for a in range(4) for b in range(a+1,4))/6
            decoh=torch.zeros(495)
            for a in range(4):
                for b in range(a+1,4):
                    decoh+=dot[idx[:,a],idx[:,b]]
            decoh/=6
            cohgap=(decoh-original_cohesion).abs()
            scores=gap+.5*cohgap
            select=int(torch.argmin(scores))
            selected=list(options[select])
            discrete={
                "sham_ids":selected,
                "angle_gap":float(gap[select]),
                "cohesion_gap":float(cohgap[select]),
                "joint_score":float(scores[select]),
                "original_candidates":495,
                "valid_physical_match":bool(
                    float(gap[select])<=.25 and float(cohgap[select])<=.50
                    and grad[mask].norm()>1e-7),
            }
            for key,val in discrete.items():
                oldvalue=item["discrete"][key]
                if isinstance(val,(int,bool,list)):
                    if oldvalue!=val:raise AssertionError("secondary discrete source sham changed")
                else:near(val,oldvalue,atol=6e-5,what="secondary discrete null")
            dsrc[mask]=original_src
            dsham[mask]=(z[mask,selected]@cols[visit][selected])*nonhub
            if not discrete["valid_physical_match"]:
                all_discrete[mask]=False
        cont,dose=independent_equal_dose(src,sham,down)
        discrete,disc_dose=independent_equal_dose(dsrc,dsham,down)
        check_dose(dose,saved["continuous_pair_dose"])
        check_dose(disc_dose,saved["discrete_pair_dose"])
        for mask in range(32):
            all_cont[mask]&=bool(dose["dose_match_within_1pct"][mask]
                                 and dose["positive_matched_dose"][mask])
            all_discrete[mask]&=bool(disc_dose["dose_match_within_1pct"][mask]
                                     and disc_dose["positive_matched_dose"][mask])
        variants["continuous4_source"][visit]=full[visit]-cont["source"]
        variants["continuous4_sham"][visit]=full[visit]-cont["sham"]
        variants["discrete4_source"][visit]=full[visit]-discrete["source"]
        variants["discrete4_sham"][visit]=full[visit]-discrete["sham"]
    if (all_cont.tolist()!=rec["continuous_available_all_visits"] or
        all_discrete.tolist()!=rec["discrete_available_all_visits"]):
        raise AssertionError("continuous or discrete invalid-case dropout changed")
    for visit in sites:
        for gate,prefix in ((all_cont,"continuous4"),(all_discrete,"discrete4")):
            for side in ("source","sham"):
                name=f"{prefix}_{side}"
                variants[name][visit]=full[visit]+(
                    variants[name][visit]-full[visit])*gate[:,None]
    traces={"clean":clean,"hubs":lesion}
    for name,vs in variants.items():
        donors={visit:{"native":vec} for visit,vec in vs.items()}
        with torch.no_grad():
            traces[name]=traced_forward(model,x,mode="native",
                orientation=orient,donors=donors,gates=gates)
    y=labels(x,prepared["world"],control=arm=="additive").tolist()
    if y!=rec["original_source_teacher_labels"]:
        raise AssertionError("full original Boolean teacher labels differ")
    correct={}
    margins={}
    for name in MODES:
        near(traces[name]["logits"].tolist(),rec["logits"][name],atol=3e-4,
             what="12-visit native recurrent output logits "+name)
        pred=traces[name]["logits"].argmax(1).tolist()
        if pred!=rec["original_predictions"][name]:
            raise AssertionError("frozen continuous source scored class changed")
        correct[name]=sum(int(a==b) for a,b in zip(pred,y))
        if name in ("full_affine","continuous4_source","continuous4_sham"):
            margins[name]=[float(z0[y0]-z0[1-y0])
                           for z0,y0 in zip(traces[name]["logits"],y)]
        drift=(traces[name]["memory"]-base["memory"]).norm(dim=-1).mean(0)
        near(drift.tolist(),rec["mean_recurrent_memory_drift_from_full_affine"][name],
             what="recurrent memory drift")
    for gate,family in ((all_cont,"continuous4"),(all_discrete,"discrete4")):
        if not torch.allclose(traces[f"{family}_source"]["logits"][~gate],
                              traces[f"{family}_sham"]["logits"][~gate],atol=1e-6):
            raise AssertionError("source/sham invalid case emitted nonzero difference")
    return correct,int(all_cont.sum()),int(all_discrete.sum()),margins


def audit(index,source,out,*,reset=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_sha=verify_lock()
    if index not in ROOTS:raise PermissionError("NEC03X eight original source roots only")
    report_path=out/"nec03x_source_root_report.json"
    report=read(report_path)
    world=worlds()[index]
    if (report["schema"]!=SCHEMA or report["world_index"]!=index
        or report["world_id"]!=world["world_id"]
        or report["world_sha256"]!=world["world_sha256"]
        or report["source_manifest_sha256"]!=context_catalog(world)[1]["source_sha256"]
        or report["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or report["four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or report["source_lock_sha256"]!=lock_sha
        or report["source_training_groups_scored"]!=list(PANEL)
        or report["source_cases_per_model_arm"]!=CASES
        or report["original_paired_source_eligible"]!=(index in ELIGIBLE)
        or report["original_ITT_zero"]!=(index in ITT_ZERO)
        or report["model_or_SAE_optimizer_updates"]!=0
        or any(report[k]!=0 for k in (
            "new_development_eval_groups_scored","reserved_groups_scored",
            "new_recruited_worlds_scored","five_input_embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"))
        or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or set(report["arms"])!={"composed","additive"}):
        raise AssertionError("NEC03X registered native source root or result malformed")
    arms={}
    for arm in ("composed","additive"):
        frozen=load_source_arm(index,arm,source,reset=reset)
        frozen["arm"]=arm
        original=report["arms"][arm]
        path=out/original["raw_file"]
        if (original["source_model_sha256"]!=frozen["model_sha256"]
            or original["source_sae_sha256"]!=frozen["sae_sha256"]
            or original["source_rank_sha256"]!=frozen["rank_sha256"]
            or original["original_source_receipt_sha256"]!=frozen["source_report_sha256"]
            or original["original_source_eligible"]!=frozen["original_arm_source_eligible"]
            or original["source_candidate_present"]!=bool(frozen["source_candidate"])
            or original["source_group_indices"]!=list(PANEL)
            or original["source_cases"]!=CASES
            or original["raw_sha256"]!=sha_file(path)):
            raise AssertionError("native/SAE frozen source checkpoint or raw JSONL changed")
        recs=[json.loads(line) for line in path.read_text().splitlines()]
        if len(recs)!=len(PANEL):
            raise AssertionError("source root has incomplete original groups")
        correct={m:0 for m in MODES}
        vcont=vdisc=0
        margins={m:[] for m in ("full_affine","continuous4_source","continuous4_sham")}
        for j,row in enumerate(recs):
            if row["group"]!=PANEL[j] or row["arm"]!=arm:
                raise AssertionError("original training source group changed")
            c,vc,vd,ms=replay_group(frozen,row,arm)
            for m in MODES:correct[m]+=c[m]
            vcont+=vc;vdisc+=vd
            for m in margins:margins[m].extend(ms[m])
        scale=max(1.,median(abs(v) for v in margins["full_affine"]))
        normalized=sum((b-a)/scale for a,b in zip(
            margins["continuous4_source"],margins["continuous4_sham"]))/CASES
        if (correct!=original["correct_counts"]
            or vcont!=original["matched_continuous_inputs"]
            or vdisc!=original["matched_discrete_inputs"]
            or correct["continuous4_sham"]-correct["continuous4_source"]!=
               original["continuous_sham_minus_source_correct"]
            or correct["discrete4_sham"]-correct["discrete4_source"]!=
               original["discrete_sham_minus_source_correct"]):
            raise AssertionError("new continuous physical sham source/root correctness changed")
        near(scale,original["secondary_within_model_margin_scale"],what="median native margin scale")
        near(normalized,original["secondary_normalized_continuous_sham_minus_source_margin"],
             what="normalized within-arm secondary margin")
        arms[arm]={
            "correct_counts":correct,
            "matched_continuous_inputs":vcont,
            "matched_discrete_inputs":vdisc,
            "continuous_correct_case_difference":
                correct["continuous4_sham"]-correct["continuous4_source"],
            "discrete_correct_case_difference":
                correct["discrete4_sham"]-correct["discrete4_source"],
            "source_model_sha256":frozen["model_sha256"],
            "SAE_checkpoint_sha256":frozen["sae_sha256"],
            "source_rank_sha256":frozen["rank_sha256"],
            "source_candidate_present":bool(frozen["source_candidate"]),
            "original_source_eligible":frozen["original_arm_source_eligible"],
            "margin_scale":scale,
            "secondary_normalized_source_sham_margin":float(normalized),
            "raw_sha256":sha_file(path),
        }
    did=(arms["composed"]["continuous_correct_case_difference"]-
         arms["additive"]["continuous_correct_case_difference"])/CASES
    if (abs(did-report["raw_continuous_primary_root_DID"])>1e-12
        or abs((did if index in ELIGIBLE else 0.)-
               report["primary_original_ITT_root_DID"])>1e-12):
        raise AssertionError("new preregistered eight-root primary effect altered")
    result={
        "schema":AUDIT_SCHEMA,
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "original_root_report_sha256":sha_file(report_path),
        "source_lock_sha256":lock_sha,
        "frozen_original_native_and_SAE_checkpoint_ancestry_verified":True,
        "all_495_disjoint_continuous_supports_and_bounded_32_step_solver_replayed":True,
        "all_source_coherent_quartets_and_control_quality_gates_recomputed":True,
        "original_secondary_discrete_null_independently_replayed":True,
        "original_32d_output_L2_doses_and_nonamplifying_gains_verified":True,
        "all_original_256_training_source_labels_per_teacher_recomputed":True,
        "all_nine_12_visit_neural_recurrent_modes_independently_replayed":True,
        "source_root_original_paired_eligible":index in ELIGIBLE,
        "primary_root_ITT_DID":did if index in ELIGIBLE else 0.,
        "source_cases_per_arm":CASES,"arms":arms,
        "new_development_eval_groups_scored":0,"reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,"five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"independent_nec03x_continuous_source_audit.json").write_bytes(encoded(result))
    print("NEC03X_INDEPENDENT_ORIGINAL_CONTINUOUS_SHAM_ROOT_PASS",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
