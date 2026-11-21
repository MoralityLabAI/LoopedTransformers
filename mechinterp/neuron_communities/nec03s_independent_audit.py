"""NEC-03S independent source-only exhaustive budget/offset audit.

Uses the immutable original reset1800 checkpoint, source TopK16 SAE
and source ranks. Independently enumerates all same-budget disjoint
shams, verifies the FOUR-WAY common-dose anchor across offset/no-offset
donors, hash-random nulls, all 26 recurrent logits and source truth
labels. Never calls the experiment's group_measure or dose helpers.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03s_factorial import (
    BUDGETS,OFFSETS,PANEL,CASES,FOCAL,MODES,ROOTS,PAIRED_ELIGIBLE,
    ITT_ZERO,HASH_SALT,EFFECT_FLOOR,ANGLE_LIMIT,SCHEMA,
    load,verify_lock,
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

AUDIT_SCHEMA="NEC-03S/independent-source-offset-budget-audit/v1"


def near(x,y,*,atol=1e-4,rtol=1e-5,description="measurement"):
    a=torch.tensor(x,dtype=torch.float64)
    b=torch.tensor(y,dtype=torch.float64)
    if a.shape!=b.shape or not torch.allclose(a,b,atol=atol,rtol=rtol):
        m=float((a-b).abs().max()) if a.shape==b.shape else -1.
        raise AssertionError(f"NEC03S independently reconstructed {description} differs (max {m})")


def directed_cos(a,g):
    return (a*g).sum(dim=-1)/(a.norm(dim=-1)*g.norm(dim=-1)).clamp_min(1e-10)


def replay_group(prepared:dict,record:dict,arm:str):
    group=record["original_source_group_index"]
    if (record["schema"]!=SCHEMA
        or record["arm"]!=arm
        or group not in PANEL
        or record["focal_mask_order"]!=list(range(FOCAL))
        or record["teacher_truth_labels_used_for_donor_routing"] is not False):
        raise AssertionError("source-only factorial group or label-blind control altered")
    original=prepared["catalog"]["train"][group*FOCAL:(group+1)*FOCAL]
    if [tuple(x) for x in record["input_histories"]]!=original:
        raise AssertionError("original source contexts or 32 focal masks modified")
    model=prepared["model"]
    x=as_tensor(original)
    baseline=traced_forward(model,x,mode="clean")
    direction=baseline["logits"].argmax(dim=1)
    if direction.tolist()!=record["clean_chosen_orientation"]:
        raise AssertionError("source route used different clean margin orientation")
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    if set(record["source_site_geometry"])!={str(v) for v in sites}:
        raise AssertionError("native source site membership changed")
    gates={}
    hub_donors={}
    for visit in sites:
        gate=torch.zeros(64)
        if visit in hubs:
            gate[sorted(hubs[visit])]=1.
        gates[visit]=gate
        hub_donors[visit]={"hubs":torch.zeros(FOCAL,64)}
    hubs_trace=traced_forward(model,x,mode="hubs",
             orientation=direction,donors=hub_donors,gates=gates,derivative=bool(sites))
    acts=clean_activations(model,x) if sites else None
    variants={name:{} for name in MODES if name not in ("clean","hubs")}
    eligible={b:torch.full((FOCAL,),bool(sites),dtype=torch.bool) for b in BUDGETS}
    random_ok={b:torch.full((FOCAL,),bool(sites),dtype=torch.bool) for b in BUDGETS}
    support_ok={b:torch.full((FOCAL,),bool(sites),dtype=torch.bool) for b in BUDGETS}
    down=model.block.mlp_down.weight.detach()
    for visit,units in sorted(sites.items()):
        record_site=record["source_site_geometry"][str(visit)]
        if (record_site["original_site_units"]!=sorted(units)
            or record_site["original_hubs"]!=sorted(hubs.get(visit,[]))):
            raise AssertionError("original native site/hub register changed")
        state=acts[:,visit,:]
        gate_nonhub=1.-gates[visit]
        ae=prepared["sae"]
        with torch.no_grad():
            code=ae.codes((state-prepared["mean"])/prepared["scale"])
            physical_cols=ae.decoder.weight.detach().T*prepared["scale"][None,:]
            feature=code@physical_cols
            offset=(prepared["mean"]+ae.decoder.bias.detach()*prepared["scale"])*gate_nonhub
            full_affine=ae.decoder(code)*prepared["scale"]+prepared["mean"]
        if not torch.allclose(full_affine,feature+(prepared["mean"]+
                  ae.decoder.bias.detach()*prepared["scale"]),atol=2e-5,rtol=1e-5):
            raise AssertionError("frozen SAE offset/decomposition does not reconstruct")
        near(offset.tolist(),record_site["offset_64d"],description="original offset")
        native=state*gate_nonhub
        zero=feature*gate_nonhub
        one=(feature+offset)*gate_nonhub
        variants["native_nonhub"][visit]=native
        variants["offset_only"][visit]=offset.unsqueeze(0).expand(FOCAL,-1)
        variants["full16_0"][visit]=zero
        variants["full16_1"][visit]=one
        near((native@down.T).norm(dim=1).tolist(),
             record_site["natural_nonhub_dose"]["native"],description="native dose")
        near((zero@down.T).norm(dim=1).tolist(),
             record_site["natural_nonhub_dose"]["full16_0"],description="full16 no-offset dose")
        near((one@down.T).norm(dim=1).tolist(),
             record_site["natural_nonhub_dose"]["full16_1"],description="full16 offset dose")
        near((offset@down.T).norm().item(),
             record_site["natural_nonhub_dose"]["offset_only"],description="offset dose")
        near(((one-native)@down.T).norm(dim=1).div(
             (native@down.T).norm(dim=1).clamp_min(1e-6)).tolist(),
             record_site["sae_full16_relative_32d_recon_error"],
             description="SAE full affine 32D residual")
        gradient=hubs_trace["grad_down"][visit]
        importance={int(f):i for i,f in enumerate(prepared["ranks"][visit])}
        for budget in BUDGETS:
            k=len(units) if budget=="orig" else min(2*len(units),8)
            if k>8 or k<1:
                raise AssertionError("unsupported original source active budget")
            logged=record_site["budget_geometry"][budget]
            if logged["k"]!=k:
                raise AssertionError("native source to TopK budget changed")
            source=torch.zeros(FOCAL,64)
            sham=torch.zeros(FOCAL,64)
            random=torch.zeros(FOCAL,64)
            this_valid=torch.ones(FOCAL,dtype=torch.bool)
            this_rnd_valid=torch.ones(FOCAL,dtype=torch.bool)
            this_supported=torch.ones(FOCAL,dtype=torch.bool)
            for mask in range(FOCAL):
                act=torch.where(code[mask]>1e-7)[0].tolist()
                if act!=logged["all_active_ids"][mask]:
                    raise AssertionError("active TopK feature IDs differ")
                ok=len(act)==16
                if logged["source_support"][mask]!=ok:
                    raise AssertionError("TopK16 active support gate differs")
                this_supported[mask]=ok
                if not ok:
                    if (logged["source_ids"][mask] or logged["sham_ids"][mask] or
                        logged["random_ids"][mask] or logged["candidate_count"][mask]!=0):
                        raise AssertionError("insufficient-code case invented controls")
                    for key in ("cosine_gap_without_offset","cosine_gap_with_offset","joint_minimax_gap"):
                        if logged[key][mask] is not None:
                            raise AssertionError("unsupported TopK code retained a cosine result")
                    this_valid[mask]=False
                    this_rnd_valid[mask]=False
                    continue
                ids=sorted(act,key=lambda j:importance[j])[:k]
                pool=sorted(set(act)-set(ids))
                options=list(combinations(pool,k))
                idx=torch.tensor(options,dtype=torch.long)
                if (ids!=logged["source_ids"][mask] or
                    len(options)!=logged["candidate_count"][mask]):
                    raise AssertionError("source-ranked IDs or disjoint search count changed")
                src=(code[mask,ids]@physical_cols[ids])*gate_nonhub
                alternatives=(code[mask,idx].unsqueeze(-1)*
                              physical_cols[idx]).sum(dim=1)*gate_nonhub
                p0=src@down.T
                p1=(src+offset)@down.T
                alt0=alternatives@down.T
                alt1=(alternatives+offset)@down.T
                grad=gradient[mask]
                gap0=(directed_cos(alt0,grad)-directed_cos(p0,grad)).abs()
                gap1=(directed_cos(alt1,grad)-directed_cos(p1,grad)).abs()
                minimax=torch.maximum(gap0,gap1)
                best=int(torch.argmin(minimax))
                selected=list(options[best])
                if selected!=logged["sham_ids"][mask]:
                    raise AssertionError("sham fails exhaustive minimax criterion")
                for key,val in (
                    ("cosine_gap_without_offset",gap0[best]),
                    ("cosine_gap_with_offset",gap1[best]),
                    ("joint_minimax_gap",minimax[best])):
                    near(float(val),logged[key][mask],atol=6e-5,
                         description="full disjoint cosine minimax")
                token=f"{prepared['world']['world_sha256']}|{arm}|{group}|{mask}|{visit}|{budget}".encode()
                h=int.from_bytes(sha256(HASH_SALT+b"|"+token).digest(),"big")
                rnd=list(options[h%len(options)])
                if rnd!=logged["random_ids"][mask]:
                    raise AssertionError("hash-selected active sham does not reproduce")
                source[mask]=src
                sham[mask]=code[mask,selected]@physical_cols[selected]*gate_nonhub
                random[mask]=code[mask,rnd]@physical_cols[rnd]*gate_nonhub
                if float(minimax[best])>ANGLE_LIMIT or float(grad.norm())<=1e-7:
                    this_valid[mask]=False
            four={
                "src0":source,"sham0":sham,
                "src1":source+offset,"sham1":sham+offset,
            }
            norms={key:(v@down.T).norm(dim=1) for key,v in four.items()}
            anchor=torch.stack(list(norms.values())).min(dim=0).values
            scaled={}
            gains={}
            dose_good=torch.ones(FOCAL,dtype=torch.bool)
            for name,vec in four.items():
                gains[name]=torch.where(
                    norms[name]>1e-10,anchor/norms[name].clamp_min(1e-10),
                    torch.zeros_like(anchor)).clamp(0,1)
                scaled[name]=vec*gains[name][:,None]
                observed=(scaled[name]@down.T).norm(dim=1)
                dose_good &= ((observed-anchor).abs()/anchor.clamp_min(1e-6))<=.01
                near(norms[name].tolist(),logged["primary_four_way_dose"]["norms"][name],
                     description="original candidate L2")
                near(gains[name].tolist(),logged["primary_four_way_dose"]["gains"][name],
                     description="common-anchor gain")
            near(anchor.tolist(),logged["primary_four_way_dose"]["shared"],
                 description="same dose across offset and no-offset conditions")
            if dose_good.tolist()!=logged["primary_four_way_dose"]["matched_within_1pct"]:
                raise AssertionError("four-way shared physical dose did not match")
            if (anchor>1e-6).tolist()!=logged["primary_four_way_dose"]["nonzero_shared"]:
                raise AssertionError("weak four-way physical dose gate changed")
            this_valid &= dose_good & (anchor>1e-6)
            eligible[budget]&=this_valid
            support_ok[budget]&=this_supported
            variants[f"natural_{budget}_0"][visit]=source
            variants[f"natural_{budget}_1"][visit]=source+offset
            for name,vec in scaled.items():
                variants[f"{budget}_{name}"][visit]=vec
            for flag in OFFSETS:
                pair=(source if flag==0 else source+offset)
                rnd=(random if flag==0 else random+offset)
                d0=(pair@down.T).norm(dim=1)
                d1=(rnd@down.T).norm(dim=1)
                shared=torch.minimum(d0,d1)
                a=torch.where(d0>1e-10,shared/d0.clamp_min(1e-10),
                              torch.zeros_like(shared)).clamp(0,1)
                b=torch.where(d1>1e-10,shared/d1.clamp_min(1e-10),
                              torch.zeros_like(shared)).clamp(0,1)
                receipt=logged["random_pair_dose"][str(flag)]
                for name,actual in (("source_norm",d0),("random_norm",d1),
                                    ("shared",shared),("source_gain",a),("random_gain",b)):
                    near(actual.tolist(),receipt[name],description="random matched dose")
                if (shared>1e-6).tolist()!=receipt["nonzero_shared"]:
                    raise AssertionError("random-sham positive physical dose gate changed")
                this_rnd_valid &= shared>1e-6
                variants[f"{budget}_random_src{flag}"][visit]=pair*a[:,None]
                variants[f"{budget}_random_sham{flag}"][visit]=rnd*b[:,None]
            random_ok[budget]&=this_rnd_valid
    if (set(record["valid_all_visits"])!=set(BUDGETS)
        or set(record["valid_random_all_visits"])!=set(BUDGETS)):
        raise AssertionError("registered original/expanded budget availability omitted")
    for visit in sites:
        for budget in BUDGETS:
            for flag in OFFSETS:
                variants[f"natural_{budget}_{flag}"][visit]*=support_ok[budget][:,None]
            for kind in ("src0","sham0","src1","sham1"):
                variants[f"{budget}_{kind}"][visit]*=eligible[budget][:,None]
            for flag in OFFSETS:
                variants[f"{budget}_random_src{flag}"][visit]*=random_ok[budget][:,None]
                variants[f"{budget}_random_sham{flag}"][visit]*=random_ok[budget][:,None]
    trace={"clean":baseline,"hubs":hubs_trace}
    for mode,vecs in variants.items():
        assignments={visit:{"native":vec} for visit,vec in vecs.items()}
        with torch.no_grad():
            trace[mode]=traced_forward(model,x,mode="native",
                   orientation=direction,donors=assignments,gates=gates)
    ground_truth=labels(x,prepared["world"],control=(arm=="additive")).tolist()
    if ground_truth!=record["true_teacher_labels"]:
        raise AssertionError("source Boolean teacher labels changed")
    totals={}
    for mode in MODES:
        near(trace[mode]["logits"].tolist(),record["logits"][mode],
             atol=3e-4,description="full recurrent source factorial logits")
        pred=trace[mode]["logits"].argmax(dim=1).tolist()
        if pred!=record["class_predictions"][mode]:
            raise AssertionError("source factorial correctness changed in independent replay")
        totals[mode]=sum(int(a==b) for a,b in zip(pred,ground_truth))
        drift=(trace[mode]["memory"]-hubs_trace["memory"]).norm(dim=-1).mean(dim=0)
        near(drift.tolist(),record["memory_drift_mean_from_hubs"][mode],
             description="tied recurrent memory drift")
    for budget in BUDGETS:
        if (eligible[budget].tolist()!=record["valid_all_visits"][budget]
            or random_ok[budget].tolist()!=record["valid_random_all_visits"][budget]):
            raise AssertionError("complete all-visit source/sham admissibility changed")
        bad=~eligible[budget]
        for flag in OFFSETS:
            a=trace[f"{budget}_src{flag}"]["logits"][bad]
            b=trace[f"{budget}_sham{flag}"]["logits"][bad]
            if not torch.allclose(a,b,atol=1e-6):
                raise AssertionError("source and sham gave nonzero effect on invalid cases")
    return totals,{budget:sum(eligible[budget].tolist()) for budget in BUDGETS}


def audit(index:int,source:Path,out:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("only eight original source root independent audits allowed")
    report_path=out/"nec03s_source_factorial_report.json"
    report=load(report_path)
    world=worlds()[index]
    if (report["schema"]!=SCHEMA or report["world_id"]!=world["world_id"]
        or report["world_index"]!=index or report["world_sha256"]!=world["world_sha256"]
        or report["source_lock_sha256"]!=sha
        or report["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or report["four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or report["original_source_manifest_sha256"]!=context_catalog(world)[1]["source_sha256"]
        or report["source_groups_scored"]!=list(PANEL)
        or report["training_source_case_count_per_arm"]!=CASES
        or report["native_and_sae_training_updates"]!=0
        or report["primary_familywise_comparisons"]!=1
        or report["minimum_meaningful_absolute_effect"]!=EFFECT_FLOOR
        or any(report[k]!=0 for k in (
            "development_eval_groups_scored","reserved_groups_scored",
            "new_source_recruited_worlds_scored","embargoed_worlds_scored",
            "original_four_input_confirmation_worlds_scored"))
        or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or set(report["arms"])!={"composed","additive"}):
        raise AssertionError("pre-registered original source-only cohort or contrast changed")
    verified={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        prepared["arm"]=arm
        scored=report["arms"][arm]
        raw=out/scored["raw_filename"]
        if (sha_file(raw)!=scored["raw_sha256"]
            or scored["model_weights_sha256"]!=prepared["model_sha256"]
            or scored["sae_checkpoint_sha256"]!=prepared["sae_sha256"]
            or scored["source_rank_sha256"]!=prepared["rank_sha256"]
            or scored["source_report_sha256"]!=prepared["source_report_sha256"]
            or scored["original_source_candidate_present"]!=bool(prepared["source_candidate"])
            or scored["original_arm_source_eligible"]!=prepared["original_arm_source_eligible"]
            or scored["source_training_histories"]!=CASES
            or scored["source_groups_scored"]!=list(PANEL)):
            raise AssertionError("source model, SAE or raw cohort checksums differ")
        items=[json.loads(line) for line in raw.read_text().splitlines()]
        if len(items)!=len(PANEL):
            raise AssertionError("incomplete source-only factorial records")
        counts={mode:0 for mode in MODES}
        valid={b:0 for b in BUDGETS}
        for i,rec in enumerate(items):
            if rec["original_source_group_index"]!=PANEL[i]:
                raise AssertionError("source training panel omitted or reordered")
            a,v=replay_group(prepared,rec,arm)
            for mode in MODES:counts[mode]+=a[mode]
            for b in BUDGETS:valid[b]+=v[b]
        def interaction(budget):
            return (counts[f"{budget}_src1"]-counts[f"{budget}_sham1"]) - (
                    counts[f"{budget}_src0"]-counts[f"{budget}_sham0"])
        if (counts!=scored["correctness"]
            or valid!=scored["available_common_dose_input_cases"]
            or any(interaction(b)!=scored["offset_interaction_correct_difference"][b]
                   for b in BUDGETS)
            or counts["clean"]-counts["hubs"]!=scored["signed_clean_hubs_gap"]):
            raise AssertionError("independent causal offset×budget counts or anchor differ")
        verified[arm]={
            "correctness":counts,"available_common_dose_input_cases":valid,
            "offset_interaction_correct_difference":{b:interaction(b) for b in BUDGETS},
            "model_weights_sha256":prepared["model_sha256"],
            "sae_sha256":prepared["sae_sha256"],
            "source_report_sha256":prepared["source_report_sha256"],
            "original_source_candidate_present":bool(prepared["source_candidate"]),
            "original_source_eligible":prepared["original_arm_source_eligible"],
            "raw_sha256":sha_file(raw),
        }
    raw_effect=(verified["composed"]["offset_interaction_correct_difference"]["orig"]-
                verified["additive"]["offset_interaction_correct_difference"]["orig"])/CASES
    eligible=index in PAIRED_ELIGIBLE
    if (abs(raw_effect-report["raw_primary_original_budget_interaction_DID"])>1e-12
        or abs((raw_effect if eligible else 0.)-report["primary_ITT_root_interaction_DID"])>1e-12
        or report["original_paired_source_eligible"]!=eligible
        or report["preregistered_source_ITT_failure"]!=(index in ITT_ZERO)):
        raise AssertionError("original n-code source only ITT interaction miscomputed")
    result={
        "schema":AUDIT_SCHEMA,
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "original_report_sha256":sha_file(report_path),
        "source_lock_sha256":sha,
        "all_original_source_groups_and_truth_labels_independently_replayed":True,
        "original_source_native_and_sae_checkpoints_immutable":True,
        "all_disjoint_minimax_angle_and_hash_random_shams_reenumerated":True,
        "all_four_offset_control_doses_identically_anchored_and_nonamplifying":True,
        "all_26_full_recurrent_logit_modes_independently_replayed":True,
        "all_16_original_model_arms_in_fixed_eight_root_denominator":True,
        "primary_original_budget_ITT_root_interaction":raw_effect if eligible else 0.,
        "raw_original_budget_interaction":raw_effect,
        "source_cases_per_arm":CASES,
        "arms":verified,
        "development_eval_groups_scored":0,"reserved_groups_scored":0,
        "new_source_recruited_worlds_scored":0,"embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"independent_nec03s_source_factorial_audit.json").write_bytes(encoded(result))
    print("NEC03S_INDEPENDENT_SOURCE_OFFSET_FACTORIAL_PASS",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
