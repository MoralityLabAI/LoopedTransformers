"""NEC-03Z-CAL independently replay original-source dose calibration.

New cohort only original groups 48..55; old immutable native/SAE. Uses
the previously independent NEC03X control search reconstruction, and
newly reconstructs each dose-scaled native 64D donor, original 32D
matching and all 14 full 12-visit recurrent logit modes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import torch

from mechinterp.neuron_communities.nec03z_cal_source import (
    ROOTS,CAL,ELIGIBLE,ITT_ZERO,CASES,MASKS,TAGS,DOSES,
    MODES,SCHEMA,read,verify_lock,
)
from mechinterp.neuron_communities.nec03x_independent_audit import (
    replay_optimal_sham,independent_equal_dose,near,
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

AUDIT_SCHEMA="NEC-03Z/independent-archive-model-dose-calibration-replay/v1"


def check_geometry(prior,computed):
    if set(prior)!=set(computed):
        raise AssertionError("original physical sham solver receipt keys changed")
    for k,v in computed.items():
        if isinstance(v,list):
            if k in ("source_ids","selected_sham_ids"):
                if v!=prior[k]:raise AssertionError("previously independent sham feature ids differ")
            else:
                near(v,prior[k],atol=6e-5,what="continuous coefficient "+k)
        elif isinstance(v,(bool,int)):
            if v!=prior[k]:
                raise AssertionError("original continuous solver or null gate identity differs")
        else:
            near(v,prior[k],atol=6e-5,what="continuous physical "+k)


def validate_group(prepared,rec,arm):
    group=rec["source_group"]
    if (rec["schema"]!=SCHEMA or group not in CAL
        or rec["arm"]!=arm or rec["focal_mask_order"]!=list(range(MASKS))
        or rec["labels_used_for_sham_selection"] is not False):
        raise AssertionError("NEC03Z original calibration cohort or blind selection changed")
    inputs=prepared["catalog"]["train"][group*MASKS:(group+1)*MASKS]
    if [tuple(v) for v in rec["source_training_histories"]]!=inputs:
        raise AssertionError("registered calibration complete source group changed")
    model=prepared["model"]
    x=as_tensor(inputs)
    clean=traced_forward(model,x,mode="clean")
    orient=clean["logits"].argmax(1)
    if orient.tolist()!=rec["clean_chosen_orientation"]:
        raise AssertionError("native original clean orientation altered")
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    if set(rec["site_geometry"])!={str(v) for v in sites}:
        raise AssertionError("original historical native recurrent source sites altered")
    gates={}
    for visit in sites:
        gates[visit]=torch.zeros(64)
        if visit in hubs:gates[visit][sorted(hubs[visit])]=1.
    lesion=traced_forward(
        model,x,mode="hubs",orientation=orient,
        donors={v:{"hubs":torch.zeros(MASKS,64)} for v in sites},gates=gates
    )
    source_acts=clean_activations(model,x) if sites else None
    full={}
    codes={}
    cols={}
    down=model.block.mlp_down.weight.detach()
    variants={m:{} for m in MODES if m not in ("clean","hubs")}
    for visit in sorted(sites):
        native=source_acts[:,visit,:]
        nonhub=1.-gates[visit]
        sae=prepared["sae"]
        with torch.no_grad():
            z=sae.codes((native-prepared["mean"])/prepared["scale"])
            dec=sae.decoder.weight.detach().T*prepared["scale"][None,:]
            aff=sae.decoder(z)*prepared["scale"]+prepared["mean"]
            offset=prepared["mean"]+sae.decoder.bias.detach()*prepared["scale"]
        if not torch.allclose(aff,z@dec+offset,atol=2e-5,rtol=1e-5):
            raise AssertionError("original native SAE decoder affine identity changed")
        full[visit]=aff*nonhub
        codes[visit]=z
        cols[visit]=dec
        variants["native_nonhub"][visit]=native*nonhub
        variants["full_affine"][visit]=full[visit]
    reference=(
        traced_forward(
            model,x,mode="native",orientation=orient,
            donors={v:{"native":a} for v,a in full.items()},
            gates=gates,derivative=True
        ) if sites else clean
    )
    valid_base=torch.full((MASKS,),bool(sites),dtype=torch.bool)
    minimum=torch.full((MASKS,),float("inf"))
    for visit,nodes in sorted(sites.items()):
        record=rec["site_geometry"][str(visit)]
        if (record["native_selected_units"]!=sorted(nodes)
            or record["original_graph_hubs"]!=sorted(hubs.get(visit,[]))
            or len(record["case_meta"])!=MASKS):
            raise AssertionError("original native graph or complete physical site metadata differs")
        nonhub=1.-gates[visit]
        z=codes[visit]
        dec=cols[visit]
        phys=(dec*nonhub[None,:])@down.T
        lengths=phys.norm(dim=1)
        cosine=phys@phys.T/(lengths[:,None]*lengths[None,:]).clamp_min(1e-10)
        rank={f:i for i,f in enumerate(prepared["ranks"][visit])}
        src=torch.zeros(MASKS,64)
        sham=torch.zeros(MASKS,64)
        for mask in range(MASKS):
            item=record["case_meta"][mask]
            active=torch.where(z[mask]>1e-7)[0].tolist()
            if item["active_ids"]!=active:
                raise AssertionError("original positive active SAE code list changed")
            if len(active)!=16:
                if item["source_ids"] or item["sham"] is not None:
                    raise AssertionError("missing source codes yielded a false sham")
                valid_base[mask]=False
                continue
            seed=min(active,key=lambda f:(rank[f],f))
            near_neighbors=sorted((f for f in active if f!=seed),
                 key=lambda f:(-float(cosine[seed,f]),rank[f],f))
            source_ids=[seed]+near_neighbors[:3]
            if source_ids!=item["source_ids"]:
                raise AssertionError("registered old coherent source quartet differs")
            support,weights,geometry=replay_optimal_sham(
                active,source_ids,z[mask],phys,cosine,
                reference["grad_down"][visit][mask]
            )
            check_geometry(item["sham"],geometry)
            src[mask]=(z[mask,source_ids]@dec[source_ids])*nonhub
            sham[mask]=(weights@dec[support])*nonhub
            if not geometry["selected_candidate_quality_valid"]:
                valid_base[mask]=False
        paired,dose=independent_equal_dose(src,sham,down)
        old=record["physical_pair_dose"]
        if set(dose)!=set(old):
            raise AssertionError("original physical 32D dose field roster changed")
        for k in dose:
            near(dose[k],old[k],atol=8e-5,what="full-source continuous removal "+k)
        near((full[visit]@down.T).norm(dim=1).tolist(),
             record["full_affine_physical_output_L2"],what="native full-affine output dose")
        v=torch.tensor(dose["shared_output_L2"])/torch.tensor(
            dose["raw_source_output_L2"]).clamp_min(1e-10)
        minimum=torch.minimum(minimum,v)
        valid_base &= torch.tensor([
            dose["positive_matched_dose"][m] and dose["dose_match_within_1pct"][m]
            for m in range(MASKS)],dtype=torch.bool
        )
        for tag,fraction in zip(TAGS,DOSES):
            variants[f"source_d{tag}"][visit]=full[visit]-fraction*paired["source"]
            variants[f"sham_d{tag}"][visit]=full[visit]-fraction*paired["sham"]
    available={}
    relative={}
    for tag,fraction in zip(TAGS,DOSES):
        v=valid_base & (minimum*fraction>1e-6)
        available[tag]=v.tolist()
        relative[tag]=torch.where(v,minimum*fraction,
                                  torch.zeros_like(minimum)).tolist()
        if available[tag]!=rec["available_per_dose_all_visits"][tag]:
            raise AssertionError("registered source/dose availability changed")
        near(relative[tag],rec["min_site_relative_physical_dose_per_case"][tag],
             atol=6e-5,what="whole-trajectory physical removal dose fraction")
        for visit in sites:
            for side in ("source","sham"):
                name=f"{side}_d{tag}"
                variants[name][visit]=full[visit]+(
                    variants[name][visit]-full[visit])*v[:,None]
    traces={"clean":clean,"hubs":lesion}
    for name,donors_by_site in variants.items():
        donors={v:{"native":a} for v,a in donors_by_site.items()}
        with torch.no_grad():
            traces[name]=traced_forward(model,x,mode="native",
                orientation=orient,donors=donors,gates=gates)
    truth=labels(x,prepared["world"],control=(arm=="additive")).tolist()
    if truth!=rec["teacher_source_truth_labels"]:
        raise AssertionError("original Boolean teacher source labels changed")
    predictions={}
    for name in MODES:
        near(traces[name]["logits"].tolist(),rec["logits"][name],
             atol=3e-4,rtol=1e-5,what="fully recurrent original source logits")
        predictions[name]=traces[name]["logits"].argmax(1).tolist()
        if predictions[name]!=rec["class_predictions"][name]:
            raise AssertionError("original source training prediction changed")
    for tag in TAGS:
        bad=~torch.tensor(available[tag])
        if not torch.allclose(
            traces[f"source_d{tag}"]["logits"][bad],
            traces[f"sham_d{tag}"]["logits"][bad],atol=1e-6):
            raise AssertionError("original calibration invalid case contributed causal sham effect")
    return truth,predictions,relative,available


def audit(index,source,out,*,reset=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("only eight original source model worlds")
    world=worlds()[index]
    path=out/"nec03z_cal_source_root_report.json"
    report=read(path)
    if (report["schema"]!=SCHEMA
        or report["world_index"]!=index or report["world_id"]!=world["world_id"]
        or report["world_sha256"]!=world["world_sha256"]
        or report["source_registry_sha256"]!=registration()["registry_sha256"]
        or report["old_four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or report["source_manifest_sha256"]!=context_catalog(world)[1]["source_sha256"]
        or report["source_lock_sha256"]!=lock_sha
        or report["calibration_source_groups_scored"]!=list(CAL)
        or report["sealed_analysis_source_groups_scored"]!=[]
        or report["source_cases_per_native_teacher"]!=CASES
        or report["original_paired_source_eligible"]!=(index in ELIGIBLE)
        or report["original_ITT_zero"]!=(index in ITT_ZERO)
        or report["new_native_SAE_training_updates"]!=0
        or any(report[x]!=0 for x in (
            "development_eval_scored","reserved_scored","recruited_worlds_scored",
            "embargoed_worlds_scored","old_four_input_confirmation_scored"))
        or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or set(report["arms"])!={"composed","additive"}):
        raise AssertionError("NEC03Z source-only calibration original root identity differed")
    arms={}
    for arm in ("composed","additive"):
        frozen=load_source_arm(index,arm,source,reset=reset)
        frozen["arm"]=arm
        old=report["arms"][arm]
        rawpath=out/old["raw_file"]
        if (old["model_sha256"]!=frozen["model_sha256"]
            or old["sae_sha256"]!=frozen["sae_sha256"]
            or old["source_rank_sha256"]!=frozen["rank_sha256"]
            or old["old_source_receipt_sha256"]!=frozen["source_report_sha256"]
            or old["original_source_candidate_present"]!=bool(frozen["source_candidate"])
            or old["original_arm_source_eligible"]!=frozen["original_arm_source_eligible"]
            or old["source_groups"]!=list(CAL)
            or old["source_cases"]!=CASES
            or sha_file(rawpath)!=old["raw_sha256"]):
            raise AssertionError("original model/SAE source ancestry or receipt bytes altered")
        raws=[json.loads(x) for x in rawpath.read_text().splitlines()]
        if len(raws)!=8:
            raise AssertionError("source-group calibration audit has missing group")
        correct={m:0 for m in MODES}
        available={t:0 for t in TAGS}
        basecorrect={t:0 for t in TAGS}
        c2w={t:0 for t in TAGS}
        w2c={t:0 for t in TAGS}
        s2w={t:0 for t in TAGS}
        relative={t:[] for t in TAGS}
        for i,rec in enumerate(raws):
            if rec["source_group"]!=CAL[i] or rec["arm"]!=arm:
                raise AssertionError("calibration source group order changed")
            truth,preds,rel,valid=validate_group(frozen,rec,arm)
            for name in MODES:
                correct[name]+=sum(int(a==b) for a,b in zip(preds[name],truth))
            for t in TAGS:
                for case in range(MASKS):
                    if not valid[t][case]:continue
                    available[t]+=1
                    base=preds["full_affine"][case]==truth[case]
                    ctrl=preds[f"sham_d{t}"][case]==truth[case]
                    src=preds[f"source_d{t}"][case]==truth[case]
                    basecorrect[t]+=int(base)
                    c2w[t]+=int(base and not ctrl)
                    w2c[t]+=int(not base and ctrl)
                    s2w[t]+=int(base and not src)
                    relative[t].append(rel[t][case])
        summaries={
            t:{
                "dose_fraction":DOSES[TAGS.index(t)],
                "available_original_source_cases":available[t],
                "baseline_correct_matched_cases":basecorrect[t],
                "sham_correct_to_wrong":c2w[t],
                "sham_wrong_to_correct":w2c[t],
                "source_correct_to_wrong":s2w[t],
                "sham_inertness_damage_fraction":(
                    c2w[t]/basecorrect[t] if basecorrect[t] else None
                ),
                "median_min_site_relative_dose":(
                    float(median(relative[t])) if relative[t] else None
                ),
            } for t in TAGS
        }
        if correct!=old["correct"] or summaries!=old["dose_calibration"]:
            raise AssertionError("source calibration sham damage and physical dose could not be replayed")
        if model_digest(frozen["model"])!=frozen["model_sha256"]:
            raise AssertionError("independent dose calibration altered native model")
        arms[arm]={
            "correct":correct,"dose_calibration":summaries,
            "model_sha256":frozen["model_sha256"],
            "sae_sha256":frozen["sae_sha256"],
            "source_rank_sha256":frozen["rank_sha256"],
            "raw_sha256":sha_file(rawpath),
            "original_source_eligible":frozen["original_arm_source_eligible"],
        }
    result={
        "schema":AUDIT_SCHEMA,"world_index":index,
        "world_id":world["world_id"],"world_sha256":world["world_sha256"],
        "source_lock_sha256":lock_sha,
        "original_root_report_sha256":sha_file(path),
        "all_original_source_calibration_truths_independently_recomputed":True,
        "all_five_dose_fractions_and_original_solver_routes_independently_reconstructed":True,
        "all_original_native_SAE_source_ancestry_verified":True,
        "all_fourteen_complete_recurrent_modes_independently_replayed":True,
        "whole_trajectory_invalid_sham_source_inputs_return_to_full_affine":True,
        "all_original_source_correctness_and_sham_damage_counts_verified":True,
        "source_cases_per_arm":CASES,
        "original_qualified_source_ITT":index in ELIGIBLE,
        "arms":arms,
        "sealed_source_analysis_groups_scored":0,
        "new_development_eval_scored":0,"reserved_scored":0,
        "recruited_worlds_scored":0,"embargoed_worlds_scored":0,
        "old_four_input_confirmation_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"independent_nec03z_cal_source_audit.json").write_bytes(encoded(result))
    print("NEC03Z_INDEPENDENT_SOURCE_SHAM_DOSE_CAL_AUDIT_PASS",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
