"""NEC-03T independent source-only full-affine SAE necessity replay.

Independently rebuild physical full SAE background and all decoder-code
removals from frozen checkpoints. Exhaustively check disjoint active
shams, min-L2 equal-dose ablations, complete 12-visit recurrent logits,
and all original 256 source truth labels per model arm and world.
Never call experiment group_measure or shared_removal.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03t_source_necessity import (
    SCHEMA, ROOTS, PANEL, SOURCE_ELIGIBLE, ITT_ZERO, CASES, FOCAL,
    BUDGETS, MODES, HASH_SALT, ANGLE_MAX, read, verify_lock,
)
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, registration, context_catalog, labels, encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from research.pef07.pef07 import as_tensor, model_digest

AUDIT_SCHEMA = "NEC-03T/independent-source-affine-necessity-replay/v1"


def near(x, y, *, atol=1e-4, rtol=1e-5, label="physical source quantity"):
    first=torch.tensor(x,dtype=torch.float64)
    second=torch.tensor(y,dtype=torch.float64)
    if first.shape!=second.shape or not torch.allclose(first,second,atol=atol,rtol=rtol):
        difference=float((first-second).abs().max()) if first.shape==second.shape else -1
        raise AssertionError(f"NEC03T independent {label} error {difference}")


def physical_cos(v, gradient):
    return (v*gradient).sum(dim=-1) / (
        v.norm(dim=-1)*gradient.norm(dim=-1)).clamp_min(1e-10)


def independently_match_removals(src, sham, down):
    """Source and sham share exact attainable min local output dose."""
    norms={"source":(src@down.T).norm(dim=1),
           "sham":(sham@down.T).norm(dim=1)}
    floor=torch.minimum(norms["source"],norms["sham"])
    gain={key:torch.where(
        v>1e-10,floor/v.clamp_min(1e-10),torch.zeros_like(floor)).clamp(0,1)
        for key,v in norms.items()}
    scaled={"source":src*gain["source"][:,None],
            "sham":sham*gain["sham"][:,None]}
    comparable=torch.ones(len(src),dtype=torch.bool)
    for key,delta in scaled.items():
        actual=(delta@down.T).norm(dim=1)
        comparable &= (actual-floor).abs()/floor.clamp_min(1e-6) <= .01
        if (gain[key]>1+1e-6).any():
            raise AssertionError("independent source/sham attenuation amplified")
    return scaled,norms,floor,gain,comparable


def replay_group(frozen, original_record, arm):
    idx=original_record["source_group"]
    if (original_record["schema"]!=SCHEMA or idx not in PANEL
        or original_record["arm"]!=arm
        or original_record["focal_mask_order"]!=list(range(FOCAL))
        or original_record["labels_used_for_routing"] is not False):
        raise AssertionError("original source-training cohort or label-blind selection changed")
    world=frozen["world"]
    source_input=frozen["catalog"]["train"][idx*FOCAL:(idx+1)*FOCAL]
    if [tuple(v) for v in original_record["input_histories"]]!=source_input:
        raise AssertionError("original complete source context group changed")
    model=frozen["model"]
    x=as_tensor(source_input)
    clean=traced_forward(model,x,mode="clean")
    orientation=clean["logits"].argmax(dim=1)
    if orientation.tolist()!=original_record["clean_chosen_orientation"]:
        raise AssertionError("original clean predicted orientation differs")
    sites=frozen["source_sites"]
    hubs=frozen["source_hubs"]
    if set(original_record["site_geometry"])!={str(v) for v in sites}:
        raise AssertionError("original source-selected recurrent visit set changed")
    gates={}
    hub_inputs={}
    for visit in sites:
        gate=torch.zeros(64)
        if visit in hubs:
            gate[sorted(hubs[visit])]=1.
        gates[visit]=gate
        hub_inputs[visit]={"hubs":torch.zeros(FOCAL,64)}
    hubs_trace=traced_forward(
        model,x,mode="hubs",orientation=orientation,donors=hub_inputs,gates=gates)
    acts=clean_activations(model,x) if sites else None
    variants={name:{} for name in MODES if name not in ("clean","hubs")}
    background_donors={}
    full={}
    code={}
    physical_columns={}
    down=model.block.mlp_down.weight.detach()
    for visit in sorted(sites):
        state=acts[:,visit,:]
        nonhub=1.-gates[visit]
        sae=frozen["sae"]
        with torch.no_grad():
            active=sae.codes((state-frozen["mean"])/frozen["scale"])
            matrix=sae.decoder.weight.detach().T*frozen["scale"][None,:]
            affine=sae.decoder(active)*frozen["scale"]+frozen["mean"]
            contributions=active@matrix
            offset=frozen["mean"]+sae.decoder.bias.detach()*frozen["scale"]
            if not torch.allclose(affine,contributions+offset,atol=2e-5,rtol=1e-5):
                raise AssertionError("full SAE reconstruction affine offset changed")
        full[visit]=affine*nonhub
        code[visit]=active
        physical_columns[visit]=matrix
        background_donors[visit]={"native":full[visit]}
        variants["native_nonhub"][visit]=state*nonhub
        variants["full_affine"][visit]=full[visit]
        variants["contrib16_only"][visit]=contributions*nonhub
    if sites:
        reference=traced_forward(
            model,x,mode="native",orientation=orientation,
            donors=background_donors,gates=gates,derivative=True)
    else:
        reference=clean
    valid={name:torch.full((FOCAL,),bool(sites),dtype=torch.bool)
           for name in BUDGETS}
    hash_valid=torch.full((FOCAL,),bool(sites),dtype=torch.bool)
    for visit,units in sorted(sites.items()):
        logged=original_record["site_geometry"][str(visit)]
        if (logged["native_source_units"]!=sorted(units)
            or logged["original_graph_hubs"]!=sorted(hubs.get(visit,[]))):
            raise AssertionError("frozen source community nodes/hubs changed")
        grad=reference["grad_down"][visit]
        z=code[visit]
        cols=physical_columns[visit]
        nonhub=1.-gates[visit]
        projected_cols=(cols*nonhub[None,:])@down.T
        near((full[visit]@down.T).norm(dim=1).tolist(),
             logged["full_affine_physical_dose"],
             label="full SAE original physical-dose")
        near((
            ((full[visit]-variants["native_nonhub"][visit])@down.T).norm(dim=1) /
            (variants["native_nonhub"][visit]@down.T).norm(dim=1).clamp_min(1e-6)
        ).tolist(),logged["full_affine_relative_32d_error"],
             label="full SAE native 32D reconstruction")
        order={i:p for p,i in enumerate(frozen["ranks"][visit])}
        for budget in BUDGETS:
            k=len(units) if budget=="orig" else min(2*len(units),8)
            route=logged["budget"][budget]
            if k<1 or k>8 or route["k"]!=k:
                raise AssertionError("unregistered source-selected code removal budget")
            raw_source=torch.zeros(FOCAL,64)
            raw_sham=torch.zeros(FOCAL,64)
            raw_random=torch.zeros(FOCAL,64)
            okay=torch.ones(FOCAL,dtype=torch.bool)
            random_ok=torch.ones(FOCAL,dtype=torch.bool)
            for mask in range(FOCAL):
                active=torch.where(z[mask]>1e-7)[0].tolist()
                if route["active_ids"][mask]!=active:
                    raise AssertionError("TopK positive source code identities changed")
                enough=len(active)==16
                if route["exact_positive16_codes"][mask]!=enough:
                    raise AssertionError("TopK16 source code support gate changed")
                if not enough:
                    if (route["source_ids"][mask] or route["angular_sham_ids"][mask]
                        or route["hash_sham_ids"][mask]
                        or route["candidate_count"][mask]!=0
                        or route["angle_gap"][mask] is not None):
                        raise AssertionError("unsupported source ablation attempted")
                    okay[mask]=False
                    random_ok[mask]=False
                    continue
                selected=sorted(active,key=lambda j:order[j])[:k]
                choices=sorted(set(active)-set(selected))
                options=list(combinations(choices,k))
                indices=torch.tensor(options,dtype=torch.long)
                if (route["source_ids"][mask]!=selected
                    or route["candidate_count"][mask]!=len(options)):
                    raise AssertionError("source ranks or exhaustive sham enumeration differed")
                native_src=(z[mask,selected]@cols[selected])*nonhub
                # Independently reconstruct candidates in the native 32D
                # output coordinate, and verify reported optimum.
                physical_candidates=(
                    z[mask,indices].unsqueeze(-1)*projected_cols[indices]).sum(dim=1)
                target_src=native_src@down.T
                angles=(physical_cos(physical_candidates,grad[mask]) -
                        physical_cos(target_src,grad[mask])).abs()
                best=int(torch.argmin(angles))
                sham_ids=route["angular_sham_ids"][mask]
                if tuple(sham_ids) not in options or abs(
                    float(angles[options.index(tuple(sham_ids))] -
                          angles[best]))>5e-5:
                    raise AssertionError("source-ablation sham not exhaustive cosine optimum")
                actual_sham=(z[mask,sham_ids]@cols[sham_ids])*nonhub
                near(float(angles[options.index(tuple(sham_ids))]),
                     route["angle_gap"][mask],atol=6e-5,
                     label="full affine gradient cosine gap")
                raw_source[mask]=native_src
                raw_sham[mask]=actual_sham
                if budget=="orig":
                    key=(f"{world['world_sha256']}|{arm}|{idx}|{mask}|{visit}|orig").encode()
                    position=int.from_bytes(sha256(HASH_SALT+b"|"+key).digest(),"big")%len(options)
                    random_ids=list(options[position])
                    if route["hash_sham_ids"][mask]!=random_ids:
                        raise AssertionError("hash-selected control reindexed")
                    raw_random[mask]=(z[mask,random_ids]@cols[random_ids])*nonhub
                elif route["hash_sham_ids"][mask]:
                    raise AssertionError("unregistered hash control at widened budget")
                if (float(angles[options.index(tuple(sham_ids))])>ANGLE_MAX
                    or float(grad[mask].norm())<=1e-7):
                    okay[mask]=False
            paired,norms,dose,gain,match=independently_match_removals(
                raw_source,raw_sham,down)
            dr=route["angular_removal_dose"]
            for key in ("source","sham"):
                near(norms[key].tolist(),dr["raw_dose"][key],
                     label="source/sham raw ablation dose")
                near(gain[key].tolist(),dr["gains"][key],
                     label="source/sham attenuation")
            near(dose.tolist(),dr["shared_dose"],
                 label="shared equal-dose ablation")
            if (match.tolist()!=dr["physical_dose_equal_within_1pct"]
                or (dose>1e-6).tolist()!=dr["nonzero_shared_dose"]):
                raise AssertionError("invalid physical 32D dose matching")
            okay &= match & (dose>1e-6)
            valid[budget] &= okay
            variants[f"ablate_{budget}_source"][visit]=full[visit]-paired["source"]
            variants[f"ablate_{budget}_sham"][visit]=full[visit]-paired["sham"]
            if budget=="orig":
                random_pair,random_norms,random_shared,random_gain,random_match=(
                    independently_match_removals(raw_source,raw_random,down))
                rh=route["hash_removal_dose"]
                for key in ("source","sham"):
                    near(random_norms[key].tolist(),rh["raw_dose"][key],
                         label="source/hash raw removal")
                    near(random_gain[key].tolist(),rh["gains"][key],
                         label="source/hash removal gain")
                near(random_shared.tolist(),rh["shared_dose"],
                     label="source/hash same 32D dose")
                if (random_match.tolist()!=rh["physical_dose_equal_within_1pct"]
                    or (random_shared>1e-6).tolist()!=rh["nonzero_shared_dose"]):
                    raise AssertionError("source/hash physical-dose status differs")
                random_ok &= random_match & (random_shared>1e-6)
                hash_valid &= random_ok
                variants["ablate_hash_source"][visit]=full[visit]-random_pair["source"]
                variants["ablate_hash_sham"][visit]=full[visit]-random_pair["sham"]
    if (set(original_record["valid_angular_all_visits"])!=set(BUDGETS)
        or any(valid[b].tolist()!=original_record["valid_angular_all_visits"][b]
               for b in BUDGETS)
        or hash_valid.tolist()!=original_record["valid_hash_all_visits"]):
        raise AssertionError("missing-code or invalid-case original source site mask changed")
    for visit in sites:
        for budget in BUDGETS:
            for side in ("source","sham"):
                mode=f"ablate_{budget}_{side}"
                variants[mode][visit]=full[visit]+(
                    variants[mode][visit]-full[visit])*valid[budget][:,None]
        for side in ("source","sham"):
            mode=f"ablate_hash_{side}"
            variants[mode][visit]=full[visit]+(
                variants[mode][visit]-full[visit])*hash_valid[:,None]
    reconstructed={"clean":clean,"hubs":hubs_trace}
    for mode,donor_by_site in variants.items():
        donors={v:{"native":d} for v,d in donor_by_site.items()}
        with torch.no_grad():
            reconstructed[mode]=traced_forward(
                model,x,mode="native",orientation=orientation,
                donors=donors,gates=gates)
    true=labels(x,world,control=(arm=="additive")).tolist()
    if true!=original_record["true_teacher_labels"]:
        raise AssertionError("true 5-bit Boolean source labels changed")
    counts={}
    for mode in MODES:
        near(reconstructed[mode]["logits"].tolist(),
             original_record["logits"][mode],atol=3e-4,rtol=1e-5,
             label="full recurrent source necessity logits")
        pred=reconstructed[mode]["logits"].argmax(dim=1).tolist()
        if pred!=original_record["class_predictions"][mode]:
            raise AssertionError("source-native ablation class prediction mismatched")
        counts[mode]=sum(int(a==b) for a,b in zip(pred,true))
        drift=(reconstructed[mode]["memory"]-reference["memory"]).norm(dim=-1).mean(dim=0)
        near(drift.tolist(),
             original_record["mean_memory_distance_from_full_affine_by_visit"][mode],
             label="recurrent memory deviation")
    for budget in BUDGETS:
        bad=~valid[budget]
        if not torch.allclose(
            reconstructed[f"ablate_{budget}_source"]["logits"][bad],
            reconstructed[f"ablate_{budget}_sham"]["logits"][bad],atol=1e-6):
            raise AssertionError("unavailable source/sham ablation was nonzero")
    if not torch.allclose(
        reconstructed["ablate_hash_source"]["logits"][~hash_valid],
        reconstructed["ablate_hash_sham"]["logits"][~hash_valid],atol=1e-6):
        raise AssertionError("unavailable source/hash ablation was nonzero")
    return counts,{b:sum(valid[b].tolist()) for b in BUDGETS},sum(hash_valid.tolist())


def audit(index:int,source:Path,output:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("only eight preadmitted source roots may be audited")
    world=worlds()[index]
    report_path=output/"nec03t_source_necessity_report.json"
    report=read(report_path)
    if (report["schema"]!=SCHEMA or report["world_id"]!=world["world_id"]
        or report["world_index"]!=index or report["world_sha256"]!=world["world_sha256"]
        or report["source_lock_sha256"]!=lock_sha
        or report["source_groups_scored"]!=list(PANEL)
        or report["source_training_cases_per_arm"]!=CASES
        or report["original_paired_source_eligible"]!=(index in SOURCE_ELIGIBLE)
        or report["original_ITT_source_failure"]!=(index in ITT_ZERO)
        or report["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or report["old_four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or report["source_manifest_sha256"]!=context_catalog(world)[1]["source_sha256"]
        or any(report[field]!=0 for field in (
            "new_native_or_SAE_training_updates",
            "new_development_eval_groups_scored","reserved_groups_scored",
            "newly_recruited_worlds_scored","embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"))
        or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or set(report["arms"])!={"composed","additive"}):
        raise AssertionError("source-only NEC03T experimental identity changed")
    arms={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        prepared["arm"]=arm
        row=report["arms"][arm]
        original_path=output/row["raw_file"]
        if (row["source_model_sha256"]!=prepared["model_sha256"]
            or row["source_sae_checkpoint_sha256"]!=prepared["sae_sha256"]
            or row["source_sae_rank_sha256"]!=prepared["rank_sha256"]
            or row["source_receipt_sha256"]!=prepared["source_report_sha256"]
            or row["original_source_candidate_present"]!=bool(prepared["source_candidate"])
            or row["original_arm_source_eligible"]!=prepared["original_arm_source_eligible"]
            or row["source_histories"]!=CASES
            or row["source_groups_scored"]!=list(PANEL)
            or sha_file(original_path)!=row["raw_sha256"]):
            raise AssertionError("source checkpoint or raw JSONL changed")
        records=[json.loads(line) for line in original_path.read_text().splitlines()]
        if len(records)!=len(PANEL):
            raise AssertionError("source root missing group record")
        totals={mode:0 for mode in MODES}
        valid={budget:0 for budget in BUDGETS}
        valid_hash=0
        for j,rec in enumerate(records):
            if rec["source_group"]!=PANEL[j] or rec["arm"]!=arm:
                raise AssertionError("source context group omitted or reordered")
            counts,v,h=replay_group(prepared,rec,arm)
            for mode in MODES:
                totals[mode]+=counts[mode]
            for budget in BUDGETS:
                valid[budget]+=v[budget]
            valid_hash+=h
        necessities={budget:totals[f"ablate_{budget}_sham"]-
                             totals[f"ablate_{budget}_source"]
                     for budget in BUDGETS}
        if (row["correct"]!=totals or
            row["valid_angular_case_count"]!=valid or
            row["valid_hash_case_count"]!=valid_hash or
            row["source_ranked_necessity_correct_count"]!=necessities):
            raise AssertionError("source-only correctness or matched-dose counts not reproduced")
        arms[arm]={
            "correct":totals,
            "matched_angular_cases":valid,
            "matched_hash_cases":valid_hash,
            "necessity_correct_count":necessities,
            "source_model_sha256":prepared["model_sha256"],
            "source_sae_checkpoint_sha256":prepared["sae_sha256"],
            "original_source_candidate_present":bool(prepared["source_candidate"]),
            "original_arm_source_eligible":prepared["original_arm_source_eligible"],
            "raw_sha256":row["raw_sha256"],
        }
    did=(arms["composed"]["necessity_correct_count"]["orig"]-
         arms["additive"]["necessity_correct_count"]["orig"])/CASES
    if (abs(did-report["raw_primary_source_necessity_DID"])>1e-12 or
        abs((did if index in SOURCE_ELIGIBLE else 0.)-
            report["primary_ITT_root_effect"])>1e-12):
        raise AssertionError("original paired-source ITT necessity contrast changed")
    audited={
        "schema":AUDIT_SCHEMA,"world_id":world["world_id"],
        "world_index":index,"world_sha256":world["world_sha256"],
        "original_report_sha256":sha_file(report_path),
        "source_lock_sha256":lock_sha,
        "all_original_source_groups_and_true_labels_verified":True,
        "all_native_and_SAE_checkpoint_provenance_verified":True,
        "full_affine_source_background_independently_reconstructed":True,
        "all_disjoint_ranked_shams_and_hash_random_controls_recomputed":True,
        "all_matched_physical_ablation_doses_and_invalid_cases_replayed":True,
        "all_eleven_full_recurrent_outcomes_independently_recomputed":True,
        "paired_root_primary_ITT_effect":did if index in SOURCE_ELIGIBLE else 0.,
        "source_cases_per_arm":CASES,"arms":arms,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,"newly_recruited_worlds_scored":0,
        "embargoed_worlds_scored":0,"old_four_input_confirmation_worlds_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (output/"independent_nec03t_source_necessity_audit.json").write_bytes(encoded(audited))
    print("NEC03T_INDEPENDENT_FULL_AFFINE_NECESSITY_AUDIT_PASS",world["world_id"])
    return audited


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
