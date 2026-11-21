"""NEC-03R independent, checkpoint-based source-only recoverability replay.

Reconstruct full SAE affine, sixteen-code contribution, ranked sparse,
native nonhub donors, exhaustive direction shams and hashed active shams
without calling the scorer's intervention or control-selection routine.
Reevaluate original source-label correctness on all 8x32 input histories.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03r_source_recoverability import (
    ROOTS,SOURCE_PANEL,CASES,MODELS,SCHEMA,ANGLE_LIMIT,RANDOM_SALT,
    read,verify_lock,load_source_arm,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.matched_dose import by_visit
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor,model_digest

AUDIT_SCHEMA="NEC-03R/independent-source-recoverability-replay/v1"


def compare(a,b,*,atol=2e-4,rtol=1e-5):
    x=torch.tensor(a,dtype=torch.float64)
    y=torch.tensor(b,dtype=torch.float64)
    if x.shape!=y.shape or not torch.allclose(x,y,atol=atol,rtol=rtol):
        m=float((x-y).abs().max()) if x.shape==y.shape else -1
        raise AssertionError(f"physical geometry or replay disagreement: {m}")


def cos32(v,g):
    return float(((v*g).sum()/(v.norm()*g.norm()).clamp_min(1e-10)).detach())


def attenuation_independent(a,b,down):
    """Original physical shared minimum norm, independent of scorer helper."""
    first=(a@down.T).norm(dim=1)
    other=(b@down.T).norm(dim=1)
    shared=torch.minimum(first,other)
    # Original helper uses EPS/clamp for near-zero attenuated donors.
    alpha=torch.where(first>1e-10,shared/first.clamp_min(1e-10),
                      torch.zeros_like(first)).clamp(0,1)
    beta=torch.where(other>1e-10,shared/other.clamp_min(1e-10),
                     torch.zeros_like(other)).clamp(0,1)
    return a*alpha[:,None],b*beta[:,None],shared,alpha,beta


def audit_group(prepared:dict,record:dict,arm:str):
    world=prepared["world"]
    group=record["source_group"]
    if (record["arm"]!=arm or group not in SOURCE_PANEL
        or record["focal_masks"]!=list(range(32))
        or record["true_teacher_labels_used_in_routing"] is not False):
        raise AssertionError("source-only group or label-blind selector changed")
    bits=prepared["catalog"]["train"][group*32:(group+1)*32]
    if [tuple(z) for z in record["original_source_input_histories"]]!=bits:
        raise AssertionError("source group mutated or partial focal block")
    model=prepared["model"]
    x=as_tensor(bits)
    clean=traced_forward(model,x,mode="clean")
    orientation=clean["logits"].argmax(1)
    if record["clean_chosen_class"]!=orientation.tolist():
        raise AssertionError("clean class orientation was not original source model output")
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    if set(map(int,record["site_geometry"]))!=set(sites):
        raise AssertionError("site selector changed from original source roster")
    gate={}
    donors={}
    for visit in sites:
        m=torch.zeros(64)
        if visit in hubs:m[sorted(hubs[visit])]=1.
        gate[visit]=m
        donors[visit]={"hubs":torch.zeros(32,64)}
    baseline=traced_forward(
        model,x,mode="hubs",orientation=orientation,donors=donors,
        gates=gate,derivative=bool(sites))
    acts=clean_activations(model,x) if sites else None
    variants={mode:{} for mode in MODELS if mode not in ("clean","hubs")}
    all_supported=torch.full((32,),bool(sites),dtype=torch.bool)
    all_angle=torch.full((32,),bool(sites),dtype=torch.bool)
    all_random=torch.full((32,),bool(sites),dtype=torch.bool)
    down=model.block.mlp_down.weight.detach()
    for visit,units in sorted(sites.items()):
        site=record["site_geometry"][str(visit)]
        n=len(units)
        if (site["n_active_per_source"]!=n or site["units"]!=sorted(units)
            or site["hubs"]!=sorted(hubs.get(visit,[]))):
            raise AssertionError("native source-site visit or equal-code budget changed")
        original=acts[:,visit,:]
        nonhub=1.-gate[visit]
        ae=prepared["sae"]
        with torch.no_grad():
            z=ae.codes((original-prepared["mean"])/prepared["scale"])
            cols=ae.decoder.weight.detach().T*prepared["scale"][None,:]
            # Source readout uses exact GEMV reconstruction with the
            # frozen SAE decoder bias and original activation mean.
            contrib16=z@cols
            # Reproduce the physical GELU affine decoder in the same
            # GEMM/bias operation order as the registered scorer. Check
            # the decomposition independently with an explicit tolerance.
            affine=ae.decoder(z)*prepared["scale"]+prepared["mean"]
            decomposed=contrib16+(
                prepared["mean"]+ae.decoder.bias.detach()*prepared["scale"])
            if not torch.allclose(affine,decomposed,atol=2e-5,rtol=1e-5):
                raise AssertionError("SAE affine decomposition changed")
        native=original*nonhub
        reconstructed=affine*nonhub
        fullcontrib=contrib16*nonhub
        ranked=torch.zeros_like(original)
        matched=torch.zeros_like(original)
        random=torch.zeros_like(original)
        local_supported=torch.zeros(32,dtype=torch.bool)
        local_angle=torch.zeros(32,dtype=torch.bool)
        local_random=torch.zeros(32,dtype=torch.bool)
        rankpos={i:p for p,i in enumerate(prepared["ranks"][visit])}
        for mask in range(32):
            active=torch.where(z[mask]>1e-7)[0].tolist()
            if active!=site["all_active_ids"][mask]:
                raise AssertionError("source activation TopK support changed")
            ok=len(active)==16
            if site["code_support_exact16"][mask]!=ok:
                raise AssertionError("TopK16 code support gate differs")
            local_supported[mask]=ok
            if not ok:
                if (site["source_active_ids"][mask] or
                    site["exhaustive_angle_sham_ids"][mask] or
                    site["hashed_disjoint_sham_ids"][mask] or
                    site["candidate_search_count"][mask]!=0):
                    raise AssertionError("missing-code case invented source direction")
                continue
            ordered=sorted(active,key=lambda i:rankpos[i])
            source_ids=ordered[:n]
            others=sorted(set(active)-set(source_ids))
            possibilities=list(combinations(others,n))
            if (source_ids!=site["source_active_ids"][mask]
                or site["candidate_search_count"][mask]!=len(possibilities)):
                raise AssertionError("not all disjoint active-code candidates were preserved")
            sel=tuple(site["exhaustive_angle_sham_ids"][mask])
            rnd=tuple(site["hashed_disjoint_sham_ids"][mask])
            if sel not in possibilities or rnd not in possibilities:
                raise AssertionError("sham overlaps source TopK components or is missing")
            token=(f"{world['world_sha256']}|{arm}|{group}|{mask}|{visit}").encode()
            pos=int.from_bytes(sha256(RANDOM_SALT+b"|"+token).digest(),"big")%len(possibilities)
            if rnd!=possibilities[pos]:
                raise AssertionError("hashed active sham modified")
            s=z[mask,source_ids]@cols[source_ids]
            good=z[mask,list(sel)]@cols[list(sel)]
            rnd64=z[mask,list(rnd)]@cols[list(rnd)]
            ranked[mask]=s
            matched[mask]=good
            random[mask]=rnd64
            projected=s@down.T
            grad=baseline["grad_down"][visit][mask]
            sourcecos=cos32(projected,grad)
            # Re-enumerate every independent disjoint feature subset
            # directly from the frozen source decoder in physical 32D.
            candidate_idx=torch.tensor(possibilities,dtype=torch.long)
            projected_columns=cols@down.T
            candidate32=(
                z[mask,candidate_idx].unsqueeze(-1)*
                projected_columns[candidate_idx]).sum(dim=1)
            coss=(candidate32@grad)/(
                candidate32.norm(dim=1)*grad.norm()).clamp_min(1e-10)
            best=float((coss-sourcecos).abs().min())
            gap=abs(cos32(good@down.T,grad)-sourcecos)
            if gap>best+5e-5:
                raise AssertionError("archived angle sham is not exhaustive nearest active control")
            compare(gap,site["absolute_cosine_gap"][mask],atol=6e-5)
            local_angle[mask]=bool(
                gap<=ANGLE_LIMIT and grad.norm()>1e-7
                and projected.norm()>1e-6 and (good@down.T).norm()>1e-6)
            local_random[mask]=bool(
                projected.norm()>1e-6 and (rnd64@down.T).norm()>1e-6)
        x_ang,y_ang,d_ang,ga,gb=attenuation_independent(ranked,matched,down)
        x_rnd,y_rnd,d_rnd,gr,gs=attenuation_independent(ranked,random,down)
        for mode,actual in {
            "native_nonhub":native,
            "sae_affine_nonhub":reconstructed,
            "sae16_contrib_nonhub":fullcontrib,
            "source_ranked_n":ranked,
        }.items():
            compare((actual@down.T).norm(dim=1).tolist(),
                    site["natural_output_dose"][mode])
        for measure,actual in [
            ("full_64d_reconstruction_relative_error",
             (affine-original).norm(dim=1)/original.norm(dim=1).clamp_min(1e-6)),
            ("full_nonhub_32d_reconstruction_relative_error",
             ((reconstructed-native)@down.T).norm(dim=1)/
             (native@down.T).norm(dim=1).clamp_min(1e-6)),
            ("topk16_contrib_32d_reconstruction_relative_error",
             ((fullcontrib-native)@down.T).norm(dim=1)/
             (native@down.T).norm(dim=1).clamp_min(1e-6)),
        ]:
            compare(actual.tolist(),site[measure],atol=1e-4)
        for name,actual in (
            ("source_vs_angle_shared_output_dose",d_ang),
            ("source_vs_angle_source_gain",ga),
            ("source_vs_angle_sham_gain",gb),
            ("source_vs_random_shared_output_dose",d_rnd),
            ("source_vs_random_source_gain",gr),
            ("source_vs_random_sham_gain",gs),
        ):
            compare(actual.tolist(),site[name],atol=1e-4)
        local_angle &= d_ang>1e-6
        local_random &= d_rnd>1e-6
        if (local_angle.tolist()!=site["angle_control_valid_site"]
            or local_random.tolist()!=site["random_control_valid_site"]):
            raise AssertionError("physical or angular source admission gates changed")
        all_supported &= local_supported
        all_angle &= local_angle
        all_random &= local_random
        for name,value in (
            ("native_nonhub",native),("sae_affine_nonhub",reconstructed),
            ("sae16_contrib_nonhub",fullcontrib),("ranked_n",ranked),
            ("ranked_angle",x_ang),("angle_sham",y_ang),
            ("ranked_random",x_rnd),("random_active",y_rnd)
        ):
            variants[name][visit]=value
    if (all_supported.tolist()!=record["valid_ranked_active_on_all_source_visits"]
        or all_angle.tolist()!=record["valid_angle_pair_on_all_source_visits"]
        or all_random.tolist()!=record["valid_random_pair_on_all_source_visits"]):
        raise AssertionError("all-source-visit availability masking changed")
    for visit in sites:
        variants["ranked_n"][visit]*=all_supported[:,None]
        for name in ("ranked_angle","angle_sham"):
            variants[name][visit]*=all_angle[:,None]
        for name in ("ranked_random","random_active"):
            variants[name][visit]*=all_random[:,None]
    trace={"clean":clean,"hubs":baseline}
    for mode,v in variants.items():
        assignments={j:{"native":vec} for j,vec in v.items()}
        with torch.no_grad():
            trace[mode]=traced_forward(
                model,x,mode="native",orientation=orientation,
                donors=assignments,gates=gate)
    y=labels(x,world,control=arm=="additive").tolist()
    if record["true_teacher_labels"]!=y:
        raise AssertionError("source Boolean teacher labels changed")
    correct={}
    for mode in MODELS:
        compare(trace[mode]["logits"].tolist(),record["logits"][mode],atol=3e-4)
        pred=trace[mode]["logits"].argmax(dim=1).tolist()
        if pred!=record["class_predictions"][mode]:
            raise AssertionError("source model correctness changed after audit replay")
        correct[mode]=sum(int(a==b) for a,b in zip(y,pred))
        mean_drift=(trace[mode]["memory"]-baseline["memory"]).norm(dim=-1).mean(dim=0)
        compare(mean_drift.tolist(),record["memory_distance_from_hubs_by_visit"][mode])
    if any(trace["ranked_angle"]["logits"][~all_angle].flatten()!=
           trace["angle_sham"]["logits"][~all_angle].flatten()):
        if not torch.allclose(
            trace["ranked_angle"]["logits"][~all_angle],
            trace["angle_sham"]["logits"][~all_angle],atol=1e-6):
            raise AssertionError("failed source-matched control generated nonzero signal")
    if not torch.allclose(
        trace["ranked_random"]["logits"][~all_random],
        trace["random_active"]["logits"][~all_random],atol=1e-6):
        raise AssertionError("failed random source-control generated nonzero signal")
    return correct


def audit(index:int,source:Path,out:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("source-only independent audit on registered eight roots only")
    report_path=out/"nec03r_source_recovery_report.json"
    result=read(report_path)
    world=worlds()[index]
    if (result["schema"]!=SCHEMA
        or result["world_index"]!=index
        or result["world_sha256"]!=world["world_sha256"]
        or result["source_lock_sha256"]!=lock_sha
        or result["source_groups_scored"]!=list(SOURCE_PANEL)
        or result["source_training_histories_per_arm"]!=CASES
        or result["original_five_input_registry_sha256"]!=registration()["registry_sha256"]
        or result["original_four_input_confirmation_sha256"]!=old_registry()["confirmation_full_spec_sha256"]
        or result["models_or_SAEs_retrained"]!=0
        or result["new_development_eval_contexts_scored"]!=0
        or result["reserved_contexts_scored"]!=0
        or result["embargoed_worlds_scored"]!=0
        or result["original_four_input_confirmation_worlds_scored"]!=0
        or set(result["arms"])!={"composed","additive"}
        or result["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("source-only population, endpoints, or seals changed")
    arms={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        row=result["arms"][arm]
        path=out/row["source_root_raw_file"]
        if (sha_file(path)!=row["source_root_raw_sha256"]
            or row["native_sha256"]!=prepared["model_sha256"]
            or row["sae_checkpoint_sha256"]!=prepared["sae_sha256"]
            or row["source_rank_sha256"]!=prepared["rank_sha256"]
            or row["source_receipt_sha256"]!=prepared["source_report_sha256"]
            or row["source_candidate_present"]!=bool(prepared["source_candidate"])
            or row["source_eligible_under_original_90pct_gate"]!=prepared["original_arm_source_eligible"]):
            raise AssertionError("source evidence checksum, candidate, or SHA changed")
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if len(records)!=len(SOURCE_PANEL):
            raise AssertionError("incomplete source-only context panel")
        counts={key:0 for key in MODELS}
        for idx,record in enumerate(records):
            if record["source_group"]!=SOURCE_PANEL[idx]:
                raise AssertionError("omitted/duplicated source training group")
            group_count=audit_group(prepared,record,arm)
            for k in MODELS:
                counts[k]+=group_count[k]
        if counts!=row["per_intervention_correct_counts"]:
            raise AssertionError("source-only clean/hub/recoverability scores did not independently replay")
        # The geometry statistics are reconstructed per complete source
        # site/input, then reduced within the same world/teacher arm.
        geometry={
            "sae_reconstruction_64d_relative":[],
            "sae_affine_nonhub_32d_relative":[],
            "sae16_contribution_nonhub_32d_relative":[],
            "native_nonhub_output_dose":[],
            "full_affine_output_dose":[],
            "sparse_ranked_output_dose":[],
            "matched_angle_source_retained_gain":[],
            "matched_angle_shared_output_dose":[],
            "matched_control_available":[]
        }
        for rec in records:
            for site in rec["site_geometry"].values():
                geometry["sae_reconstruction_64d_relative"].extend(site[
                    "full_64d_reconstruction_relative_error"])
                geometry["sae_affine_nonhub_32d_relative"].extend(site[
                    "full_nonhub_32d_reconstruction_relative_error"])
                geometry["sae16_contribution_nonhub_32d_relative"].extend(site[
                    "topk16_contrib_32d_reconstruction_relative_error"])
                geometry["native_nonhub_output_dose"].extend(site[
                    "natural_output_dose"]["native_nonhub"])
                geometry["full_affine_output_dose"].extend(site[
                    "natural_output_dose"]["sae_affine_nonhub"])
                geometry["sparse_ranked_output_dose"].extend(site[
                    "natural_output_dose"]["source_ranked_n"])
                geometry["matched_angle_source_retained_gain"].extend(site[
                    "source_vs_angle_source_gain"])
                geometry["matched_angle_shared_output_dose"].extend(site[
                    "source_vs_angle_shared_output_dose"])
                geometry["matched_control_available"].extend(
                    int(b) for b in site["angle_control_valid_site"])
        median_geometry={
            k:(median(values) if values else None)
            for k,values in geometry.items() if k!="matched_control_available"}
        median_geometry["source_site_input_observations"]=len(
            geometry["matched_control_available"])
        median_geometry["angle_control_available_fraction"]=(
            sum(geometry["matched_control_available"])/len(geometry["matched_control_available"])
            if geometry["matched_control_available"] else None)
        gap=counts["clean"]-counts["hubs"]
        expected={mode:((counts[mode]-counts["hubs"])/gap if gap>0 else None)
                  for mode in MODELS if mode not in ("clean","hubs")}
        if (expected!=row["correctness_gap_fraction_recovered_if_positive"]
            or row["clean_minus_hubs_correct_gap"]!=gap
            or row["ranked_source_minus_angle_sham_correct"]!=counts["ranked_angle"]-counts["angle_sham"]
            or row["ranked_source_minus_random_sham_correct"]!=counts["ranked_random"]-counts["random_active"]):
            raise AssertionError("recoverable clean gap or matched control effect misreported")
        arms[arm]={
            "counts":counts,"clean_hubs_gap":gap,"fraction_if_gap_positive":expected,
            "verified_source_geometry_medians":median_geometry,
            "source_candidate_present":bool(prepared["source_candidate"]),
            "original_90pct_native_and_SAE_source_eligible":prepared["original_arm_source_eligible"],
            "raw_sha256":sha_file(path),
            "native_model_sha256":prepared["model_sha256"],
            "sae_sha256":prepared["sae_sha256"],
        }
    observed=(arms["composed"]["counts"]["ranked_angle"]-
              arms["composed"]["counts"]["angle_sham"]-
              arms["additive"]["counts"]["ranked_angle"]+
              arms["additive"]["counts"]["angle_sham"])/CASES
    admitted=index in (0,1,2,4,6,7)
    if (abs(observed-result["raw_root_angle_sham_DID"])>1e-12
        or abs((observed if admitted else 0)-result["primary_ITT_angle_sham_DID"])>1e-12):
        raise AssertionError("original world-level primary control DID differs")
    report={
        "schema":AUDIT_SCHEMA,
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "source_report_sha256":sha_file(report_path),
        "all_eight_complete_source_groups_per_arm_independently_replayed":True,
        "source_full_SAE_and_native_nonhub_reconstruction_verified":True,
        "all_disjoint_angle_shams_and_hashed_random_controls_recomputed":True,
        "all_recurrent_native_SAE_source_and_control_logits_replayed":True,
        "all_original_training_source_truth_labels_recomputed":True,
        "arm_count":2,"source_cases_per_arm":CASES,
        "source_eligible_root_in_prior_study":admitted,
        "root_ITT_angle_DID":observed if admitted else 0.,
        "raw_root_angle_DID":observed,
        "new_development_eval_contexts_scored":0,
        "reserved_contexts_scored":0,
        "embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":arms,
    }
    (out/"independent_nec03r_source_recovery_audit.json").write_bytes(encoded(report))
    print("NEC03R_SOURCE_ONLY_INDEPENDENT_RECOVERABILITY_AUDIT_PASS",world["world_id"])
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.world_index,a.source_root,a.out,reset=a.reset_root)
