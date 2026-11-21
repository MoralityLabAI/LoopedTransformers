"""NEC-03M independent five-input *source-only* neural and geometry audit.

Every 12-bit input belongs to the original NEC03L source-training split.
Reload trained native checkpoint, score all original 80 source groups
only, recalculate TopK16 reconstruction and frozen feature ranks on
disjoint 40/40 source groups. Independently enumerate each source-only
direction-matched active-code subset from the archived 32D contributions,
verify its physical output dose, and retain all missing/failed arms.

No development-evaluation, reserved, embargoed or original confirmation
model or context is evaluated.
"""
from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    registration,worlds,context_catalog,labels,additive_control_truth,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03k_source_audit import (
    cosine,vector_sum,norm,close,file_hash,
)
from mechinterp.neuron_communities.topk_sae import TopKSAE,dev_ranks
from mechinterp.neuron_communities.source_sae import (
    collect_source,seal_array,
)
from research.pef07.pef07 import Looper,as_tensor,model_digest

SCHEMA="NEC-03M/five-input-40-40-source-native-sae-competence-v1"
INDICES=tuple(range(4))
ANG_LIMIT=.25


def read(path:Path):
    return json.loads(path.read_text())


def audit_arm(folder:Path,w:dict,catalog:dict,source:dict,entry:dict,arm:str):
    trained=entry["native"]
    ck=folder/trained["checkpoint_file"]
    if (trained["teacher_arm"]!=arm or trained["source_training_steps"]!=900
        or not trained["source_accuracy_is_in_sample"]
        or trained["training_source_histories"]!=2560
        or not trained["no_development_reserved_or_confirmation_data_used"]):
        raise AssertionError("wrong five-input source training receipt")
    if file_hash(ck)!=trained["checkpoint_sha256"]:
        raise AssertionError("native checkpoint changed")
    payload=torch.load(ck,map_location="cpu",weights_only=True)
    if (payload["world_sha256"]!=w["world_sha256"] or
        payload["seed"]!=w["native_model_seed"]+(100000 if arm=="additive" else 0)
        or payload["steps"]!=900):
        raise AssertionError("trained source model belongs to wrong five-input world")
    model=Looper()
    model.load_state_dict(payload["model"],strict=True)
    model.eval()
    if model_digest(model)!=trained["final_model_weights_sha256"]:
        raise AssertionError("train receipt model-weight hash differs")
    bits=as_tensor(catalog["train"])
    y=labels(bits,w,control=(arm=="additive"))
    true_mask=(w["truth_table_32"] if arm=="composed"
               else additive_control_truth(w["truth_table_32"]))
    if trained["teacher_truth_table"]!=true_mask:
        raise AssertionError("correctness labels are not from registered source teacher")
    correct=0
    with torch.no_grad():
        for i in range(0,len(bits),128):
            correct+=int((model.logits(bits[i:i+128]).argmax(1)==y[i:i+128]).sum())
    if correct!=trained["source_correct"] or abs(
            trained["source_accuracy"]-correct/2560)>1e-10:
        raise AssertionError("in-sample native correctness did not independently replay")
    if entry["source_training_clean_90pct"]!=(correct/2560>=.9):
        raise AssertionError("source clean learning admission misrepresented")
    sitefile=folder/f"{arm}.source_roster.json"
    if file_hash(sitefile)!=entry["source_roster_sha256"]:
        raise AssertionError("frozen native source-roster checksum changed")
    ro=read(sitefile)
    if (ro["world_sha256"]!=w["world_sha256"]
        or ro["source_sha256"]!=source["source_sha256"]
        or ro["model_sha256"]!=trained["final_model_weights_sha256"]
        or ro["arm"]!=arm):
        raise AssertionError("source native roster belongs to wrong frozen model")
    roster=ro["source_roster"]
    if (roster["source_context_groups"]!=40
        or roster["focal_states_per_context"]!=32
        or roster["source_selection_depends_on_scored_development_data"]):
        raise AssertionError("source graph did not use complete 40×32 source groups")
    candidate=roster["source_candidate"]
    if entry["source_candidate"]!=candidate:
        raise AssertionError("source site candidate changed after fitting")
    status=("fallback" if roster["fallback_used"] else "graph") if candidate else "none"
    if entry["source_site_kind"]!=status:
        raise AssertionError("wrong native source-site status")
    if candidate:
        nodes=candidate["nodes"]
        if len(nodes)!=len(set(nodes)) or len(nodes)<2:
            raise AssertionError("native neuron site duplicated")
        count={}
        for n in nodes:
            visit,neuron=map(int,n.split(":"))
            if visit not in range(12) or neuron not in range(64):
                raise AssertionError("native selected wrong physical GELU coordinate")
            count[visit]=count.get(visit,0)+1
        if any(v>8 for v in count.values()):
            raise AssertionError("native site exceeds sparse control budget")
        if set(nodes)&set(roster["source_graph"]["hubs"]):
            raise AssertionError("source native site collides with ubiquitous hubs")
        if status=="fallback":
            if (sorted(count)!=w["focal_positions"][2:4] or
                sorted(count.values())!=[2,2]):
                raise AssertionError("five-input fallback not source-fixed cardinality")
    sae_receipt=entry["SAE_source"]
    angle=entry["angle_source"]
    if candidate is None:
        if sae_receipt is not None or angle is not None or entry["source_model_eligible"]:
            raise AssertionError("invented source SAE/angle for no-candidate native model")
        return {
            "arm":arm,"source_correct":correct,"source_total":2560,
            "source_site_kind":status,"source_SAE_present":False,
            "angle_source_feasible":False,"root_model_source_eligible":False,
            "native_checkpoint_sha256":trained["checkpoint_sha256"]}
    sae_file=folder/sae_receipt["source_dictionary_file"]
    if file_hash(sae_file)!=sae_receipt["source_dictionary_sha256"]:
        raise AssertionError("source-fitted TopK decoder changed")
    saved=torch.load(sae_file,map_location="cpu",weights_only=True)
    if (saved["world_sha256"]!=w["world_sha256"] or saved["arm"]!=arm
        or saved["model_sha256"]!=trained["final_model_weights_sha256"]
        or saved["source_fit_context_groups"]!=40
        or saved["source_rank_context_groups"]!=40):
        raise AssertionError("SAE was fitted or ranked using unsupported cohort")
    sae=TopKSAE()
    sae.load_state_dict(saved["state"],strict=True)
    sae.eval()
    mean,scale=saved["mean"],saved["scale"]
    if (seal_array(mean)!=sae_receipt["normalization_mean_sha256"]
        or seal_array(scale)!=sae_receipt["normalization_scale_sha256"]):
        raise AssertionError("source normalization changed")
    fit_contexts=catalog["train"][:40*32]
    rank_contexts=catalog["train"][40*32:]
    activations=collect_source(model,fit_contexts).reshape(-1,64)
    normed=(activations-mean)/scale
    with torch.no_grad():
        recon,z=sae(normed)
        mse=float((recon-normed).square().mean())
        l0=float((z>1e-7).float().sum(1).mean())
    if (abs(sae_receipt["reconstruction_mse_normalized"]-mse)>2e-6
        or abs(sae_receipt["observed_mean_active_codes"]-l0)>1e-6
        or sae_receipt["sae_training_steps"]!=480
        or sae_receipt["source_contexts_fit"]!=40
        or sae_receipt["source_rank_context_groups"]!=40
        or sae_receipt["development_or_reserved_contexts_seen"]!=0):
        raise AssertionError("SAE source reconstruction or fit group ownership differs")
    fresh_rank,rank_sha=dev_ranks(model,sae,mean,scale,rank_contexts)
    if fresh_rank!=saved["rank"] or rank_sha!=saved["rank_sha256"]:
        raise AssertionError("source-only SAE rank differs from independent 40-group replay")
    if not angle or angle["source_contexts"]!=8 or angle[
            "heldout_development_reserved_or_embargoed_inputs_seen"]!=0:
        raise AssertionError("angle control opened forbidden inputs")
    if angle["source_training_label_free_control_selection"] is not True:
        raise AssertionError("angle matching used true teacher labels")
    if len(angle["source_contexts_and_site_receipts"])!=8:
        raise AssertionError("missing source-only 8 context groups")
    n_angle=total=positive=0
    retained=[]
    selected_per_visit={}
    for n in candidate["nodes"]:
        v,j=map(int,n.split(":"))
        selected_per_visit[v]=selected_per_visit.get(v,0)+1
    for i,group in enumerate(angle["source_contexts_and_site_receipts"]):
        if (group["source_group"]!=i
            or tuple(group["source_context_bits"])!=tuple(catalog["train"][32*i])
            or group["all_32_focal_input_masks"]!=list(range(32))):
            raise AssertionError("angle preflight read outside source group or partial masks")
        for visit,site in group["sites"].items():
            v=int(visit)
            if v not in selected_per_visit or len(site["routes"])!=32:
                raise AssertionError("unregistered site or incomplete source input")
            n=selected_per_visit[v]
            rankpos={fid:j for j,fid in enumerate(fresh_rank[v])}
            dose=site["direct_dose"]
            for k,route in enumerate(site["routes"]):
                active=route["all_active"]
                source_codes=route["source_top_active"]
                control_codes=route["selected_disjoint_control"]
                proj={o["id"]:o["vector"] for o in route["active_output_contributions32"]}
                if (len(active)!=16 or set(proj)!=set(active) or
                    len(source_codes)!=n or len(control_codes)!=n or
                    set(source_codes)&set(control_codes)
                    or source_codes!=sorted(active,key=lambda f:rankpos[f])[:n]):
                    raise AssertionError("source SAE active routing differs from saved 40-group rank")
                available=sorted(set(active)-set(source_codes))
                possibilities=list(combinations(available,n))
                if len(possibilities)!=route["candidate_count"]:
                    raise AssertionError("control angle search omitted admissible active-code subsets")
                gradient=route["hub_gradient32"]
                source_out=vector_sum(source_codes,proj)
                control_out=vector_sum(control_codes,proj)
                src_cos=cosine(source_out,gradient)
                alt_cos=cosine(control_out,gradient)
                gap=abs(src_cos-alt_cos)
                best=min(abs(cosine(vector_sum(ids,proj),gradient)-src_cos)
                         for ids in possibilities)
                if gap>best+5e-5:
                    raise AssertionError("source control not minimum-angle active candidate")
                close(route["absolute_cosine_gap"],gap,atol=4e-5)
                a=norm(source_out)
                b=norm(control_out)
                shared=min(a,b)
                for key,value in (
                    ("full_left_dose",a),("full_right_dose",b),
                    ("shared_achievable_dose",shared),
                    ("realized_left_dose",shared),
                    ("realized_right_dose",shared)):
                    close(dose[key][k],value,atol=1e-4)
                ratio=shared/a if a>1e-6 else 0.
                close(dose["retained_fraction_of_full_left_dose"][k],ratio)
                if dose["left_gain"][k]>1.000001 or dose["right_gain"][k]>1.000001:
                    raise AssertionError("source-only control amplified native output")
                allowed=gap<=.25 and norm(gradient)>1e-7 and a>1e-6 and b>1e-6
                if allowed!=route["within_angle_gate"]:
                    raise AssertionError("source angular null gate changed")
                n_angle+=int(allowed)
                total+=1
                positive+=int(shared>1e-6)
                retained.append(ratio)
    med=median(retained)
    if (abs(angle["angle_match_fraction"]-n_angle/total)>1e-6
        or abs(angle["physical_dose"]["median_fraction_full_left_dose_preserved"]-med)>1e-6
        or abs(angle["physical_dose"]["fraction_with_positive_shared_dose"]-positive/total)>1e-6):
        raise AssertionError("source angle/physical signal summary differs from raw routes")
    feasible=(n_angle/total>=.8 and positive/total>=.8 and med>=.25)
    if angle["source_null_feasible"]!=feasible:
        raise AssertionError("wrong source null feasibility")
    eligible=(correct/2560>=.9 and mse<=.03 and feasible)
    if eligible!=entry["source_model_eligible"]:
        raise AssertionError("five-input native/SAE/angular combined gate mislabeled")
    return {
        "arm":arm,"source_correct":correct,"source_total":2560,
        "source_site_kind":status,"source_SAE_present":True,
        "source_SAE_MSE":mse,
        "source_SAE_mean_l0":l0,
        "source_angular_coverage":n_angle/total,
        "source_median_left_dose_retained":med,
        "source_angle_cases_verified":total,
        "angle_source_feasible":feasible,
        "root_model_source_eligible":eligible,
        "native_checkpoint_sha256":trained["checkpoint_sha256"],
        "sae_checkpoint_sha256":sae_receipt["source_dictionary_sha256"],
        "source_rank_sha256":rank_sha}


def audit_root(folder:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt=read(folder/"source_pilot_receipt.json")
    index=receipt["world_index"]
    if index not in INDICES:
        raise AssertionError("new 5-input source training on unauthorized world")
    world=worlds()[index]
    catalog,source=context_catalog(world)
    if (read(folder/"world_spec.json")!=world
        or read(folder/"source_split_manifest.json")!=source
        or receipt["world_sha256"]!=world["world_sha256"]
        or receipt["source_manifest_sha256"]!=source["source_sha256"]
        or receipt["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or receipt["original_four_input_confirmation_spec_sha256"]!=
            old_registry()["confirmation_full_spec_sha256"]
        or receipt["development_evaluation_context_groups_scored"]!=0
        or receipt["reserved_context_groups_scored"]!=0
        or receipt["new_embargoed_worlds_scored"]!=0
        or receipt["original_confirmation_worlds_scored"]!=0):
        raise AssertionError("five-input source-cohort membership or original commitments changed")
    arms={arm:audit_arm(folder,world,catalog,source,receipt["arms"][arm],arm)
          for arm in ("composed","additive")}
    ready=all(z["root_model_source_eligible"] for z in arms.values())
    if receipt["root_ready_for_future_development_scoring"]!=ready:
        raise AssertionError("source pilot root eligibility differs from independent audit")
    audit={
        "schema":"NEC-03M/independent-source-only-neural-and-angle-audit/v1",
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "source_manifest_sha256":source["source_sha256"],
        "raw_receipt_sha256":file_hash(folder/"source_pilot_receipt.json"),
        "native_checkpoint_and_SAE_bytes_verified":True,
        "native_source_training_labels_and_all_2560_input_scores_replayed":True,
        "source_SAE_reconstruction_and_40_group_attribution_rank_recomputed":True,
        "source_direction_active_candidates_independently_reenumerated":True,
        "source_root_ready_for_development":ready,
        "development_evaluation_contexts_scored":0,
        "reserved_context_groups_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "arms":arms,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY"
    }
    (folder/"independent_source_audit.json").write_text(
        json.dumps(audit,sort_keys=True,indent=2)+"\n")
    print("NEC03M_INDEPENDENT_NEW_FIVE_INPUT_SOURCE_CHECKPOINT_AUDIT_PASS")
    return audit


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    args=p.parse_args(argv)
    audit_root(args.folder)


if __name__=="__main__":
    main()
