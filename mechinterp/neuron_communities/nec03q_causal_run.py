"""NEC-03Q: frozen development-only causal pilot with equal-dose active shams.

Six NEC-03O source-eligible five-input development roots (0,1,2,4,6,7).
The two source-ineligible roots (3,5) enter the separate eight-root ITT
aggregate as zero. Every development context is complete: 24 groups x
32 focal counterfactuals per native arm. Confirmation stays sealed.

This runner is part of the experimental source lock, frozen before its
first execution. The prior NEC-03P design manifest is a fixed parent.
"""
from __future__ import annotations

import argparse
from hashlib import sha1,sha256
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03p_causal_manifest import (
    development_pilot_manifest,canonical,PREVIOUS_SOURCE_ELIGIBLE,
    PAIRED_SOURCE_FAILURES,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03o_source_refit import restore_model
from mechinterp.neuron_communities.nec03ob_source_audit import read_model
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from mechinterp.neuron_communities.nec03k_source_geometry import exhaustive_direction_match
from mechinterp.neuron_communities.nec03e_pairs import attenuate_pair
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations,seal_array
from mechinterp.neuron_communities.topk_sae import TopKSAE
from mechinterp.neuron_communities.matched_dose import by_visit
from research.pef07.pef07 import as_tensor,model_digest

SCHEMA="NEC-03Q/five-input-development-causal-DID/v1"
LOCK="NEC03Q_DEVELOPMENT_EXECUTION_LOCK.json"
ROOTS=PREVIOUS_SOURCE_ELIGIBLE
ALL_ROOTS=tuple(range(8))
DEVELOPMENT_GROUPS=24
FOCAL_MASKS=32
DEV_CASES=DEVELOPMENT_GROUPS*FOCAL_MASKS
MIN_MEANINGFUL_EFFECT=.02
EXPECTED_MANIFEST_SHA256=sha256(canonical(development_pilot_manifest())).hexdigest()


def read(path:Path):
    return json.loads(path.read_text())


def verify_lock():
    path=Path(__file__).with_name(LOCK)
    cfg=read(path)
    required={
        "schema":"NEC-03Q/frozen-five-input-development-causal-execution/v1",
        "prior_design_manifest_sha256":EXPECTED_MANIFEST_SHA256,
        "five_input_registry_sha256":registration()["registry_sha256"],
        "old_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "development_root_indices":list(ROOTS),
        "eight_root_ITT_denominator":list(ALL_ROOTS),
        "unavailable_root_indices_assigned_zero":[3,5],
        "development_groups_per_eligible_arm":24,
        "focal_masks_per_group":32,
        "effect_floor":MIN_MEANINGFUL_EFFECT,
        "familywise_primary_comparisons":1,
        "angle_gap_max":.25,
        "physically_equalized_l2_dose_required":True,
        "development_outcome_selected_after_source_only_lock":True,
        "new_native_training_updates":0,
        "new_SAE_training_updates":0,
        "reserved_contexts_scored":0,
        "embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise AssertionError("NEC03Q frozen execution contract changed")
    for name,expected in cfg["git_blob_sha1"].items():
        blob=(Path(__file__).resolve().parents[2]/name).read_bytes()
        git_sha=sha1(b"blob "+str(len(blob)).encode()+b"\x00"+blob).hexdigest()
        if git_sha!=expected:
            raise AssertionError("NEC03Q executable identity changed "+name)
    return cfg,sha_file(path)


def load_frozen_arm(index:int, arm:str, source:Path, *, reset:Path|None=None):
    """Load only existing source-audited reset1800 weights and new SAE."""
    if index not in ROOTS or arm not in ("composed","additive"):
        raise PermissionError("attempt to use unselected model or root")
    world=worlds()[index]
    catalog,manifest=context_catalog(world)
    if index<4:
        if reset is None:
            raise ValueError("old source cohort needs separate reset checkpoint archive")
        receipt=read(source/"nec03o_source_refit_receipt.json")
        audit=read(source/"independent_nec03o_source_audit.json")
        native=receipt["arms"][arm]["native_reset1800"]
        upstream=read(reset/"checkpoint_reset_source_report.json")["arms"][arm]
        if (audit["source_refit_receipt_sha256"]!=sha_file(source/"nec03o_source_refit_receipt.json")
            or audit["world_sha256"]!=world["world_sha256"]
            or not audit["original_900_native_and_reset1800_model_provenance_verified"]
            or not audit["fresh_1800_specific_source_SAE_and_rank_audited"]
            or audit["arms"][arm]["source_model_eligible"] is not True
            or native["new_model_sha256"]!=upstream["new_native_model_sha256"]
            or native["new_checkpoint_sha256"]!=upstream["new_checkpoint_sha256"]):
            raise AssertionError("NEC03O source checkpoint or independent audit not eligible")
        model=restore_model(reset/upstream["new_checkpoint_file"],world,arm,upstream)
    else:
        receipt=read(source/"nec03ob_source_receipt.json")
        audit=read(source/"independent_nec03ob_source_audit.json")
        native=receipt["arms"][arm]["native"]
        if (audit["source_receipt_sha256"]!=sha_file(source/"nec03ob_source_receipt.json")
            or audit["world_sha256"]!=world["world_sha256"]
            or not audit["both_stage_checkpoint_bytes_and_training_source_labels_audited"]
            or not audit["both_arm_source_SAEs_and_geometry_independently_checked"]
            or audit["arms"][arm]["source_eligible"] is not True):
            raise AssertionError("prospective source model or independent audit not eligible")
        model,_=read_model(
            source/native["reset1800_checkpoint_file"],
            native["reset1800_checkpoint_sha256"],world,arm,
            native["reset1800_model_sha256"],old=False)
    if (receipt["world_sha256"]!=world["world_sha256"]
        or receipt["source_manifest_sha256"]!=manifest["source_sha256"]
        or not receipt["arms"][arm]["source_model_eligible"]
        or not receipt["both_arms_source_eligible"]):
        raise AssertionError("source-only ITT/admission provenance mismatch")
    roster_path=source/f"{arm}.source_roster.json"
    row=read(roster_path)
    if sha_file(roster_path)!=receipt["arms"][arm]["source_roster_sha256"]:
        raise AssertionError("frozen source candidate roster altered")
    candidate=receipt["arms"][arm]["source_candidate"]
    chosen=by_visit(candidate["nodes"])
    data=row["roster"] if index<4 else row["source_roster"]
    if (data["source_candidate"]!=candidate
        or row["world_sha256"]!=world["world_sha256"]
        or (row.get("source_sha256") or row.get("source_manifest_sha256"))!=manifest["source_sha256"]):
        raise AssertionError("frozen native candidate selected outside source policy")
    hubs=by_visit(data["source_graph"]["hubs"])
    sae_info=receipt["arms"][arm]["SAE_source"]
    checkpoint=source/sae_info["source_dictionary_file"]
    if sha_file(checkpoint)!=sae_info["source_dictionary_sha256"]:
        raise AssertionError("source-only SAE checkpoint bytes altered")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (payload["world_sha256"]!=world["world_sha256"]
        or payload["arm"]!=arm or payload["model_sha256"]!=model_digest(model)
        or payload["source_fit_context_groups"]!=40
        or payload["source_rank_context_groups"]!=40):
        raise AssertionError("SAE belongs to the wrong trained model or source split")
    sae=TopKSAE().eval()
    sae.load_state_dict(payload["state"],strict=True)
    mean,scale,ranks=payload["mean"],payload["scale"],payload["rank"]
    if (seal_array(mean)!=sae_info["normalization_mean_sha256"]
        or seal_array(scale)!=sae_info["normalization_scale_sha256"]
        or payload["rank_sha256"]!=sae_info["source_rank_sha256"]):
        raise AssertionError("SAE source normalization or ranking changed")
    if set(chosen.keys())-set(range(12)) or len(chosen)<1:
        raise AssertionError("bad selected recurrent visits")
    return {
        "model":model,"sae":sae,"mean":mean,"scale":scale,
        "ranks":ranks,"candidate":candidate,"chosen":chosen,"hubs":hubs,
        "catalog":catalog,"world":world,"manifest":manifest,
        "native_sha256":model_digest(model),
        "sae_checkpoint_sha256":sae_info["source_dictionary_sha256"],
        "sae_rank_sha256":sae_info["source_rank_sha256"],
        "source_receipt_sha256":sha_file(
            source/("nec03o_source_refit_receipt.json" if index<4
                    else "nec03ob_source_receipt.json")),
    }


def paired_dev_group(loaded:dict, bits):
    """Fit no weights; choose null before teacher labels; do 32 paired cases."""
    model=loaded["model"]
    x=as_tensor(bits)
    if x.shape!=(32,12):
        raise ValueError("whole 32-state original source-disjoint group is required")
    clean=traced_forward(model,x,mode="clean")
    chosen=clean["logits"].argmax(1)
    acts=clean_activations(model,x)
    native_sites=loaded["chosen"]
    hubs=loaded["hubs"]
    donors, gates={},{}
    for visit in native_sites:
        gates[visit]=torch.zeros(64)
        if visit in hubs:
            gates[visit][sorted(hubs[visit])]=1.
        donors[visit]={"hubs":torch.zeros(32,64)}
    hubs_trace=traced_forward(
        model,x,mode="hubs",orientation=chosen,donors=donors,
        gates=gates,derivative=True)
    valid=torch.ones(32,dtype=torch.bool)
    site_data={}
    all_vectors={}
    down=model.block.mlp_down.weight.detach()
    for visit,units in sorted(native_sites.items()):
        # A development input with fewer than 16 positive codes is an
        # unavailable control and receives zero contrast, never exclusion.
        with torch.no_grad():
            z=loaded["sae"].codes(
                (acts[:,visit,:]-loaded["mean"])/loaded["scale"])
            supported=(z>1e-7).sum(1)==16
        left=torch.zeros(32,64)
        right=torch.zeros(32,64)
        routes=[{
            "within_angle_gate":False,"source_top_active":[],
            "selected_disjoint_control":[],"all_active":[],
            "absolute_cosine_gap":1.0,"candidate_count":0
        } for _ in range(32)]
        if supported.any():
            kept=torch.where(supported)[0].tolist()
            la,ra,raudit=exhaustive_direction_match(
                loaded["sae"],loaded["mean"],loaded["scale"],
                acts[kept,visit,:],loaded["ranks"][visit],
                hubs_trace["grad_down"][visit][kept],down,
                count=len(units))
            for j,i in enumerate(kept):
                left[i]=la[j]
                right[i]=ra[j]
                routes[i]=raudit[j]
        left,right,doses=attenuate_pair(left,right,down)
        here=[
            bool(route["within_angle_gate"] and doses["shared_nonzero"][i]
                 and doses["matched_within_1pct"][i])
            for i,route in enumerate(routes)]
        valid &= torch.tensor(here,dtype=torch.bool)
        all_vectors[visit]=(left,right)
        site_data[str(visit)]={
            "n_components":len(units),
            "source_feature_ids":[r["source_top_active"] for r in routes],
            "disjoint_sham_feature_ids":[r["selected_disjoint_control"] for r in routes],
            "all_active_feature_ids":[r["all_active"] for r in routes],
            "cosine_gap":[r["absolute_cosine_gap"] for r in routes],
            "angle_gate":[r["within_angle_gate"] for r in routes],
            "candidate_count":[r["candidate_count"] for r in routes],
            "code_support_exact16":supported.tolist(),
            "shared_direct_output_dose":doses["shared_achievable_dose"],
            "source_gain":doses["left_gain"],
            "sham_gain":doses["right_gain"],
            "dose_match_1pct":doses["matched_within_1pct"],
            "shared_nonzero":doses["shared_nonzero"],
        }
    for visit,(left,right) in all_vectors.items():
        keep=valid.to(left.dtype).unsqueeze(1)
        donors[visit]["native"]=left*keep
        donors[visit]["sham"]=right*keep
    with torch.no_grad():
        source=traced_forward(model,x,mode="native",orientation=chosen,donors=donors,gates=gates)
        sham=traced_forward(model,x,mode="sham",orientation=chosen,donors=donors,gates=gates)
    if not torch.allclose(source["logits"][~valid],sham["logits"][~valid],atol=1e-6):
        raise AssertionError("invalid sham has a nonzero causal contrast")
    return {
        "source_logits":source["logits"].tolist(),
        "sham_logits":sham["logits"].tolist(),
        "clean_logits":clean["logits"].tolist(),
        "hubs_logits":hubs_trace["logits"].tolist(),
        "valid_input_control":valid.tolist(),
        "clean_chosen_class":chosen.tolist(),
        "site_geometry":site_data,
        "full_recurrent_memory_distance":(
            source["memory"]-sham["memory"]).norm(dim=-1).mean(dim=1).tolist(),
        "true_labels_used_in_donor_selection":False,
    }


def run(index:int,source:Path,out:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("only six preadmitted development-world roots can open their evaluation contexts")
    frozen=development_pilot_manifest()
    if frozen["source_eligible_roots"]!=list(ROOTS) or frozen["pilot_readout_status"]!="SEALED_UNOPENED":
        raise AssertionError("prior source-selected developmental pilot plan changed")
    prepared={}
    for arm in ("composed","additive"):
        prepared[arm]=load_frozen_arm(index,arm,source,reset=reset)
    world=worlds()[index]
    context_catalogue,source_split=context_catalog(world)
    if not all(prepared[a]["manifest"]["source_sha256"]==source_split["source_sha256"] for a in prepared):
        raise AssertionError("inconsistent source context partitions between paired arms")
    out.mkdir(parents=True,exist_ok=True)
    arms={}
    for arm in ("composed","additive"):
        locked=prepared[arm]
        counts={"source":0,"sham":0,"clean":0,"hubs":0,"valid_controls":0}
        raw=[]
        for context in range(DEVELOPMENT_GROUPS):
            rows=context_catalogue["development"][context*32:(context+1)*32]
            if len(rows)!=32:
                raise AssertionError("evaluation group missing focal states")
            item=paired_dev_group(locked,rows)
            # Task truth is introduced ONLY after the label-blind donor/routing decision.
            y=labels(as_tensor(rows),world,control=arm=="additive").tolist()
            predicted={}
            for key in ("source","sham","clean","hubs"):
                z=item[f"{key}_logits"]
                predicted[key]=[int(pair[1]>pair[0]) for pair in z]
                counts[key]+=sum(int(a==b) for a,b in zip(predicted[key],y))
            valid=item["valid_input_control"]
            if any(predicted["source"][i]!=predicted["sham"][i]
                   for i,ok in enumerate(valid) if not ok):
                raise AssertionError("site unavailability leaked into source-versus-sham correctness")
            counts["valid_controls"]+=sum(valid)
            raw.append({
                "schema":SCHEMA,"world_id":world["world_id"],
                "arm":arm,"development_group":context,
                "original_split_group_index":80+context,
                "focal_masks":list(range(32)),
                "input_histories":rows,
                "true_labels":y,
                "source_class_prediction":predicted["source"],
                "sham_class_prediction":predicted["sham"],
                "clean_class_prediction":predicted["clean"],
                "hubs_class_prediction":predicted["hubs"],
                **item,
            })
        file=out/f"{arm}.development_causal.jsonl"
        rawbytes=b"".join(encoded(z) for z in raw)
        file.write_bytes(rawbytes)
        arms[arm]={
            "native_sha256":locked["native_sha256"],
            "sae_checkpoint_sha256":locked["sae_checkpoint_sha256"],
            "source_rank_sha256":locked["sae_rank_sha256"],
            "source_receipt_sha256":locked["source_receipt_sha256"],
            "source_candidate":locked["candidate"],
            "source_correct":counts["source"],"sham_correct":counts["sham"],
            "clean_correct":counts["clean"],"hubs_correct":counts["hubs"],
            "valid_input_controls":counts["valid_controls"],
            "total_input_cases":DEV_CASES,
            "signed_source_minus_sham_correct":counts["source"]-counts["sham"],
            "raw_file":file.name,"raw_sha256":sha256(rawbytes).hexdigest(),
        }
    d=(arms["composed"]["signed_source_minus_sham_correct"]
       -arms["additive"]["signed_source_minus_sham_correct"])/DEV_CASES
    result={
        "schema":SCHEMA,"world_id":world["world_id"],
        "world_index":index,"world_sha256":world["world_sha256"],
        "old_source_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "original_source_manifest_sha256":source_split["source_sha256"],
        "frozen_execution_lock_sha256":lock_sha,
        "frozen_prior_manifest_sha256":EXPECTED_MANIFEST_SHA256,
        "native_and_sae_weights_updated":False,
        "true_labels_used_for_control_selection":False,
        "development_groups_evaluated":DEVELOPMENT_GROUPS,
        "focal_masks_per_group":FOCAL_MASKS,
        "root_ITT_effect":d,
        "root_ITT_effect_definition":"((composed_source - composed_sham)-(additive_source - additive_sham))/768",
        "source_eligible_in_original_source_only_study":True,
        "primary_comparison_family_size":1,
        "minimum_meaningful_effect":MIN_MEANINGFUL_EFFECT,
        "reserved_contexts_scored":0,"embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":arms,
    }
    (out/"nec03q_development_causal_report.json").write_bytes(encoded(result))
    print("NEC03Q_REGISTERED_DEVELOPMENT_CAUSAL_ROOT_WRITTEN",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.world_index,a.source_root,a.out,reset=a.reset_root)
