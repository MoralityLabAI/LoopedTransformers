"""NEC-03V: coherent SAE feature-group necessity on original source worlds.

Eight historical dev5-000..007 worlds, fixed original source training
groups 32..39 and previously trained reset1800 model/TopK16 SAEs.
Select a decoder-coherent quartet, remove it or a matched disjoint
cohesive angular null from the same full-affine SAE recurrent background.
No new training, root or source-graph adaptation, held-out scoring.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor,model_digest

SCHEMA="NEC-03V/source-only-coherent-group-necessity/v1"
LOCK_NAME="NEC03V_SOURCE_LOCK.json"
ROOTS=tuple(range(8))
PANEL=tuple(range(32,40))
QUALIFIED=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
FOCAL=32
CASES=256
GROUPS={"four":4,"six":6}
ANGLE_LIMIT=.25
COHESION_LIMIT=.50
EFFECT_FLOOR=.02
HASH_SALT=b"NEC03V/source-only-coherent-quartet-hash-sham/v1"
MODES=(
    "clean","hubs","native_nonhub","full_affine",
    "contrib16_only","coherent4_source","coherent4_sham",
    "coherent6_source","coherent6_sham",
    "hash4_source","hash4_sham",
)


def read(path):
    return json.loads(Path(path).read_text())


def verify_lock():
    path=Path(__file__).with_name(LOCK_NAME)
    lock=read(path)
    checks={
        "schema":"NEC-03V/frozen-source-group-necessity-registration/v1",
        "original_eight_roots":list(ROOTS),
        "source_training_groups":list(PANEL),
        "complete_focal_masks":FOCAL,
        "source_inputs_per_arm":CASES,
        "paired_qualified_roots":list(QUALIFIED),
        "preexisting_ITT_zero_roots":list(ITT_ZERO),
        "original_A_native_run":44898361742,
        "original_A_SAE_run":44899750508,
        "original_B_native_SAE_run":44900628360,
        "five_input_registry_sha256":registration()["registry_sha256"],
        "old_four_input_spec_sha256":old_registry()["confirmation_full_spec_sha256"],
        "original_sae_TopK":16,
        "group_budgets":GROUPS,
        "source_group_rule":"seed source-highest-ranked active feature, add k-1 active projected decoder columns in descending physical cosine to seed; source rank and feature ID break ties",
        "physical_cohesion":"mean pairwise cosine of original nonhub-masked 32D mlp_down decoder feature columns",
        "sham_rule":"exhaust all disjoint equal-k active subsets, minimize absolute gradient-cosine gap + 0.5 times physical cohesion gap",
        "gradient_reference":"frozen full-affine SAE-restored clean-chosen logit margin at 32D post-MLP-down",
        "maximum_angle_cosine_gap":ANGLE_LIMIT,
        "maximum_physical_cohesion_gap":COHESION_LIMIT,
        "minimum_positive_shared_dose":1e-6,
        "maximum_L2_relative_mismatch":.01,
        "hash_salt":HASH_SALT.decode(),
        "intervention_modes":list(MODES),
        "original_primary":"((composed sham4_correct - source4_correct)-(additive sham4_correct - source4_correct))/256, paired source failures 3/5 zero, mean all 8",
        "primary_comparisons":1,
        "primary_minimum_meaningful_effect":EFFECT_FLOOR,
        "root_resolution_rule":"mean>+0.02 and at least five of six source-qualified world roots positive and source/sham valid coverage >=0.80 per qualified teacher arm",
        "within_model_secondary_margin_scale":"max(1.0,median(abs(full_affine_true_label_signed_margin) over 256 original source cases per arm))",
        "native_or_SAE_training_updates":0,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(lock.get(k)!=v for k,v in checks.items()):
        raise AssertionError("NEC03V registered source-only group protocol changed")
    root=Path(__file__).resolve().parents[2]
    for filename,pin in lock["git_blob_sha1"].items():
        blob=(root/filename).read_bytes()
        actual=sha1(b"blob "+str(len(blob)).encode()+b"\x00"+blob).hexdigest()
        if actual!=pin:
            raise AssertionError("NEC03V prospective source blob changed: "+filename)
    return lock,sha_file(path)


def cosine(a,b):
    return (a*b).sum(dim=-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-10)


def physical_cohesion(ids,physical_cosines):
    k=len(ids)
    if k<2:
        raise ValueError("need at least two distinct physical feature directions")
    return sum(physical_cosines[ids[i],ids[j]]
               for i in range(k) for j in range(i+1,k))/(k*(k-1)//2)


def source_coherent_group(active,rank_map,physical_cosines,k):
    if len(active)!=16 or k not in (4,6):
        raise ValueError("exactly 16 active codes and fixed coherent budgets required")
    ordered=sorted(active,key=lambda f:(rank_map[f],f))
    seed=ordered[0]
    partners=sorted((f for f in active if f!=seed),
        key=lambda f:(-float(physical_cosines[seed,f]),rank_map[f],f))
    return [seed]+partners[:k-1]


def hashed_sham(options,*,world_sha,arm,group,mask,visit):
    if not options:
        raise ValueError("no disjoint source group sham")
    key=f"{world_sha}|{arm}|{group}|{mask}|{visit}|four".encode()
    h=int.from_bytes(sha256(HASH_SALT+b"|"+key).digest(),"big")
    return list(options[h%len(options)])


def equal_output_dose(source,control,down):
    if source.ndim!=2 or source.shape!=control.shape or source.shape[1]!=64:
        raise ValueError("matched 64D source/sham group shape required")
    d0=(source@down.T).norm(dim=1)
    d1=(control@down.T).norm(dim=1)
    target=torch.minimum(d0,d1)
    g0=torch.where(d0>1e-10,target/d0.clamp_min(1e-10),
                   torch.zeros_like(d0)).clamp(0,1)
    g1=torch.where(d1>1e-10,target/d1.clamp_min(1e-10),
                   torch.zeros_like(d1)).clamp(0,1)
    result={"source":source*g0[:,None],"sham":control*g1[:,None]}
    actual0=(result["source"]@down.T).norm(dim=1)
    actual1=(result["sham"]@down.T).norm(dim=1)
    matched=((actual0-actual1).abs()/target.clamp_min(1e-6))<=.01
    if bool((g0>1+1e-6).any() or (g1>1+1e-6).any()):
        raise AssertionError("positive gain must not exceed 1")
    return result,{
        "raw_source_32d_dose":d0.tolist(),
        "raw_sham_32d_dose":d1.tolist(),
        "shared_output_32d_dose":target.tolist(),
        "source_gain":g0.tolist(),"sham_gain":g1.tolist(),
        "matched_1pct":matched.tolist(),
        "nonzero_dose":(target>1e-6).tolist(),
    }


def original_case_signed_margin(logits,labels):
    return [float(z[y]-z[1-y]) for z,y in zip(logits,labels)]


def group_measure(prepared,group,inputs):
    if group not in PANEL or len(inputs)!=FOCAL:
        raise PermissionError("NEC03V limited to complete original source groups 32..39")
    world=prepared["world"]
    arm=prepared["arm"]
    model=prepared["model"]
    x=as_tensor(inputs)
    clean=traced_forward(model,x,mode="clean")
    orientation=clean["logits"].argmax(1)
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    gates={}
    for visit in sites:
        gate=torch.zeros(64)
        if visit in hubs:gate[sorted(hubs[visit])]=1.
        gates[visit]=gate
    donors0={visit:{"hubs":torch.zeros(FOCAL,64)} for visit in sites}
    hubs_trace=traced_forward(
        model,x,mode="hubs",orientation=orientation,donors=donors0,gates=gates)
    variants={name:{} for name in MODES if name not in ("clean","hubs")}
    valid={name:torch.full((FOCAL,),bool(sites),dtype=torch.bool) for name in GROUPS}
    random_valid=torch.full((FOCAL,),bool(sites),dtype=torch.bool)
    acts=clean_activations(model,x) if sites else None
    down=model.block.mlp_down.weight.detach()
    full={}
    codes={}
    columns={}
    geometry={}
    for visit,nodes in sorted(sites.items()):
        native=acts[:,visit,:]
        nonhub=1.-gates[visit]
        ae=prepared["sae"]
        with torch.no_grad():
            z=ae.codes((native-prepared["mean"])/prepared["scale"])
            affine=ae.decoder(z)*prepared["scale"]+prepared["mean"]
            decoder=ae.decoder.weight.detach().T*prepared["scale"][None,:]
            feature16=z@decoder
            offset=prepared["mean"]+ae.decoder.bias.detach()*prepared["scale"]
        if not torch.allclose(affine,feature16+offset,atol=2e-5,rtol=1e-5):
            raise AssertionError("frozen physical SAE affine reconstruction failed")
        full[visit]=affine*nonhub
        codes[visit]=z
        columns[visit]=decoder
        variants["native_nonhub"][visit]=native*nonhub
        variants["full_affine"][visit]=full[visit]
        variants["contrib16_only"][visit]=feature16*nonhub
    # All control directions reference full-affine restored recurrent
    # trajectory. Only clean-predicted orientation, never task truth.
    if sites:
        full_donors={v:{"native":vec} for v,vec in full.items()}
        ref=traced_forward(model,x,mode="native",orientation=orientation,
                           donors=full_donors,gates=gates,derivative=True)
    else:
        ref=clean
    for visit,nodes in sorted(sites.items()):
        nonhub=1.-gates[visit]
        cols=columns[visit]
        projected=(cols*nonhub[None,:])@down.T
        physical_cosines=projected@projected.T/(
            projected.norm(dim=1)[:,None]*projected.norm(dim=1)[None,:]).clamp_min(1e-10)
        z=codes[visit]
        rank_map={feature:order for order,feature in enumerate(prepared["ranks"][visit])}
        geom={
            "native_selected_units":sorted(nodes),
            "original_graph_hubs":sorted(hubs.get(visit,[])),
            "full_affine_32d_output_dose":(full[visit]@down.T).norm(dim=1).tolist(),
            "full_affine_32d_native_relative_error":(
                ((full[visit]-variants["native_nonhub"][visit])@down.T).norm(dim=1)/
                (variants["native_nonhub"][visit]@down.T).norm(dim=1).clamp_min(1e-6)
            ).tolist(),
            "groups":{},
        }
        for budget,k in GROUPS.items():
            source=torch.zeros(FOCAL,64)
            sham=torch.zeros(FOCAL,64)
            random=torch.zeros(FOCAL,64)
            meta={
                "k":k,"active_ids":[],"source_ids":[],"sham_ids":[],
                "hash_sham_ids":[],"exhaustive_control_count":[],
                "source_physical_cohesion":[],"sham_physical_cohesion":[],
                "cohesion_gap":[],"angular_gap":[],"sham_joint_score":[],
                "exactly_16_positive_active_codes":[],
            }
            for mask in range(FOCAL):
                active=[int(f) for f in torch.where(z[mask]>1e-7)[0].tolist()]
                meta["active_ids"].append(active)
                available=len(active)==16
                meta["exactly_16_positive_active_codes"].append(available)
                if not available:
                    valid[budget][mask]=False
                    if budget=="four": random_valid[mask]=False
                    meta["source_ids"].append([]);meta["sham_ids"].append([])
                    meta["hash_sham_ids"].append([])
                    meta["exhaustive_control_count"].append(0)
                    for key in ("source_physical_cohesion","sham_physical_cohesion",
                                "cohesion_gap","angular_gap","sham_joint_score"):
                        meta[key].append(None)
                    continue
                src_ids=source_coherent_group(active,rank_map,physical_cosines,k)
                others=sorted(set(active)-set(src_ids))
                opts=list(combinations(others,k))
                idx=torch.tensor(opts,dtype=torch.long)
                src=(z[mask,src_ids]@cols[src_ids])*nonhub
                control32=(z[mask,idx].unsqueeze(-1)*projected[idx]).sum(dim=1)
                src32=src@down.T
                grad=ref["grad_down"][visit][mask]
                src_cos=cosine(src32,grad)
                angle_gap=(cosine(control32,grad)-src_cos).abs()
                s_cohesion=physical_cohesion(src_ids,physical_cosines)
                cohesion=torch.zeros(len(opts))
                for i in range(k):
                    for j in range(i+1,k):
                        cohesion+=physical_cosines[idx[:,i],idx[:,j]]
                cohesion/=k*(k-1)//2
                cohesion_gap=(cohesion-s_cohesion).abs()
                joint=angle_gap+.5*cohesion_gap
                best=int(torch.argmin(joint))
                control_ids=list(opts[best])
                rnd_ids=hashed_sham(opts,world_sha=world["world_sha256"],arm=arm,
                                    group=group,mask=mask,visit=visit) if budget=="four" else []
                source[mask]=src
                sham[mask]=(z[mask,control_ids]@cols[control_ids])*nonhub
                if budget=="four":
                    random[mask]=(z[mask,rnd_ids]@cols[rnd_ids])*nonhub
                meta["source_ids"].append(src_ids)
                meta["sham_ids"].append(control_ids)
                meta["hash_sham_ids"].append(rnd_ids)
                meta["exhaustive_control_count"].append(len(opts))
                meta["source_physical_cohesion"].append(float(s_cohesion))
                meta["sham_physical_cohesion"].append(float(cohesion[best]))
                meta["cohesion_gap"].append(float(cohesion_gap[best]))
                meta["angular_gap"].append(float(angle_gap[best]))
                meta["sham_joint_score"].append(float(joint[best]))
                if (float(grad.norm())<=1e-7 or
                    float(angle_gap[best])>ANGLE_LIMIT or
                    float(cohesion_gap[best])>COHESION_LIMIT):
                    valid[budget][mask]=False
            matched,dose=equal_output_dose(source,sham,down)
            valid[budget] &= torch.tensor([
                dose["matched_1pct"][i] and dose["nonzero_dose"][i]
                for i in range(FOCAL)],dtype=torch.bool)
            variants[f"coherent{k}_source"][visit]=full[visit]-matched["source"]
            variants[f"coherent{k}_sham"][visit]=full[visit]-matched["sham"]
            meta["angular_physical_dose"]=dose
            if budget=="four":
                hash_pair,hash_dose=equal_output_dose(source,random,down)
                random_valid &= torch.tensor([
                    hash_dose["matched_1pct"][i] and hash_dose["nonzero_dose"][i]
                    for i in range(FOCAL)],dtype=torch.bool)
                variants["hash4_source"][visit]=full[visit]-hash_pair["source"]
                variants["hash4_sham"][visit]=full[visit]-hash_pair["sham"]
                meta["hash_physical_dose"]=hash_dose
            geom["groups"][budget]=meta
        geometry[str(visit)]=geom
    for visit in sites:
        for budget,k in GROUPS.items():
            for side in ("source","sham"):
                mode=f"coherent{k}_{side}"
                variants[mode][visit]=full[visit]+(
                    variants[mode][visit]-full[visit])*valid[budget][:,None]
        for side in ("source","sham"):
            mode=f"hash4_{side}"
            variants[mode][visit]=full[visit]+(
                variants[mode][visit]-full[visit])*random_valid[:,None]
    trace={"clean":clean,"hubs":hubs_trace}
    for mode,arr in variants.items():
        injected={visit:{"native":vec} for visit,vec in arr.items()}
        with torch.no_grad():
            trace[mode]=traced_forward(model,x,mode="native",
                orientation=orientation,donors=injected,gates=gates)
    for budget,k in GROUPS.items():
        bad=~valid[budget]
        if not torch.allclose(trace[f"coherent{k}_source"]["logits"][bad],
                              trace[f"coherent{k}_sham"]["logits"][bad],atol=1e-6):
            raise AssertionError("invalid coherent group sham had a nonzero effect")
    if not torch.allclose(trace["hash4_source"]["logits"][~random_valid],
                          trace["hash4_sham"]["logits"][~random_valid],atol=1e-6):
        raise AssertionError("invalid hash group control had a nonzero effect")
    return {
        "schema":SCHEMA,"source_group":group,"source_input_histories":inputs,
        "focal_masks":list(range(FOCAL)),
        "clean_chosen_orientation":orientation.tolist(),
        "teacher_labels_used_for_group_selection":False,
        "site_geometry":geometry,
        "valid_group_source_sham_on_all_visits":{k:v.tolist() for k,v in valid.items()},
        "valid_hash_on_all_visits":random_valid.tolist(),
        "logits":{key:arr["logits"].tolist() for key,arr in trace.items()},
        "memory_drift_from_full_affine_by_visit":{
            key:(arr["memory"]-ref["memory"]).norm(dim=-1).mean(dim=0).tolist()
            for key,arr in trace.items()},
    }


def run(index,source,out,*,reset=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("NEC03V only original eight source-trained worlds")
    world=worlds()[index]
    out.mkdir(parents=True,exist_ok=True)
    arms={}
    for arm in ("composed","additive"):
        model=load_source_arm(index,arm,source,reset=reset)
        model["arm"]=arm
        assert model["world"]["world_sha256"]==world["world_sha256"]
        data=[]
        correct={key:0 for key in MODES}
        valid={key:0 for key in GROUPS}
        hash_valid=0
        signed_margins={key:[] for key in MODES}
        raw_margin_deltas=[]
        for group in PANEL:
            histories=model["catalog"]["train"][group*FOCAL:(group+1)*FOCAL]
            rec=group_measure(model,group,histories)
            y=labels(as_tensor(histories),world,control=arm=="additive").tolist()
            predictions={}
            for mode in MODES:
                z=rec["logits"][mode]
                p=[int(v[1]>v[0]) for v in z]
                predictions[mode]=p
                correct[mode]+=sum(int(a==b) for a,b in zip(p,y))
                signed_margins[mode].extend(original_case_signed_margin(z,y))
            for key in GROUPS:
                valid[key]+=sum(rec["valid_group_source_sham_on_all_visits"][key])
            hash_valid+=sum(rec["valid_hash_on_all_visits"])
            rec["arm"]=arm
            rec["original_source_true_labels"]=y
            rec["source_predictions"]=predictions
            data.append(rec)
        blob=b"".join(encoded(x) for x in data)
        filename=f"{arm}.nec03v_source_coherent_groups.jsonl"
        (out/filename).write_bytes(blob)
        scale=max(1.,median(abs(v) for v in signed_margins["full_affine"]))
        paired=[(h-s)/scale for h,s in zip(
            signed_margins["coherent4_sham"],signed_margins["coherent4_source"])]
        arms[arm]={
            "source_model_sha256":model["model_sha256"],
            "source_sae_checkpoint_sha256":model["sae_sha256"],
            "source_sae_rank_sha256":model["rank_sha256"],
            "source_parent_receipt_sha256":model["source_report_sha256"],
            "original_source_candidate_present":bool(model["source_candidate"]),
            "original_arm_source_eligible":model["original_arm_source_eligible"],
            "source_groups_scored":list(PANEL),
            "cases_per_arm":CASES,
            "correct_counts":correct,
            "matched_group_valid_cases":valid,
            "hash_valid_cases":hash_valid,
            "four_source_ranked_necessity_correct_cases":
                correct["coherent4_sham"]-correct["coherent4_source"],
            "six_source_ranked_necessity_correct_cases":
                correct["coherent6_sham"]-correct["coherent6_source"],
            "within_arm_full_affine_median_margin_scale":scale,
            "secondary_normalized_source_vs_sham_margin_mean":
                float(sum(paired)/len(paired)),
            "secondary_normalized_margin_not_confirmatory":True,
            "raw_filename":filename,"raw_sha256":sha256(blob).hexdigest(),
        }
        if model_digest(model["model"])!=model["model_sha256"]:
            raise AssertionError("source native model weights changed")
    root_effect=(arms["composed"]["four_source_ranked_necessity_correct_cases"]-
                 arms["additive"]["four_source_ranked_necessity_correct_cases"])/CASES
    qualified=index in QUALIFIED
    report={
        "schema":SCHEMA,"world_index":index,"world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "five_input_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "source_manifest_sha256":context_catalog(world)[1]["source_sha256"],
        "source_lock_sha256":lock_sha,
        "source_groups_scored":list(PANEL),"source_cases_per_model_arm":CASES,
        "paired_qualified_in_original_source_roster":qualified,
        "original_ITT_source_failure":index in ITT_ZERO,
        "raw_root_primary_coherent4_necessity_DID":root_effect,
        "registered_ITT_primary_root_DID":root_effect if qualified else 0.,
        "arms":arms,"primary_comparison_family_size":1,
        "minimum_meaningful_primary":EFFECT_FLOOR,
        "new_native_or_SAE_training_updates":0,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec03v_source_group_root_report.json").write_bytes(encoded(report))
    print("NEC03V_SOURCE_COHERENT_GROUP_ROOT_WRITTEN",world["world_id"])
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",required=True,type=int)
    p.add_argument("--source-root",required=True,type=Path)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.world_index,a.source_root,a.out,reset=a.reset_root)
