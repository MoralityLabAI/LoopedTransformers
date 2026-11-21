"""NEC-03M: five-input Boolean source-only native/TopK16 competence pilot.

Only NEW five-focal-bit development worlds dev5-000..003 are allowed.
No new development-evaluation, reserved, embargoed or original four-input
confirmation labels are read. All native model learning and 128-feature
TopK16 SAE fitting take place inside each world's 80 source-training
seven-bit context groups, with 40 SAE fit groups and disjoint 40 rank
groups. The same 64D native GELU -> 32D mlp_down coordinate contract is
preserved. All failed learners and missing source sites are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.core import build_graph,discover,decode,node
from mechinterp.neuron_communities.source_sae import (
    WIDTH,L,collect_source,clean_activations,seal_array,
)
from mechinterp.neuron_communities.topk_sae import TopKSAE,dev_ranks
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
    SOURCE_FIT_CONTEXTS,SOURCE_RANK_CONTEXTS,FOCAL_MASKS,
)
from mechinterp.neuron_communities.nec03k_source_geometry import (
    exhaustive_direction_match,
)
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.nec03e_pairs import attenuate_pair,pair_dose_stats
from mechinterp.neuron_communities.matched_dose import by_visit
from research.pef07.pef07 import Looper,model_digest,as_tensor

SCHEMA="NEC-03M/five-input-40-40-source-native-sae-competence-v1"
INDICES=(0,1,2,3)
NATIVE_STEPS=900
SAE_STEPS=480
SOURCE_SITE_GROUPS=40
SOURCE_ANGLE_GROUPS=8
SAE_FEATURES=128
SAE_ACTIVE_K=16
SOURCE_CLEAN_FLOOR=0.90
SAE_RECON_FLOOR=0.03
ANGLE_MAX_GAP=0.25
ANGLE_COVERAGE_FLOOR=0.80
DOSE_COVERAGE_FLOOR=0.80
DOSE_MEDIAN_RETAIN_FLOOR=0.25
SOURCE_RANK_GROUPS=40


def sha_file(path:Path):
    h=sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def verify_m_lock():
    path=Path(__file__).with_name("NEC03M_SOURCE_LOCK.json")
    obj=json.loads(path.read_text())
    if (obj.get("schema")!="NEC-03M/source-native-SAE-smoke-lock-v1"
        or obj.get("five_source_registry_sha256")!=registration()["registry_sha256"]
        or obj.get("old_confirmation_world_spec_sha256")!=
            old_registry()["confirmation_full_spec_sha256"]
        or obj.get("source_development_world_indices")!=list(INDICES)
        or obj.get("native_steps")!=NATIVE_STEPS
        or obj.get("sae_steps")!=SAE_STEPS
        or obj.get("source_SAE_fit_groups")!=SOURCE_FIT_CONTEXTS
        or obj.get("source_feature_rank_groups")!=SOURCE_RANK_GROUPS
        or obj.get("source_only_angle_contexts")!=SOURCE_ANGLE_GROUPS
        or obj.get("heldout_development_worlds_scored")!=0
        or obj.get("five_input_embargoed_worlds_scored")!=0
        or obj.get("original_confirmation_worlds_scored")!=0):
        raise AssertionError("five-input source-only pilot preregistration changed")
    root=Path(__file__).parent
    for name,expected in obj["git_blob_sha1"].items():
        content=(root/name).read_bytes()
        digest=__import__("hashlib").sha1(
            b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()
        if digest!=expected:
            raise AssertionError("NEC03M source preflight implementation changed: "+name)
    obj["lock_sha256"]=sha_file(path)
    return obj


def native_teacher(model_seed,world,source,root:Path,arm):
    """All 900 optimizer steps use ONLY the 80 original source groups."""
    if arm not in ("composed","additive") or world["cohort"]!="development":
        raise PermissionError("invalid model teacher/source cohort")
    seed=model_seed+(100000 if arm=="additive" else 0)
    torch.manual_seed(seed)
    model=Looper().train()
    initial=model_digest(model)
    x=as_tensor(source)
    y=labels(x,world,control=arm=="additive")
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
    generator=torch.Generator().manual_seed(seed*7919+59)
    first=last=None
    for _ in range(NATIVE_STEPS):
        idx=torch.randint(len(x),(128,),generator=generator)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(x[idx]),y[idx])
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite source-only five-input native loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if first is None: first=float(loss.detach())
        last=float(loss.detach())
    model.eval()
    frozen=model_digest(model)
    ckpt=root/f"{arm}.native5.pt"
    torch.save({
        "world_id":world["world_id"],"world_sha256":world["world_sha256"],
        "arm":arm,"seed":seed,"steps":NATIVE_STEPS,
        "model_weights_sha256":frozen,"model":model.state_dict()},ckpt)
    with torch.no_grad():
        correct=0
        source_prediction=[]
        for offset in range(0,len(source),128):
            x_chunk=x[offset:offset+128]
            y_chunk=y[offset:offset+128]
            predicted=model.logits(x_chunk).argmax(dim=1)
            correct+=int((predicted==y_chunk).sum())
            source_prediction.extend(predicted.tolist())
    if len(source_prediction)!=80*32:
        raise AssertionError("missing native source-training-case predictions")
    return model,{
        "teacher_arm":arm,"training_seed":seed,
        "source_training_steps":NATIVE_STEPS,
        "teacher_truth_table":world["truth_table_32"] if arm=="composed" else
             __import__("mechinterp.neuron_communities.nec03l_five_input_worlds",
             fromlist=["additive_control_truth"]).additive_control_truth(world["truth_table_32"]),
        "initial_model_weights_sha256":initial,
        "final_model_weights_sha256":frozen,
        "checkpoint_file":ckpt.name,"checkpoint_sha256":sha_file(ckpt),
        "training_source_histories":len(source),
        "source_correct":correct,
        "source_total":len(source),
        "source_accuracy":correct/len(source),
        "source_prediction_order":"80 source context groups × 32 focal masks, little-endian",
        "source_prediction_labels":source_prediction,
        "source_accuracy_is_in_sample":True,
        "initial_train_loss":first,"final_train_loss":last,
        "no_development_reserved_or_confirmation_data_used":True,
    }


def native_attributions32(model,bits):
    """Top-2 GELU contributions aggregated across a COMPLETE 32-state block."""
    if bits.shape!=(32,12):
        raise ValueError("five-input world attribution requires all 32 focal counterfactuals")
    captured=[]
    handle=model.block.mlp_down.register_forward_pre_hook(
        lambda module,args: captured.append(args[0]))
    try:
        logits=model.logits(bits)
        chosen=logits.argmax(1)
        margin=(logits.gather(1,chosen[:,None]).squeeze(1)-
                logits.gather(1,(1-chosen)[:,None]).squeeze(1))
        if len(captured)!=L:
            raise AssertionError("source native visit capture incomplete")
        gradients=torch.autograd.grad(margin.sum(),captured)
    finally:
        handle.remove()
    output=[]
    for visit,(state,grad) in enumerate(zip(captured,gradients)):
        if state.shape!=(32,2,64) or grad.shape!=(32,2,64):
            raise AssertionError("physical GELU channels not 64D native")
        rank=(state[:,1,:]*grad[:,1,:]).abs().mean(dim=0)
        units=rank.topk(2).indices.tolist()
        output.extend(node(visit,int(n)) for n in units)
    return output


def source_roster(model,catalog,world):
    rows=[catalog["train"][i:i+32] for i in
          range(0,SOURCE_SITE_GROUPS*32,32)]
    attributions=[native_attributions32(model,as_tensor(g)) for g in rows]
    graph=build_graph(attributions,hub_rate=.9,min_pair_count=3,min_phi=.12)
    eligible=[]
    for cand in discover(graph,max_size=12,max_candidates=8,seed=0):
        by=Counter(decode(n)[0] for n in cand["nodes"])
        if all(1<=count<=8 for count in by.values()):
            eligible.append(cand)
    chosen=eligible[0] if eligible else None
    used_fallback=False
    if chosen is None:
        used_fallback=True
        nodes=[]
        for visit in world["focal_positions"][2:4]:
            options=[(node(visit,j),graph["frequency"].get(node(visit,j),0))
                     for j in range(64)
                     if node(visit,j) not in graph["hubs"]
                     and graph["frequency"].get(node(visit,j),0)>=2]
            options.sort(key=lambda x:(-x[1],decode(x[0])[1]))
            nodes.extend(k for k,n in options[:2])
        if len(nodes)==4:
            chosen={"id":"nec03m-source-fallback-"+sha256(
                    ",".join(sorted(nodes,key=decode)).encode()).hexdigest()[:12],
                    "nodes":sorted(nodes,key=decode),
                    "visits":world["focal_positions"][2:4],
                    "source_strategy":"top-two frequent nonhub at third/fourth focal visit"}
    report={
        "original_source_truth_world_sha256":world["world_sha256"],
        "source_context_groups":SOURCE_SITE_GROUPS,
        "focal_states_per_context":FOCAL_MASKS,
        "graph_has_qualifying_community":bool(eligible),
        "fallback_used":used_fallback,
        "source_candidate":chosen,
        "source_graph":graph,
        "source_selection_depends_on_scored_development_data":False,
    }
    return chosen,graph,report


def fit_five_topk(model,source_groups,seed):
    """40 complete 32-state source groups; exact 480-step TopK16 optimizer."""
    if len(source_groups)!=SOURCE_FIT_CONTEXTS*FOCAL_MASKS:
        raise ValueError("wrong five-input SAE fit contexts")
    original=model_digest(model)
    states=collect_source(model,source_groups,context_batch=8).reshape(-1,64)
    mean=states.mean(0)
    scale=states.std(0,unbiased=False).clamp_min(1e-3)
    data=(states-mean)/scale
    torch.manual_seed(seed+50331)
    sae=TopKSAE()
    opt=torch.optim.AdamW(sae.parameters(),lr=.001,weight_decay=0.)
    gen=torch.Generator().manual_seed(seed+50331+89)
    first=last=None
    for _ in range(SAE_STEPS):
        ids=torch.randint(len(data),(256,),generator=gen)
        recon,codes=sae(data[ids])
        loss=(recon-data[ids]).square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite five-input TopK SAE source fit")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:first=float(loss.detach())
        last=float(loss.detach())
    sae.eval()
    with torch.no_grad():
        recon,codes=sae(data)
        mse=float((recon-data).square().mean())
        mean_active=float((codes>1e-7).float().sum(1).mean())
        exact=float(((codes>1e-7).sum(1)==16).float().mean())
    if model_digest(model)!=original:
        raise AssertionError("five-input source SAE training mutated native weights")
    return sae,mean,scale,{
        "source_contexts_fit":SOURCE_FIT_CONTEXTS,
        "focal_states_per_context":FOCAL_MASKS,
        "source_input_histories_fit":len(source_groups),
        "training_activation_samples":len(data),
        "source_only":True,
        "model_weights_sha256":original,
        "sae_features":SAE_FEATURES,
        "active_codes_target":SAE_ACTIVE_K,
        "sae_training_steps":SAE_STEPS,
        "reconstruction_mse_normalized":mse,
        "observed_mean_active_codes":mean_active,
        "fraction_exactly_16_active":exact,
        "loss_initial":first,"loss_final":last,
        "normalization_mean_sha256":seal_array(mean),
        "normalization_scale_sha256":seal_array(scale),
        "development_or_reserved_contexts_seen":0,
    }


def source_direction_coverage(model,catalog,world,candidate,graph,
                              sae,mean,scale,rank):
    """Source-only 8×32 input null feasibility; never scored test labels."""
    chosen=by_visit(candidate["nodes"])
    hubs=by_visit(graph["hubs"])
    down=model.block.mlp_down.weight.detach()
    all_doses=[]
    n_angle=total=0
    context_receipts=[]
    for i in range(SOURCE_ANGLE_GROUPS):
        bits=as_tensor(catalog["train"][i*32:(i+1)*32])
        clean=traced_forward(model,bits,mode="clean")
        orientation=clean["logits"].argmax(1)
        activations=clean_activations(model,bits)
        donors={}
        masks={}
        for v in chosen:
            donors[v]={"hubs":torch.zeros(32,64)}
            mask=torch.zeros(64)
            if v in hubs:
                mask[list(sorted(hubs[v]))]=1.
            masks[v]=mask
        reference=traced_forward(
            model,bits,mode="hubs",orientation=orientation,
            donors=donors,gates=masks,derivative=True)
        sites={}
        doses_by_visit={}
        for visit,units in sorted(chosen.items()):
            left,right,routes=exhaustive_direction_match(
                sae,mean,scale,activations[:,visit,:],rank[visit],
                reference["grad_down"][visit],down,count=len(units))
            left,right,dose=attenuate_pair(left,right,down)
            n_angle+=sum(row["within_angle_gate"] for row in routes)
            total+=len(routes)
            doses_by_visit[visit]=dose
            sites[str(visit)]={
                # Preserve the complete 16-active-code physical output and
                # gradient proof for an independently re-enumerating auditor.
                "routes":routes,
                "direct_dose":dose,
                "angle_cosine_gaps":[r["absolute_cosine_gap"] for r in routes],
                "angle_matched":[r["within_angle_gate"] for r in routes],
                "candidate_subset_counts":[r["candidate_count"] for r in routes],
                "matched_shared_output_dose":dose["shared_achievable_dose"],
                "original_left_output_dose":dose["full_left_dose"],
                "original_right_output_dose":dose["full_right_dose"],
                "left_gain":dose["left_gain"],"right_gain":dose["right_gain"],
            }
        all_doses.append(doses_by_visit)
        context_receipts.append({
            "source_group":i,
            "source_context_bits":catalog["train"][i*32],
            "all_32_focal_input_masks":list(range(32)),
            "clean_chosen_class":orientation.tolist(),
            "clean_logits":clean["logits"].detach().double().tolist(),
            "sites":sites,
        })
    physical=pair_dose_stats(all_doses)
    # Prior dose helper records 48 heldout groups; this is exactly 8 source groups.
    physical["independent_source_context_count"]=SOURCE_ANGLE_GROUPS
    physical["population"]="five-input source-training-only, 8 context groups"
    return {
        "source_contexts":SOURCE_ANGLE_GROUPS,
        "focal_histories_per_source_group":32,
        "total_site_input_measurements":total,
        "angle_match_fraction":n_angle/total,
        "physical_dose":physical,
        "source_null_feasible":n_angle/total>=ANGLE_COVERAGE_FLOOR and
            physical["candidate_physically_comparable_and_signal_sufficient"],
        "source_training_label_free_control_selection":True,
        "source_contexts_and_site_receipts":context_receipts,
        "heldout_development_reserved_or_embargoed_inputs_seen":0,
    }


def run(index:int,out:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=verify_m_lock()
    if index not in INDICES:
        raise PermissionError("NEC03M source preflight restricted to first four 5-input development roots")
    w=worlds()[index]
    if w["cohort"]!="development":
        raise PermissionError("embargoed world cannot train")
    if registration()["new_five_input_development_world_specs"][index]!=w:
        raise AssertionError("five-input source registry moved")
    catalog,source=context_catalog(w)
    out.mkdir(parents=True,exist_ok=True)
    (out/"world_spec.json").write_bytes(encoded(w))
    (out/"source_split_manifest.json").write_bytes(encoded(source))
    arms={}
    for arm in ("composed","additive"):
        model,train=native_teacher(w["native_model_seed"],w,catalog["train"],out,arm)
        candidate,graph,roster=source_roster(model,catalog,w)
        frozen_bytes=encoded({
            "world_sha256":w["world_sha256"],"model_sha256":train["final_model_weights_sha256"],
            "source_sha256":source["source_sha256"],"arm":arm,"source_roster":roster})
        (out/f"{arm}.source_roster.json").write_bytes(frozen_bytes)
        site_kind=("fallback" if roster["fallback_used"] else "graph") if candidate else "none"
        sae_report=None
        angle=None
        if candidate is not None:
            fit_rows=catalog["train"][:SOURCE_FIT_CONTEXTS*32]
            rank_rows=catalog["train"][SOURCE_FIT_CONTEXTS*32:80*32]
            sae,mean,scale,sae_report=fit_five_topk(
                model,fit_rows,seed=train["training_seed"] if "training_seed" in train
                else w["native_model_seed"]+(100000 if arm=="additive" else 0))
            rank,rank_sha=dev_ranks(model,sae,mean,scale,rank_rows)
            sae_path=out/f"{arm}.topk16_source.pt"
            torch.save({
                "world_sha256":w["world_sha256"],"arm":arm,
                "model_sha256":train["final_model_weights_sha256"],
                "source_fit_context_groups":40,
                "source_rank_context_groups":40,
                "state":sae.state_dict(),
                "mean":mean,"scale":scale,
                "rank":rank,"rank_sha256":rank_sha},sae_path)
            sae_report["source_dictionary_file"]=sae_path.name
            sae_report["source_dictionary_sha256"]=sha_file(sae_path)
            sae_report["source_rank_sha256"]=rank_sha
            sae_report["source_feature_ranks_per_visit"]=rank
            sae_report["source_rank_context_groups"]=40
            sae_report["source_focal_masks_per_context"]=32
            angle=source_direction_coverage(
                model,catalog,w,candidate,graph,sae,mean,scale,rank)
        if model_digest(model)!=train["final_model_weights_sha256"]:
            raise AssertionError("five-input native checkpoint changed after source calibration")
        arms[arm]={
            "native":train,
            "source_site_kind":site_kind,
            "source_candidate":candidate,
            "source_roster_sha256":sha_file(out/f"{arm}.source_roster.json"),
            "source_training_clean_90pct":train["source_accuracy"]>=SOURCE_CLEAN_FLOOR,
            "SAE_source":sae_report,
            "angle_source":angle,
            "source_model_eligible":(
                train["source_accuracy"]>=SOURCE_CLEAN_FLOOR and
                candidate is not None and sae_report is not None and
                sae_report["reconstruction_mse_normalized"]<=SAE_RECON_FLOOR and
                angle is not None and angle["source_null_feasible"])
        }
    receipt={
        "schema":SCHEMA,"world_id":w["world_id"],"world_index":index,
        "world_sha256":w["world_sha256"],
        "five_input_registry_sha256":registration()["registry_sha256"],
        "original_four_input_confirmation_spec_sha256":
            old_registry()["confirmation_full_spec_sha256"],
        "source_manifest_sha256":source["source_sha256"],
        "NEC03M_source_lock_sha256":lock["lock_sha256"],
        "native_steps_per_arm":NATIVE_STEPS,
        "SAE_steps_per_candidate_arm":SAE_STEPS,
        "source_focal_bit_masks":FOCAL_MASKS,
        "source_parameter_context_groups":80,
        "source_SAE_fit_context_groups":40,
        "source_SAE_rank_context_groups":40,
        "source_direction_context_groups":SOURCE_ANGLE_GROUPS,
        "teacher_models_trained_from_independent_seeds":True,
        "source_results_only":True,
        "development_evaluation_context_groups_scored":0,
        "reserved_context_groups_scored":0,
        "new_embargoed_worlds_scored":0,
        "original_confirmation_worlds_scored":0,
        "arms":arms,
        "root_ready_for_future_development_scoring":all(
            a["source_model_eligible"] for a in arms.values()),
        "evidence_class":"source training competence and null feasibility only",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"source_pilot_receipt.json").write_bytes(encoded(receipt))
    print("NEC03M_NEW_FIVE_INPUT_SOURCE_ONLY_PILOT_WRITTEN")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--development-world-index",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    run(args.development_world_index,args.out)


if __name__=="__main__":
    main()
