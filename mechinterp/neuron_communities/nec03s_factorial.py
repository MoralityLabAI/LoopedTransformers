"""NEC-03S: registered source-only affine-offset x TopK active-budget factorial.

All eight original dev5-000..007 model roots, two model arms each. Exact
NEC03O reset1800 native and source SAE checkpoints; 8 original native
source-training groups 16..23 x 32 focal masks, never evaluation groups.

Primary: difference-in-differences (source-sham with common affine
offset minus source-sham without offset), composed minus additive,
original source n budget, 8-root ITT including frozen zero roots 3,5.
All four primary donor vectors use the SAME 32D output-L2 dose per
visit/input, anchored to their pre-outcome minimum. No dose amplification.
"""
from __future__ import annotations

import argparse
from hashlib import sha1,sha256
from itertools import combinations
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.matched_dose import by_visit
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor,model_digest

SCHEMA="NEC-03S/source-only-offset-budget-four-way-physical-dose/v1"
LOCK_NAME="NEC03S_SOURCE_LOCK.json"
ROOTS=tuple(range(8))
PANEL=tuple(range(16,24))
PAIRED_ELIGIBLE=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
CASES=256
FOCAL=32
ANGLE_LIMIT=.25
EFFECT_FLOOR=.02
HASH_SALT=b"NEC03S-fixed-hash-disjoint-source-budget-v1"
BUDGETS=("orig","wide")
OFFSETS=(0,1)
BASE_MODES=(
    "clean","hubs","native_nonhub","offset_only","full16_0",
    "full16_1","natural_orig_0","natural_orig_1",
    "natural_wide_0","natural_wide_1",
)
FACTOR_MODES=tuple(
    f"{budget}_{suffix}" for budget in BUDGETS for suffix in (
        "src0","sham0","src1","sham1",
        "random_src0","random_sham0","random_src1","random_sham1",
    )
)
MODES=BASE_MODES+FACTOR_MODES


def load(path:Path):
    return json.loads(path.read_text())


def verify_lock():
    path=Path(__file__).with_name(LOCK_NAME)
    cfg=load(path)
    required={
        "schema":"NEC-03S/frozen-source-offset-active-budget-registration/v1",
        "five_input_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "original_world_roots":list(ROOTS),
        "original_source_group_indices":list(PANEL),
        "original_source_history_count_per_arm":CASES,
        "focal_states_per_group":FOCAL,
        "paired_source_eligible_roots":list(PAIRED_ELIGIBLE),
        "paired_source_ineligible_ITT_zero_roots":list(ITT_ZERO),
        "sae_active_k":16,
        "native_source_a_run":44898361742,
        "sae_source_a_run":44899750508,
        "native_sae_source_b_run":44900628360,
        "budgets":list(BUDGETS),
        "offset_conditions":list(OFFSETS),
        "intervention_mode_names":list(MODES),
        "sham_search":"full disjoint k-combinations; minimize maximum of no-offset and offset 32D cosine gaps",
        "random_hash_salt":HASH_SALT.decode(),
        "dose_anchor":"minimum of four source/sham offset/no-offset physical output L2 norms; one anchor across all four",
        "offset_masked_to_native_nonhubs":True,
        "disjoint_pair_budget_max":8,
        "angle_cosine_gate_both_offsets":ANGLE_LIMIT,
        "primary_comparisons":1,
        "primary_minimum_meaningful_correctness_fraction":EFFECT_FLOOR,
        "native_and_sae_training_updates":0,
        "development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_source_recruited_worlds_scored":0,
        "embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise AssertionError("NEC03S source-only factorial protocol changed")
    root=Path(__file__).resolve().parents[2]
    for file,pinned in cfg["git_blob_sha1"].items():
        raw=(root/file).read_bytes()
        digest=sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()
        if digest!=pinned:
            raise AssertionError("NEC03S frozen source code changed: "+file)
    return cfg,sha_file(path)


def budget_width(n:int,name:str)->int:
    if not 1<=n<=8 or name not in BUDGETS:
        raise ValueError("unsupported source-site active-code budget")
    return n if name=="orig" else min(2*n,8)


def hash_control(options:list[tuple[int,...]],*,world:str,arm:str,
                 group:int,mask:int,visit:int,budget:str)->tuple[int,...]:
    if not options:
        raise AssertionError("cannot sample a nonexistent disjoint active control")
    info=f"{world}|{arm}|{group}|{mask}|{visit}|{budget}".encode()
    selected=int.from_bytes(sha256(HASH_SALT+b"|"+info).digest(),"big")%len(options)
    return options[selected]


def cosine32(v:torch.Tensor,g:torch.Tensor)->torch.Tensor:
    return (v*g).sum(dim=-1)/(
        v.norm(dim=-1)*g.norm(dim=-1)).clamp_min(1e-10)


def attenuate_to_shared_four(donors:dict[str,torch.Tensor],down:torch.Tensor):
    """Attenuate all four donors to their single shared min 32D output dose."""
    if set(donors)!={"src0","sham0","src1","sham1"}:
        raise ValueError("four-source-donor equal-dose family required")
    first=donors["src0"]
    if any(x.shape!=first.shape for x in donors.values()):
        raise ValueError("four matched donor tensors need identical shape")
    norms={k:(v@down.T).norm(dim=1) for k,v in donors.items()}
    shared=torch.stack(list(norms.values())).min(dim=0).values
    gains={k:torch.where(
        norms[k]>1e-10,shared/norms[k].clamp_min(1e-10),
        torch.zeros_like(shared)).clamp(0,1)
        for k in donors}
    scaled={k:donors[k]*gains[k][:,None] for k in donors}
    physical={(k):(v@down.T).norm(dim=1) for k,v in scaled.items()}
    matched=torch.ones(len(shared),dtype=torch.bool)
    for k in donors:
        mismatch=(physical[k]-shared).abs()/shared.clamp_min(1e-6)
        matched &= mismatch<=.01
        if bool((gains[k]>1+1e-6).any()):
            raise AssertionError("source/directional sham was amplified")
    return scaled,{
        "norms":{k:v.tolist() for k,v in norms.items()},
        "shared":shared.tolist(),
        "gains":{k:v.tolist() for k,v in gains.items()},
        "matched_within_1pct":matched.tolist(),
        "nonzero_shared":(shared>1e-6).tolist(),
    }


def attenuate_random_pair(source:torch.Tensor,random:torch.Tensor,down:torch.Tensor):
    norms=torch.stack(((source@down.T).norm(dim=1),
                       (random@down.T).norm(dim=1)))
    shared=norms.min(dim=0).values
    g0=torch.where(norms[0]>1e-10,shared/norms[0].clamp_min(1e-10),
                   torch.zeros_like(shared)).clamp(0,1)
    g1=torch.where(norms[1]>1e-10,shared/norms[1].clamp_min(1e-10),
                   torch.zeros_like(shared)).clamp(0,1)
    return source*g0[:,None],random*g1[:,None],{
        "source_norm":norms[0].tolist(),"random_norm":norms[1].tolist(),
        "shared":shared.tolist(),"source_gain":g0.tolist(),
        "random_gain":g1.tolist(),"nonzero_shared":(shared>1e-6).tolist(),
    }


def group_measure(prepared:dict,group:int,rows):
    if group not in PANEL or len(rows)!=FOCAL:
        raise PermissionError("only 8 complete original source groups 16..23")
    world=prepared["world"]
    arm=prepared["arm"]
    model=prepared["model"]
    x=as_tensor(rows)
    clean=traced_forward(model,x,mode="clean")
    direction=clean["logits"].argmax(dim=1)
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    donor_hub={}
    gates={}
    for visit in sites:
        mask=torch.zeros(64)
        if visit in hubs:mask[sorted(hubs[visit])]=1.
        gates[visit]=mask
        donor_hub[visit]={"hubs":torch.zeros(FOCAL,64)}
    baseline=traced_forward(
        model,x,mode="hubs",orientation=direction,
        donors=donor_hub,gates=gates,derivative=bool(sites))
    acts=clean_activations(model,x) if sites else None
    variants={mode:{} for mode in MODES if mode not in ("clean","hubs")}
    valid={name:torch.full((FOCAL,),bool(sites),dtype=torch.bool)
           for name in BUDGETS}
    valid_random={name:torch.full((FOCAL,),bool(sites),dtype=torch.bool)
                  for name in BUDGETS}
    supported={name:torch.full((FOCAL,),bool(sites),dtype=torch.bool)
               for name in BUDGETS}
    geometry={}
    down=model.block.mlp_down.weight.detach()
    for visit,site_units in sorted(sites.items()):
        states=acts[:,visit,:]
        nonhub=(1-gates[visit])
        sae=prepared["sae"]
        with torch.no_grad():
            z=sae.codes((states-prepared["mean"])/prepared["scale"])
            physical_decoder=sae.decoder.weight.detach().T*prepared["scale"][None,:]
            feature_state=z@physical_decoder
            offset=(prepared["mean"]+
                    sae.decoder.bias.detach()*prepared["scale"])*nonhub
        offset_full=offset.unsqueeze(0).expand(FOCAL,-1)
        raw_native=states*nonhub
        full0=feature_state*nonhub
        full1=(feature_state+offset)*nonhub
        variants["native_nonhub"][visit]=raw_native
        variants["offset_only"][visit]=offset_full
        variants["full16_0"][visit]=full0
        variants["full16_1"][visit]=full1
        grad=baseline["grad_down"][visit]
        rank={f:i for i,f in enumerate(prepared["ranks"][visit])}
        all_active=[]
        route_info={}
        for budget in BUDGETS:
            k=budget_width(len(site_units),budget)
            vals={kind:torch.zeros(FOCAL,64) for kind in (
                "src0","sham0","src1","sham1","random0","random1")}
            meta={
                "source_ids":[],"sham_ids":[],"random_ids":[],
                "all_active_ids":[],"candidate_count":[],
                "cosine_gap_without_offset":[],"cosine_gap_with_offset":[],
                "joint_minimax_gap":[],"source_support":[],
            }
            for mask in range(FOCAL):
                active=torch.where(z[mask]>1e-7)[0].tolist()
                meta["all_active_ids"].append(active)
                enough=len(active)==16
                meta["source_support"].append(enough)
                if not enough:
                    for key in ("source_ids","sham_ids","random_ids"):
                        meta[key].append([])
                    meta["candidate_count"].append(0)
                    for key in ("cosine_gap_without_offset","cosine_gap_with_offset",
                                "joint_minimax_gap"):meta[key].append(None)
                    supported[budget][mask]=False
                    valid[budget][mask]=False
                    valid_random[budget][mask]=False
                    continue
                src_ids=sorted(active,key=lambda i:rank[i])[:k]
                others=sorted(set(active)-set(src_ids))
                options=list(combinations(others,k))
                if not options:
                    raise AssertionError("predeclared disjoint control budget unavailable")
                src=(z[mask,src_ids]@physical_decoder[src_ids])*nonhub
                idx=torch.tensor(options,dtype=torch.long)
                choices=(z[mask,idx].unsqueeze(-1)*
                         physical_decoder[idx]).sum(dim=1)*nonhub
                p_src0=src@down.T
                p_src1=(src+offset)@down.T
                p_choice0=choices@down.T
                p_choice1=(choices+offset)@down.T
                g=grad[mask]
                cos_src0=cosine32(p_src0,g)
                cos_src1=cosine32(p_src1,g)
                gaps0=(cosine32(p_choice0,g)-cos_src0).abs()
                gaps1=(cosine32(p_choice1,g)-cos_src1).abs()
                minimax=torch.maximum(gaps0,gaps1)
                selected=int(torch.argmin(minimax))
                ids=list(options[selected])
                rnd_ids=list(hash_control(options,world=world["world_sha256"],
                        arm=arm,group=group,mask=mask,visit=visit,budget=budget))
                sham=z[mask,ids]@physical_decoder[ids]*nonhub
                random=z[mask,rnd_ids]@physical_decoder[rnd_ids]*nonhub
                vals["src0"][mask]=src
                vals["sham0"][mask]=sham
                vals["src1"][mask]=src+offset
                vals["sham1"][mask]=sham+offset
                vals["random0"][mask]=random
                vals["random1"][mask]=random+offset
                meta["source_ids"].append(src_ids)
                meta["sham_ids"].append(ids)
                meta["random_ids"].append(rnd_ids)
                meta["candidate_count"].append(len(options))
                meta["cosine_gap_without_offset"].append(float(gaps0[selected]))
                meta["cosine_gap_with_offset"].append(float(gaps1[selected]))
                meta["joint_minimax_gap"].append(float(minimax[selected]))
                if (float(minimax[selected])>ANGLE_LIMIT or
                    float(g.norm())<=1e-7):
                    valid[budget][mask]=False
            variants[f"natural_{budget}_0"][visit]=vals["src0"]
            variants[f"natural_{budget}_1"][visit]=vals["src1"]
            four={k:vals[k] for k in ("src0","sham0","src1","sham1")}
            scaled,dose=attenuate_to_shared_four(four,down)
            good=torch.tensor([
                bool(dose["matched_within_1pct"][i] and
                     dose["nonzero_shared"][i])
                for i in range(FOCAL)])
            valid[budget] &= good
            for name in ("src0","sham0","src1","sham1"):
                variants[f"{budget}_{name}"][visit]=scaled[name]
            random_doses={}
            for offset_flag in OFFSETS:
                source,sham,receipt=attenuate_random_pair(
                    vals[f"src{offset_flag}"],vals[f"random{offset_flag}"],down)
                variants[f"{budget}_random_src{offset_flag}"][visit]=source
                variants[f"{budget}_random_sham{offset_flag}"][visit]=sham
                valid_random[budget] &= torch.tensor(receipt["nonzero_shared"])
                random_doses[str(offset_flag)]=receipt
            route_info[budget]={
                "k":k,**meta,"primary_four_way_dose":dose,
                "random_pair_dose":random_doses,
            }
        geometry[str(visit)]={
            "original_site_units":sorted(site_units),
            "original_hubs":sorted(hubs.get(visit,[])),
            "offset_64d":offset.tolist(),
            "natural_nonhub_dose":{
                "native":(raw_native@down.T).norm(dim=1).tolist(),
                "full16_0":(full0@down.T).norm(dim=1).tolist(),
                "full16_1":(full1@down.T).norm(dim=1).tolist(),
                "offset_only":(offset@down.T).norm().item(),
            },
            "sae_full16_relative_32d_recon_error":(
                ((full1-raw_native)@down.T).norm(dim=1)/
                (raw_native@down.T).norm(dim=1).clamp_min(1e-6)).tolist(),
            "budget_geometry":route_info,
        }
    for visit in sites:
        for budget in BUDGETS:
            allow=supported[budget][:,None]
            variants[f"natural_{budget}_0"][visit]*=allow
            variants[f"natural_{budget}_1"][visit]*=allow
            for kind in ("src0","sham0","src1","sham1"):
                variants[f"{budget}_{kind}"][visit]*=valid[budget][:,None]
            for offset_flag in OFFSETS:
                variants[f"{budget}_random_src{offset_flag}"][visit]*=valid_random[budget][:,None]
                variants[f"{budget}_random_sham{offset_flag}"][visit]*=valid_random[budget][:,None]
    traces={"clean":clean,"hubs":baseline}
    for mode,by_visit in variants.items():
        donors={visit:{"native":val} for visit,val in by_visit.items()}
        with torch.no_grad():
            traces[mode]=traced_forward(model,x,mode="native",
                    orientation=direction,donors=donors,gates=gates)
    for budget in BUDGETS:
        bad=~valid[budget]
        if not torch.allclose(traces[f"{budget}_src0"]["logits"][bad],
                              traces[f"{budget}_sham0"]["logits"][bad],atol=1e-6):
            raise AssertionError("invalid no-offset input generated primary contrast")
        if not torch.allclose(traces[f"{budget}_src1"]["logits"][bad],
                              traces[f"{budget}_sham1"]["logits"][bad],atol=1e-6):
            raise AssertionError("invalid offset input generated primary contrast")
    return {
        "schema":SCHEMA,"original_source_group_index":group,
        "input_histories":rows,"focal_mask_order":list(range(FOCAL)),
        "clean_chosen_orientation":direction.tolist(),
        "teacher_truth_labels_used_for_donor_routing":False,
        "source_site_geometry":geometry,
        "valid_all_visits":{k:v.tolist() for k,v in valid.items()},
        "valid_random_all_visits":{k:v.tolist() for k,v in valid_random.items()},
        "logits":{mode:traces[mode]["logits"].tolist() for mode in MODES},
        "memory_drift_mean_from_hubs":{
            mode:(traces[mode]["memory"]-baseline["memory"]).norm(dim=-1).mean(dim=0).tolist()
            for mode in MODES},
    }


def run(index:int,source:Path,out:Path,*,reset:Path|None=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock,locksha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("NEC03S source-only eight-world cohort locked")
    world=worlds()[index]
    out.mkdir(parents=True,exist_ok=True)
    arms={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        prepared["arm"]=arm
        if prepared["world"]["world_sha256"]!=world["world_sha256"]:
            raise AssertionError("wrong source root checkpoint")
        raw=[]
        totals={mode:0 for mode in MODES}
        validity={budget:0 for budget in BUDGETS}
        for group in PANEL:
            rows=prepared["catalog"]["train"][group*FOCAL:(group+1)*FOCAL]
            record=group_measure(prepared,group,rows)
            y=labels(as_tensor(rows),world,control=(arm=="additive")).tolist()
            predicted={}
            for mode in MODES:
                pred=torch.tensor(record["logits"][mode]).argmax(dim=1).tolist()
                predicted[mode]=pred
                totals[mode]+=sum(int(p==q) for p,q in zip(pred,y))
            for budget in BUDGETS:
                validity[budget]+=sum(record["valid_all_visits"][budget])
            record["true_teacher_labels"]=y
            record["class_predictions"]=predicted
            record["arm"]=arm
            raw.append(record)
        payload=b"".join(encoded(row) for row in raw)
        path=out/f"{arm}.nec03s_source_factorial.jsonl"
        path.write_bytes(payload)
        def delta(budget):
            return ((totals[f"{budget}_src1"]-totals[f"{budget}_sham1"])-
                    (totals[f"{budget}_src0"]-totals[f"{budget}_sham0"]))
        arms[arm]={
            "arm":arm,"model_weights_sha256":prepared["model_sha256"],
            "sae_checkpoint_sha256":prepared["sae_sha256"],
            "source_rank_sha256":prepared["rank_sha256"],
            "source_report_sha256":prepared["source_report_sha256"],
            "original_source_candidate_present":bool(prepared["source_candidate"]),
            "original_arm_source_eligible":prepared["original_arm_source_eligible"],
            "source_training_histories":CASES,
            "source_groups_scored":list(PANEL),
            "correctness":totals,
            "available_common_dose_input_cases":validity,
            "offset_interaction_correct_difference":{k:delta(k) for k in BUDGETS},
            "signed_clean_hubs_gap":totals["clean"]-totals["hubs"],
            "raw_filename":path.name,"raw_sha256":sha256(payload).hexdigest(),
        }
        if model_digest(prepared["model"])!=prepared["model_sha256"]:
            raise AssertionError("frozen native source model mutated during offset factorial")
    raw_did=(arms["composed"]["offset_interaction_correct_difference"]["orig"]-
             arms["additive"]["offset_interaction_correct_difference"]["orig"])/CASES
    paired=index in PAIRED_ELIGIBLE
    report={
        "schema":SCHEMA,
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "five_input_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "original_source_manifest_sha256":context_catalog(world)[1]["source_sha256"],
        "source_lock_sha256":locksha,
        "source_groups_scored":list(PANEL),
        "training_source_case_count_per_arm":CASES,
        "arms":arms,
        "original_paired_source_eligible":paired,
        "preregistered_source_ITT_failure":index in ITT_ZERO,
        "raw_primary_original_budget_interaction_DID":raw_did,
        "primary_ITT_root_interaction_DID":raw_did if paired else 0.,
        "primary_familywise_comparisons":1,
        "minimum_meaningful_absolute_effect":EFFECT_FLOOR,
        "native_and_sae_training_updates":0,
        "development_eval_groups_scored":0,"reserved_groups_scored":0,
        "new_source_recruited_worlds_scored":0,
        "embargoed_worlds_scored":0,"original_four_input_confirmation_worlds_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec03s_source_factorial_report.json").write_bytes(encoded(report))
    print("NEC03S_ORIGINAL_SOURCE_OFFSET_FACTORIAL_COMPLETE",world["world_id"])
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.world_index,a.source_root,a.out,reset=a.reset_root)
