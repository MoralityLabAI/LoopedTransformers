"""NEC-03G: CPU development-root pilot on fresh Boolean-composition worlds.

One world has a unique Boolean function and focal-position quartet, an
independently seeded native PEF-07-sized tied model, and an additive
prethreshold-class-balance control. Each model gets source-trained TopK16
SAE and fixed source-only neuron community. The confirmation cohort is
never admitted by this runner. It never opens a world reserved split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.core import build_graph,discover,decode
from mechinterp.neuron_communities.nec03_development import hub_disjoint_sham
from mechinterp.neuron_communities.nec03f_active import build_pair
from mechinterp.neuron_communities.nec03e_pairs import pair_dose_stats
from mechinterp.neuron_communities.matched_dose import graft_forward
from mechinterp.neuron_communities.pef07_adapter import context_attributions
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.topk_sae import fit_topk,dev_ranks
from mechinterp.neuron_communities.nec03g_worlds import (
    DEV_COUNT, MIN_CLEAN_ACC, PRIMARY_MEANINGFUL_DELTA, SCHEMA,
    additive_control_truth, context_catalog, digest, encoded, registry,
    target_bits, worlds,
)
from research.pef07.pef07 import Looper, as_tensor, model_digest

PAIRS=(("native","sham"),("native","topk"),("topk","topk_alt"))
ROSTER_CONTEXTS=80
GRAPH_MIN_PHI=0.12
GRAPH_MIN_JOINT=3
MAX_COMMUNITY_SIZE=12
MAX_COMMUNITIES_INSPECTED=8
SAE_TOPK=16


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def train_one(world: dict, catalog: dict, root: Path, *, arm: str,
              steps: int | None = None) -> tuple[Looper, dict]:
    """Freeze native weights on source-training examples before ANY eval."""
    if arm not in ("composed","additive"):
        raise ValueError("unknown teacher arm")
    if world["cohort"]!="development":
        raise PermissionError("confirmation world training is sealed")
    total=world["trunk_steps"] if steps is None else steps
    if not 1<=total<=world["trunk_steps"]:
        raise ValueError("invalid CPU optimizer steps")
    seed=world["trunk_seed"]+(100000 if arm=="additive" else 0)
    torch.manual_seed(seed)
    model=Looper().train()
    initial=model_digest(model)
    source=as_tensor(catalog["train"])
    labels=target_bits(source,world,control=arm=="additive")
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0)
    gen=torch.Generator().manual_seed(seed*7919+59)
    first=last=None
    for step in range(total):
        ids=torch.randint(len(source),(128,),generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(source[ids]),labels[ids])
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite native train loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if first is None:
            first=float(loss.detach())
        last=float(loss.detach())
    model.eval()
    frozen_hash=model_digest(model)
    root.mkdir(parents=True,exist_ok=True)
    saved=root/f"{arm}.native.pt"
    payload={"model":model.state_dict(),"world_id":world["world_id"],
             "world_sha256":world["world_sha256"],"seed":seed,
             "teacher":arm,"steps":total,"model_sha256":frozen_hash,
             "trunk_class":"PEF07/Looper/32-D-12-tied-visits"}
    torch.save(payload,saved)
    manifest={
        "arm":arm,"teacher_truth_table":
            additive_control_truth(world["truth_table_16"])
            if arm=="additive" else world["truth_table_16"],
        "checkpoint_name":saved.name,
        "checkpoint_sha256":sha_file(saved),
        "model_weights_sha256":frozen_hash,
        "seed":seed,"initial_weights_sha256":initial,
        "steps":total,"first_loss":first,"final_loss":last,
        "frozen_before_discovery":True,
        "trained_on_source_train_only":True}
    return model,manifest


@torch.no_grad()
def clean_score(model, rows, world, *, arm):
    correct=0
    truth=0
    for start in range(0,len(rows),16*8):
        bits=as_tensor(rows[start:start+16*8])
        y=target_bits(bits,world,control=arm=="additive")
        logits=model.logits(bits)
        correct+=int((logits.argmax(1)==y).sum())
        truth+=len(y)
    return {"correct":correct,"total":truth,"accuracy":correct/truth}


def source_community(model, catalog: dict):
    """One highest-ranked source-only community; never search eval scores."""
    rows=[catalog["train"][j:j+16] for j in range(0,16*ROSTER_CONTEXTS,16)]
    attributions=[context_attributions(model,as_tensor(r),top_k=2)
                  for r in rows]
    graph=build_graph(attributions,hub_rate=0.90,
                      min_pair_count=GRAPH_MIN_JOINT,
                      min_phi=GRAPH_MIN_PHI)
    candidates=discover(graph,max_size=MAX_COMMUNITY_SIZE,
                        max_candidates=MAX_COMMUNITIES_INSPECTED)
    selected=None
    for c in candidates:
        counts=Counter(decode(k)[0] for k in c["nodes"])
        if all(n<=SAE_TOPK//2 for n in counts.values()):
            selected=c
            break
    return selected,graph,{
        "graph":graph,"candidates_observed_before_eval":len(candidates),
        "selection_rule":"first source graph candidate with <=8 native occurrences per visit",
        "candidate":selected,
        "candidate_selection_contexts":ROSTER_CONTEXTS,
        "source_only":True,
    }


def run_arm(world: dict, catalog: dict, source: dict, out: Path,
            *, arm: str, steps: int, sae_steps: int):
    model,trained=train_one(world,catalog,out,arm=arm,steps=steps)
    native_hash=model_digest(model)
    # First freeze the source-only graph, before any development evaluation.
    candidate,graph,source_discovery=source_community(model,catalog)
    graph_bytes=encoded({"world_sha256":world["world_sha256"],
                         "model_sha256":native_hash,
                         "source_sha256":source["sha256"],
                         "arm":arm,"source_discovery":source_discovery})
    (out/f"{arm}.community_source_freeze.json").write_bytes(graph_bytes)
    graph_sha=sha_bytes(graph_bytes)
    if candidate is None:
        score=clean_score(model,catalog["development"],world,arm=arm)
        report={"arm":arm,"train":trained,"clean":score,
                "source_graph_freeze_sha256":graph_sha,
                "source_discovery":source_discovery,
                "candidate_status":"NO_SOURCE_COMMUNITY",
                "pairs":[],"primary_development_effect":"0",
                "source_manifest_sha256":source["sha256"],
                "raw_paired_context_file":None,
                "raw_paired_context_sha256":None,
                "raw_paired_records":0,
                "world_sha256":world["world_sha256"]}
        (out/f"{arm}.report.json").write_bytes(encoded(report))
        return report
    model_path=out/f"{arm}.sae.pt"
    sae,sae_receipt,mean,scale=fit_topk(
        model,catalog["train"][:80*16],seed=trained["seed"],steps=sae_steps)
    ranks,rank_sha=dev_ranks(model,sae,mean,scale,
                             catalog["train"][80*16:160*16])
    torch.save({"state":sae.state_dict(),"mean":mean,"scale":scale,
                "source_contexts_fit":source["contexts_ordered"][:80],
                "source_contexts_rank":source["contexts_ordered"][80:160],
                "model_sha256":native_hash},model_path)
    sae_receipt=dict(sae_receipt)
    sae_receipt["source_sae_sha256"]=sha_file(model_path)
    sae_receipt["source_feature_rank_sha256"]=rank_sha
    sae_receipt["source_fit_contexts"]=80
    sae_receipt["source_rank_contexts"]=80
    sae_receipt["test_and_reserved_contexts_used"]=0
    policy={"hub_disjoint_native_sham":hub_disjoint_sham(candidate,graph)}
    # Frozen policy/SAE/graph receipt goes to disk *before* opening eval.
    freeze={
        "schema":SCHEMA,"world_sha256":world["world_sha256"],
        "model_sha256":native_hash,
        "source_manifest_sha256":source["sha256"],
        "source_graph_candidate":candidate,"graph_hubs":graph["hubs"],
        "graph_full_sha256":digest(graph),
        "source_sae":sae_receipt,
        "source_ranking_per_visit":ranks,
        "native_sham_policy":policy,
        "no_eval_or_reserved_contexts_used_for_selection":True}
    frozen_raw=encoded(freeze)
    freeze_path=out/f"{arm}.source_freeze.json"
    freeze_path.write_bytes(frozen_raw)
    frozen_sha=sha_bytes(frozen_raw)
    score=clean_score(model,catalog["development"],world,arm=arm)
    # Measure every pair, including dose-infeasible and negative endpoints.
    effects=[]
    raw=[]
    for pair in PAIRS:
        errors={"clean":0,"hubs":0,"left":0,"right":0}
        context_effects=[]
        dosage=[]
        for group_index in range(48):
            quartet=catalog["development"][16*group_index:16*(group_index+1)]
            bits=as_tensor(quartet)
            y=target_bits(bits,world,control=arm=="additive")
            with torch.no_grad():
                z=model.logits(bits)
            act=clean_activations(model,bits)
            donors,doses,hubs,selected=__import__(
                "mechinterp.neuron_communities.nec03f_active",
                fromlist=["build_pair"]).build_pair(
                    model,act,candidate,policy,ranks,sae,mean,scale,
                    graph["hubs"],pair)
            with torch.no_grad():
                h=graft_forward(model,bits,donors,hubs,"hubs")
                left=graft_forward(model,bits,donors,hubs,"native")
                right=graft_forward(model,bits,donors,hubs,"sham")
            for name,logit in (("clean",z),("hubs",h),
                               ("left",left),("right",right)):
                errors[name]+=int((logit.argmax(1)!=y).sum())
            context_effects.append(float(
                ((left.argmax(1)==y).float()-(right.argmax(1)==y).float()).mean()))
            dosage.append(doses)
            context_positions=source["context_positions"]
            raw.append({"world_id":world["world_id"],"arm":arm,
                        "pair":list(pair),"source_context_index":group_index,
                        "source_context_bits":[quartet[0][k] for k in context_positions],
                        "focal_mask_order":list(range(16)),
                        "labels":y.tolist(),
                        "clean_logits":z.detach().double().tolist(),
                        "hubs_logits":h.double().tolist(),
                        "left_logits":left.double().tolist(),
                        "right_logits":right.double().tolist(),
                        "local_physical_doses":{str(k):v for k,v in doses.items()},
                        "active_features":{str(k):v for k,v in selected.items()}})
        site=pair_dose_stats(dosage)
        effects.append({
            "pair":list(pair),
            "errors_of_768":errors,
            "left_minus_right_correct":errors["right"]-errors["left"],
            "left_minus_right_accuracy":(errors["right"]-errors["left"])/768,
            "physical_dose":site,
            "paired_context_differences":context_effects,
        })
    raw_bytes=b"".join(encoded(r) for r in raw)
    filename=f"{arm}.raw_pairs.jsonl"
    (out/filename).write_bytes(raw_bytes)
    if model_digest(model)!=native_hash:
        raise AssertionError("frozen PEF-07 native weights changed during causal replay")
    report={
        "arm":arm,"train":trained,"clean":score,
        "source_discovery":source_discovery,
        "candidate_status":"SOURCE_SELECTED",
        "source_freeze_sha256":frozen_sha,
        "source_graph_freeze_sha256":graph_sha,
        "source_only_sae":sae_receipt,
        "source_manifest_sha256":source["sha256"],
        "world_sha256":world["world_sha256"],
        "pairs":effects,
        "raw_paired_context_file":filename,
        "raw_paired_context_sha256":sha_bytes(raw_bytes),
        "raw_paired_records":len(raw),
        "development_contexts":48,
        "reserved_contexts_opened":0,
        "all_selection_frozen_before_evaluation":True}
    (out/f"{arm}.report.json").write_bytes(encoded(report))
    return report


def root_effects(world, reports):
    """Intention-to-treat: invalid competency, dose or source => zero effect."""
    reasons=[]
    for arm in ("composed","additive"):
        r=reports[arm]
        if r["clean"]["accuracy"] < MIN_CLEAN_ACC:
            reasons.append(arm + "_incompetent_clean_head")
        if r["candidate_status"]!="SOURCE_SELECTED":
            reasons.append(arm + "_no_source_candidate")
    extracted={}
    if not reasons:
        for arm in ("composed","additive"):
            pairs={tuple(p["pair"]):p for p in reports[arm]["pairs"]}
            for label,pair in (("primary",("topk","topk_alt")),
                               ("native_sham",("native","sham")),
                               ("native_topk",("native","topk"))):
                record=pairs[pair]
                if not record["physical_dose"]["candidate_physically_comparable_and_signal_sufficient"]:
                    reasons.append(f"{arm}_{label}_insufficient_physical_signal")
                extracted[(arm,label)]=record["left_minus_right_accuracy"]
    if reasons:
        return {"root_primary_effect":"0","sham_effects":["0"],
                "observed_primary_development_effect":None,
                "observed_control_development_effect":None,
                "planned_world_valid":False,"failure_reasons":sorted(set(reasons)),
                "effect_class":"intention-to-treat-zero-for-failed-world"}
    primary=extracted["composed","primary"]
    control=extracted["additive","primary"]
    sham_native=extracted["composed","native_sham"]
    sham_additive=extracted["additive","native_sham"]
    return {"root_primary_effect":str(round(primary,12)),
            "sham_effects":[str(round(x,12))
                            for x in (control,sham_native,sham_additive)],
            "observed_primary_development_effect":primary,
            "observed_control_development_effect":control,
            "observed_specificity_difference_in_differences":primary-control,
            "observed_native_vs_topk_composed":
                extracted["composed","native_topk"],
            "planned_world_valid":True,"failure_reasons":[],
            "effect_class":"composed-teacher-topk-versus-active-alternative"}


def run(world_index: int, out: Path, *, trunk_steps: int=450,
        sae_steps: int=480):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if world_index not in range(DEV_COUNT):
        raise PermissionError("confirmation world run forbidden by this development runner")
    world=worlds()[world_index]
    if world["cohort"]!="development":
        raise AssertionError("world cohort mismatch")
    cohort=registry()
    if cohort["dev_full_world_specs"][world_index]!=world:
        raise AssertionError("source registry mismatch")
    out.mkdir(parents=True,exist_ok=True)
    catalog,source=context_catalog(world)
    (out/"context_source_manifest.json").write_bytes(encoded(source))
    (out/"world_spec.json").write_bytes(encoded(world))
    arm_reports={}
    for arm in ("composed","additive"):
        arm_reports[arm]=run_arm(
            world,catalog,source,out,arm=arm,
            steps=trunk_steps,sae_steps=sae_steps)
    effect=root_effects(world,arm_reports)
    full=(trunk_steps==world["trunk_steps"] and sae_steps==world["sae_steps"])
    if not full:
        effect["effect_class"]="engineering-short-run-not-for-power"
    summary={
        "schema":SCHEMA,"world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "cohort":"development",
        "registry_sha256":cohort["registry_sha256"],
        "source_sha256":source["sha256"],
        "trunk_steps":trunk_steps,"sae_steps":sae_steps,
        "full_registered_development_recipe":full,
        "model_and_control_trained_independently":True,
        "original_confirmation_worlds_evaluated":0,
        "world_reserved_context_groups_evaluated":0,
        "trunk_model_clean_scores":{
            arm:arm_reports[arm]["clean"] for arm in arm_reports},
        "candidate_source_status":{
            arm:arm_reports[arm]["candidate_status"] for arm in arm_reports},
        "primary_effect":effect,
        "independent_world_id":world["world_id"],
        "failure_included_in_intention_to_treat":True,
        "FSI_01_class":"DEVELOPMENT_PILOT_ONLY",
    }
    (out/"root_summary.json").write_bytes(encoded(summary))
    print(json.dumps({"world":world["world_id"],"steps":trunk_steps,
                      "recipe_full":full,
                      "clean":{a:r["clean"]["correct"] for a,r in arm_reports.items()},
                      "candidates":summary["candidate_source_status"],
                      "primary_world_effect":effect["root_primary_effect"],
                      "valid":effect["planned_world_valid"],
                      "reasons":effect["failure_reasons"]},sort_keys=True),flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-world-index",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--trunk-steps",type=int,default=450)
    p.add_argument("--sae-steps",type=int,default=480)
    args=p.parse_args(argv)
    run(args.development_world_index,args.out,
        trunk_steps=args.trunk_steps,sae_steps=args.sae_steps)


if __name__=="__main__":
    main()
