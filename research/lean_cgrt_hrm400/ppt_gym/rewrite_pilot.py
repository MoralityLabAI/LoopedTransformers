"""LSA-02K matched CPU equality/transport pilot and independent Lean verification.

The teacher is a verified witness sequence (not an exhaustive shortest-path
label set). The fixed Plan B architecture learns to rank visible rewrites.
The shuffled sham preserves every candidate menu and witness-label count.
All evaluation arms are budget-matched structural search; Lean independently
compiles the original theorem for each successful terminal rewrite path.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import torch

from .core import Config, PlasticProofTransformer
from .native_train import _model_fingerprint
from .rewrite_gym import (
    Case, COUNTS, DEPTH, GENERATOR_SEED, WORK, TEACHER_WORK, Move,
    action_token, canonical, certify, digest, file_sha, freeze, load_split,
    neighbors, public_packet, replay, search, write_json, write_rows,
)


SCHEMA="lsa02k.equality_rewrite_cpu_pilot.v1"
SEEDS=(197,199)
RANDOM_SEEDS=(17,31,47,61,83)
MODES=("learned","sham")
STEPS=720


def build_teacher(suite,out):
    cases,manifest,witnesses=load_split(suite,"train",witnesses=True)
    if len(cases)!=COUNTS["train"]:
        raise ValueError("registered 64 training roots required")
    frames=[]
    proofs=[]
    found=0
    for case in cases:
        result=search(case,"bidir",budget=TEACHER_WORK,depth=DEPTH)
        found+=int(result["closed"])
        path=result["path"] if result["closed"] else witnesses[case.root]
        states=replay(case,path)
        proofs.append((case,path,"train_"+case.root))
        for i in range(len(path)):
            menu=neighbors(case,states[i])
            successor=states[i+1]
            # Multiple occurrence/rule actions reaching the same witnessed
            # successor are all positive. Other actions are *unselected
            # witness alternatives*, not certified dead/invalid.
            labels=[int(x.successor==successor) for x in menu]
            if not any(labels):
                raise ValueError("native witness step missing in public menu")
            frames.append({
                "root":case.root,"lineage":case.lineage,"step":i,
                "packet":public_packet(case,states[i],DEPTH-i),
                "candidates":[action_token(x) for x in menu],
                "labels":labels,"checks_remaining":min(8,DEPTH-i),
            })
    out.mkdir(parents=True)
    cert=certify(proofs,out/"certified_teacher")
    if cert["compiled_axiom_free"]!=len(cases):
        raise ValueError("incomplete native source teacher certification")
    write_rows(out/"frames.jsonl",frames)
    result={
        "suite_sha256":file_sha(Path(suite)/"manifest.json"),
        "native_certificate_manifest_sha256":file_sha(out/"certified_teacher"/"manifest.json"),
        "root_count":len(cases),"frames":len(frames),
        "teacher_bidir_found":found,"generator_witness_fallback":len(cases)-found,
        "discriminating_witness_frames":sum(0<sum(x["labels"])<len(x["labels"]) for x in frames),
        "candidate_actions":sum(len(x["labels"]) for x in frames),
        "source_only":True,"native_compiled":True,"shortest_labels_claimed":False,
        "frames_sha256":file_sha(out/"frames.jsonl"),
    }
    write_json(out/"manifest.json",result)
    return frames,result


def fit(frames,out,seed,mode):
    if mode not in MODES or out.exists():
        raise ValueError("fresh source-only model fit required")
    torch.manual_seed(seed)
    model=PlasticProofTransformer(Config(plan="b",width=32,heads=4,max_bytes=384)).cpu().float()
    if model.parameter_count()!=57475:
        raise ValueError("frozen Plan B parameter contract changed")
    initial=_model_fingerprint(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    byroot=defaultdict(list)
    for row in frames:
        byroot[row["root"]].append(row)
    roots=sorted(byroot)
    rng=random.Random(seed)
    trace=[]
    started=time.monotonic()
    for step in range(STEPS):
        if time.monotonic()-started>300:
            raise TimeoutError("registered CPU fit cap reached")
        root=roots[rng.randrange(len(roots))]
        row=byroot[root][rng.randrange(len(byroot[root]))]
        labels=list(row["labels"])
        if mode=="sham":
            shuffle=random.Random(int(digest([seed,root,row["step"],"sham"])[:16],16))
            shuffle.shuffle(labels)
        outputs=model(row["packet"],row["candidates"],depth=2,
                      checks_remaining=row["checks_remaining"],middle_order="ab",return_all=True)
        mask=torch.tensor(labels,dtype=torch.bool)
        loss=torch.stack([
            torch.logsumexp(o["logits"],0)-torch.logsumexp(o["logits"][mask],0)
            for o in outputs
        ]).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite witness ranking loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        trace.append({"step":step+1,"root":root,"source_step":row["step"],"loss":float(loss.detach())})
    model.eval()
    out.mkdir(parents=True)
    torch.save({
        "schema":"lsa02k.rewrite_witness_ranker.v1","seed":seed,"mode":mode,
        "config":asdict(model.cfg),"weights":model.state_dict(),"steps":STEPS,
    },out/"checkpoint.pt")
    write_rows(out/"optimizer.jsonl",trace)
    receipt={"seed":seed,"mode":mode,"initial":initial,"final":_model_fingerprint(model),
             "checkpoint":file_sha(out/"checkpoint.pt"),"trace":file_sha(out/"optimizer.jsonl"),
             "schedule":digest([(r["root"],r["source_step"]) for r in trace]),
             "steps":STEPS,"parameters":model.parameter_count()}
    write_json(out/"manifest.json",receipt)
    return model,receipt


@torch.no_grad()
def ranker(model):
    def call(case,word,menu,remaining):
        outputs=model(public_packet(case,word,remaining),[action_token(x) for x in menu],
                      depth=2,checks_remaining=min(8,max(1,remaining)),middle_order="ab")
        return outputs[-1]["logits"].tolist()
    return call


def bootstrap(values):
    rng=random.Random(20261204)
    n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [means[50],means[1949]]


def campaign(suite,out):
    if out.exists():
        raise ValueError("fresh CPU experiment destination required")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    frames,teacher=build_teacher(suite,out/"teacher")
    train_lineages={f["lineage"] for f in frames}
    models={}
    fits=[]
    for seed in SEEDS:
        first=[]
        for mode in MODES:
            model,a=fit(frames,out/"fits"/f"{seed}_{mode}",seed,mode)
            _,b=fit(frames,out/"replicas"/f"{seed}_{mode}",seed,mode)
            if a!=b:
                raise ValueError("same-worker duplicate model/trace fingerprints differ")
            for p in model.parameters():
                p.requires_grad_(False)
            models[(seed,mode)]=model
            first.append(a)
            fits.append({**a,"duplicate_bitwise":True})
        if first[0]["initial"]!=first[1]["initial"] or first[0]["schedule"]!=first[1]["schedule"]:
            raise ValueError("learned and sham source schedules/initialization differ")
    # Calibration public cases are loaded only after the model fits. Their
    # generator witnesses remain unread until every search has finished.
    cases,meta=load_split(suite,"calibration")
    if len(cases)!=COUNTS["calibration"] or train_lineages & {c.lineage for c in cases}:
        raise ValueError("calibration theorem roots overlap training")
    root_episodes=[]
    proof_items=[]
    for case in cases:
        for policy in ("astar","bidir"):
            result=search(case,policy,budget=WORK,depth=DEPTH)
            result["seed"]=None
            root_episodes.append(result)
        for seed in RANDOM_SEEDS:
            result=search(case,"uniform",budget=WORK,depth=DEPTH,
                          seed=int(digest([seed,case.root])[:16],16))
            result["seed"]=seed
            root_episodes.append(result)
        for seed in SEEDS:
            for mode in MODES:
                model=models[(seed,mode)]
                result=search(case,mode,budget=WORK,depth=DEPTH,ranker=ranker(model))
                result["seed"]=seed
                root_episodes.append(result)
    # No reference witness enters the online ranking or symbolic policies.
    by_root={c.root:c for c in cases}
    for i,result in enumerate(root_episodes):
        result["native_certified"]=False
        if result["closed"]:
            case=by_root[result["root"]]
            replay(case,result["path"])
            proof_items.append((case,result["path"],f"policy_{i}"))
    model_proofs=len(proof_items)
    # The sealed calibration reference witnesses are opened only now.
    _,_,references=load_split(suite,"calibration",witnesses=True)
    for case in cases:
        proof_items.append((case,references[case.root],"reference_"+case.root))
    receipt=certify(proof_items,out/"native_calibration_certificates")
    if receipt["compiled_axiom_free"]!=len(proof_items):
        raise ValueError("a successful policy or reference lacks a native certificate")
    for result in root_episodes:
        result["native_certified"]=bool(result["closed"])
    for fit_receipt in fits:
        key=(fit_receipt["seed"],fit_receipt["mode"])
        if _model_fingerprint(models[key])!=fit_receipt["final"]:
            raise ValueError("model updated during held-out proof search")
    byroot=defaultdict(list)
    for r in root_episodes:
        byroot[r["root"]].append(r)
    roots=[]
    policies=("astar","bidir","uniform","learned","sham")
    for case in cases:
        rows=byroot[case.root]
        rates={}
        for policy in policies:
            matching=[x for x in rows if x["policy"]==policy]
            expected=5 if policy=="uniform" else 2 if policy in MODES else 1
            if len(matching)!=expected:
                raise ValueError("missing root-paired policy trials")
            rates[policy]=sum(x["native_certified"] for x in matching)/expected
        roots.append({"root":case.root,"lineage":case.lineage,"family":case.family,"rates":rates})
    rates={p:sum(r["rates"][p] for r in roots)/len(roots) for p in policies}
    costs={}
    for p in policies:
        selected=[r for r in root_episodes if r["policy"]==p]
        closed=[r for r in selected if r["native_certified"]]
        costs[p]={"episodes":len(selected),"closed":len(closed),
                  "mean_work":sum(r["work"] for r in selected)/len(selected),
                  "mean_work_closed":sum(r["work"] for r in closed)/len(closed) if closed else None,
                  "mean_path_closed":sum(len(r["path"]) for r in closed)/len(closed) if closed else None,
                  "model_calls":sum(r["model_calls"] for r in selected),
                  "expanded":sum(r["expanded"] for r in selected)}
    effects={}
    for p in ("astar","bidir","sham","uniform"):
        vals=[r["rates"]["learned"]-r["rates"][p] for r in roots]
        effects[f"learned_minus_{p}"]={"mean":sum(vals)/len(vals),
                                      "root_bootstrap95_descriptive":bootstrap(vals)}
    family_rates={}
    for family in ("equality","dependent_transport"):
        fr=[r for r in roots if r["family"]==family]
        family_rates[family]={"roots":len(fr),"rates":{
            p:sum(r["rates"][p] for r in fr)/len(fr) for p in policies}}
    best_symbolic=max(rates["astar"],rates["bidir"])
    screen=(rates["learned"]-best_symbolic>=.05 and rates["learned"]-rates["sham"]>=.05)
    report={
        "schema":SCHEMA,"generator_seed":GENERATOR_SEED,"suite_sha256":file_sha(Path(suite)/"manifest.json"),
        "teacher":teacher,"fits":fits,"train_roots":len(train_lineages),"calibration_roots":len(roots),
        "seed_roster":list(SEEDS),"random_seed_roster":list(RANDOM_SEEDS),
        "work_budget":WORK,"depth_limit":DEPTH,
        "native_compiler":receipt["lean_version"],"native_policy_proofs":model_proofs,
        "native_reference_proofs":len(cases),"all_proofs_axiom_free":True,
        "native_online_tactic_calls":0,"policy_rates":rates,"work_costs":costs,
        "family_rates":family_rates,"paired_effects":effects,
        "symbolic_headroom_screen":best_symbolic<.95,
        "registered_development_screen_passed":screen,
        "learned_competence_preflight":rates["learned"]>=.40 and rates["learned"]-best_symbolic>=.10 and rates["learned"]-rates["uniform"]>=.10,
        "root_results":roots,"same_worker_duplicates":True,"torch_version":torch.__version__,
        "host_platform":platform.platform(),"frozen_model_inference":True,
        "ordinary_ood_outcomes_opened":False,"calibration_training_updates":0,
        "scientific_eligible":False,"architecture_promotion_eligible":False,
        "limitation":"First-order unary equality rewriting plus genuine indexed Type transport; symbolic transition engine, native Lean terminal certificates; no live tactic search or general Mathlib transfer.",
    }
    write_rows(out/"episodes.jsonl",root_episodes)
    report["episodes_sha256"]=file_sha(out/"episodes.jsonl")
    write_json(out/"manifest.json",report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    report=campaign(a.suite,a.out)
    print(canonical({k:report[k] for k in (
        "train_roots","calibration_roots","teacher","policy_rates","work_costs","family_rates",
        "paired_effects","native_policy_proofs","native_reference_proofs",
        "symbolic_headroom_screen","registered_development_screen_passed","learned_competence_preflight"
    )}))
    return report


if __name__=="__main__":
    main()
