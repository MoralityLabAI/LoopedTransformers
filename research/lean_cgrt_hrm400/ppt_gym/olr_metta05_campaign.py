"""Registered history-alias campaign: native source proofs, locked fits, fresh recall."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical,digest,sha256_file
from .olr_metta05_native import (
    SEED,SPLITS,roots,root_row,from_row,capture,grade,execute,lookup,compile_proofs,
)
from .olr_metta05_model import (
    ARMS,SEEDS,make,fit,choose,flip_receipt,packets,weight_hash,
)

REG="ppt.olr-metta05-registry-v1"
FIT="ppt.olr-metta05-fit-v1"
SCORE="ppt.olr-metta05-score-v1"


def write(path,rows):
    path.write_text("".join(canonical(r)+"\n" for r in rows),encoding="utf-8")


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def manifest(path,content):
    path.write_text(canonical(content)+"\n",encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(path,names):
    return {n:sha256_file(path/n) for n in names}


def freeze(out):
    if out.exists():raise ValueError("create-only registered root bank required")
    cohorts=roots()
    out.mkdir(parents=True)
    for split,group in cohorts.items():
        write(out/(split+".jsonl"),[root_row(r) for r in group])
    m={"schema":REG,"seed":SEED,"split_worlds":{s:len(v) for s,v in cohorts.items()},
       "split_load":{s:k for s,(_,k) in SPLITS.items()},
       "lean_sha256":sha256_file(config.LEAN_EXE),
       "repl_sha256":sha256_file(config.REPL_EXE),
       "registered_before_native_scoring":True,
       "files":hashes(out,[s+".jsonl" for s in SPLITS])}
    manifest(out/"manifest.json",m);return m


def read_bank(bank):
    m=load(bank/"manifest.json")
    if (m["schema"]!=REG or m["seed"]!=SEED or
            m["lean_sha256"]!=sha256_file(config.LEAN_EXE) or
            m["repl_sha256"]!=sha256_file(config.REPL_EXE)):
        raise ValueError("registered native toolchain changed")
    expected=roots()
    cohorts={}
    lineages=set()
    for split,(n,k) in SPLITS.items():
        path=bank/(split+".jsonl")
        if sha256_file(path)!=m["files"][path.name]:raise ValueError("root shard altered")
        rows=read(path)
        if len(rows)!=2*n:raise ValueError("wrong registered root count")
        group=tuple(from_row(r) for r in rows)
        if group!=expected[split]:raise ValueError("registered root generation drift")
        for a,b in zip(group[::2],group[1::2],strict=True):
            if (a.lineage() in lineages or a.pair!=b.pair or a.projection()!=b.projection()
                    or a.probe_order!=b.probe_order or a.probe_branches!=b.probe_branches
                    or any(x.winner==y.winner for x,y in zip(a.slots,b.slots,strict=True))):
                raise ValueError("paired alias registration invalid")
            lineages.add(a.lineage())
        cohorts[split]=group
    return m,cohorts


def setup_torch():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)


def source(bank,out,*,steps=160):
    if out.exists():raise ValueError("new source fit directory required")
    setup_torch()
    _,cohorts=read_bank(bank)
    rows=[];proofs=[]
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):raise ValueError("pinned Lean 4.19 required")
        for root in cohorts["train"]:
            receipt=grade(lean,root)
            rows.append({"root_id":root.root_id,"pair":root.pair,"world":root.world,
                         "recall_order":list(root.recall_order),**receipt})
            proofs.append((root,receipt["labels"]))
    out.mkdir(parents=True)
    cert=compile_proofs(proofs,out/"source.lean")
    write(out/"source.jsonl",rows)
    models={};meta={}
    for seed in SEEDS:
        for arm in ARMS:
            model,stats=fit(rows,seed=seed,arm=arm,steps=steps)
            key=f"{seed}:{arm}"
            models[key]=model;meta[key]=stats
        a,d=(meta[f"{seed}:{arm}"] for arm in ("addressed","dense"))
        if (a["initial_sha256"]!=d["initial_sha256"] or
                a["parameters"]!=d["parameters"] or a["sampler_sha256"]!=d["sampler_sha256"]):
            raise ValueError("matched memory initialization or source schedule differs")
    with (out/"models.pt").open("xb") as f:
        torch.save({key:m.state_dict() for key,m in models.items()},f)
    m={"schema":FIT,"source_worlds":len(rows),"source_certificate":cert,
       "source_registry_sha256":sha256_file(bank/"manifest.json"),
       "steps":steps,"seeds":SEEDS,"arms":ARMS,"models":meta,
       "source_only_training":True,
       "artifacts":hashes(out,["source.lean","source.jsonl","models.pt"])}
    manifest(out/"manifest.json",m);return m


def load_models(bank,fit_path):
    m=load(fit_path/"manifest.json")
    if (m["schema"]!=FIT or m["source_registry_sha256"]!=sha256_file(bank/"manifest.json")
            or tuple(m["seeds"])!=SEEDS or tuple(m["arms"])!=ARMS):
        raise ValueError("source fit identity differs")
    for n,sha in m["artifacts"].items():
        if sha256_file(fit_path/n)!=sha:raise ValueError("source artifact changed")
    states=torch.load(fit_path/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{seed}:{arm}";model=make(arm)
            model.load_state_dict(states[key],strict=True);model.eval()
            if weight_hash(model)!=m["models"][key]["weights_sha256"]:
                raise ValueError("source tensor identity changed")
            models[seed,arm]=model
    return m,models


def projections(root,receipts):
    h,q,_=packets(receipts,root.recall_order)
    return digest(q.tolist()),digest(h.tolist())


def choices_for(root,receipts,models):
    ref=lookup(receipts);oracle=[ref[a] for a in root.recall_order]
    rows=[]
    def put(seed,arm,picks,passes,*,focal=False):
        if len(picks)!=len(root.slots) or any(b not in (0,1) for b in picks):
            raise ValueError("invalid frozen action vector")
        rows.append({"root_id":root.root_id,"pair":root.pair,"world":root.world,
                     "split":root.split,"seed":seed,"arm":arm,
                     "choices":picks,"attention_visits":passes,
                     "focal_index":root.recall_order.index(root.focal),
                     "target_only":focal})
    put(0,"exact_map",oracle,0)
    put(0,"fixed_r0",[0]*len(oracle),0)
    for seed in SEEDS:
        for arm in ARMS:
            model=models[seed,arm]
            vec,visits=choose(model,arm,receipts,root.recall_order)
            put(seed,arm,vec,visits)
            if arm!="addressed":continue
            erased,cost=choose(model,arm,receipts,root.recall_order,erase=True)
            put(seed,"erased",erased,cost)
            index=root.recall_order.index(root.focal)
            for name,stream in (("focal_natural",receipts),
                    ("focal_target_flip",flip_receipt(receipts,root.focal)),
                    ("focal_sham_flip",flip_receipt(receipts,root.sham))):
                prediction,visits=choose(model,arm,stream,root.recall_order)
                mixed=list(oracle);mixed[index]=prediction[index]
                put(seed,name,mixed,visits,focal=True)
    return rows


def audit_alias(root_map,histories,decisions):
    index={(r["root_id"],r["seed"],r["arm"]):r for r in decisions}
    pairs=defaultdict(list)
    for root in root_map.values():pairs[root.pair].append(root)
    for group in pairs.values():
        a,b=sorted(group,key=lambda r:r.world)
        left,right=histories[a.root_id],histories[b.root_id]
        if (left["projection_sha256"]!=right["projection_sha256"] or
                left["history_sha256"]==right["history_sha256"]):
            raise ValueError("current projection failed paired alias")
        la,lb=lookup(left["receipts"]),lookup(right["receipts"])
        if any(la[i]==lb[i] for i in la):
            raise ValueError("paired native receipts failed opposite labels")
        for seed in SEEDS:
            x=index[a.root_id,seed,"erased"]["choices"]
            y=index[b.root_id,seed,"erased"]["choices"]
            if x!=y:raise ValueError("memoryless reader leaked paired truth")
    return len(pairs)


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["split"],r["seed"],r["arm"]].append(r)
    out={}
    for (split,seed,arm),v in sorted(groups.items()):
        paired=defaultdict(list)
        for r in v:paired[r["pair"]].append(r)
        out[f"{split}:{seed}:{arm}"]={
            "worlds":len(v),"proofs":sum(r["closed"] for r in v),
            "complete_pairs":sum(all(r["closed"] for r in g) for g in paired.values()),
            "correct_addresses":sum(r["correct_addresses"] for r in v),
            "addresses":sum(len(r["choices"]) for r in v),
            "focal_correct":sum(r["focal_correct"] for r in v),
            "tactic_calls":sum(r["calls"] for r in v),
            "focuses":sum(r["focuses"] for r in v),"restores":sum(r["restores"] for r in v),
            "attention_visits":sum(r["attention_visits"] for r in v),
        }
    return out


def contrast(rows,split,left,right,key,repetitions=10000):
    values={(r["pair"],r["world"],r["seed"],r["arm"]):float(r[key])
            for r in rows if r["split"]==split}
    pair_ids=sorted({i[0] for i in values})
    diffs=[]
    for p in pair_ids:
        diffs.append(sum(values[p,w,s,left]-values[p,w,s,right]
                         for w in (0,1) for s in SEEDS)/6)
    rng=random.Random(f"05/{split}/{left}/{right}/{key}")
    n=len(diffs)
    draws=sorted(sum(rng.choice(diffs) for _ in range(n))/n for _ in range(repetitions))
    return {"difference":sum(diffs)/n,"ci95":[draws[250],draws[9750]],"pair_clusters":n}


def score(bank,fit_path,out):
    if out.exists():raise ValueError("create-only query result directory required")
    setup_torch();_,cohorts=read_bank(bank);fit_receipt,models=load_models(bank,fit_path)
    root_map={r.root_id:r for split in ("dev","transfer") for r in cohorts[split]}
    histories={};decisions=[];start=time.monotonic()
    # Actual native receipt history is observed before choosing. The outcome
    # of the final continuation and the alternate probe are still unopened.
    with LeanRepl(timeout=60) as lean:
        for root in root_map.values():
            n,receipts,metta=capture(lean,root)
            q,h=projections(root,receipts)
            histories[root.root_id]={"root_id":root.root_id,"receipts":receipts,
                "prefix":n.receipt(),"metta":metta,"projection_sha256":q,"history_sha256":h}
            decisions.extend(choices_for(root,receipts,models))
    alias_pairs=audit_alias(root_map,histories,decisions)
    out.mkdir(parents=True)
    write(out/"observed_histories.jsonl",histories.values())
    write(out/"frozen_choices.jsonl",decisions)
    byroot=defaultdict(list)
    for d in decisions:byroot[d["root_id"]].append(d)
    rows=[];native=[];grading=[];proofs=[]
    with LeanRepl(timeout=60) as lean:
        for root_id,root in root_map.items():
            observed=histories[root_id]["receipts"]
            g=grade(lean,root)
            if g["receipts"]!=observed:raise ValueError("new native grade history drifted")
            labels=g["labels"];grading.append({"root_id":root_id,**g})
            proofs.append((root,labels))
            outcomes={}
            for d in byroot[root_id]:
                picks=tuple(d["choices"])
                if picks not in outcomes:
                    ep=execute(lean,root,picks,observed)
                    outcomes[picks]=ep
                    native.append({"root_id":root_id,**ep})
                ep=outcomes[picks]["native"]
                correct=sum(a==b for a,b in zip(picks,labels,strict=True))
                if ep["closed"]!=(correct==len(labels)):
                    raise ValueError("native closure differs from independently verified branch table")
                focal=d["focal_index"]
                rows.append({**d,"closed":ep["closed"],"calls":ep["calls"],
                             "focuses":ep["focuses"],"restores":ep["restores"],
                             "correct_addresses":correct,"focal_correct":int(picks[focal]==labels[focal]),
                             "native_sha256":digest(ep)})
            if time.monotonic()-start>750:raise TimeoutError("bounded native evaluation elapsed")
    cert=compile_proofs(proofs,out/"query.lean")
    write(out/"grading.jsonl",grading)
    write(out/"native.jsonl",native)
    write(out/"score_rows.jsonl",rows)
    summary=summarize(rows)
    for split in ("dev","transfer"):
        for seed in SEEDS:
            e=summary[f"{split}:{seed}:erased"]
            if e["correct_addresses"]*2!=e["addresses"]:
                raise ValueError("memoryless 50% paired address ceiling failed")
            if summary[f"{split}:{seed}:addressed"]["attention_visits"]!=summary[f"{split}:{seed}:dense"]["attention_visits"]:
                raise ValueError("matched attention visit count differs")
        e=summary[f"{split}:0:exact_map"]
        if e["proofs"]!=e["worlds"]:raise ValueError("explicit native receipt ceiling failed")
    contrasts={}
    for split in ("dev","transfer"):
        for left,right,key in (("addressed","dense","closed"),
                ("focal_natural","focal_target_flip","focal_correct"),
                ("focal_natural","focal_sham_flip","focal_correct"),
                ("focal_sham_flip","focal_target_flip","focal_correct")):
            contrasts[f"{split}:{left}-{right}:{key}"]=contrast(rows,split,left,right,key)
    m={"schema":SCORE,"registry_sha256":sha256_file(bank/"manifest.json"),
       "fit_sha256":sha256_file(fit_path/"manifest.json"),"query_worlds":len(root_map),
       "alias_pairs":alias_pairs,"policy_rows":len(rows),"distinct_native_policy_executions":len(native),
       "distinct_native_policy_replays":len(native),"independent_grade_replays":len(grading),
       "query_certificate":cert,"all_policy_choices_frozen_before_final_proof":True,
       "addressed_dense_compute_matched_per_event":True,
       "policy_calls_per_full_proof":{"dev":20,"transfer":34},
       "scorecard":summary,"contrasts":contrasts,"architecture_promotion_eligible":False,
       "artifacts":hashes(out,["observed_histories.jsonl","frozen_choices.jsonl","grading.jsonl",
                      "native.jsonl","score_rows.jsonl","query.lean"])}
    manifest(out/"manifest.json",m);return m


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="phase",required=True)
    a=sub.add_parser("freeze");a.add_argument("--out",type=Path,required=True)
    b=sub.add_parser("fit");b.add_argument("--bank",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True);b.add_argument("--steps",type=int,default=160)
    c=sub.add_parser("score");c.add_argument("--bank",type=Path,required=True)
    c.add_argument("--fit",type=Path,required=True);c.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if args.phase=="freeze":m=freeze(args.out)
    elif args.phase=="fit":m=source(args.bank,args.out,steps=args.steps)
    else:m=score(args.bank,args.fit,args.out)
    print(canonical(m))


if __name__=="__main__":main()
