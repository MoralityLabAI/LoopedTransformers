"""Frozen synthetic paid-relation experiment; no Lean proof claims."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import platform
from pathlib import Path
import random
import time

import torch

from .olr06_queries import (
    COUNTS,SEED,Q,STOP,COST_WEIGHT,bank,from_row,Planner,execute,delayed_example,
)
from .olr06_query_model import ARMS,SEEDS,VISITS,packet,stack_packets,fit,make,hash_weights,choose_many

REG="olr06b-registry-v1"
FIT="olr06b-source-fit-v1"
EVAL="olr06b-expected-eval-v1"
REFS=("stop","greedy","lookahead2","optimal")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,data):
    path.write_text(json.dumps(data,sort_keys=True,separators=(",",":"),
                               allow_nan=False)+"\n",encoding="utf-8")


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def setup():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)


def freeze(out):
    if out.exists():raise ValueError("new registry directory required")
    out.mkdir(parents=True)
    groups=bank()
    for split,rows in groups.items():dump(out/(split+".json"),rows)
    m={"schema":REG,"seed":SEED,"counts":COUNTS,
       "files":{s+".json":sha(out/(s+".json")) for s in COUNTS},
       "utility":"resolved_fraction - .01 * attempted_cost",
       "split_policy":"canonical weighted graph signature + structural load shift"}
    dump(out/"manifest.json",m);return m


def load_bank(path):
    m=read(path/"manifest.json")
    if m["schema"]!=REG or m["seed"]!=SEED or m["counts"]!=COUNTS:
        raise ValueError("registry contract drift")
    expected=bank();seen=set();out={}
    for split,count in COUNTS.items():
        name=split+".json"
        if sha(path/name)!=m["files"][name]:raise ValueError("registered bytes altered")
        rows=read(path/name)
        if len(rows)!=count:raise ValueError("wrong cohort count")
        # JSON roundtrip normalizes tuples; compare canonical serialization.
        if json.dumps(rows,sort_keys=True)!=json.dumps(expected[split],sort_keys=True):
            raise ValueError("registered generator output drift")
        for row in rows:
            root=from_row(row["root"])
            sig=root.signature()
            if sig!=row["signature"] or sig in seen or row["split"]!=split:
                raise ValueError("canonical root leakage or identity drift")
            seen.add(sig)
        out[split]=rows
    return m,out


def reference(planner,arm):
    if arm=="stop":return lambda a,s:STOP
    depth={"greedy":1,"lookahead2":2,"optimal":None}[arm]
    return lambda a,s:planner.choose(a,s,depth)


def source(bank_dir,out):
    if out.exists():raise ValueError("new source fit directory required")
    setup();_,groups=load_bank(bank_dir)
    out.mkdir(parents=True)
    xs=[];targets=[];identities=[];t0=time.perf_counter()
    for row in groups["train"]:
        p=Planner(from_row(row["root"]))
        all_states=p.states()
        rng=random.Random(row["id"]+"/train_states")
        others=[z for z in all_states if z!=(0,0)]
        selected=[(0,0),*rng.sample(others,min(31,len(others)))]
        for a,s in selected:
            q=p.values(a,s)
            base=p.stop_value(a,s)
            x,inc,mask=packet(p,a,s)
            y=torch.zeros(Q+1)
            for j,v in q.items():y[j]=v-base
            if set(torch.nonzero(mask).flatten().tolist())!=set(q):
                raise ValueError("source legal menu differs from Bellman reference")
            xs.append((x,inc,mask));targets.append(y)
            identities.append({"root":row["id"],"attempted":a,"succeeded":s,
                               "exact_values":q,"stop_value":base})
    tensors=(*stack_packets(xs),torch.stack(targets))
    dump(out/"source_targets.json",identities)
    models={};meta={}
    for seed in SEEDS:
        for arm in ARMS:
            tick=time.perf_counter()
            model,stats=fit(tensors,seed=seed,arm=arm)
            key=f"{seed}:{arm}"
            models[key]=model.state_dict()
            stats["training_seconds"]=time.perf_counter()-tick
            meta[key]=stats
            print("OLR06B fit",key,stats["source_value_regret"],flush=True)
        a,b,c=[meta[f"{seed}:{v}"] for v in ARMS[:3]]
        for name in ("parameters","initial_sha256","sampler_sha256","decision_macs"):
            if len({a[name],b[name],c[name]})!=1:
                raise ValueError("matched direction control changed initialization or compute")
    torch.save(models,out/"models.pt")
    m={"schema":FIT,"registry_sha256":sha(bank_dir/"manifest.json"),
       "source_worlds":len(groups["train"]),"source_states":len(xs),
       "seeds":SEEDS,"arms":ARMS,"steps":240,"models":meta,
       "source_only":True,"elapsed_seconds":time.perf_counter()-t0,
       "runtime":{"python":platform.python_version(),"torch":str(torch.__version__),
                  "threads":torch.get_num_threads(),"mkldnn":False,"deterministic":True},
       "files":{n:sha(out/n) for n in ("models.pt","source_targets.json")}}
    dump(out/"manifest.json",m);return m


def load_models(bank_dir,fit_dir):
    m=read(fit_dir/"manifest.json")
    if (m["schema"]!=FIT or m["registry_sha256"]!=sha(bank_dir/"manifest.json")
            or tuple(m["arms"])!=ARMS or tuple(m["seeds"])!=SEEDS
            or m["steps"]!=240 or not m["source_only"]):
        raise ValueError("source fit does not match registration")
    for name,h in m["files"].items():
        if sha(fit_dir/name)!=h:raise ValueError("source-fit bytes changed")
    weights=torch.load(fit_dir/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{seed}:{arm}";model=make(arm)
            model.load_state_dict(weights[key],strict=True);model.eval()
            if hash_weights(model)!=m["models"][key]["weights_sha256"]:
                raise ValueError("source weight hash drift")
            models[seed,arm]=model
    return m,models


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["split"],r["arm"]].append(r)
    metrics=("coverage","utility","spent","full","queries","regret")
    return {f"{s}:{a}":{"root_trials":len(v),
             **{k:sum(row[k] for row in v)/len(v) for k in metrics},
             "sampled_coverage":sum(row["sampled"]["coverage"] for row in v)/len(v),
             "expected_model_macs":sum(row["expected_model_macs"] for row in v)/len(v),
             "per_seed":{str(seed):{k:sum(r[k] for r in v if r["seed"]==seed)/
                  sum(r["seed"]==seed for r in v) for k in ("coverage","utility","regret")}
                  for seed in sorted({r["seed"] for r in v})}}
            for (s,a),v in sorted(groups.items())}


def contrast(rows,split,left,right,metric):
    subset=[r for r in rows if r["split"]==split]
    roots=sorted({r["id"] for r in subset})
    idx=defaultdict(list)
    for r in subset:idx[r["id"],r["arm"]].append(r[metric])
    def mean(x):return sum(x)/len(x)
    diffs=[mean(idx[id,left])-mean(idx[id,right]) for id in roots]
    rng=random.Random(f"olr06b/{split}/{left}/{right}/{metric}")
    draws=sorted(mean([rng.choice(diffs) for _ in roots]) for _ in range(10000))
    return {"mean":mean(diffs),"ci95":[draws[250],draws[9750]],"roots":len(roots)}


def delayed_audit():
    p=Planner(delayed_example())
    return {arm:{"first":p.choose(0,0,{"greedy":1,"lookahead2":2,"optimal":None}[arm]),
                 **p.expectation(reference(p,arm))}
            for arm in REFS[1:]}


def score(bank_dir,fit_dir,out):
    if out.exists():raise ValueError("new evaluation directory required")
    setup();_,groups=load_bank(bank_dir);fm,models=load_models(bank_dir,fit_dir)
    t0=time.perf_counter();frozen=[];roots=[]
    # Freeze every public-state action, including actions for histories that
    # may never occur. No simulated hidden outcome table is used here.
    for split in ("dev","transfer"):
        for row in groups[split]:
            root=from_row(row["root"]);p=Planner(root);states=p.states()
            tensors=stack_packets([packet(p,a,s) for a,s in states])
            policies={}
            for seed,arm in models:
                start=time.perf_counter()
                actions=choose_many(models[seed,arm],tensors)
                policy=dict(zip(states,actions,strict=True))
                if any(j not in p.legal(a) for (a,s),j in policy.items()):
                    raise ValueError("illegal frozen policy")
                policies[seed,arm]=policy
                frozen.append({"id":row["id"],"split":split,"seed":seed,"arm":arm,
                    "states":[[a,s,policy[a,s]] for a,s in states],
                    "batch_score_seconds":time.perf_counter()-start})
            roots.append((row,root,p,policies))
    out.mkdir(parents=True)
    dump(out/"frozen_choices.json",frozen)
    rows=[]
    # Only after serialization of the complete frozen policy, evaluate
    # expectations and sample one common actual outcome table per root.
    for row,root,p,policies in roots:
        policies={(seed,arm):(lambda a,s,table=table:table[a,s])
                  for (seed,arm),table in policies.items()}
        policies.update({(-1,arm):reference(p,arm) for arm in REFS})
        optimum=p.value()
        for (seed,arm),policy in policies.items():
            expected=p.expectation(policy)
            regret=optimum-expected["utility"]
            if regret<-1e-8:raise ValueError("policy exceeds exact Bellman optimum")
            sampled=execute(root,policy,int(row["id"][:16],16))
            cost=0 if seed<0 else fm["models"][f"{seed}:{arm}"]["decision_macs"]
            rows.append({"id":row["id"],"split":row["split"],"seed":seed,"arm":arm,
                         **expected,"regret":max(0.,regret),"sampled":sampled,
                         "expected_model_macs":cost*(expected["queries"]+1)})
    dump(out/"root_scores.json",rows)
    scorecard=summarize(rows)
    contrasts={}
    for split in ("dev","transfer"):
        for a,b in (("bi","direct"),("bi","component_to_query"),
                    ("bi","query_to_component"),("bi","dense"),
                    ("bi","greedy"),("bi","lookahead2")):
            for metric in ("coverage","utility"):
                contrasts[f"{split}:{a}-{b}:{metric}"]=contrast(rows,split,a,b,metric)
    m={"schema":EVAL,"source_fit_sha256":sha(fit_dir/"manifest.json"),
       "registry_sha256":sha(bank_dir/"manifest.json"),
       "query_roots":len(roots),"policy_trials":len(rows),
       "frozen_neural_state_tables":len(frozen),
       "scorecard":scorecard,"contrasts":contrasts,
       "delayed_example":delayed_audit(),
       "exact_expectation":True,"independent_signed_sample_audit":True,
       "native_lean":False,"architecture_promotion_eligible":False,
       "elapsed_seconds":time.perf_counter()-t0,
       "files":{n:sha(out/n) for n in ("frozen_choices.json","root_scores.json")}}
    dump(out/"manifest.json",m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sp=parser.add_subparsers(dest="phase",required=True)
    for mode in ("freeze","fit","score"):
        p=sp.add_parser(mode);p.add_argument("--out",type=Path,required=True)
        if mode!="freeze":p.add_argument("--bank",type=Path,required=True)
        if mode=="score":p.add_argument("--fit",type=Path,required=True)
    args=parser.parse_args()
    if args.phase=="freeze":m=freeze(args.out)
    elif args.phase=="fit":m=source(args.bank,args.out)
    else:m=score(args.bank,args.fit,args.out)
    print(json.dumps(m,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":main()
