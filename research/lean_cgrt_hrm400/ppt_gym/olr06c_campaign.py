"""Registered native semantic applicability preflight and frozen-policy scoring."""
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
from .olr06c_native import (
    COUNTS,SEED,Q,bank,from_row,sha,symbolic,outcomes,native_table,compile_certificates,closure,UNKNOWN
)
from .olr06c_model import ARMS,SEEDS,SemanticPrior,fit,prior,packet,weight_hash
from .olr06c_planner import execute

REG="olr06c-native-registry-v1"
FIT="olr06c-native-fit-v1"
SCORE="olr06c-native-score-v1"


def dump(path,data):
    path.write_text(json.dumps(data,sort_keys=True,separators=(",",":"),
                               allow_nan=False)+"\n",encoding="utf-8")


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def files(path,names):return {n:sha(path/n) for n in names}


def setup():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)


def freeze(out):
    if out.exists():raise ValueError("create-only native registry")
    groups=bank();out.mkdir(parents=True)
    for s,rows in groups.items():dump(out/(s+".json"),rows)
    m={"schema":REG,"seed":SEED,"counts":COUNTS,"lean_sha256":sha(config.LEAN_EXE),
       "repl_sha256":sha(config.REPL_EXE),"files":files(out,[s+".json" for s in COUNTS])}
    dump(out/"manifest.json",m);return m


def load_bank(path):
    m=read(path/"manifest.json")
    if (m["schema"]!=REG or m["counts"]!=COUNTS or m["seed"]!=SEED
            or m["lean_sha256"]!=sha(config.LEAN_EXE)
            or m["repl_sha256"]!=sha(config.REPL_EXE)):
        raise ValueError("registered native toolchain or allocation drift")
    expected=bank();seen=set();groups={}
    for s,count in COUNTS.items():
        name=s+".json"
        if sha(path/name)!=m["files"][name]:raise ValueError("registry bytes changed")
        rows=read(path/name)
        if len(rows)!=count or json.dumps(rows,sort_keys=True)!=json.dumps(expected[s],sort_keys=True):
            raise ValueError("source generation drift")
        for row in rows:
            root=from_row(row["root"])
            if row["split"]!=s or root.signature()!=row["signature"] or set(root.exprs)&seen:
                raise ValueError("canonical expression leakage or identity drift")
            seen.update(root.exprs)
        groups[s]=rows
    return m,groups


def check_roots(rows):
    result=[]
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):raise ValueError("Lean 4.19 required")
        for row in rows:
            root=from_row(row["root"])
            rs=native_table(lean,root)
            result.append({"id":row["id"],"receipts":rs})
    return result


def count_receipts(records):
    return sum(len(r["receipts"]) for r in records)


def source(bank_path,out):
    if out.exists():raise ValueError("create-only native fit")
    setup();_,groups=load_bank(bank_path);rows=groups["train"]
    t0=time.perf_counter()
    receipts=check_roots(rows)
    out.mkdir(parents=True)
    dump(out/"source_receipts.json",receipts)
    items=[];labels=[];cert_items=[]
    for row,r in zip(rows,receipts,strict=True):
        root=from_row(row["root"]);rs=r["receipts"]
        items.append(packet(root))
        y=torch.zeros(Q)
        for j,receipt in enumerate(rs):y[j]=float(receipt["accepted"])
        labels.append(y)
        accepted=sum(1<<j for j,v in enumerate(rs) if v["accepted"])
        if any(x==UNKNOWN for x in closure(root,accepted).values()):
            raise ValueError("registered root lacks full positive transport")
        cert_items.append((root,rs,[accepted]))
    cert=compile_certificates(cert_items,out/"source.lean")
    if cert["full"]!=len(rows):raise ValueError("not all source roots certified")
    weights={};meta={}
    for seed in SEEDS:
        for arm in ARMS:
            model,stats=fit(items,labels,seed,arm)
            key=f"{seed}:{arm}";weights[key]=model.state_dict();meta[key]=stats
            print("fit",key,stats["source_nll"],stats["source_query_accuracy"],flush=True)
        a,b=(meta[f"{seed}:{arm}"] for arm in ("bi","one_way"))
        if any(a[k]!=b[k] for k in ("parameters","initial_sha256","sampler_sha256")):
            raise ValueError("matched direction source fit drift")
    torch.save(weights,out/"models.pt")
    m={"schema":FIT,"registry_sha256":sha(bank_path/"manifest.json"),
       "source_roots":len(rows),"native_queries":count_receipts(receipts),
       "native_replays":count_receipts(receipts),"source_certificate":cert,
       "seeds":SEEDS,"arms":ARMS,"models":meta,"source_only":True,
       "elapsed_seconds":time.perf_counter()-t0,
       "files":files(out,["source_receipts.json","source.lean","models.pt"])}
    dump(out/"manifest.json",m);return m


def load_models(bank_path,fit_path):
    m=read(fit_path/"manifest.json")
    if (m["schema"]!=FIT or m["registry_sha256"]!=sha(bank_path/"manifest.json")
            or tuple(m["arms"])!=ARMS or tuple(m["seeds"])!=SEEDS or not m["source_only"]):
        raise ValueError("source fit identity differs")
    for n,h in m["files"].items():
        if sha(fit_path/n)!=h:raise ValueError("source artifact changed")
    w=torch.load(fit_path/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{seed}:{arm}";model=SemanticPrior(arm)
            model.load_state_dict(w[key],strict=True);model.eval()
            if weight_hash(model)!=m["models"][key]["weight_sha256"]:
                raise ValueError("frozen source weights differ")
            models[seed,arm]=model
    return m,models


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["split"],r["arm"]].append(r)
    out={}
    for (s,a),v in groups.items():
        out[f"{s}:{a}"]={"trials":len(v),"full":sum(r["full"] for r in v),
            **{k:sum(r[k] for r in v)/len(v) for k in ("coverage","queries","utility","planner_calls",
                            "query_accuracy","query_nll","sign_accuracy")},
            "per_seed":{str(seed):{"full":sum(r["full"] for r in v if r["seed"]==seed),
                "trials":sum(r["seed"]==seed for r in v)}
                for seed in sorted({r["seed"] for r in v})}}
    return out


def contrast(rows,split,left,right,key):
    idx=defaultdict(list)
    for r in rows:
        if r["split"]==split:idx[r["id"],r["arm"]].append(float(r[key]))
    ids=sorted({p[0] for p in idx})
    mean=lambda v:sum(v)/len(v)
    diffs=[mean(idx[i,left])-mean(idx[i,right]) for i in ids]
    rng=random.Random(f"06c/{split}/{left}/{right}/{key}")
    draws=sorted(mean([rng.choice(diffs) for _ in ids]) for _ in range(10000))
    return {"difference":mean(diffs),"ci95":[draws[250],draws[9750]],"roots":len(ids)}


def score(bank_path,fit_path,out):
    if out.exists():raise ValueError("create-only native evaluation")
    setup();_,groups=load_bank(bank_path);_,models=load_models(bank_path,fit_path)
    t0=time.perf_counter();query_rows=[r for s in ("dev","transfer") for r in groups[s]]
    frozen=[]
    for row in query_rows:
        root=from_row(row["root"])
        # All priors computed from public syntax only and frozen before
        # independent query-side native verification.
        policies=[{"seed":seed,"arm":arm,"prior":prior(model,root),"depth":2}
                  for (seed,arm),model in models.items()]
        exact=[float(v==1) for v in symbolic(root)]
        policies.extend(({"seed":-1,"arm":"uniform2","prior":[.5]*root.n,"depth":2},
                         {"seed":-1,"arm":"uniform_full","prior":[.5]*root.n,"depth":None},
                         {"seed":-1,"arm":"symbolic2","prior":exact,"depth":2},
                         {"seed":-1,"arm":"symbolic_full","prior":exact,"depth":None}))
        frozen.append({"id":row["id"],"policies":policies})
    out.mkdir(parents=True)
    dump(out/"frozen_priors.json",frozen)
    receipts=check_roots(query_rows);dump(out/"native_receipts.json",receipts)
    rows=[];cert_items=[]
    import math
    for row,f,r in zip(query_rows,frozen,receipts,strict=True):
        if not row["id"]==f["id"]==r["id"]:raise ValueError("query receipt identity drift")
        root=from_row(row["root"]);rs=r["receipts"];masks=set()
        for policy in f["policies"]:
            p=policy["prior"];start=time.perf_counter()
            ep=execute(root,rs,p,policy["depth"])
            scores=[]
            for o in root.offers:
                pu,pv=p[o.u],p[o.v]
                q=pu if o.kind=="anchor" else pu*pv+(1-pu)*(1-pv)
                scores.append(q if o.sign==1 else 1-q)
            native=[v["accepted"] for v in rs]
            nll=-sum(math.log(max(1e-12,q if ok else 1-q)) for q,ok in zip(scores,native))/len(rs)
            ep.update({"id":row["id"],"split":row["split"],"arm":policy["arm"],"seed":policy["seed"],
                "query_accuracy":sum((q>=.5)==ok for q,ok in zip(scores,native))/len(rs),
                "query_nll":nll,"sign_accuracy":sum((q>=.5)==(s==1) for q,s in zip(p,symbolic(root)))/root.n,
                "planner_seconds":time.perf_counter()-start})
            masks.add(ep["accepted_mask"]);rows.append(ep)
        cert_items.append((root,rs,masks))
    cert=compile_certificates(cert_items,out/"query.lean")
    dump(out/"policy_rows.json",rows)
    summary=summarize(rows)
    for s in ("dev","transfer"):
        if summary[f"{s}:symbolic_full"]["full"]!=COUNTS[s]:
            raise ValueError("public finite evaluator failed registered full-plan ceiling")
    contrasts={}
    for s in ("dev","transfer"):
        for other in ("one_way","dense","local","full","uniform2","uniform_full"):
            for key in ("full","utility"):
                contrasts[f"{s}:bi-{other}:{key}"]=contrast(rows,s,"bi",other,key)
    m={"schema":SCORE,"registry_sha256":sha(bank_path/"manifest.json"),
       "fit_sha256":sha(fit_path/"manifest.json"),"query_roots":len(query_rows),
       "native_queries":count_receipts(receipts),"native_replays":count_receipts(receipts),
       "policy_trials":len(rows),"independent_certificate":cert,"scorecard":summary,
       "contrasts":contrasts,"frozen_priors_before_native_scoring":True,
       "positive_receipt_authority":True,"architecture_promotion_eligible":False,
       "elapsed_seconds":time.perf_counter()-t0,
       "files":files(out,["frozen_priors.json","native_receipts.json","policy_rows.json","query.lean"])}
    dump(out/"manifest.json",m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest="op",required=True)
    for op in ("freeze","fit","score"):
        a=sp.add_parser(op);a.add_argument("--out",type=Path,required=True)
        if op!="freeze":a.add_argument("--bank",type=Path,required=True)
        if op=="score":a.add_argument("--fit",type=Path,required=True)
    args=p.parse_args()
    if args.op=="freeze":m=freeze(args.out)
    elif args.op=="fit":m=source(args.bank,args.out)
    else:m=score(args.bank,args.fit,args.out)
    print(json.dumps(m,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":main()
