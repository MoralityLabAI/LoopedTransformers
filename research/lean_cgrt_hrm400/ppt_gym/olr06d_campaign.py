"""OLR-06D source compilation, calibrated residual fits and native search audit."""
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
from .equality_search import plan,path_moves,run_live,replay,BUDGET,EXACT_CAP
from .equality_preflight import compile_scripts
from .olr06d_search import (
    COUNTS,CAPS,FRAMES,freeze,load_bank,make_case,source_frames,PatternDB,hamming,search,
)
from .olr06d_model import ARMS,SEEDS,DistanceReader,PacketMaker,encode,fit,weight_hash,residual_reader

FIT="olr06d.opaque_source_fit.v1"
SCORE="olr06d.opaque_native_score.v1"


def write(p,obj):p.write_text(canonical(obj)+"\n",encoding="utf-8")
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def hashes(path,names):return {n:sha256_file(path/n) for n in names}


def setup():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)


def native():
    return LeanRepl(imports=("Lean",),timeout=60,exe=config.REPL_EXE,
                    cwd=config.LEAN_DIR,env=config.lean_env())


def source(bank,out):
    if out.exists():raise ValueError("create-only source fit")
    setup();_,groups=load_bank(bank);out.mkdir(parents=True);t0=time.perf_counter()
    all_records={};items=[];episodes=[];teacher_exp=teacher_edges=0
    with native() as lean:
        if str(lean.version)!="4.19.0":raise ValueError("pinned Lean required")
        for split in ("train","calibration"):
            records=[]
            for case in groups[split]:
                frames,exp,edges=source_frames(case);teacher_exp+=exp;teacher_edges+=edges
                rr=[]
                for k,(word,distance,words) in enumerate(frames):
                    sample=make_case(case.equations,case.source,word,split)
                    moves=path_moves(sample,words)
                    script=[m.tactic(w) for m,w in zip(moves,words,strict=False)]+["exact w"]
                    if len(script)!=distance+1 or len(script)>BUDGET:
                        raise ValueError("teacher path cost differs")
                    items.append((sample,"source",script))
                    rr.append({"word":word,"distance":distance,"words":words,
                               "case":sample.root,"script":script})
                    if k==0:
                        ep=run_live(lean,case,"bidirectional_bfs",words=words)
                        if not ep["live_kernel"] or not replay(lean,case,ep):
                            raise ValueError("source root failed native replay")
                        episodes.append(ep)
                if len(rr)!=FRAMES:raise ValueError("source frame allocation drift")
                records.append(rr)
            all_records[split]=records
    certificate=compile_scripts(items,out,timeout=300)
    if certificate["passed"]!=sum(COUNTS[s]*FRAMES for s in ("train","calibration")):
        raise ValueError("source certificate count differs")
    write(out/"source_frames.json",all_records);write(out/"reference_episodes.json",episodes)
    train=encode(groups["train"],all_records["train"])
    calibration=encode(groups["calibration"],all_records["calibration"])
    weights={};meta={}
    for seed in SEEDS:
        for arm in ARMS:
            model,stats=fit(train,calibration,seed,arm)
            key=f"{seed}:{arm}";weights[key]=model.state_dict();meta[key]=stats
            print("fit",key,stats["scale"],stats["calibration_mae"],flush=True)
        a,b=(meta[f"{seed}:{arm}"] for arm in ("bi","one_way"))
        if any(a[k]!=b[k] for k in ("parameters","initial_sha256","sampler_sha256")):
            raise ValueError("matched direction initialization/budget drift")
    torch.save(weights,out/"models.pt")
    m={"schema":FIT,"registry_sha256":sha256_file(bank/"manifest.json"),
       "train_frames":len(train[0]),"calibration_frames":len(calibration[0]),
       "source_reference_native_replays":len(episodes),"source_certificate":certificate["passed"],
       "teacher_expansions":teacher_exp,"teacher_edges":teacher_edges,
       "lean_sha256":sha256_file(config.LEAN_EXE),"repl_sha256":sha256_file(config.REPL_EXE),
       "models":meta,"seeds":SEEDS,"arms":ARMS,"elapsed_seconds":time.perf_counter()-t0,
       "files":hashes(out,["source_frames.json","reference_episodes.json","standalone.lean",
                  "compiler.log","compiler_receipt.json","models.pt"])}
    write(out/"manifest.json",m);return m


def load_models(bank,source_dir):
    m=read(source_dir/"manifest.json")
    if (m["schema"]!=FIT or m["registry_sha256"]!=sha256_file(bank/"manifest.json")
            or m["lean_sha256"]!=sha256_file(config.LEAN_EXE)
            or m["repl_sha256"]!=sha256_file(config.REPL_EXE)
            or tuple(m["seeds"])!=SEEDS or tuple(m["arms"])!=ARMS):
        raise ValueError("source fit identity differs")
    for n,h in m["files"].items():
        if sha256_file(source_dir/n)!=h:raise ValueError("source fit bytes changed")
    states=torch.load(source_dir/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{seed}:{arm}";model=DistanceReader(arm)
            model.load_state_dict(states[key],strict=True);model.eval()
            if weight_hash(model)!=m["models"][key]["weights_sha256"]:
                raise ValueError("frozen weights differ")
            models[seed,arm]=(model,m["models"][key]["scale"])
    return m,models


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["split"],r["arm"]].append(r)
    result={}
    for (split,arm),v in sorted(groups.items()):
        means={k:sum(x[k] for x in v)/len(v) for k in
               ("expansions","candidate_edges_enumerated","model_batches","model_rows","planning_seconds")}
        result[f"{split}:{arm}"]={"trials":len(v),
            **{f"closed{cap}":sum(x["certified"] and x["expansions"]<=cap for x in v)
               for cap in CAPS},"certified":sum(x["certified"] for x in v),
            "native_tactics":sum(x["native_tactics"] for x in v),
            **means,"per_seed":{str(s):{f"closed{cap}":sum(
                x["certified"] and x["expansions"]<=cap for x in v if x["seed"]==s)
                for cap in CAPS} for s in sorted({x["seed"] for x in v})}}
    return result


def contrast(rows,split,left,right,cap):
    idx=defaultdict(list)
    for r in rows:
        if r["split"]==split:
            idx[r["root"],r["arm"]].append(int(r["certified"] and r["expansions"]<=cap))
    ids=sorted({k[0] for k in idx})
    mean=lambda v:sum(v)/len(v)
    diffs=[mean(idx[i,left])-mean(idx[i,right]) for i in ids]
    rng=random.Random(f"olr06d/{split}/{left}/{right}/{cap}")
    draws=sorted(mean([rng.choice(diffs) for _ in ids]) for _ in range(5000))
    return {"difference":mean(diffs),"ci95":[draws[125],draws[4875]],"roots":len(ids)}


def score(bank,source_dir,out):
    if out.exists():raise ValueError("create-only evaluation")
    setup();_,groups=load_bank(bank);m,models=load_models(bank,source_dir)
    out.mkdir(parents=True);t0=time.perf_counter();pending=[];setup_rows=[]
    cases={}
    # Choose all bounded paths from public data before opening native queries
    # or computing any unrestricted reference for these roots.
    for split in ("dev","transfer"):
        for case in groups[split]:
            cases[case.root]=case
            pdb=PatternDB(case.equations,case.source)
            maker=PacketMaker(case,pdb)
            setup_rows.append({"split":split,"root":case.root,"expansions":pdb.expansions,
                               "edges":pdb.edges,"seconds":pdb.setup_seconds})
            plans=[(-1,"hamming",search(case,lambda w:hamming(w,case.source),cap=128)),
                   (-1,"pdb",search(case,pdb.bound,cap=128)),
                   (-1,"bidir",plan(case,cap=128))]
            for (seed,arm),(model,scale) in models.items():
                result=search(case,pdb.bound,cap=128,batch_rank=residual_reader(model,maker,scale))
                plans.append((seed,arm,result))
            for seed,arm,p in plans:
                pending.append({"root":case.root,"split":split,"seed":seed,"arm":arm,
                                "model_batches":0,"model_rows":0,**p})
    write(out/"frozen_plans.json",pending)
    # Exact unrestricted reference is evaluated independently after freeze.
    for split in ("dev","transfer"):
        for case in groups[split]:
            p=plan(case,cap=EXACT_CAP)
            if not p["found"] or p["rewrite_distance"]+1>BUDGET:
                raise ValueError("registered exact reference unavailable")
            pending.append({"root":case.root,"split":split,"seed":-1,"arm":"exact",
                            "model_batches":0,"model_rows":0,**p})
    certs={};items=[];episodes=[];native_seconds=0.
    with native() as lean:
        if str(lean.version)!="4.19.0":raise ValueError("native runtime differs")
        for p in pending:
            p["certified"]=False;p["native_tactics"]=0
            if not p["found"]:continue
            if p["rewrite_distance"]+1>BUDGET:
                # Censored overlong proposal is not a native proof failure.
                p["status"]="censored_native_budget";continue
            key=p["root"],tuple(p["words"])
            if key not in certs:
                case=cases[p["root"]];start=time.perf_counter()
                ep=run_live(lean,case,"bidirectional_bfs",words=p["words"])
                if not ep["live_kernel"] or not replay(lean,case,ep):
                    raise ValueError("public proposed path failed native replay")
                native_seconds+=time.perf_counter()-start
                label=f"p{len(items)}";items.append((case,label,ep["script"]))
                certs[key]=(len(episodes),ep["effort"]);episodes.append(ep)
            p["native_episode"],p["native_tactics"]=certs[key];p["certified"]=True
    cert=compile_scripts(items,out,timeout=300)
    if cert["passed"]!=len(episodes):raise ValueError("native closure/standalone count differs")
    write(out/"root_scores.json",pending);write(out/"native_episodes.json",episodes)
    write(out/"pdb_setup.json",setup_rows)
    result=summarize(pending)
    contrasts={f"{split}:bi-{arm}:{cap}":contrast(pending,split,"bi",arm,cap)
               for split in ("dev","transfer") for arm in
               ("one_way","local","dense","full","hamming","pdb","bidir") for cap in CAPS}
    report={"schema":SCORE,"registry_sha256":sha256_file(bank/"manifest.json"),
            "fit_sha256":sha256_file(source_dir/"manifest.json"),"query_roots":len(cases),
            "policy_trials":len(pending),"frozen_bounded_plans":len(pending)-len(cases),
            "native_independent_replays":len(episodes),"independent_certificates":cert["passed"],
            "native_seconds":native_seconds,"elapsed_seconds":time.perf_counter()-t0,
            "pdb_abstract_expansions":sum(r["expansions"] for r in setup_rows),
            "pdb_abstract_edges":sum(r["edges"] for r in setup_rows),
            "scorecard":result,"contrasts":contrasts,"full_public_information":True,
            "query_failures_censored":True,"architecture_promotion_eligible":False,
            "files":hashes(out,["frozen_plans.json","root_scores.json","native_episodes.json",
                          "pdb_setup.json","standalone.lean","compiler.log","compiler_receipt.json"])}
    for split in ("dev","transfer"):
        if result[f"{split}:exact"]["certified"]!=COUNTS[split]:
            raise ValueError("exact native ceiling failed")
    write(out/"manifest.json",report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="phase",required=True)
    for phase in ("freeze","fit","score"):
        a=sub.add_parser(phase);a.add_argument("--out",type=Path,required=True)
        if phase=="freeze":a.add_argument("--old-manifest",type=Path,required=True)
        else:a.add_argument("--bank",type=Path,required=True)
        if phase=="score":a.add_argument("--fit",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="freeze":r=freeze(args.out,args.old_manifest)
    elif args.phase=="fit":r=source(args.bank,args.out)
    else:r=score(args.bank,args.fit,args.out)
    print(canonical(r),flush=True)


if __name__=="__main__":main()
