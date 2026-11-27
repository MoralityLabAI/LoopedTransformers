"""Reissue the exact original OLR-07A and 07B selected positions after scope repair.

Original archives are read-only, authenticated; corrected outputs are separate.
No network training and no substitute theorem states.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import zipfile

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from . import olr07a_library as A
from . import olr07b_cpu as B

VERSION="pre_command_v1"
SCHEMA="olr07.scope_reissue.v1"
SOURCE=A.COMMIT
PINS={
    "07a":{"manifest":"manifest.json","manifest_sha":"846958ede542bedd510d865b3559e6f4711a97963583084c2d1028e70da2328e",
           "registry":"bank/states.json","registry_sha":"029b8d591851c7034736db1c1ec09c6e956c80e339fefb8c42e7a0429ce18151",
           "receipts":"native/receipts.json","count":207,"human_hb":500,"minimum":(20,8,20)},
    "07b":{"manifest":"audit_manifest.json","manifest_sha":"b388335c52ffb4b40b63e3cc77a37202c8c70553901a115635dc74f1d735b1ad",
           "registry":"registry.json","registry_sha":"d5cde5012d5312305b0b6e19b3ea823db8e55bdb95ef83dddfc71dfb2e8c212c",
           "receipts":"all_receipts.json","count":1441,"human_hb":20000,"minimum":(100,30,100)},
}
ROLES=("train","calibration","evaluation")

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(path):return B.sha(path)
def read(path):return B.read(path)
def write(path,obj):B.write(path,obj)

def archive(path,kind):
    pin=PINS[kind]
    with zipfile.ZipFile(path) as z:
        raw=z.read(pin["manifest"])
        if sha_bytes(raw)!=pin["manifest_sha"]:
            raise ValueError("original manifest bytes differ")
        m=json.loads(raw)
        if m["files"].get(pin["registry"])!=pin["registry_sha"]:
            raise ValueError("original registry pin differs")
        for n,h in m["files"].items():
            if sha_bytes(z.read(n))!=h:raise ValueError("original file hash mismatch "+n)
        roots=json.loads(z.read(pin["registry"]))
        rows=json.loads(z.read(pin["receipts"]))
    if len(roots)!=pin["count"] or len(rows)!=len(roots):
        raise ValueError("original root census changed")
    if len({r["id"] for r in roots})!=len(roots) or {r["id"] for r in roots}!={r["id"] for r in rows}:
        raise ValueError("old receipt/registry identity mismatch")
    return roots,{r["id"]:r for r in rows},{"archive_sha256":sha(path),"verified_files":len(m["files"]),
                      "manifest_sha256":pin["manifest_sha"],"registry_sha256":pin["registry_sha"]}

def source_check(source,root):
    rev=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if rev!=SOURCE:raise ValueError("pinned source revision required")
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("modified Lean source checkout")
    for r in root:
        p=source/r["file"];raw=p.read_bytes();lf=raw.replace(b"\r\n",b"\n")
        if r["file_sha256"] not in {sha_bytes(raw),sha_bytes(lf),sha_bytes(lf.replace(b"\n",b"\r\n"))}:
            raise ValueError("source bytes differ "+r["file"])

def native(path,dst,timeout,**kwargs):
    start=time.perf_counter()
    with LeanRepl(imports=("Lean",),timeout=timeout,exe=config.REPL_EXE,cwd=config.LEAN_DIR,
                  env=config.lean_env()) as lean:
        if str(lean.version)!="4.19.0":raise ValueError("wrong Lean version")
        rep=lean.request(op="trace",path=str(path.resolve()),out=str(dst.resolve()),**kwargs)
    if rep.get("ok") is not True or rep.get("scope_version")!=VERSION or rep.get("errors"):
        raise ValueError("native pre-command trace failed "+str(rep)[:1000])
    write(dst.with_suffix(".report.json"),rep)
    return rep,time.perf_counter()-start

def select(path,out,items,kind,timeout):
    groups=[]
    hb=PINS[kind]["human_hb"]
    if hb==A.HB:
        groups.append(("combined",A.HB,lambda r:list(dict.fromkeys((*A.MENU,r["human"])))))
    else:
        groups.extend((("menu",A.HB,lambda r:list(A.MENU)),("human",hb,lambda r:[r["human"]])))
    results={};times={}
    for label,cap,menu in groups:
        sel=[{"line":r["line"],"col":r["col"],"tactics":menu(r)} for r in items]
        for repeat in range(2):
            dst=out/f"{label}{repeat}.jsonl"
            _,seconds=native(path,dst,timeout,select=sel,cand_heartbeats=cap)
            rows=A.read_lines(dst)
            index={(r["line"],r["col"]):r for r in rows}
            if len(index)!=len(rows):raise ValueError("duplicate selected position")
            results[label,repeat]=index;times[f"{label}{repeat}"]=seconds
    return results,times

def state_status(x,r):
    if x is None:return "unreached"
    if x.get("scope_version")!=VERSION:
        raise ValueError("old or missing native scope marker")
    if x.get("env_scope")=="before" and x.get("decl_in_env_before") is not False:
        raise ValueError("self visibility in purported clean scope")
    if x.get("decl")!=r["decl"] or x.get("goals_before")!=[r["goal"]] or x.get("goals")!=1:
        return "source_state_mismatch"
    if x.get("env_scope")!="before":return "not_portable_before_command"
    if x.get("main_prop") is not True or x.get("main_closed") is not True:return "uncertifiable"
    return "scored"

def grade_file(source,out,items,kind,timeout):
    rel=items[0]["file"];role=items[0]["role"];path=source/rel
    dest=out/"native"/role/(path.parent.name+"_"+path.stem)
    def work(d):
        runs,times=select(path,d,items,kind,timeout)
        rows=[]
        for r in items:
            pos=(r["line"],r["col"])
            xs={k:v.get(pos) for k,v in runs.items()}
            statuses={state_status(x,r) for x in xs.values()}
            if len(statuses)!=1:
                raise ValueError("scope/goal replay disagreement "+r["id"]+" "+str(statuses))
            status=statuses.pop()
            if status!="scored":
                rows.append({**r,"scored":False,"reason":status});continue
            maps={k:{z["tactic"]:z for z in x["results"]} for k,x in xs.items()}
            if kind=="07a":
                a,b=maps["combined",0],maps["combined",1]
                h0,h1=a,b;expected=set((*A.MENU,r["human"]))
                if set(a)!=expected or set(b)!=expected:raise ValueError("candidate roster changed")
            else:
                a,b=maps["menu",0],maps["menu",1];h0,h1=maps["human",0],maps["human",1]
                if set(a)!=set(A.MENU) or set(b)!=set(A.MENU) or set(h0)!={r["human"]} or set(h1)!={r["human"]}:
                    raise ValueError("candidate roster changed")
            for u,v,ks in ((a,b,A.MENU),(h0,h1,(r["human"],))):
                for t in ks:
                    if (A.closed(u[t]),A.category(u[t]))!=(A.closed(v[t]),A.category(v[t])):
                        raise ValueError("native receipt replay disagreement "+r["id"]+" "+t)
            rows.append({**r,"scored":True,"scope_version":VERSION,"env_scope":"before",
                         "decl_in_env_before":False,"human_closed":A.closed(h0[r["human"]]),
                         "human_receipts":[h0[r["human"]],h1[r["human"]]],
                         "receipts":[a[t] for t in A.MENU],"replay_receipts":[b[t] for t in A.MENU]})
        write(d/"rows.json",rows)
        return {"file":rel,"role":role,"seconds":times,"selected":len(items),"scored":sum(r["scored"] for r in rows)}
    m=B.checkpoint(dest,work);rows=read(dest/"rows.json")
    if [r["id"] for r in rows]!=[r["id"] for r in items]:raise ValueError("reissue checkpoint roster changed")
    print("REISSUE_FILE",kind,json.dumps(m["stats"]),flush=True)
    return rows

def envelope(row):return any(A.closed(x) for x in row["receipts"])

def paired(old,rows):
    both=[r for r in rows if r["scored"] and old[r["id"]]["scored"]]
    byfile=defaultdict(list)
    for r in both:
        byfile[r["file"]].append(int(envelope(old[r["id"]]))-int(envelope(r)))
    fs=sorted(byfile);rng=random.Random(20270123);draws=[]
    if fs:
        for _ in range(4000):
            v=[d for f in rng.choices(fs,k=len(fs)) for d in byfile[f]]
            draws.append(sum(v)/len(v))
        draws.sort()
    return {"roots":len(both),"files":len(fs),
            "old_any":sum(envelope(old[r["id"]]) for r in both),
            "clean_any":sum(envelope(r) for r in both),
            "old_minus_clean_ci95":[draws[100],draws[3899]] if draws else None,
            "old_per_action":{t:sum(A.closed(old[r["id"]]["receipts"][j]) for r in both) for j,t in enumerate(A.MENU)},
            "clean_per_action":{t:sum(A.closed(r["receipts"][j]) for r in both) for j,t in enumerate(A.MENU)}}

def score(kind,old,rows):
    subsets={s:[r for r in rows if r["role"]==s] for s in ROLES}
    scored={s:[r for r in v if r["scored"]] for s,v in subsets.items()}
    census={s:B.census(v) for s,v in subsets.items()}
    valid=all(census[s]["scored"]>=n and census[s]["human_rate"]>=.95 for s,n in zip(ROLES,PINS[kind]["minimum"]))
    report={"schema":SCHEMA,"scope_version":VERSION,"kind":kind,"native_valid":valid,
            "census":census,"paired":{s:paired(old,v) for s,v in subsets.items()},
            "clean_ceiling":{s:sum(envelope(r) for r in v) for s,v in scored.items()},
            "old_receipts_superseded":True,"model_training_updates":0,"architecture_promotion_eligible":False}
    episodes=[]
    if valid:
        tr=scored["train"];report["fixed_order"]=[A.MENU[j] for j in A.fixed_order(tr)]
        for role in ("calibration","evaluation"):
            ep,stats,best=B.classical(tr,scored[role]);episodes.extend(ep)
            report[role]={"arms":stats,"best":best,"best_full":max(z["closed"] for z in stats.values()),
                          "menu_gap":(report["clean_ceiling"][role]-max(z["closed"] for z in stats.values()))/len(scored[role])}
        cal=report["calibration"]
        report["selected_on_calibration"]=cal["best"]
        report["source_calibration_gate"]=cal["menu_gap"]>=.1 and cal["best_full"]<=.8*len(scored["calibration"])
        report["fresh_gap_gate"]=report["evaluation"]["menu_gap"]>=.1 and report["evaluation"]["best_full"]<=.8*len(scored["evaluation"])
    else:
        report["source_calibration_gate"]=False;report["fresh_gap_gate"]=False
    return report,episodes

def sentinel(out,timeout):
    out.mkdir(parents=True,exist_ok=True)
    path=out/"scope_sentinel.lean"
    text="import Lean\nnamespace ScopeSentinel\n theorem earlier (n : Nat) : n + 0 = n := Nat.add_zero n\n @[simp] theorem current (n : Nat) : n + 0 = n := by\n  exact earlier n\n theorem later (n : Nat) : n + 0 = n := earlier n\nend ScopeSentinel\n"
    path.write_text(text,encoding="utf-8")
    native(path,out/"trace.jsonl",timeout,cand_every=0)
    source=[r for r in A.read_lines(out/"trace.jsonl") if r["decl"]=="ScopeSentinel.current" and r["tactic"].strip()=="exact earlier n"]
    if not source:raise ValueError("sentinel tactic state missing")
    r=source[0]
    menu=["exact ScopeSentinel.earlier n","exact ScopeSentinel.current n",
          "exact ScopeSentinel.later n","simp only [ScopeSentinel.current]"]
    sel=[{"line":r["line"],"col":r["col"],"tactics":menu}]
    native(path,out/"selected.jsonl",timeout,select=sel,cand_heartbeats=500)
    result=A.read_lines(out/"selected.jsonl")
    if len(result)!=1:raise ValueError("sentinel position mismatch")
    s=result[0];xs={x["tactic"]:x for x in s["results"]}
    if (s["env_scope"]!="before" or s["decl_in_env_before"] is not False
            or set(xs)!=set(menu) or not A.closed(xs[menu[0]])
            or any(A.closed(xs[t]) for t in menu[1:]) or xs[menu[2]].get("ok") is True
            or any(xs[t].get("ok") is True and
                   (xs[t].get("cert") or {}).get("kernel") is not False for t in (menu[1],menu[3]))):
        raise ValueError("native self/future visibility sentinel FAILED "+str(s))
    write(out/"PASS.json",{"scope_version":VERSION,"earlier_certified":True,"current_rejected":True,
                           "later_rejected":True,"simp_current_rejected":True,"results":xs})

def run(source,old_path,out,kind,workers,timeout):
    if out.exists():raise ValueError("create-only scope reissue output")
    roots,old,custody=archive(old_path,kind)
    source_check(source,roots)
    out.mkdir(parents=True)
    write(out/"frozen_registry.json",roots)
    write(out/"contract.json",{"kind":kind,"scope_version":VERSION,"source_commit":SOURCE,
              "custody":custody,"menu":A.MENU,"candidate_hb":A.HB,"human_hb":PINS[kind]["human_hb"],
              "lean_sha256":sha(config.LEAN_EXE),"repl_sha256":sha(config.REPL_EXE),
              "tracer_sha256":sha(config.LEAN_DIR/"Trace.lean"),"runner_sha256":sha(Path(__file__))})
    sentinel(out/"sentinel",timeout)
    groups=defaultdict(list)
    for r in roots:groups[r["file"]].append(r)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=[r for batch in pool.map(lambda v:grade_file(source,out,v,kind,timeout),groups.values()) for r in batch]
    if len(rows)!=len(roots):raise ValueError("root loss during reissue")
    write(out/"receipts.json",rows)
    report,episodes=score(kind,old,rows)
    write(out/"policy_episodes.json",episodes)
    report["custody"]=custody;report["elapsed_seconds"]=time.perf_counter()-start
    report["files"]={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(out/"manifest.json",report)
    print("REISSUE_RESULT",kind,json.dumps({k:v for k,v in report.items() if k!="files"},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--old",type=Path)
    p.add_argument("--kind",choices=tuple(PINS))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--sentinel-only",action="store_true")
    a=p.parse_args()
    if a.sentinel_only:sentinel(a.out,a.timeout)
    else:
        if a.old is None or a.kind is None:p.error("--old and --kind required")
        run(a.source,a.old,a.out,a.kind,a.workers,a.timeout)

if __name__=="__main__":main()
