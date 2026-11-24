"""OLR-07A: authentic Lean terminal states, paid failure feedback and exact empirical trees.

The existing Lean trace/select backend runs every candidate from the original
goal snapshot, and certifies closures. A nonclosing attempt is censored with
respect to mathematical provability. Offline outcome matrices are grading data.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
import time

from lcgrt import config
from lcgrt.lean_repl import LeanRepl

COMMIT="6caaee842e9495688c1567e78c0e68dbb96942aa"
SEED=20270111
FAMILIES={"train":("Nat","List","Option"),"calibration":("Int",),"evaluation":("Array","Fin","Sum")}
MENU=("rfl","assumption","simp","simp_all","simp [*]","decide","omega","trivial",
      "contradiction","exact?","constructor","ac_rfl")
ARMS=("fixed","binary","diagnostic","local_binary","local_diagnostic")
BUDGET=4
PER_FILE=20
HB=500
AXIOMS={"propext","Classical.choice","Quot.sound"}
FORBIDDEN=re.compile(r"\b(sorry|admit|native_decide|run_tac|run_cmd|run_meta|by_elab|set_option|unsafe|trustCompiler)\b|#|IO\.")
SCHEMA="olr07a.library_receipt.v1"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def read_lines(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def closed(r):
    cert=r.get("cert") or {}
    return (r.get("ok") is True and r.get("goals_after")==0 and
            cert.get("kernel") is True and isinstance(cert.get("axioms"),list)
            and set(cert["axioms"])<=AXIOMS)


def category(r):
    if closed(r):return "closed"
    if r.get("ok") is True:
        n=r.get("goals_after")
        return "uncertified" if n==0 else "open:"+str(min(3,n)) if isinstance(n,int) and n>0 else "malformed"
    msg=str(r.get("error","")).lower()
    if "heartbeat" in msg or "recursion depth" in msg or "timeout" in msg:
        return "resource"
    if msg.startswith("parse:"):return "parse"
    if any(s in msg for s in ("type mismatch","unif","application type","failed to synthesize","does not have type")):
        return "type"
    if any(s in msg for s in ("no goals","did not find","no matching","made no progress","nothing to rewrite","failed to close")):
        return "no_match"
    return "rejected"


def human_text(r):
    first,*rest=r["tactic"].split("\n")
    return "\n".join([first]+[s[min(r["col"],len(s)-len(s.lstrip(" "))):] for s in rest])


def native_trace(path,out,**kwargs):
    start=time.perf_counter()
    with LeanRepl(imports=("Lean",),timeout=180,exe=config.REPL_EXE,cwd=config.LEAN_DIR,
                  env=config.lean_env()) as lean:
        if str(lean.version)!="4.19.0":raise ValueError("pinned Lean required")
        report=lean.request(op="trace",path=str(path.resolve()),out=str(out.resolve()),**kwargs)
    if report.get("ok") is not True:raise ValueError("native trace failed: "+str(report)[:500])
    return report,time.perf_counter()-start


def freeze(source,out):
    rev=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if rev!=COMMIT:raise ValueError("source revision differs from pinned toolchain")
    if out.exists():raise ValueError("create-only registry")
    out.mkdir(parents=True)
    rows=[];seen_goals=set();audit=[]
    for role,families in FAMILIES.items():
        for fam in families:
            for name in ("Basic","Lemmas"):
                rel=f"src/Init/Data/{fam}/{name}.lean";p=source/rel
                if not p.is_file():raise ValueError("missing registered source: "+rel)
                key=f"{fam}_{name}"
                raw=out/(key+".jsonl")
                rep,seconds=native_trace(p,raw,cand_every=0)
                write(out/(key+".trace.json"),rep)
                ok={d["decl"] for d in rep["decls"] if d["in_env"] and isinstance(d["axioms"],list)
                    and set(d["axioms"])<=AXIOMS and d["decl"] not in ("","[anonymous]","_anonymous")}
                candidates=[]
                for r in read_lines(raw):
                    if (r.get("decl") not in ok or len(r.get("goals_before",[]))!=1
                            or r.get("goals_after")!=[] or not r.get("tactic")
                            or "…" in r["goals_before"][0] or "…" in r["tactic"]):
                        continue
                    text=human_text(r)
                    if FORBIDDEN.search(text):continue
                    identity=[rel,r["decl"],r["line"],r["col"],r["goals_before"]]
                    candidates.append((digest([SEED,identity]),r,text))
                picked=[];decls=set();positions=set()
                for ident,r,human in sorted(candidates,key=lambda x:x[0]):
                    goal=r["goals_before"][0];pos=(r["line"],r["col"])
                    if r["decl"] in decls or goal in seen_goals or pos in positions:continue
                    decls.add(r["decl"]);seen_goals.add(goal);positions.add(pos)
                    picked.append({"id":ident,"role":role,"family":fam,"file":rel,"file_sha256":sha(p),
                                   "decl":r["decl"],"line":r["line"],"col":r["col"],
                                   "goal":goal,"human":human})
                    if len(picked)==PER_FILE:break
                rows.extend(picked)
                audit.append({"file":rel,"role":role,"source_sha256":sha(p),
                              "trace_seconds":seconds,"file_errors":len(rep.get("errors",[])),
                              "eligible_candidates":len(candidates),"selected":len(picked)})
    write(out/"states.json",rows);write(out/"trace_audit.json",audit)
    m={"schema":SCHEMA,"source_commit":COMMIT,"seed":SEED,"menu":MENU,"budget":BUDGET,
       "heartbeats_thousands":HB,"roots":len(rows),
       "roles":dict(Counter(r["role"] for r in rows)),
       "lean_sha256":sha(config.LEAN_EXE),"repl_sha256":sha(config.REPL_EXE),
       "states_sha256":sha(out/"states.json")}
    write(out/"manifest.json",m);return m


def grade(source,bank,out):
    if out.exists():raise ValueError("create-only native receipt output")
    m=json.loads((bank/"manifest.json").read_text())
    if (m["schema"]!=SCHEMA or m["states_sha256"]!=sha(bank/"states.json")
            or tuple(m["menu"])!=MENU or m["source_commit"]!=COMMIT
            or m["lean_sha256"]!=sha(config.LEAN_EXE) or m["repl_sha256"]!=sha(config.REPL_EXE)):
        raise ValueError("registry/toolchain identity changed")
    roots=json.loads((bank/"states.json").read_text());groups=defaultdict(list)
    for r in roots:groups[r["file"]].append(r)
    out.mkdir(parents=True);rows=[];audit=[]
    for rel,items in groups.items():
        p=source/rel
        if any(sha(p)!=r["file_sha256"] for r in items):raise ValueError("source file changed")
        sel=[{"line":r["line"],"col":r["col"],"tactics":list(dict.fromkeys((*MENU,r["human"])))}
             for r in items]
        runs=[];times=[]
        for repeat in range(2):
            dest=out/(Path(rel).parent.name+"_"+Path(rel).stem+f".{repeat}.jsonl")
            rep,seconds=native_trace(p,dest,select=sel,cand_heartbeats=HB)
            write(dest.with_suffix(".report.json"),rep)
            raw=read_lines(dest)
            index={(r["line"],r["col"]):r for r in raw}
            if len(index)!=len(raw):raise ValueError("duplicate native select positions")
            runs.append(index);times.append(seconds)
        audit.append({"file":rel,"seconds":times,"selected":len(items),"reached":[len(a) for a in runs]})
        for r in items:
            pos=(r["line"],r["col"])
            a,b=(run.get(pos) for run in runs)
            def valid(x):
                return x is not None and x.get("decl")==r["decl"] and x.get("goals_before")==[r["goal"]] and x.get("goals")==1
            if valid(a)!=valid(b):raise ValueError("native state replay drift")
            if not valid(a):
                rows.append({**r,"scored":False,"reason":"unreached_or_source_state_mismatch"});continue
            if any(x.get("scope_version")!="pre_command_v1" for x in (a,b)):
                raise ValueError("unsafe legacy tracer: rebuild before collecting receipts")
            scopes=[x.get("env_scope")=="before" and x.get("decl_in_env_before") is False for x in (a,b)]
            if scopes[0]!=scopes[1]:raise ValueError("native scope replay drift")
            if not all(scopes):
                rows.append({**r,"scored":False,"reason":"not_portable_before_command"});continue
            def certifiable(x):return x.get("main_prop") is True and x.get("main_closed") is True
            if certifiable(a)!=certifiable(b):raise ValueError("native certifiability drift")
            if not certifiable(a):
                rows.append({**r,"scored":False,"reason":"uncertifiable"});continue
            aa,bb=({v["tactic"]:v for v in x["results"]} for x in (a,b))
            expected=set((*MENU,r["human"]))
            if set(aa)!=expected or set(bb)!=expected:raise ValueError("native candidate roster drift")
            if any((closed(aa[t]),category(aa[t]))!=(closed(bb[t]),category(bb[t])) for t in expected):
                raise ValueError("native receipt replay disagreement at "+r["id"])
            rows.append({**r,"scored":True,"human_closed":closed(aa[r["human"]]),
                         "receipts":[aa[t] for t in MENU],"replay_receipts":[bb[t] for t in MENU]})
    write(out/"receipts.json",rows);write(out/"native_audit.json",audit)
    return rows


def tokens(text):
    return set(re.findall(r"\w+|[^\w\s]",text.lower(),flags=re.UNICODE))


class Empirical:
    """Finite empirical belief tree, with exact success-count objective."""
    def __init__(self,rows,diagnostic):
        self.rows=rows;self.diagnostic=diagnostic
        self.all=(1<<len(rows))-1;self.masks=[defaultdict(int) for _ in MENU]
        self.goal_tokens=[tokens(r["goal"]) for r in rows]
        for i,r in enumerate(rows):
            for j,v in enumerate(r["receipts"]):
                self.masks[j][self.label(v)]|=1<<i
        self.cache={}

    def label(self,r):
        c=category(r)
        return c if self.diagnostic or c=="closed" else "nonclosure"

    def initial(self,goal,local):
        if not local:return self.all
        q=tokens(goal)
        order=sorted(range(len(self.rows)),key=lambda i:(
            -len(q&self.goal_tokens[i])/max(1,len(q|self.goal_tokens[i])),self.rows[i]["id"]))
        return sum(1<<i for i in order[:24])

    def best(self,belief,tried,left):
        key=belief,tried,left
        if key in self.cache:return self.cache[key]
        if not belief or not left:return (0,0,-1)
        best=None
        for j,parts in enumerate(self.masks):
            if tried>>j&1:continue
            gain=(belief&parts.get("closed",0)).bit_count()
            cost=belief.bit_count()
            for label,mask in parts.items():
                b=belief&mask
                if b and label!="closed":
                    a,c,_=self.best(b,tried|1<<j,left-1)
                    gain+=a;cost+=c
            cand=(gain,cost,j)
            if best is None or (gain,-cost,-j)>(best[0],-best[1],-best[2]):best=cand
        if best is None:best=(0,0,-1)
        self.cache[key]=best;return best

    def reveal(self,belief,history):
        j,c=history[-1];b=belief&self.masks[j].get(c,0)
        if b:return b
        b=self.all
        for j,c in history:b&=self.masks[j].get(c,0)
        return b or self.all


def fixed_order(rows):
    masks=[sum(1<<i for i,r in enumerate(rows) if closed(r["receipts"][j])) for j in range(len(MENU))]
    def coverage(ids):
        m=0
        for j in ids:m|=masks[j]
        return m.bit_count()
    chosen=max(combinations(range(len(MENU)),BUDGET),key=lambda a:(coverage(a),tuple(-j for j in a)))
    remain=set(chosen);seen=0;out=[]
    while remain:
        j=max(remain,key=lambda j:((masks[j]&~seen).bit_count(),-j))
        out.append(j);remain.remove(j);seen|=masks[j]
    return tuple(out)


def episode(row,arm,order,trees):
    t=time.perf_counter();history=[];events=[];tried=0
    tree=trees.get("diagnostic" if arm.endswith("diagnostic") else "binary")
    belief=tree.initial(row["goal"],arm.startswith("local")) if arm!="fixed" else 0
    for k in range(BUDGET):
        j=order[k] if arm=="fixed" else tree.best(belief,tried,BUDGET-k)[2]
        if j<0 or tried>>j&1:raise ValueError("illegal empirical decision")
        tried|=1<<j
        r=row["receipts"][j]  # only selected receipt revealed after paying
        events.append({"action":MENU[j],"status":category(r),"closed":closed(r)})
        if closed(r):break
        if arm!="fixed":
            history.append((j,tree.label(r)));belief=tree.reveal(belief,history)
    return {"id":row["id"],"role":row["role"],"file":row["file"],"arm":arm,
            "closed":events[-1]["closed"],"calls":len(events),"events":events,
            "policy_seconds":time.perf_counter()-t}


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    bank=out/"bank";freeze(source,bank)
    receipts=out/"native";rows=grade(source,bank,receipts)
    scored={s:[r for r in rows if r["role"]==s and r["scored"]] for s in FAMILIES}
    n={s:len(v) for s,v in scored.items()}
    human={s:sum(r["human_closed"] for r in v)/len(v) if v else 0. for s,v in scored.items()}
    valid=n["train"]>=20 and n["calibration"]>=8 and n["evaluation"]>=20 and min(human.values())>=.95
    report={"schema":SCHEMA,"source_commit":COMMIT,"scored":n,
            "selected":dict(Counter(r["role"] for r in rows)),
            "human_close1":human,"native_valid":valid,"model_training_updates":0,
            "architecture_promotion_eligible":False,"results":{}}
    if valid:
        train=scored["train"];order=fixed_order(train)
        trees={"binary":Empirical(train,False),"diagnostic":Empirical(train,True)}
        episodes=[]
        for s in ("calibration","evaluation"):
            v=scored[s];part=[episode(r,a,order,trees) for r in v for a in ARMS]
            episodes.extend(part)
            results={}
            for a in ARMS:
                p=[r for r in part if r["arm"]==a]
                results[a]={"closed":sum(r["closed"] for r in p),"roots":len(p),
                            "mean_calls":sum(r["calls"] for r in p)/len(p)}
            envelope=sum(any(closed(x) for x in r["receipts"]) for r in v)
            best=max(results[a]["closed"] for a in ARMS)
            report["results"][s]={"arms":results,"envelope":envelope,"best":best,
                                   "optimistic_gap":(envelope-best)/len(v)}
            if s=="calibration":
                selected=max(ARMS,key=lambda a:(results[a]["closed"],-results[a]["mean_calls"],-ARMS.index(a)))
                report["calibration_selected"]=selected
        report["fixed_order"]=[MENU[j] for j in order]
        ev=report["results"]["evaluation"]
        report["headroom"]=all(v["optimistic_gap"]>=.1 for v in report["results"].values()) and ev["best"]<=.8*n["evaluation"]
        report["selected_fresh"]=ev["arms"][selected]
        write(out/"policy_episodes.json",episodes)
    else:report["headroom"]=False
    report["files"]={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(out/"manifest.json",report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lean-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.lean_source,a.out),sort_keys=True),flush=True)


if __name__=="__main__":main()
