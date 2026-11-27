"""OLR-06H: native-certified counterfactual frontier-component utility.

One shared demand prefix; choose a single live component for a block, then
continue the exact demand controller. All graph state survives the decision.
This is a zero-training headroom screen, not a learned scheduler.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
import math
from pathlib import Path
import random
import time

from .data import canonical, sha256_file
from .equality_suite import ALPHABET, RULE_COUNT, BANDS, choose_shell, system_lineage
from .olr06e_multigoal import Root, Meter, ProposalLedger, bounded, certify, read, write
from .olr06f_anytime import path_audit
from .olr06g_portfolio import PAIRS, load as load_old, schedule, utility

SEED=20270107
OLD_SHA="581f2a247e0e7cd2f8abd452dcd24cb46e2caf253d6e5dac9963c1f94a74b5ce"
COUNTS={"train":{"dev":32,"transfer":32},"calibration":{"dev":8,"transfer":8},
        "evaluation":{"dev":16,"transfer":16}}
PREFIX=128
BLOCK=64
CAP=512
SLOTS=6
SAMPLE=8
CONTROLS=("continue","balanced_block","demand_block","contact_block","largest_block","cache128_demand")
REG="olr06h.frontier_registry.v1"
SOURCE="olr06h.frontier_source.v1"
SCORE="olr06h.frontier_score.v1"


def from_row(row):
    if set(row)!=set(Root.__dataclass_fields__):
        raise ValueError("unexpected public root metadata")
    return Root(row["split"],tuple(tuple(e) for e in row["equations"]),
                row["source"],tuple(row["targets"])).validate()


def freeze(old,out):
    if out.exists() or sha256_file(old/"manifest.json")!=OLD_SHA:
        raise ValueError("pinned 06G bank and create-only output required")
    _,old_groups=load_old(old)
    historical=set(read(old/"excluded.json"))
    previous={r.lineage for rs in old_groups.values() for r in rs}
    if len(historical)!=300 or len(previous)!=128 or historical&previous:
        raise ValueError("prior lineage roster changed")
    out.mkdir(parents=True)
    groups={r:list(old_groups[r]) for r in ("train","calibration")}
    excluded=historical|previous
    used=set(excluded)
    groups["evaluation"]=[]
    pairs=list(combinations(PAIRS,2))
    for split,count in COUNTS["evaluation"].items():
        rng=random.Random(f"olr06h/{SEED}/{split}")
        got=[];k=3 if split=="dev" else 5
        for _ in range(count*1000):
            if len(got)==count:break
            rules=tuple((a,b) if rng.randrange(2) else (b,a)
                        for a,b in rng.sample(pairs,RULE_COUNT))
            lineage=system_lineage(rules)
            if lineage in used:continue
            n=7+len(got)%3 if split=="dev" else 10
            source="".join(rng.choice(ALPHABET) for _ in range(n))
            targets=[]
            for j in range(k):
                t=choose_shell(source,rules,BANDS[j%3],rng)
                if t is None or t in targets:break
                targets.append(t)
            if len(targets)!=k:continue
            rng.shuffle(targets)
            got.append(Root(split,rules,source,tuple(targets)).validate())
            used.add(lineage)
        if len(got)!=count:raise ValueError("fresh root generator exhausted")
        groups["evaluation"].extend(got)
    files={}
    for role,rs in groups.items():
        p=out/(role+".json");write(p,[asdict(r) for r in rs]);files[p.name]=sha256_file(p)
    write(out/"excluded.json",sorted(excluded))
    files["excluded.json"]=sha256_file(out/"excluded.json")
    m={"schema":REG,"seed":SEED,"old_manifest_sha256":OLD_SHA,
       "excluded_lineages":428,"counts":COUNTS,"prefix":PREFIX,"block":BLOCK,"cap":CAP,
       "files":files}
    write(out/"manifest.json",m)
    return m


def load(bank):
    m=read(bank/"manifest.json")
    if (m["schema"]!=REG or m["seed"]!=SEED or m["old_manifest_sha256"]!=OLD_SHA
            or m["counts"]!=COUNTS or (m["prefix"],m["block"],m["cap"])!=(PREFIX,BLOCK,CAP)):
        raise ValueError("unregistered 06H bank")
    for name,h in m["files"].items():
        if sha256_file(bank/name)!=h:raise ValueError("frozen bank file changed")
    excluded=set(read(bank/"excluded.json"))
    if len(excluded)!=428:raise ValueError("lineage exclusion census changed")
    seen=set();groups={}
    for role,panels in COUNTS.items():
        roots=[from_row(r) for r in read(bank/(role+".json"))]
        if {s:sum(r.split==s for r in roots) for s in panels}!=panels:
            raise ValueError("registered role counts changed")
        for r in roots:
            if r.lineage in seen or ((r.lineage in excluded)!=(role!="evaluation")):
                raise ValueError("source role or fresh lineage violation")
            seen.add(r.lineage)
        groups[role]=roots
    return m,groups


class State:
    def __init__(self,root):
        self.root=root
        self.meter=Meter(root.equations,CAP,False)
        self.graph=ProposalLedger(root)
        self.pending=set(root.targets)
        self.found={}

    def advance(self,limit,focus=None):
        """Advance without resetting any graph state or meter.

        Focus is an endpoint word; after a merge its representative follows
        the union-find component. If its frontier exhausts, fall back.
        """
        if not self.meter.expansions<=limit<=CAP:
            raise ValueError("invalid continuation budget")
        while self.pending and self.meter.expansions<limit:
            w=None
            if focus is not None:
                c=self.graph.find(focus)
                front=self.graph.front[c]
                if front:
                    w=min(front,key=lambda t:(self.graph.level[t],t))
                else:
                    focus=None
            if w is None:w=self.graph.select(self.pending,"demand")
            if w is None:break
            self.graph.expand(w,self.meter)
            for t,path in self.graph.connected_paths(self.pending).items():
                self.found[t]={"words":path,"found_at":self.meter.required_cap,
                               "fresh_at":self.meter.expansions}
                self.pending.remove(t)

    def active(self):
        if not self.pending or self.meter.expansions>=CAP:return {}
        endpoints=(self.root.source,*self.root.targets)
        result={}
        seen=set()
        for slot,w in enumerate(endpoints):
            if slot and w not in self.pending:continue
            c=self.graph.find(w)
            if c not in seen and self.graph.front[c]:
                result[slot]=c;seen.add(c)
        return result

    def result(self,seconds):
        row={"found":{t:{**q,"abstract_at":0} for t,q in self.found.items()},
             **self.meter.stats(),"abstract_expansions":0,"abstract_edges":0,
             "rekeys":0,"pdb_seconds":0.,"total_work":self.meter.expansions,"seconds":seconds}
        path_audit(self.root,row,CAP)
        return row


def verify_identity(state,limit):
    """Audit against original 06E implementation; charge only as offline audit."""
    old=bounded(state.root,"demand",limit)
    if state.found!=old["found"] or any(
            state.meter.stats()[k]!=old[k] for k in state.meter.stats()):
        raise ValueError("unmodified demand path/counter identity failed")
    return old["expansions"],old["seconds"]


def observe(state):
    """Public checkpoint graph, with no extra neighbor enumerations."""
    start=time.perf_counter()
    root=state.root;g=state.graph;n=len(root.source)
    endpoints=(root.source,*root.targets)
    source=g.find(root.source)
    active=state.active()
    comps=[None]*SLOTS
    for slot,c in active.items():
        front=sorted(g.front[c],key=lambda t:(g.level[t],t))
        members=[i for i,w in enumerate(endpoints) if g.find(w)==c]
        demand=len(state.pending) if c==source else sum(
            g.find(t)==c for t in state.pending)
        levels=[g.level[w] for w in front]
        if not demand or not front:
            raise ValueError("inactive component offered")
        comps[slot]={"slot":slot,"members":members,"witness":c==source,
                    "pending_demand":demand,"size":g.size[c],"frontier":len(front),
                    "expanded":g.size[c]-len(front),"min_level":min(levels),
                    "mean_level":sum(levels)/len(levels),"max_level":max(levels),
                    "samples":front[:SAMPLE]}
    contacts=[];comparisons=0
    for i,j in combinations(range(SLOTS),2):
        a,b=comps[i],comps[j]
        if a is None or b is None:
            contacts.append({"endpoints":[i,j],"active":False,
                             "min_hamming":0,"mean_hamming":0.,"comparisons":0})
            continue
        ds=[sum(x!=y for x,y in zip(u,v,strict=True))
            for u in a["samples"] for v in b["samples"]]
        comparisons+=len(ds)
        contacts.append({"endpoints":[i,j],"active":True,
                         "min_hamming":min(ds),"mean_hamming":sum(ds)/(len(ds)*n),
                         "min_normalized":min(ds)/n,"comparisons":len(ds)})
    packet={"public_root":asdict(root),"root":root.root,"split":root.split,
            "prefix_work":state.meter.expansions,"remaining_work":CAP-state.meter.expansions,
            "prefix_counters":state.meter.stats(),"found_mask":[t in state.found for t in root.targets],
            "pending":len(state.pending),"components":comps,"contacts":contacts,
            "legal":[True]+[c is not None for c in comps],"pair_comparisons":comparisons,
            "feature_seconds":time.perf_counter()-start}
    return packet


def controls(packet):
    """Decisions derived entirely from the public prefix packet."""
    legal=[c for c in packet["components"] if c is not None]
    if not legal:return {a:0 for a in CONTROLS if a!="cache128_demand"}
    n=len(packet["public_root"]["source"])
    distance={c["slot"]:n for c in legal}
    for p in packet["contacts"]:
        if p["active"]:
            a,b=p["endpoints"];h=p["min_hamming"]
            distance[a]=min(distance[a],h);distance[b]=min(distance[b],h)
    def choose(fn):return 1+min(legal,key=lambda c:(fn(c),c["slot"]))["slot"]
    out={"continue":0,
         "balanced_block":choose(lambda c:c["frontier"]),
         "demand_block":choose(lambda c:Fraction(c["frontier"],c["pending_demand"])),
         "contact_block":choose(lambda c:Fraction(
             (1+distance[c["slot"]])*c["frontier"],n*c["pending_demand"])),
         "largest_block":choose(lambda c:-c["pending_demand"])}
    if any(not packet["legal"][a] for a in out.values()):
        raise ValueError("control selected illegal component")
    return out


def prefix(root):
    s=State(root);start=time.perf_counter();s.advance(PREFIX)
    seconds=time.perf_counter()-start
    return s,seconds


def suffix(checkpoint,action,prefix_seconds,legal):
    """Teacher branches clone actual state. Per-policy prefix remains paid."""
    if type(action) is not int or not 0<=action<len(legal) or not legal[action]:
        raise ValueError("illegal block action")
    start=time.perf_counter()
    s=deepcopy(checkpoint)
    clone_seconds=time.perf_counter()-start
    start=time.perf_counter()
    before=s.meter.expansions
    if action:
        rep=(s.root.source,*s.root.targets)[action-1]
        s.advance(min(CAP,before+BLOCK),focus=rep)
    after=s.meter.expansions
    s.advance(CAP)
    seconds=time.perf_counter()-start
    r=s.result(prefix_seconds+seconds)
    r.update({"action":action,"prefix_work":before,
              "block_work":after-before,"continuation_work":s.meter.expansions-after,
              "teacher_clone_seconds":clone_seconds})
    if r["total_work"]!=before+r["block_work"]+r["continuation_work"]:
        raise ValueError("cloned prefix work lost")
    return s,r


def propose(roots,out):
    """Freeze all decisions before computing candidate continuation outcomes."""
    checkpoints=[];packets=[];decisions=[];audit_exp=0;audit_seconds=0.
    for r in roots:
        s,seconds=prefix(r);p=observe(s)
        a,t=verify_identity(s,PREFIX);audit_exp+=a;audit_seconds+=t
        checkpoints.append((r,s,seconds,p))
        packets.append(p);decisions.append({"root":r.root,"split":r.split,"choices":controls(p)})
    write(out/"prefix_packets.json",packets)
    write(out/"frozen_control_choices.json",decisions)
    rows=[]
    for root,s,seconds,p in checkpoints:
        for action,allowed in enumerate(p["legal"]):
            if not allowed:continue
            terminal,result=suffix(s,action,seconds,p["legal"])
            if action==0:
                a,t=verify_identity(terminal,CAP);audit_exp+=a;audit_seconds+=t
            rows.append({"root":root.root,"split":root.split,"arm":f"action_{action}",**result})
        # The fixed restart reference executes from scratch and pays its own
        # first stage. It receives neither checkpoint nor counterfactual table.
        r=schedule(root,"cache128_demand")
        rows.append({"root":root.root,"split":root.split,"arm":"cache128_demand",**r})
    write(out/"frozen_plans.json",rows)
    return packets,decisions,rows,{"identity_audit_expansions":audit_exp,
                                   "identity_audit_seconds":audit_seconds}


def grade(roots,out):
    packets,decisions,rows,audit=propose(roots,out)
    certificate=certify(roots,rows,out)
    write(out/"native_counterfactual_scores.json",rows)
    root_map={r.root:r for r in roots}
    index={(r["root"],r["arm"]):r for r in rows}
    controls_rows=[];oracle_rows=[];targets=[]
    for p,d in zip(packets,decisions,strict=True):
        rid=p["root"];k=len(root_map[rid].targets)
        y=[None]*len(p["legal"]);full=[None]*len(p["legal"])
        for action,allowed in enumerate(p["legal"]):
            if allowed:
                r=index[rid,f"action_{action}"]
                y[action]=utility(r,k);full[action]=bool(r["joint_lemma"])
        legal=[a for a,v in enumerate(y) if v is not None]
        best=max(legal,key=lambda a:(y[a],-a))
        targets.append({"root":rid,"split":p["split"],"legal":p["legal"],
                        "utility":y,"full":full,"best_action":best,
                        "actionable":bool(p["pending"] and len(legal)>=2),
                        "different_full":len({full[a] for a in legal})>1})
        def row(arm,selected,action):
            return {"root":rid,"split":p["split"],"arm":arm,"action":action,
                    "full":bool(selected["joint_lemma"]),"coverage":len(selected["found"])/k,
                    "utility":utility(selected,k),"total_work":selected["total_work"],
                    "search_seconds":selected["seconds"],
                    "feature_seconds":p["feature_seconds"] if arm not in ("continue","cache128_demand") else 0.,
                    "pair_comparisons":p["pair_comparisons"] if arm not in ("continue","cache128_demand") else 0,
                    "native_tactics":sum(q["native_tactics"] for q in selected["found"].values())}
        for arm,a in d["choices"].items():
            controls_rows.append(row(arm,index[rid,f"action_{a}"],a))
        controls_rows.append(row("cache128_demand",index[rid,"cache128_demand"],None))
        oracle_rows.append(row("hindsight",index[rid,f"action_{best}"],best))
    write(out/"teacher_targets.json",targets)
    return packets,controls_rows,oracle_rows,targets,certificate,audit


def select_source(train,controls_rows):
    index={r.root for r in train}
    return {s:max(CONTROLS,key=lambda a:(
        sum(r["utility"] for r in controls_rows if r["root"] in index
            and r["split"]==s and r["arm"]==a),-CONTROLS.index(a)))
            for s in ("dev","transfer")}


def add_fixed(rows,selection):
    return [*rows,*[{**r,"arm":"source_fixed","selected_control":r["arm"]}
                     for r in rows if r["arm"]==selection[r["split"]]]]


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["split"],r["arm"]].append(r)
    out={}
    for (split,arm),v in sorted(groups.items()):
        out[f"{split}:{arm}"]={"trials":len(v),"full":sum(r["full"] for r in v),
            **{f"mean_{k}":sum(r[k] for r in v)/len(v) for k in
               ("coverage","utility","total_work","search_seconds","feature_seconds",
                "pair_comparisons","native_tactics")},
            "actions":{str(a):sum(r["action"]==a for r in v)
                       for a in sorted({r["action"] for r in v if r["action"] is not None})}}
    return out


def screen(roots,rows,targets):
    ids={r.root for r in roots};subset=[r for r in rows if r["root"] in ids]
    ts=[r for r in targets if r["root"] in ids]
    score=summarize(subset)
    controls=(*CONTROLS,"source_fixed")
    best={s:max(controls,key=lambda a:(score[f"{s}:{a}"]["full"],-controls.index(a)))
          for s in ("dev","transfer")}
    oracle=sum(r["full"] for r in subset if r["arm"]=="hindsight")
    best_full=sum(score[f"{s}:{a}"]["full"] for s,a in best.items())
    n=len(roots);actionable=sum(r["actionable"] for r in ts)
    differing=sum(r["different_full"] for r in ts)
    gap=(oracle-best_full)/n
    # Descriptive paired root interval, using hindsight and posthoc best
    # control per panel: an optimistic screen, not a fitted-policy effect.
    index={(r["root"],r["arm"]):r for r in subset}
    diffs=[int(index[r.root,"hindsight"]["full"])-int(index[r.root,best[r.split]]["full"])
           for r in roots]
    rng=random.Random(f"06h/{roots[0].root}/headroom")
    draws=sorted(sum(rng.choice(diffs) for _ in roots)/n for _ in range(5000))
    return {"roots":n,"best_control_by_panel":best,"best_full":best_full,
            "hindsight_full":oracle,"hindsight_gap":gap,"actionable":actionable,
            "different_full_labels":differing,"descriptive_ci95":[draws[125],draws[4875]],
            "headroom":best_full<=.8*n and gap>=.1 and actionable>=.2*n,"scorecard":score}


def hashes(path):
    names=("prefix_packets.json","frozen_control_choices.json","frozen_plans.json",
           "native_counterfactual_scores.json","teacher_targets.json","policy_rows.json",
           "native_episodes.json","conjunctions.lean","conjunction_compiler.log",
           "leaves/standalone.lean","leaves/compiler.log","leaves/compiler_receipt.json")
    return {n:sha256_file(path/n) for n in names}


def source(bank,out):
    if out.exists():raise ValueError("create-only source output")
    start=time.perf_counter();_,groups=load(bank);out.mkdir(parents=True)
    roots=groups["train"]+groups["calibration"]
    packets,cr,oracle,targets,cert,audit=grade(roots,out)
    selected=select_source(groups["train"],cr)
    rows=add_fixed(cr,selected)+oracle;write(out/"policy_rows.json",rows)
    summary={role:screen(groups[role],rows,targets) for role in ("train","calibration")}
    m={"schema":SOURCE,"registry_sha256":sha256_file(bank/"manifest.json"),
       "train_roots":len(groups["train"]),"calibration_roots":len(groups["calibration"]),
       "counterfactual_trials":sum(sum(r["legal"]) for r in targets),
       "fixed_restart_trials":len(roots),"source_fixed":selected,
       "source_gate":summary["train"]["headroom"],"panels":summary,"certificate":cert,
       "offline_identity_audit":audit,"model_training_updates":0,
       "architecture_promotion_eligible":False,"elapsed_seconds":time.perf_counter()-start,
       "files":hashes(out)}
    write(out/"manifest.json",m);return m


def score(bank,teacher,out):
    if out.exists():raise ValueError("create-only evaluation output")
    start=time.perf_counter();_,groups=load(bank)
    fm=read(teacher/"manifest.json")
    if (fm["schema"]!=SOURCE or fm["registry_sha256"]!=sha256_file(bank/"manifest.json")
            or fm["model_training_updates"]!=0):
        raise ValueError("source teacher identity changed")
    for name,h in fm["files"].items():
        if sha256_file(teacher/name)!=h:raise ValueError("source teacher artifact changed")
    roots=groups["evaluation"];out.mkdir(parents=True)
    # Source choice is frozen before any fresh prefix or candidate outcome.
    write(out/"source_fixed.json",fm["source_fixed"])
    packets,cr,oracle,targets,cert,audit=grade(roots,out)
    rows=add_fixed(cr,fm["source_fixed"])+oracle
    write(out/"policy_rows.json",rows);result=screen(roots,rows,targets)
    m={"schema":SCORE,"registry_sha256":sha256_file(bank/"manifest.json"),
       "source_sha256":sha256_file(teacher/"manifest.json"),"evaluation_roots":len(roots),
       "counterfactual_trials":sum(sum(t["legal"]) for t in targets),
       "fixed_restart_trials":len(roots),"source_fixed":fm["source_fixed"],
       "source_gate":fm["source_gate"],"fresh_gate":result["headroom"],
       "reader_fit_eligible":fm["source_gate"] and result["headroom"],
       "screen":result,"certificate":cert,"offline_identity_audit":audit,
       "model_training_updates":0,"architecture_promotion_eligible":False,
       "elapsed_seconds":time.perf_counter()-start,
       "files":{**hashes(out),"source_fixed.json":sha256_file(out/"source_fixed.json")}}
    write(out/"manifest.json",m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="phase",required=True)
    for phase in ("freeze","source","score"):
        a=sub.add_parser(phase);a.add_argument("--out",type=Path,required=True)
        if phase=="freeze":a.add_argument("--old-bank",type=Path,required=True)
        else:a.add_argument("--bank",type=Path,required=True)
        if phase=="score":a.add_argument("--teacher",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="freeze":m=freeze(args.old_bank,args.out)
    elif args.phase=="source":m=source(args.bank,args.out)
    else:m=score(args.bank,args.teacher,args.out)
    print(canonical(m),flush=True)


if __name__=="__main__":main()
