"""LBF-20E native branching goals and context-scoped proof reuse."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from lbf20e_horn import Edge, masks_for, audit_table, tree_cost, tree_witness, min_dag, audit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lean_cgrt_hrm400"))
from lcgrt.lean_repl import LeanRepl

SEED = 2026112512
DOMAIN = range(3)
CALL_CAP = 96
WALL_CAP = 4.
ARMS = ("native_order", "native_tree", "native_dag", "deferred_tree", "deferred_dag")
NP = 17


def atom(a):
    return f"p{a[0]} {a[1]}"


@dataclass(frozen=True)
class Rule:
    # Template atom = (predicate id, argument-variable index, Nat offset).
    head: tuple
    body: tuple
    arity: int

    def formula(self):
        def expr(a):
            p,v,d = a
            x = f"x{v}" if d == 0 else f"(x{v} + {d})"
            return f"p{p} {x}"
        return "∀ (" + " ".join(f"x{j}" for j in range(self.arity)) + " : Nat), " + " → ".join(
            [expr(a) for a in self.body] + [expr(self.head)])

    def instances(self, i):
        for args in itertools.product(DOMAIN, repeat=self.arity):
            def ground(a):
                return a[0], args[a[1]] + a[2]
            yield Edge(ground(self.head), tuple(ground(a) for a in self.body), i, args)


@dataclass(frozen=True)
class Case:
    root: str
    family: str
    facts: tuple
    local: tuple
    target: tuple
    rules: tuple
    witness_rule: int
    index: int
    witness: int

    def statement(self):
        binders = ["(" + " ".join(f"p{i}" for i in range(NP)) + " : Nat → Prop)"]
        binders += [f"(h{i} : {atom(a)})" for i,a in enumerate(self.facts)]
        binders += [f"(r{i} : {r.formula()})" for i,r in enumerate(self.rules)]
        return "∀ " + " ".join(binders) + ", " + " ∧ ".join(f"({atom(a)} → {atom(self.target)})" for a in self.local)

    def intros(self):
        return "intro " + " ".join([*(f"p{i}" for i in range(NP)),
                                   *(f"h{i}" for i in range(len(self.facts))),
                                   *(f"r{i}" for i in range(len(self.rules)))])

    def edges(self):
        return tuple(e for i,r in enumerate(self.rules) for e in r.instances(i))

    def context(self, branch):
        return {**{a:f"h{i}" for i,a in enumerate(self.facts)}, self.local[branch]:f"ctx{branch}"}


def generate():
    rng = random.Random(SEED)
    cases = []
    for i in range(24):
        # Shared chain -> diamond -> witness-bearing goal. A/B are branch local.
        A,B,C0,C,D,E,F,W,V,G,U,*chain = rng.sample(range(NP),NP)
        assert len(chain)==6
        k,w = rng.randrange(3),rng.randrange(3)
        family = ("shared","witness_choice","reuse_tradeoff")[i%3]
        def t(p,d=0,v=0): return (p,v,d)
        rules = [
            Rule(t(C0), (t(A),), 1), Rule(t(C0), (t(B),), 1),
            Rule(t(C,1), (t(C0),), 1),
            Rule(t(D,2), (t(C,1),), 1), Rule(t(E,2), (t(C,1),), 1),
            Rule(t(F,3), (t(D,2), t(E,2)), 1),
            Rule(t(G,4), (t(F,3), t(W,0,1), t(V,0,1)), 2),
            Rule(t(G,4), (t(U),), 1),
            Rule(t(C0), (t(D,2),), 1),
            Rule(t(F,3), (t(U),t(C,1)), 1),
        ]
        witness_rule = rules[6]
        facts = [(W,w),(V,w)]
        if family != "shared":
            others = [j for j in DOMAIN if j!=w]
            facts.extend([(W,others[0]),(V,others[1])])
        if family=="reuse_tradeoff":
            rules.extend([Rule(t(chain[0]),(t(A),),1),Rule(t(chain[0]),(t(B),),1)])
            rules.extend(Rule(t(q), (t(p),),1) for p,q in zip(chain,chain[1:]))
            rules.append(Rule(t(G,4),(t(chain[-1]),),1))
        rng.shuffle(facts);rng.shuffle(rules)
        cases.append(Case(f"dev_{i:02d}",family,tuple(facts),((A,k),(B,k)),(G,k+4),
                          tuple(rules),rules.index(witness_rule),k,w))
    return cases


def parse_atom(target):
    raw = target.get("raw","").strip()
    m = re.fullmatch(r"p(\d+)\s+(.+)", raw, flags=re.S)
    if m is None:
        raise ValueError(f"unrecognized native atom: {target}")
    expr = m[2]
    tokens = re.findall(r"Nat\.succ|\d+|[()+]",expr)
    if "".join(tokens) != re.sub(r"\s+","",expr):
        raise ValueError(f"unexpected Nat expression: {expr}")
    pos = 0
    def term():
        nonlocal pos
        tok=tokens[pos];pos+=1
        if tok=="(":
            v=add()
            assert tokens[pos]==")"
            pos+=1
            return v
        if tok=="Nat.succ":
            return 1+term()
        assert tok.isdecimal(),(raw,tok)
        return int(tok)
    def add():
        nonlocal pos
        n=term()
        while pos<len(tokens) and tokens[pos]=="+":
            pos+=1;n+=term()
        return n
    value=add()
    assert pos==len(tokens),(raw,tokens[pos:])
    return int(m[1]),value


def hyps(repl,g):
    return repl.contexts.get(g["ctx"],g.get("hyps",[]))


def decode(repl,case,state):
    out=[]
    for g in state["goals"]:
        ty=g["target"]
        if "imp" in ty:
            a,b=map(parse_atom,ty["imp"])
            assert a in case.local and b==case.target
            out.append(("intro",case.local.index(a),b))
        else:
            a=parse_atom(ty)
            local=[j for j in range(2) if any(h["name"]==f"ctx{j}" and parse_atom(h["type"])==case.local[j]
                                             for h in hyps(repl,g))]
            assert len(local)==1,(ty,hyps(repl,g))
            out.append(("atom",local[0],a))
    return tuple(out)


def application(e, args=()):
    return "("+" ".join([f"r{e.rule}",*(str(x) for x in e.args),*args])+")"


def compile_tree(edges, facts, choice, goal):
    if goal in facts:
        return facts[goal]
    e=edges[choice[goal]]
    return application(e,[compile_tree(edges,facts,choice,a) for a in e.body])


def dag_bindings(edges, facts, plan, goal, branch):
    names=dict(facts);out=[]
    for j,i in enumerate(plan):
        e=edges[i]
        assert e.head not in names and all(a in names for a in e.body)
        name=f"d{branch}_{j}"
        expr=application(e,[names[a] for a in e.body])
        out.append((name,e.head,expr))
        names[e.head]=name
    return out,names[goal]


def dag_term(bindings,tail):
    return "("+" ".join(f"let {n} : {atom(a)} := {expr};" for n,a,expr in bindings)+" "+tail+")"


def plan_case(case):
    edges=case.edges();plans=[]
    for j in range(2):
        facts=case.context(j)
        dist,choice=tree_cost(edges,facts)
        table,rounds=masks_for(edges,facts)
        count=audit_table(edges,facts,table)
        tree=tree_witness(edges,facts,dist,choice,case.target)
        dag=min_dag(edges,facts,table,case.target)
        assert dag is not None and len(tree)==dist[case.target]
        bindings,tail=dag_bindings(edges,facts,dag,case.target,j)
        plans.append({"dist":dist,"choice":choice,"tree":tree,"dag":dag,
                      "bindings":bindings,"tail":tail,"masks_checked":count,
                      "rounds":rounds,"frontier":len(table[case.target])})
    return edges,plans


class Episode:
    def __init__(self,repl,case,arm):
        self.repl,self.case,self.arm=repl,case,arm
        self.start=time.perf_counter();self.calls=0;self.native_seconds=0.;self.kernel_seconds=0.
        self.planning_seconds=0.;self.events=[];self.rollbacks=0;self.receipt=None
        self.script=[];self.status="exhausted"
        repl.reset()
        self.root=repl.init(case.statement())
        self.init_seconds=time.perf_counter()-self.start
        assert len(self.root["goals"])==1 and "and" in self.root["goals"][0]["target"],self.root

    def alive(self):
        return self.calls<CALL_CAP and time.perf_counter()-self.start<WALL_CAP

    def run(self,state,tac,kind):
        if not self.alive():return None
        t0=time.perf_counter()
        o=self.repl.run(state["sid"],tac)
        dt=time.perf_counter()-t0
        self.calls+=1;self.native_seconds+=dt
        assert type(o.get("ok")) is bool,o
        row={"kind":kind,"tactic":tac,"accepted":o["ok"],"seconds":dt,
             "before":[g["target"] for g in state["goals"]]}
        if o["ok"]:
            row["after"]=[g["target"] for g in o["state"]["goals"]]
        else:
            row["error"]=str(o.get("error",""))[:240]
        self.events.append(row)
        return o["state"] if o["ok"] else False

    def certify(self,state,script):
        assert state["closed"] and not state["goals"]
        t0=time.perf_counter()
        cert=self.repl.kernel(state["sid"])
        self.kernel_seconds+=time.perf_counter()-t0
        assert cert.get("ok") is True and cert.get("axioms")==[],cert
        self.receipt={"axioms":[],"term_chars":len(cert.get("term",""))}
        self.status="closed" if time.perf_counter()-self.start<=WALL_CAP else "late_closed"
        self.script=list(script)

    def finish(self,extra):
        wall=time.perf_counter()-self.start
        if self.status=="exhausted":
            self.status="call_cap" if self.calls>=CALL_CAP else "time_cap" if wall>=WALL_CAP else "exhausted"
        return {"root":self.case.root,"family":self.case.family,"arm":self.arm,"status":self.status,
                "closed":self.status=="closed","calls":self.calls,"wall_seconds":wall,
                "init_seconds":self.init_seconds,"native_seconds":self.native_seconds,
                "kernel_seconds":self.kernel_seconds,"planning_seconds":self.planning_seconds,
                "rollbacks":self.rollbacks,"rejects":sum(not e["accepted"] for e in self.events),
                "kernel":self.receipt,"script":self.script,
                "script_chars":sum(len(x) for x in self.script),"events":self.events,**extra}


def constructor(ep):
    s=ep.run(ep.root,"constructor","constructor")
    if not s:return None
    assert collections.Counter(decode(ep.repl,ep.case,s))==collections.Counter(
        ("intro",j,ep.case.target) for j in range(2))
    return s


def intro(ep,state,desc):
    kind,j,a=desc
    assert kind=="intro"
    s=ep.run(state,f"intro ctx{j}","intro")
    if s:
        expected=[("atom",j,a),*decode(ep.repl,ep.case,state)[1:]]
        assert collections.Counter(decode(ep.repl,ep.case,s))==collections.Counter(expected)
    return s


def live_tree(ep,edges,plans):
    case=ep.case
    state=constructor(ep)
    if state is None:return
    def visit(s,script,ancestors):
        if s["closed"]:
            ep.certify(s,script);return True
        if not ep.alive():return False
        goals=decode(ep.repl,case,s)
        if goals in ancestors:return False
        kind,j,a=goals[0]
        if kind=="intro":
            after=intro(ep,s,goals[0])
            return bool(after) and visit(after,(*script,f"intro ctx{j}"),ancestors|{goals})
        facts=case.context(j)
        if a in facts:
            candidates=[("exact "+facts[a],())]
        else:
            t0=time.perf_counter()
            menu=[(i,e) for i,e in enumerate(edges) if e.head==a]
            if ep.arm=="native_tree":
                d=plans[j]["dist"]
                menu.sort(key=lambda pair:(sum(d.get(x,float("inf")) for x in pair[1].body),pair[0]))
                menu=[(i,e) for i,e in menu if all(x in d for x in e.body)]
            candidates=[("apply "+application(e),e.body) for _,e in menu]
            ep.planning_seconds+=time.perf_counter()-t0
        for tac,body in candidates:
            if not ep.alive():break
            after=ep.run(s,tac,"apply" if body else "exact_or_fact")
            if not after:continue
            expected=[*(("atom",j,x) for x in body),*goals[1:]]
            assert collections.Counter(decode(ep.repl,case,after))==collections.Counter(expected),(tac,after,expected)
            if visit(after,(*script,tac),ancestors|{goals}):return True
            ep.rollbacks+=1
        return False
    visit(state,("constructor",),set())


def live_dag(ep,plans):
    case=ep.case;state=constructor(ep)
    if state is None:return
    script=["constructor"]
    for _ in range(2):
        desc=decode(ep.repl,case,state)[0]
        _,j,_=desc
        state=intro(ep,state,desc)
        if not state:return
        script.append(f"intro ctx{j}")
        for name,a,expr in plans[j]["bindings"]:
            old=decode(ep.repl,case,state)
            tac=f"have {name} : {atom(a)} := {expr}"
            state=ep.run(state,tac,"have")
            if not state:return
            assert decode(ep.repl,case,state)==old
            assert any(h["name"]==name and parse_atom(h["type"])==a for h in hyps(ep.repl,state["goals"][0]))
            script.append(tac)
        tac="exact "+plans[j]["tail"]
        state=ep.run(state,tac,"exact")
        if not state:return
        script.append(tac)
    ep.certify(state,script)


def deferred(ep,edges,plans):
    case=ep.case;terms=[]
    t0=time.perf_counter()
    for j,p in enumerate(plans):
        if ep.arm=="deferred_tree":
            expr=compile_tree(edges,case.context(j),p["choice"],case.target)
        else:
            expr=dag_term(p["bindings"],p["tail"])
        terms.append(f"(fun ctx{j} => {expr})")
    tac="exact And.intro "+" ".join(terms)
    ep.planning_seconds+=time.perf_counter()-t0
    after=ep.run(ep.root,tac,"deferred")
    assert after,ep.events[-1]
    ep.certify(after,[tac])


def audit_native(repl,case,plans):
    """Separate audit calls. No controller uses these verdicts."""
    calls=0
    def req(s,t):
        nonlocal calls
        calls+=1;return repl.run(s["sid"],t)
    repl.reset();root=repl.init(case.statement())
    s=req(root,"constructor")["state"]
    j=decode(repl,case,s)[0][1]
    s=req(s,f"intro ctx{j}")["state"]
    inf=[]
    # Try implicit substitution, partially specified witness, fully instantiated.
    for tactic in (f"apply r{case.witness_rule}",
                   f"apply (r{case.witness_rule} {case.index})",
                   f"apply (r{case.witness_rule} {case.index} {case.witness})"):
        o=req(s,tactic)
        inf.append({"tactic":tactic,"accepted":o["ok"],
                    "goals":[g["target"] for g in o["state"]["goals"]] if o["ok"] else [],
                    "error":str(o.get("error",""))[:240]})
    assert inf[-1]["accepted"],inf
    assert len(inf[-1]["goals"])==4,inf # three premises plus second implication
    p=plans[j]
    expr=dag_term(p["bindings"],p["tail"])
    s=req(s,f"have branch_only : {atom(case.target)} := {expr}")["state"]
    s=req(s,"exact branch_only")["state"]
    j2=decode(repl,case,s)[0][1];assert j2!=j
    s=req(s,f"intro ctx{j2}")["state"]
    o=req(s,"exact branch_only")
    assert o["ok"] is False,o
    p=plans[j2]
    o2=req(s,"exact "+dag_term(p["bindings"],p["tail"]))
    assert o2["ok"] and o2["state"]["closed"],o2
    cert=repl.kernel(o2["state"]["sid"])
    assert cert["ok"] and cert["axioms"]==[],cert
    return {"root":case.root,"inference":inf,"cross_context_cache_rejected":True,
            "corrected_proof_kernel_closed":True,"tactic_calls":calls}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    au=audit()
    print("HORN_AUDIT",json.dumps(au),flush=True)
    cases=generate();rows=[];native_audits=[];proofs=["import Lean",""]
    case_receipts=[]
    with LeanRepl(imports=("Lean",),timeout=20) as repl:
        assert str(repl.version).startswith("4.19.0"),repl.version
        for i,case in enumerate(cases):
            # Full reference plans for audit. Never pass them into native_order.
            t0=time.perf_counter();edges,plans=plan_case(case)
            reference_time=time.perf_counter()-t0
            metrics={"tree_rule_applications":sum(len(p["tree"]) for p in plans),
                     "dag_rule_applications":sum(len(p["dag"]) for p in plans),
                     "min_dag_masks_checked":sum(p["masks_checked"] for p in plans),
                     "ground_instances":len(edges),
                     "target_frontiers":[p["frontier"] for p in plans]}
            if case.family=="reuse_tradeoff":
                assert all(len(p["tree"])==7 and len(p["dag"])==6 for p in plans),(case,metrics)
            else:
                assert all(len(p["tree"])==8 and len(p["dag"])==6 for p in plans),(case,metrics)
            case_receipts.append({"root":case.root,"family":case.family,"statement":case.statement(),
                                  "reference_audit_seconds":reference_time,**metrics})
            offset=i%len(ARMS)
            for arm in ARMS[offset:]+ARMS[:offset]:
                ep=Episode(repl,case,arm)
                t0=time.perf_counter()
                # Charge actual planner work to each arm; no shared warm reference.
                if arm=="native_order":
                    es=case.edges();ps=None
                elif arm in ("native_tree","deferred_tree"):
                    es=case.edges();ps=[]
                    for j in range(2):
                        f=case.context(j);d,ch=tree_cost(es,f)
                        ps.append({"dist":d,"choice":ch})
                else:
                    es=case.edges();ps=[]
                    for j in range(2):
                        f=case.context(j);tab,_=masks_for(es,f)
                        path=min_dag(es,f,tab,case.target)
                        bindings,tail=dag_bindings(es,f,path,case.target,j)
                        ps.append({"bindings":bindings,"tail":tail})
                ep.planning_seconds+=time.perf_counter()-t0
                if arm.startswith("deferred"):deferred(ep,es,ps)
                elif arm=="native_dag":live_dag(ep,ps)
                else:live_tree(ep,es,ps)
                row=ep.finish(metrics);rows.append(row)
                if row["kernel"] is not None:
                    name=f"lbf20e_{case.root}_{arm}"
                    proofs += [f"theorem {name} : {case.statement()} := by","  "+case.intros(),
                               *("  "+t for t in row["script"]),f"#print axioms {name}",""]
                print(case.root,arm,row["status"],row["calls"],round(row["wall_seconds"],4),flush=True)
            native_audits.append(audit_native(repl,case,plans))
    (out/"Witnesses.lean").write_text("\n".join(proofs),encoding="utf-8")
    (out/"episodes.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
    (out/"cases.json").write_text(json.dumps(case_receipts,indent=2)+"\n")
    (out/"native_audit.json").write_text(json.dumps(native_audits,indent=2)+"\n")
    ag={}
    for arm in ARMS:
        sub=[r for r in rows if r["arm"]==arm]
        closed=[r for r in sub if r["closed"]]
        ag[arm]={"roots":len(sub),"closed":len(closed),"mean_calls":statistics.mean(r["calls"] for r in sub),
                 "mean_wall_seconds":statistics.mean(r["wall_seconds"] for r in sub),
                 "mean_planning_seconds":statistics.mean(r["planning_seconds"] for r in sub),
                 "mean_rollbacks":statistics.mean(r["rollbacks"] for r in sub),
                 "mean_rejects":statistics.mean(r["rejects"] for r in sub),
                 "mean_script_chars_closed":statistics.mean(r["script_chars"] for r in closed) if closed else None,
                 "mean_kernel_term_chars_closed":statistics.mean(r["kernel"]["term_chars"] for r in closed) if closed else None,
                 "closure_at_calls":{str(k):sum(r["closed"] and r["calls"]<=k for r in sub) for k in (1,8,16,32,96)},
                 "status_counts":dict(collections.Counter(r["status"] for r in sub))}
    summary={"schema":"lbf20e.context-horn-dag.v1","scope":"development_only","roots":len(cases),
             "episodes":len(rows),"native_tactic_calls":sum(r["calls"] for r in rows),
             "native_kernel_certificates":sum(r["kernel"] is not None for r in rows),
             "separate_audit_tactic_calls":sum(r["tactic_calls"] for r in native_audits),
             "context_cache_rejections":len(native_audits),
             "implicit_apply_accepts":sum(r["inference"][0]["accepted"] for r in native_audits),
             "partial_apply_accepts":sum(r["inference"][1]["accepted"] for r in native_audits),
             "full_apply_accepts":sum(r["inference"][2]["accepted"] for r in native_audits),
             "mean_tree_rules":statistics.mean(r["tree_rule_applications"] for r in case_receipts),
             "mean_dag_rules":statistics.mean(r["dag_rule_applications"] for r in case_receipts),
             "strict_dag_advantage_roots":sum(r["dag_rule_applications"]<r["tree_rule_applications"] for r in case_receipts),
             "audit":au,"arms":ag,"new_training_fits":0}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("RESULTS",json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
