"""OLR-METTA-05: multi-goal native proof histories and typed receipts.

The policy projection deliberately masks Lean's complete hypothesis context.
All genuine receipt bits come from native tactic responses; MeTTa checks
their declared provenance/eligibility, not mathematical truth.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from pathlib import Path
import random
import re
import subprocess
import tempfile

from lcgrt import config
from .data import digest,sha256_file
from .live_lean import public_state,candidate_menu
from .metta_obligations import _native_metta

SCHEMA="ppt.olr-metta05-history-v1"
N=5
SEED=20261208
SPLITS={"train":(32,3),"dev":(12,3),"transfer":(12,5)}
TAIL=("rfl","exact h0 _","exact h1 _ (h0 _)")
ALLOWED={"constructor",*TAIL,*[f"apply r{i}" for i in range(2*N)]}
QUERY=("!(match &self (Receipt $id Current Unique Native $addr $branch $verdict) "
       "(Offer $id $addr $branch $verdict))")
OFFER=re.compile(r"^\(Offer n([0-4]) a([0-4]) b([01]) v([01])\)$")


@dataclass(frozen=True)
class Slot:
    addr:int
    a:int
    b:int
    delta:int
    winner:int

    def __post_init__(self):
        if not(0<=self.addr<N and 45<=self.a<=60 and 45<=self.b<=60
               and 1<=self.delta<=3 and self.winner in (0,1)):
            raise ValueError("bounded indexed Lean slot required")


@dataclass(frozen=True)
class Root:
    split:str
    pair:str
    world:int
    slots:tuple[Slot,...]
    probe_order:tuple[int,...]
    probe_branches:tuple[int,...]
    recall_order:tuple[int,...]
    focal:int
    sham:int

    def __post_init__(self):
        if self.split not in SPLITS or self.world not in (0,1):
            raise ValueError("invalid registered split/world")
        k=SPLITS[self.split][1]
        addresses=tuple(s.addr for s in self.slots)
        if (len(addresses)!=k or tuple(sorted(set(addresses)))!=addresses
                or set(self.probe_order)!=set(addresses) or len(self.probe_order)!=k
                or set(self.recall_order)!=set(addresses) or len(self.recall_order)!=k
                or len(self.probe_branches)!=k
                or any(b not in (0,1) for b in self.probe_branches)
                or self.focal not in addresses or self.sham not in addresses
                or self.focal==self.sham):
            raise ValueError("invalid matched address/receipt schedule")

    @property
    def root_id(self):
        return digest([SCHEMA,self.pair,self.world])

    def binders(self):
        names=" ".join(f"R{s.addr}" for s in self.slots)
        b=[f"(P Q : Nat → Prop) ({names} : Prop)",
           "(h0 : ∀ n : Nat, P n)","(h1 : ∀ n : Nat, P n → Q n)"]
        for s in self.slots:
            for branch in (0,1):
                c=s.a+s.b+(0 if branch==s.winner else s.delta)
                b.append(f"(r{2*s.addr+branch} : ({s.a} + {s.b} = {c}) → P {c} → Q {c} → R{s.addr})")
        return " ".join(b)

    def goal(self):
        targets=[f"R{s.addr}" for s in self.slots]
        out=targets[-1]
        for t in reversed(targets[:-1]):
            out=f"{t} ∧ ({out})"
        return out

    def statement(self):
        return "∀ "+self.binders()+", "+self.goal()

    def lineage(self):
        return tuple((s.addr,s.a,s.b,s.delta) for s in self.slots)

    def projection(self):
        # Only this typed projection is seen at recall, not full Lean contexts.
        return {"addresses":[s.addr for s in self.slots],
                "recall_order":list(self.recall_order),"budget":7*len(self.slots)-1}


def roots():
    used=set()
    result={}
    for split,(pairs,k) in SPLITS.items():
        rng=random.Random(f"{SCHEMA}/{SEED}/{split}")
        group=[]
        for i in range(pairs):
            while True:
                ids=sorted(rng.sample(range(N),k))
                spec=tuple((j,rng.randint(45,60),rng.randint(45,60),rng.randint(1,3)) for j in ids)
                if spec not in used:
                    used.add(spec);break
            bits=[rng.randrange(2) for _ in ids]
            probes=tuple(rng.sample(ids,k))
            branches=tuple(rng.randrange(2) for _ in probes)
            recall=tuple(rng.sample(ids,k))
            focal,sham=rng.sample(ids,2)
            pair=digest([SCHEMA,SEED,split,i,spec])
            for w in (0,1):
                slots=tuple(Slot(*v,bit^w) for v,bit in zip(spec,bits,strict=True))
                group.append(Root(split,pair,w,slots,probes,branches,recall,focal,sham))
        result[split]=tuple(group)
    return result


def root_row(root):
    return {**asdict(root),"root_id":root.root_id,
            "statement":root.statement(),"projection":root.projection()}


def from_row(row):
    r=Root(row["split"],row["pair"],row["world"],
           tuple(Slot(**s) for s in row["slots"]),
           tuple(row["probe_order"]),tuple(row["probe_branches"]),
           tuple(row["recall_order"]),row["focal"],row["sham"])
    if row["root_id"]!=r.root_id or row["statement"]!=r.statement() or row["projection"]!=r.projection():
        raise ValueError("frozen root row changed")
    return r


class Native:
    """One original theorem; immutable saved joint state and charged tactics."""
    def __init__(self,lean,root):
        self.lean,self.root=lean,root
        lean.reset()
        self.state=lean.init(root.statement(),pp=False)
        self.base=None
        self.events=[]
        self.calls=0;self.focuses=0;self.restores=0
        self.initial=self.sha()
        self.closed=False

    def public(self):return public_state(self.lean,self.state)

    def sha(self):return digest(self.public())

    def record(self,kind,before,**kw):
        self.events.append({"kind":kind,"before":before,"after":self.sha(),**kw})

    def tactic(self,tac):
        if tac not in ALLOWED:
            raise ValueError("unregistered Lean tactic")
        before=self.sha()
        result=self.lean.run(self.state["sid"],tac)
        if type(result.get("ok")) is not bool:
            raise ValueError("malformed native response")
        if result["ok"]:self.state=result["state"]
        self.calls+=1
        self.record("tactic",before,tactic=tac,ok=result["ok"],calls=self.calls)
        return result["ok"]

    def focus(self,index):
        if index==0:return
        before=self.sha()
        result=self.lean.focus(self.state["sid"],index)
        if result.get("ok") is not True:raise ValueError("native focus failed")
        self.state=result["state"];self.focuses+=1
        self.record("focus",before,index=index)

    def address(self,addr):
        targets=[g["target"] for g in self.public()["goals"]]
        if {"a":f"R{addr}"} not in targets:
            raise ValueError("pending obligation absent from native state")
        self.focus(targets.index({"a":f"R{addr}"}))
        expected=(f"apply r{2*addr}",f"apply r{2*addr+1}")
        if candidate_menu(self.lean,self.state)!=expected:
            raise ValueError("native address candidate menu changed")

    def prepare(self):
        k=len(self.root.slots)
        for _ in range(k-1):
            goals=self.public()["goals"]
            index=next((i for i,g in enumerate(goals) if "and" in g["target"]),None)
            if index is None:raise ValueError("conjunction exhausted early")
            self.focus(index)
            if not self.tactic("constructor"):raise ValueError("native constructor rejected")
        goals=self.public()["goals"]
        if sorted(g["target"].get("a","") for g in goals)!=[
                f"R{s.addr}" for s in self.root.slots]:
            raise ValueError("conjunction did not expose registered addresses")
        self.base=self.state
        self.record("save",self.sha())

    def restore(self):
        if self.base is None:raise ValueError("no joint state saved")
        before=self.sha();self.state=self.base;self.restores+=1
        self.record("restore",before)

    def probe(self,addr,branch):
        self.address(addr)
        if not self.tactic(f"apply r{2*addr+branch}"):
            raise ValueError("registered opening rejected before guard")
        if len(self.public()["goals"])!=len(self.root.slots)+2:
            raise ValueError("wrong indexed pending goal shape")
        verdict=self.tactic("rfl")
        self.restore()
        return {"addr":addr,"branch":branch,"ok":verdict}

    def complete(self,choices):
        for addr in self.root.recall_order:
            branch=choices[addr]
            if branch not in (0,1):raise ValueError("invalid final branch")
            self.address(addr)
            if not self.tactic(f"apply r{2*addr+branch}"):
                raise ValueError("native final opening refused")
            if not self.tactic("rfl"):
                return
            for tac in TAIL[1:]:
                if not self.tactic(tac):raise ValueError("indexed proof tail rejected")
        if self.public()["goals"]:raise ValueError("complete vector left open goals")
        before=self.sha()
        k=self.lean.kernel(self.state["sid"])
        if k.get("ok") is not True or k.get("axioms")!=[]:
            raise ValueError("native conjunction kernel or axiom failure")
        self.closed=True
        self.record("kernel",before,axioms=[])

    def receipt(self):
        return {"root_id":self.root.root_id,"initial":self.initial,
                "closed":self.closed,"calls":self.calls,"focuses":self.focuses,
                "restores":self.restores,"events":self.events}


def admit(receipts):
    """Ground provenance query; stale synthetic aliases have no model payload."""
    if len({r["addr"] for r in receipts})!=len(receipts):
        raise ValueError("receipt addresses must be unique")
    lines=[]
    expected=[]
    for i,r in enumerate(receipts):
        a,b,v=r["addr"],r["branch"],int(r["ok"])
        if not(0<=a<N and b in (0,1) and type(r["ok"]) is bool):
            raise ValueError("invalid actual native receipt")
        lines.append(f"(Receipt n{i} Current Unique Native a{a} b{b} v{v})")
        lines.append(f"(Receipt old{i} Stale Repeated Synthetic a{a} b{b} v{1-v})")
        expected.append(tuple(map(str,(i,a,b,v))))
    got,sha=_native_metta("\n".join((*lines,QUERY))+"\n",offer_pattern=OFFER)
    if set(got)!=set(expected):
        raise ValueError("native MeTTa receipt admission differs from independent reference")
    return {"program_sha256":sha,"admitted":len(got),"vetoed_aliases":len(receipts)}


def capture(lean,root):
    n=Native(lean,root);n.prepare()
    receipts=[n.probe(a,b) for a,b in zip(root.probe_order,root.probe_branches,strict=True)]
    if n.calls!=3*len(root.slots)-1 or n.restores!=len(root.slots):
        raise ValueError("native receipt prefix cost changed")
    return n,receipts,admit(receipts)


def lookup(receipts):
    return {r["addr"]:r["branch"] if r["ok"] else 1-r["branch"] for r in receipts}


def grade(lean,root):
    """Independent native grading: test the other branch too; compile winners."""
    n,receipts,metta=capture(lean,root)
    other=[n.probe(r["addr"],1-r["branch"]) for r in receipts]
    if any(a["ok"]==b["ok"] for a,b in zip(receipts,other,strict=True)):
        raise ValueError("unique-winner assumption failed native test")
    labels=lookup(receipts)
    # Generator truth is used only to audit the native label.
    if labels!={s.addr:s.winner for s in root.slots}:
        raise ValueError("native receipt truth conflicts with root generator")
    n.complete(labels)
    if not n.closed or n.calls!=9*len(root.slots)-1:
        raise ValueError("native grading failed reference theorem or call meter")
    ep=n.receipt()
    replay(lean,root,ep)
    return {"receipts":receipts,"opposite":other,"metta":metta,
            "labels":[labels[a] for a in root.recall_order],"grade":ep}


def execute(lean,root,choices,expected_receipts):
    n,receipts,metta=capture(lean,root)
    if receipts!=expected_receipts:raise ValueError("native history drifted")
    n.complete(dict(zip(root.recall_order,choices,strict=True)))
    ep=n.receipt();k=len(root.slots)
    if ep["calls"]>7*k-1 or ep["closed"] and ep["calls"]!=7*k-1:
        raise ValueError("native policy exceeded its exact call budget")
    replay(lean,root,ep)
    return {"native":ep,"metta":metta,"choices":list(choices)}


def replay(lean,root,ep):
    """Event-driven replay, separately restores saved native state and checks receipts."""
    if ep["root_id"]!=root.root_id:raise ValueError("native replay root differs")
    lean.reset();state=lean.init(root.statement(),pp=False);base=None
    calls=focuses=restores=0;closed=False
    def sha():return digest(public_state(lean,state))
    if sha()!=ep["initial"]:raise ValueError("native initial state changed")
    for e in ep["events"]:
        if sha()!=e["before"]:raise ValueError("native replay predecessor changed")
        kind=e["kind"]
        if kind=="tactic":
            if e["tactic"] not in ALLOWED:raise ValueError("tactic outside safe grammar")
            q=lean.run(state["sid"],e["tactic"]);calls+=1
            if q["ok"]!=e["ok"] or e["calls"]!=calls:raise ValueError("native verdict/cost changed")
            if q["ok"]:state=q["state"]
        elif kind=="focus":
            q=lean.focus(state["sid"],e["index"])
            if q.get("ok") is not True:raise ValueError("native replay focus failed")
            state=q["state"];focuses+=1
        elif kind=="save":
            if base is not None:raise ValueError("repeated joint save")
            base=state
        elif kind=="restore":
            if base is None:raise ValueError("restore without joint save")
            state=base;restores+=1
        elif kind=="kernel":
            cert=lean.kernel(state["sid"])
            if cert.get("ok") is not True or cert.get("axioms")!=[] or e["axioms"]!=[]:
                raise ValueError("native replay kernel failure")
            closed=True
        else:raise ValueError("unregistered replay event")
        if sha()!=e["after"]:raise ValueError("native replay successor changed")
    if (calls,focuses,restores,closed)!=(ep["calls"],ep["focuses"],ep["restores"],ep["closed"]):
        raise ValueError("native replay final accounting changed")
    return True


def compile_proofs(items,out:Path):
    """Independent compiler uses original theorem and each verified branch."""
    lines=["namespace OLRHistory"]
    for i,(root,choices) in enumerate(items):
        choices=dict(zip(root.recall_order,choices,strict=True))
        if set(choices)!={s.addr for s in root.slots} or any(b not in (0,1) for b in choices.values()):
            raise ValueError("unsafe compiled choice vector")
        lines.append(f"theorem h_{i} {root.binders()} : {root.goal()} := by")
        def branch(addr,indent):
            for t in (f"apply r{2*addr+choices[addr]}",*TAIL):
                lines.append(indent+t)
        def conjunction(addrs,indent):
            if len(addrs)==1:
                branch(addrs[0],indent);return
            lines.append(indent+"constructor")
            lines.append(indent+"·")
            branch(addrs[0],indent+"  ")
            lines.append(indent+"·")
            conjunction(addrs[1:],indent+"  ")
        conjunction([s.addr for s in root.slots],"  ")
    lines.append("end OLRHistory")
    lines.extend(f"#print axioms OLRHistory.h_{i}" for i in range(len(items)))
    source="\n".join(lines)+"\n"
    out.write_text(source,encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"History.lean";p.write_text(source,encoding="utf-8")
        run=subprocess.run([str(config.LEAN_EXE),str(p)],env=config.lean_env(),
              capture_output=True,text=True,encoding="utf-8",timeout=180)
    detail=run.stdout+run.stderr
    clean=sum(f"'OLRHistory.h_{i}' does not depend on any axioms" in detail for i in range(len(items)))
    if run.returncode or clean!=len(items):
        raise ValueError(f"independent Lean compilation failed {clean}/{len(items)}: {detail[:3000]}")
    return {"compiled":len(items),"axiom_free":clean,"sha256":sha256_file(out)}
