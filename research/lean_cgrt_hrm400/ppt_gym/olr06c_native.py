"""OLR-06C native scoped Boolean relation and anchor receipts.

All expression syntax is public. The eight-assignment evaluator is a strong
same-information baseline, not the source of learned labels. Lean must
accept each positive receipt and independently compile the transported proof.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from collections import deque
import hashlib
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .live_lean import public_state
from .olr06_exact_ledger import solve, UNKNOWN, CONFLICT

N=5
Q=12
STOP=Q
SEED=20261216
COUNTS={"train":64,"dev":24,"transfer":24}
TOKENS=("pad","x","y","z","false","true","not","and","or")
MAX_TOKENS=127
SCOPE="(x y z : Bool)"
MACRO="cases x <;> cases y <;> cases z <;> rfl"
REL_MACRO="cases x <;> cases y <;> cases z <;> constructor <;> rfl"


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(tokens):
    """Parse the entire unambiguous public prefix sequence into an AST."""
    seq=tuple(tokens)
    def read(i):
        if i>=len(seq):raise ValueError("truncated AST")
        t=seq[i]
        if t in ("x","y","z","false","true"):return (t,),i+1
        if t=="not":
            a,j=read(i+1);return (t,a),j
        if t in ("and","or"):
            a,j=read(i+1);b,k=read(j);return (t,a,b),k
        raise ValueError("unregistered AST symbol")
    tree,i=read(0)
    if i!=len(seq) or len(seq)>MAX_TOKENS:raise ValueError("invalid AST length")
    return tree


def flatten(e):
    return (e[0],)+tuple(t for sub in e[1:] for t in flatten(sub))


def render(e):
    op=e[0]
    if len(e)==1:return op
    if op=="not":return "(Bool.not "+render(e[1])+")"
    if op in ("and","or"):return "(Bool."+op+" "+render(e[1])+" "+render(e[2])+")"
    raise ValueError("unknown Boolean operation")


def evaluate(e,env):
    op=e[0]
    if op=="true":return True
    if op=="false":return False
    if op in ("x","y","z"):return env[op]
    if op=="not":return not evaluate(e[1],env)
    if op=="and":return evaluate(e[1],env) and evaluate(e[2],env)
    if op=="or":return evaluate(e[1],env) or evaluate(e[2],env)
    raise ValueError("unknown Boolean operation")


def truth_table(tokens):
    e=parse(tokens)
    return tuple(evaluate(e,dict(zip(("x","y","z"),bits)))
                 for bits in product((False,True),repeat=3))


def constant(tokens):
    t=truth_table(tokens)
    if len(set(t))!=1:raise ValueError("not a constant Boolean function")
    return 1 if t[0] else -1


@dataclass(frozen=True)
class Offer:
    kind:str
    u:int
    v:int
    sign:int

    def __post_init__(self):
        if (self.kind not in ("anchor","relation") or
                type(self.u) is not int or type(self.v) is not int or
                not 0<=self.u<N or not 0<=self.v<N or
                (self.u==self.v)!=(self.kind=="anchor") or
                type(self.sign) is not int or self.sign not in (-1,1)):
            raise ValueError("invalid signed native query")


@dataclass(frozen=True)
class Root:
    exprs:tuple[tuple[str,...],...]
    offers:tuple[Offer,...]
    budget:int
    family:str

    def __post_init__(self):
        n=len(self.exprs)
        if not 2<=n<=N or not n<=self.budget<=n+1 or not 1<=len(self.offers)<=Q:
            raise ValueError("invalid scope or native budget")
        if len(set(self.exprs))!=n:raise ValueError("duplicate public expression")
        for e in self.exprs:parse(e)
        if any(o.u>=n or o.v>=n for o in self.offers):
            raise ValueError("query outside root scope")
        if len(set((o.kind,min(o.u,o.v),max(o.u,o.v),o.sign) for o in self.offers))!=len(self.offers):
            raise ValueError("duplicate native offer")

    @property
    def n(self):return len(self.exprs)

    def terms(self):return [render(parse(t)) for t in self.exprs]

    def signature(self):return digest(sorted(self.exprs))

    def query(self,j):
        o=self.offers[j];u,v=self.terms()[o.u],self.terms()[o.v]
        def neg(s):return "(Bool.not "+s+")" if o.sign<0 else s
        if o.kind=="anchor":
            return f"∀ {SCOPE}, {u} = "+("true" if o.sign==1 else "false")
        return f"∀ {SCOPE}, ({v} = {neg(u)}) ∧ ({u} = {neg(v)})"

    def macro(self,j):return MACRO if self.offers[j].kind=="anchor" else REL_MACRO

    def goal(self,nodes=None):
        terms=self.terms()
        nodes=tuple(range(self.n)) if nodes is None else tuple(nodes)
        if not nodes:raise ValueError("empty certificate")
        return " ∧ ".join(f"(∃ b : Bool, ∀ {SCOPE}, {terms[i]} = b)" for i in nodes)


def from_row(row):
    return Root(tuple(tuple(e) for e in row["exprs"]),
                tuple(Offer(**o) for o in row["offers"]),row["budget"],row["family"])


def _expression(rng,depth):
    if depth<=0 or rng.random()<.21:
        return (rng.choice(("x","y","z","true","false")),)
    op=rng.choice(("not","and","or"))
    a=_expression(rng,depth-1)
    return (op,a) if op=="not" else (op,a,_expression(rng,depth-1))


def bank():
    used=set();out={}
    for split,count in COUNTS.items():
        rng=random.Random(f"olr06c-v1/{SEED}/{split}")
        n=5 if split=="transfer" else 4
        rows=[]
        for i in range(count):
            exprs=[]
            for j in range(n):
                target=rng.choice((-1,1))
                # Generate balanced signs by conditioning independent public
                # Boolean syntax; syntax, not the target, enters the policy.
                for attempt in range(20000):
                    e=flatten(_expression(rng,rng.randint(4,5) if n==5 else rng.randint(3,4)))
                    if (len(e)>=7 and len(set(e)&{"x","y","z"})>=2 and
                            len(e)<=MAX_TOKENS and e not in used and
                            all(x==(target==1) for x in truth_table(e))):
                        used.add(e);exprs.append(e);break
                else:raise ValueError("unique constant expression generation exhausted")
            family=rng.choice(("chain","branch")) if n==5 else rng.choice(("chain","fork"))
            edges=[(i,i+1) for i in range(n-1)]
            if family=="fork":edges=[(0,1),(0,2),(2,3)]
            if family=="branch":edges=[(0,1),(1,2),(1,3),(3,4)]
            offers=[Offer("relation",u,v,s) for u,v in edges for s in (-1,1)]
            offers.extend(Offer("anchor",0,0,s) for s in (-1,1))
            if len(offers)<Q and rng.random()<.65:
                v=rng.randrange(1,n);offers.append(Offer("anchor",v,v,rng.choice((-1,1))))
            perm=rng.sample(range(n),n);inv={old:new for new,old in enumerate(perm)}
            offers=[Offer(o.kind,inv[o.u],inv[o.v],o.sign) for o in offers]
            rng.shuffle(offers)
            root=Root(tuple(exprs[j] for j in perm),tuple(offers),n+rng.randrange(2),family)
            sig=root.signature()
            rows.append({"id":digest([SEED,split,i,sig]),"split":split,"signature":sig,"root":asdict(root)})
        out[split]=rows
    return out


def outcomes(root,signs):
    if len(signs)!=root.n or any(v not in (-1,1) for v in signs):
        raise ValueError("bad assignment")
    return tuple(signs[o.u]==o.sign if o.kind=="anchor" else
                 signs[o.u]*signs[o.v]==o.sign for o in root.offers)


def symbolic(root):
    return tuple(constant(e) for e in root.exprs)


def native_check(lean,root,j):
    """One actual compound native proof attempt, including kernel audit."""
    lean.reset();start=time.perf_counter()
    state=lean.init(root.query(j),pp=False)
    initial=digest(public_state(lean,state))
    answer=lean.run(state["sid"],root.macro(j))
    if type(answer.get("ok")) is not bool:raise ValueError("malformed native verdict")
    ok=answer["ok"]
    final=None;term=None
    if ok:
        after=answer["state"]
        if public_state(lean,after)["goals"]:
            raise ValueError("accepted native macro left goals")
        cert=lean.kernel(after["sid"])
        if cert.get("ok") is not True or cert.get("axioms")!=[]:
            raise ValueError("native receipt failed kernel or axiom check")
        final=digest(public_state(lean,after));term=cert.get("term")
    return {"query":j,"statement_sha256":digest(root.query(j)),
            "initial_sha256":initial,"final_sha256":final,"accepted":ok,
            "proof_term":term,"error":None if ok else str(answer.get("error",""))[:300],
            "seconds":time.perf_counter()-start}


def native_table(lean,root):
    receipts=[]
    # Every offer is checked and independently repeated from a reset state.
    for j in range(len(root.offers)):
        a=native_check(lean,root,j);b=native_check(lean,root,j)
        fields=("query","statement_sha256","initial_sha256","final_sha256","accepted")
        if any(a[k]!=b[k] for k in fields):
            raise ValueError("native deterministic receipt replay differs")
        a["replay_seconds"]=b["seconds"];receipts.append(a)
    actual=tuple(r["accepted"] for r in receipts)
    if actual!=outcomes(root,symbolic(root)):
        raise ValueError("native proof macro disagrees with exhaustive public Boolean semantics")
    return receipts


def closure(root,accepted):
    edges=[];anchors=[]
    for j,o in enumerate(root.offers):
        if not accepted>>j&1:continue
        if o.kind=="anchor":anchors.append((o.u,o.sign))
        else:edges.append((o.u,o.v,o.sign))
    _,labels=solve(range(root.n),edges,anchors)
    if CONFLICT in labels.values():raise ValueError("inconsistent positive receipt set")
    return labels


def transport(root,accepted,prefix):
    """Build explicit proof terms using only accepted scoped receipt lemmas."""
    known=closure(root,accepted)
    signs={};proof={};adj=[[] for _ in root.exprs];todo=deque()
    for j,o in enumerate(root.offers):
        if not accepted>>j&1:continue
        q=f"({prefix}_q{j} x y z)"
        if o.kind=="anchor":
            if o.u not in signs:
                signs[o.u]=o.sign;proof[o.u]=q;todo.append(o.u)
        else:
            adj[o.u].append((o.v,o.sign,q+".1"))
            adj[o.v].append((o.u,o.sign,q+".2"))
    while todo:
        u=todo.popleft()
        for v,sign,rel in adj[u]:
            if v in signs:continue
            signs[v]=sign*signs[u]
            parent=proof[u] if sign==1 else f"(congrArg Bool.not ({proof[u]}))"
            proof[v]=f"(Eq.trans ({rel}) ({parent}))"
            todo.append(v)
    if signs!={i:s for i,s in known.items() if s!=UNKNOWN}:
        raise ValueError("proof transport differs from exact signed closure")
    nodes=tuple(sorted(proof))
    terms=[f"(Exists.intro {'true' if signs[i]==1 else 'false'} (fun x y z => {proof[i]}))"
           for i in nodes]
    if not terms:return (),None
    term=terms[-1]
    for t in reversed(terms[:-1]):term=f"(And.intro {t} {term})"
    return nodes,term


def compile_certificates(items,out):
    """Compile all positive query lemmas plus selected nonempty witness subsets.

    items: (root, native receipts, distinct selected accepted masks). No
    private generator signs or symbolic proof search in the transported terms.
    """
    lines=["namespace OLR06C"]
    names=[];full=partial=receipt_count=0
    for i,(root,receipts,masks) in enumerate(items):
        prefix=f"r{i}"
        allowed=sum(1<<r["query"] for r in receipts if r["accepted"])
        for j,r in enumerate(receipts):
            if not r["accepted"]:continue
            name=f"{prefix}_q{j}"
            lines.append(f"theorem {name} : {root.query(j)} := by\n  intro x y z\n  {root.macro(j)}")
            names.append(name);receipt_count+=1
        for k,mask in enumerate(sorted(set(masks))):
            if mask&~allowed:raise ValueError("unverified receipt in final proof")
            nodes,term=transport(root,mask,prefix)
            if not nodes:continue
            name=f"{prefix}_p{k}"
            lines.append(f"theorem {name} : {root.goal(nodes)} := by\n  exact {term}")
            names.append(name)
            if len(nodes)==root.n:full+=1
            else:partial+=1
    lines.append("end OLR06C")
    lines.extend(f"#print axioms OLR06C.{n}" for n in names)
    out.write_text("\n".join(lines)+"\n",encoding="utf-8")
    run=subprocess.run([str(config.LEAN_EXE),str(out)],env=config.lean_env(),
                       capture_output=True,text=True,encoding="utf-8",timeout=300)
    output=run.stdout+run.stderr
    clean=sum(f"'OLR06C.{n}' does not depend on any axioms" in output for n in names)
    if run.returncode or clean!=len(names):
        raise ValueError(f"standalone certificate gate failed ({clean}/{len(names)}): {output[:3500]}")
    return {"receipt_theorems":receipt_count,"full":full,"partial":partial,
            "axiom_free":clean,"source_sha256":sha(out)}
