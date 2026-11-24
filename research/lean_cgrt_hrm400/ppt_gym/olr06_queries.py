"""OLR-06B synthetic paid-relation gym and exact same-information planner.

Nodes are quotient components of an already verified signed ledger.
Positive queries reveal a consistent relation or anchor. Failed queries
are censored and never install a negative fact. Independent Bernoulli
success probabilities, integer costs and topology are public. Actual
successes and witness signs are environment-private until a paid query.

Since the registered reward depends on resolved coverage, not sign,
the observed success subset is a sufficient statistic for planning.
This is an algebraic simulation, not a native Lean proof environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations, product
import hashlib
import json
import random

from .olr06_exact_ledger import solve, UNKNOWN, CONFLICT

N = 6
Q = 8
STOP = Q
SEED = 20261212
COUNTS = {"train": 160, "dev": 40, "transfer": 40}
COST_WEIGHT = .01


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, order=True)
class Offer:
    kind: str
    u: int
    v: int
    cost: int
    p4: int

    def __post_init__(self):
        if (self.kind not in ("anchor","relation") or
                type(self.u) is not int or not 0 <= self.u < N or
                type(self.v) is not int or not 0 <= self.v < N or
                (self.u == self.v) != (self.kind == "anchor") or
                type(self.cost) is not int or not 1 <= self.cost <= 3 or
                type(self.p4) is not int or not 1 <= self.p4 <= 4):
            raise ValueError("invalid public query offer")

    @property
    def p(self):
        return self.p4 / 4


@dataclass(frozen=True)
class Root:
    weights: tuple[int, ...]
    anchored: tuple[int, ...]
    offers: tuple[Offer, ...]
    budget: int
    family: str

    def __post_init__(self):
        n=len(self.weights)
        if not (2 <= n <= N and 1 <= len(self.offers) <= Q
                and 1 <= self.budget <= 6):
            raise ValueError("invalid bounded graph or budget")
        if any(type(w) is not int or not 1 <= w <= 5 for w in self.weights):
            raise ValueError("invalid obligation component weight")
        if tuple(sorted(set(self.anchored))) != self.anchored or any(
                a < 0 or a >= n for a in self.anchored):
            raise ValueError("invalid anchored component set")
        if any(o.u >= n or o.v >= n for o in self.offers):
            raise ValueError("offer endpoint outside active graph")
        if len({(o.kind,min(o.u,o.v),max(o.u,o.v))
                for o in self.offers}) != len(self.offers):
            raise ValueError("duplicate query")

    @property
    def total(self):
        return sum(self.weights)

    def row(self):
        return asdict(self)

    def signature(self):
        """Exact public graph canonicalization under component relabeling.

        Component size, initial anchor status, candidate incidence, cost,
        probability and total budget are retained. Offer ordering is ignored.
        Family names, random IDs and private witnesses are excluded.
        """
        groups={}
        anchored=set(self.anchored)
        for i,w in enumerate(self.weights):
            groups.setdefault((w,int(i in anchored)),[]).append(i)
        keys=sorted(groups)
        labels=tuple(k for k in keys for _ in groups[k])
        best=None
        for parts in product(*(permutations(groups[k]) for k in keys)):
            order=tuple(i for part in parts for i in part)
            inv={old:new for new,old in enumerate(order)}
            edges=tuple(sorted((o.kind,min(inv[o.u],inv[o.v]),
                                max(inv[o.u],inv[o.v]),o.cost,o.p4)
                               for o in self.offers))
            if best is None or edges<best:
                best=edges
        return digest([labels,best,self.budget])


def from_row(row):
    return Root(tuple(row["weights"]),tuple(row["anchored"]),
                tuple(Offer(**o) for o in row["offers"]),row["budget"],
                row["family"])


def _make(rng, transfer=False):
    # A chain/fork can connect a cheap anchor to a heavy obligation component.
    # The distractor is a separate immediately useful anchor. Further offers
    # vary costs/reliabilities, including competing direct anchors.
    m=4 if transfer else 3
    n=m+2
    family=rng.choice(("long_chain","diamond")) if transfer else rng.choice(("chain","fork"))
    graph=[(i,i+1) for i in range(m-1)]
    if family=="fork":
        graph=[(0,1),(0,2)]
    if family=="diamond":
        graph=[(0,1),(0,2),(1,3),(2,3)]
    w=[rng.randint(1,2) for _ in range(m-1)]+[rng.randint(3,5),
       rng.randint(2,4),rng.randint(1,3)]
    offers=[Offer("relation",u,v,rng.choice((1,1,2)),rng.choice((2,3,4,4)))
            for u,v in graph]
    offers.extend((Offer("anchor",0,0,1,rng.choice((3,4,4))),
                   Offer("anchor",m,m,rng.choice((1,1,2)),rng.choice((2,3,4))),
                   Offer("anchor",m+1,m+1,rng.choice((1,2,3)),rng.choice((2,3,4)))))
    if rng.random()<.6:
        offers.append(Offer("anchor",m-1,m-1,rng.choice((2,3)),rng.choice((1,2,3))))
    if rng.random()<.4 and len(offers)<Q:
        offers.append(Offer("relation",m,m+1,rng.choice((1,2)),rng.choice((2,3,4))))
    anchored=(m+1,) if rng.random()<.25 else ()
    budget=rng.choice((4,5,6)) if transfer else rng.choice((2,3,4))
    order=rng.sample(range(n),n)
    inv={old:new for new,old in enumerate(order)}
    renamed=[Offer(o.kind,inv[o.u],inv[o.v],o.cost,o.p4) for o in offers]
    rng.shuffle(renamed)
    return Root(tuple(w[i] for i in order),tuple(sorted(inv[i] for i in anchored)),
                tuple(renamed),budget,family)


def bank():
    result={};used=set()
    for split,count in COUNTS.items():
        rng=random.Random(f"olr06b-v1/{SEED}/{split}")
        rows=[]
        while len(rows)<count:
            root=_make(rng,split=="transfer")
            sig=root.signature()
            if sig in used:
                continue
            used.add(sig)
            rows.append({"id":digest([SEED,split,len(rows),sig]),"split":split,
                         "signature":sig,"root":root.row()})
        result[split]=rows
    return result


def delayed_example():
    # Three unknown main components (1,1,3), one distractor (2).
    # Main path must use two relation queries and an anchor, total gain 5.
    # Any immediate-gain greedy policy takes the distractor first;
    # two-step lookahead also spends on it. Three-step optimum gets 5.
    return Root((1,1,3,2),(),(
        Offer("relation",0,1,1,4),Offer("relation",1,2,1,4),
        Offer("anchor",0,0,1,4),Offer("anchor",3,3,1,4)),3,"delayed")


class Planner:
    def __init__(self,root:Root):
        self.root=root
        self._closure={};self._cost={};self._value={}

    def closure(self,succeeded):
        """Exact weighted known-component closure, memoized by positive receipts."""
        if succeeded in self._closure:
            return self._closure[succeeded]
        r=self.root
        parent=list(range(len(r.weights)))
        def find(u):
            while parent[u]!=u:
                u=parent[u]
            return u
        for j,o in enumerate(r.offers):
            if succeeded>>j&1 and o.kind=="relation":
                a,b=find(o.u),find(o.v)
                if a!=b:parent[max(a,b)]=min(a,b)
        anchored={find(i) for i in r.anchored}
        anchored.update(find(o.u) for j,o in enumerate(r.offers)
                        if succeeded>>j&1 and o.kind=="anchor")
        members={}
        for u in range(len(r.weights)):members.setdefault(find(u),[]).append(u)
        groups=tuple(tuple(v) for _,v in sorted(members.items()))
        index={u:i for i,g in enumerate(groups) for u in g}
        weights=tuple(sum(r.weights[u] for u in g) for g in groups)
        resolved=tuple(find(g[0]) in anchored for g in groups)
        coverage=sum(w for w,a in zip(weights,resolved) if a)
        info=(groups,index,weights,resolved,coverage)
        self._closure[succeeded]=info
        return info

    def spent(self,attempted):
        if attempted not in self._cost:
            self._cost[attempted]=sum(o.cost for j,o in enumerate(self.root.offers)
                                    if attempted>>j&1)
        return self._cost[attempted]

    def legal(self,attempted):
        remaining=self.root.budget-self.spent(attempted)
        return tuple(j for j,o in enumerate(self.root.offers)
                     if not attempted>>j&1 and o.cost<=remaining)+(STOP,)

    def stop_value(self,attempted,succeeded):
        return self.closure(succeeded)[-1]/self.root.total-COST_WEIGHT*self.spent(attempted)

    def branches(self,attempted,succeeded,j):
        if j==STOP or j not in self.legal(attempted) or succeeded&~attempted:
            raise ValueError("illegal or inconsistent state/action")
        q=self.root.offers[j]
        a=attempted|(1<<j)
        branches=[(q.p,a,succeeded|(1<<j))]
        if q.p<1:branches.append((1-q.p,a,succeeded))
        return branches

    def values(self,a,s,depth=None):
        """Same-information Bellman Q-values of terminal expected utility.

        Unknown successful signs need not be integrated separately: every
        consistent sign assignment has identical weighted resolved coverage
        for a fixed set of successful query IDs. Independence is registered.
        """
        if s&~a or self.spent(a)>self.root.budget:
            raise ValueError("invalid evidence/budget state")
        out={STOP:self.stop_value(a,s)}
        if depth==0:return out
        d=None if depth is None else depth-1
        for j in self.legal(a):
            if j!=STOP:
                out[j]=sum(p*self.value(aa,ss,d)
                           for p,aa,ss in self.branches(a,s,j))
        return out

    def value(self,a=0,s=0,depth=None):
        key=(a,s,depth)
        if key not in self._value:
            self._value[key]=max(self.values(a,s,depth).values())
        return self._value[key]

    def choose(self,a,s,depth=None):
        q=self.values(a,s,depth)
        # Prefer STOP when exactly indifferent, then smaller immediate cost.
        best=max(q.values())
        return min((j for j,v in q.items() if v>=best-1e-10),
                   key=lambda j:(0 if j==STOP else self.root.offers[j].cost,j))

    def states(self):
        """All reachable public outcome states, regardless of teacher action."""
        stack=[(0,0)];seen={(0,0)}
        while stack:
            a,s=stack.pop()
            for j in self.legal(a):
                if j==STOP:continue
                for p,aa,ss in self.branches(a,s,j):
                    if (aa,ss) not in seen:
                        seen.add((aa,ss));stack.append((aa,ss))
        return tuple(sorted(seen))

    def expectation(self,policy):
        """Exact policy expectation, independent of actual sampled test outcomes."""
        @lru_cache(None)
        def visit(a,s):
            j=policy(a,s)
            if j not in self.legal(a):raise ValueError("policy chose unaffordable query")
            if j==STOP:
                coverage=self.closure(s)[-1]
                return (coverage,self.spent(a),float(coverage==self.root.total),0.)
            v=[(p,visit(aa,ss)) for p,aa,ss in self.branches(a,s,j)]
            return tuple(sum(p*x[k] for p,x in v)+(1. if k==3 else 0.)
                         for k in range(4))
        coverage,spent,full,queries=visit(0,0)
        return {"resolved":coverage,"coverage":coverage/self.root.total,
                "spent":spent,"full":full,"queries":queries,
                "utility":coverage/self.root.total-COST_WEIGHT*spent,
                "evaluated_states":visit.cache_info().currsize}


def execute(root,policy,seed):
    """Run an actual synthetic episode; only environment holds hidden signs.

    The queried outcome is generated after the chosen action. The independent
    signed solver verifies every new closure against the public quotient.
    Censored queries install no receipt. No hidden signs are passed to policy.
    """
    rng=random.Random(seed)
    hidden=tuple(rng.choice((-1,1)) for _ in root.weights)
    outcomes=tuple(rng.random()<o.p for o in root.offers)
    a=s=0;p=Planner(root);events=[];edges=[]
    anchors=[(i,hidden[i]) for i in root.anchored]
    while True:
        j=policy(a,s)
        if j not in p.legal(a):raise ValueError("illegal paid query")
        if j==STOP:break
        o=root.offers[j]
        a|=1<<j
        if outcomes[j]:
            s|=1<<j
            if o.kind=="anchor":
                receipt=(o.u,hidden[o.u]);anchors.append(receipt)
            else:
                receipt=(o.u,o.v,hidden[o.u]*hidden[o.v]);edges.append(receipt)
        else:receipt=None
        _,labels=solve(range(len(root.weights)),edges,anchors)
        if any(v==CONFLICT or v!=UNKNOWN and v!=hidden[i] for i,v in labels.items()):
            raise ValueError("inconsistent or false synthetic evidence")
        coverage=sum(root.weights[i] for i,v in labels.items() if v!=UNKNOWN)
        if coverage!=p.closure(s)[-1]:raise ValueError("signed and quotient closure disagree")
        events.append({"query":j,"status":"accepted" if outcomes[j] else "censored",
                       "receipt":receipt,"spent":p.spent(a),"resolved":coverage,
                       "attempted":a,"succeeded":s})
    coverage=p.closure(s)[-1]
    return {"resolved":coverage,"coverage":coverage/root.total,"spent":p.spent(a),
            "full":coverage==root.total,"queries":len(events),
            "utility":p.stop_value(a,s),"events":events}
