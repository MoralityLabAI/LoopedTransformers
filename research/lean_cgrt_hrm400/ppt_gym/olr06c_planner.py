"""Same query planner for every OLR-06C prior.

The prior factorizes over signs. Exact posterior conditioning on observed
macro verdicts is licensed ONLY by the finite Boolean semantics verified
in this gym. Proof authority is separate: only positive receipts can resolve
nodes in the signed ledger. No query outcome table is read by the planner.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
import math

from .olr06c_native import STOP,outcomes,closure,UNKNOWN

COST=.01


class Planner:
    def __init__(self,root,probabilities):
        if len(probabilities)!=root.n or any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
            raise ValueError("invalid public predicted sign probabilities")
        self.root=root
        self.assignments=tuple(product((-1,1),repeat=root.n))
        self.weights=tuple(math.prod(p if s==1 else 1-p for p,s in zip(probabilities,w))
                           for w in self.assignments)
        self.yes=[0]*len(root.offers)
        self.support=0
        for i,w in enumerate(self.assignments):
            if self.weights[i]>0:self.support|=1<<i
            for j,ok in enumerate(outcomes(root,w)):
                if ok:self.yes[j]|=1<<i
        self._mass={0:0.}
        self._coverage={}
        self._value={}
        self._possible={(0,0):self.support}
        self.planner_calls=0

    def mass(self,mask):
        if mask not in self._mass:
            total=0.;bits=mask
            while bits:
                bit=bits&-bits;bits-=bit
                total+=self.weights[bit.bit_length()-1]
            self._mass[mask]=total
        return self._mass[mask]

    def possible(self,a,s):
        key=(a,s)
        if key in self._possible:return self._possible[key]
        if s&~a or a.bit_count()>self.root.budget:
            raise ValueError("invalid native query state")
        mask=self.support
        for j,yes in enumerate(self.yes):
            if a>>j&1:mask&=yes if s>>j&1 else ~yes
        if not mask:raise ValueError("observations outside prior support")
        self._possible[key]=mask
        return mask

    def coverage(self,s):
        if s not in self._coverage:
            self._coverage[s]=sum(v!=UNKNOWN for v in closure(self.root,s).values())
        return self._coverage[s]

    def legal(self,a):
        return (tuple(j for j in range(len(self.yes)) if not a>>j&1)
                if a.bit_count()<self.root.budget else ())+(STOP,)

    def stop_value(self,a,s):
        c=self.coverage(s)
        return float(c==self.root.n)+.25*c/self.root.n-COST*a.bit_count()

    def branches(self,a,s,j):
        m=self.possible(a,s);total=self.mass(m)
        aa=a|1<<j
        yes=m&self.yes[j];no=m&~self.yes[j]
        out=[]
        if yes:
            ss=s|1<<j;self._possible[aa,ss]=yes
            out.append((self.mass(yes)/total,aa,ss))
        if no:
            self._possible[aa,s]=no
            out.append((self.mass(no)/total,aa,s))
        return out

    def values(self,a,s,depth=None):
        self.planner_calls+=1
        result={STOP:self.stop_value(a,s)}
        if depth==0:return result
        d=None if depth is None else depth-1
        for j in self.legal(a):
            if j!=STOP:
                result[j]=sum(p*self.value(aa,ss,d) for p,aa,ss in self.branches(a,s,j))
        return result

    def value(self,a,s,depth=None):
        key=(a,s,depth)
        if key not in self._value:self._value[key]=max(self.values(a,s,depth).values())
        return self._value[key]

    def choose(self,a,s,depth=2):
        q=self.values(a,s,depth);best=max(q.values())
        return min((j for j,v in q.items() if v>=best-1e-10),
                   key=lambda j:(j!=STOP,j))


def execute(root,receipts,probabilities,depth=2):
    """Adaptive policy with frozen prior and shared posterior search.

    The deterministic cached native verdict table is accessed only at the
    selected legal index, after charging that compound proof attempt.
    """
    planner=Planner(root,probabilities)
    a=s=0;events=[]
    while True:
        j=planner.choose(a,s,depth)
        if j not in planner.legal(a):raise ValueError("illegal query")
        if j==STOP:break
        a|=1<<j
        verdict=receipts[j]["accepted"]
        if verdict:s|=1<<j
        # Verify the observation is in support (learned priors are clipped).
        planner.possible(a,s)
        events.append({"query":j,"accepted":verdict,"attempted":a,"positive":s,
                       "resolved":planner.coverage(s)})
    n=planner.coverage(s)
    return {"accepted_mask":s,"attempted_mask":a,"queries":a.bit_count(),
            "resolved":n,"coverage":n/root.n,"full":n==root.n,
            "utility":planner.stop_value(a,s),"planner_calls":planner.planner_calls,
            "events":events}
