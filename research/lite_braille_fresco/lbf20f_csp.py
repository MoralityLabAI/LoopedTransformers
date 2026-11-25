"""LBF-20F: finite shared-witness CSPs, sound local propagation and exact labels."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, Counter
import itertools
import random
import time

D=4
FULL=(1<<D)-1
FAMILIES=("parity_ring","sparse","dense")


def bits(m):
    while m:
        b=m&-m
        yield b.bit_length()-1
        m^=b


def transpose(rows):
    return tuple(sum(1<<a for a in range(D) if rows[a]&(1<<b)) for b in range(D))


@dataclass(frozen=True)
class Case:
    root: str
    family: str
    n: int
    domains: tuple
    edges: tuple

    def relations(self):
        return {key:v for i,j,r in self.edges for key,v in (((i,j),r),((j,i),transpose(r)))}

    def valid(self,values):
        return (len(values)==self.n and all(0<=v<D and self.domains[i]&(1<<v) for i,v in enumerate(values))
                and all(r[values[i]]&(1<<values[j]) for i,j,r in self.edges))


def generate(split,n,count,seed,*,allow_unsat=False):
    rng=random.Random(seed)
    out=[]
    seen=set()
    while len(out)<count:
        k=len(out); family=FAMILIES[k%3]
        domains=[FULL]*n
        if family=="parity_ring":
            # Two disjoint 2-lifts of the same cycle. Mode 0 has even parity,
            # mode 1 odd parity. Each row/column has exactly one support.
            perms=[rng.sample(range(D),D) for _ in range(n)]
            even=[rng.randrange(2) for _ in range(n-1)]
            even.append(sum(even)%2)
            odd=[rng.randrange(2) for _ in range(n-1)]
            odd.append(1-(sum(odd)%2))
            if allow_unsat and (k//3)%4==3:
                even[-1]^=1
            edges=[]
            for i in range(n):
                j=(i+1)%n
                rows=[0]*D
                for mode,parities in enumerate((even,odd)):
                    for bit in range(2):
                        a=perms[i][2*mode+bit]
                        b=perms[j][2*mode+(bit^parities[i])]
                        rows[a]=1<<b
                edges.append((i,j,tuple(rows)) if i<j else (j,i,transpose(rows)))
        else:
            plant=[rng.randrange(D) for _ in range(n)]
            pairs={(min(i,(i+1)%n),max(i,(i+1)%n)) for i in range(n)}
            density=.20 if family=="sparse" else .55
            for i,j in itertools.combinations(range(n),2):
                if rng.random()<density:
                    pairs.add((i,j))
            edges=[]
            for i,j in sorted(pairs):
                prob=rng.uniform(.30,.63) if family=="sparse" else rng.uniform(.40,.70)
                rows=[sum(1<<b for b in range(D) if rng.random()<prob) for _ in range(D)]
                if not (allow_unsat and k%11==10):
                    rows[plant[i]]|=1<<plant[j]
                edges.append((i,j,tuple(rows)))
            for i in range(n):
                if rng.random()<.24:
                    domains[i]=sum(1<<a for a in range(D) if a==plant[i] or rng.random()<.65)
        edges=tuple(sorted(edges))
        sig=(tuple(domains),edges)
        if sig in seen: continue
        seen.add(sig)
        out.append(Case(f"{split}_{k:03d}",family,n,tuple(domains),edges))
    return out


class Solver:
    def __init__(self,c):
        self.c=c
        self.rel=c.relations()
        self.nb=[tuple(j for j in range(c.n) if (i,j) in self.rel) for i in range(c.n)]

    def ac(self,dom,stats=None):
        ds=list(dom)
        if not all(ds):return None
        queue=deque(sorted(self.rel))
        pending=set(queue)
        while queue:
            i,j=queue.popleft();pending.remove((i,j))
            if stats is not None:stats["arc_revisions"]+=1
            r=self.rel[i,j]
            new=sum(1<<a for a in bits(ds[i]) if r[a]&ds[j])
            if not new:return None
            if new!=ds[i]:
                ds[i]=new
                for k in self.nb[i]:
                    if k!=j and (k,i) not in pending:
                        queue.append((k,i));pending.add((k,i))
        return tuple(ds)

    def pc(self,dom,stats=None):
        # Full pair-relation consistency, including diagonal identity.
        # Every surviving solution supplies the same witness at intermediate k.
        n=self.c.n
        a=[[tuple((1<<v if i==j else self.rel.get((i,j),(FULL,)*D)[v])&dom[j]
                  if dom[i]&(1<<v) else 0 for v in range(D)) for j in range(n)] for i in range(n)]
        while True:
            changed=False
            for k in range(n):
                for i in range(n):
                    left=a[i][k]
                    for j in range(n):
                        if stats is not None:stats["pair_revisions"]+=1
                        right=a[k][j]
                        old=a[i][j]
                        new=[]
                        for v,row in enumerate(old):
                            support=0
                            for u in bits(left[v]):
                                support|=right[u]
                            new.append(row&support)
                        new=tuple(new)
                        if not any(new):return None
                        if new!=old:
                            a[i][j]=new;changed=True
            if not changed:
                return tuple(sum(1<<v for v in range(D) if a[i][i][v]&(1<<v)) for i in range(n))

    def variable(self,dom):
        return min((i for i in range(self.c.n) if dom[i].bit_count()>1),
                   key=lambda i:(dom[i].bit_count(),-len(self.nb[i]),i))

    def lcv(self,dom,i):
        return [sum((self.rel[i,j][a]&dom[j]).bit_count() for j in self.nb[i]) for a in range(D)]

    def bp(self,dom,i,passes=4):
        messages={key:[1.]*D for key in self.rel}
        for _ in range(passes):
            nxt={}
            for (u,v),r in self.rel.items():
                arr=[0.]*D
                for a in bits(dom[u]):
                    w=1.
                    for k in self.nb[u]:
                        if k!=v:w*=messages[k,u][a]
                    for b in bits(r[a]&dom[v]):arr[b]+=w
                z=sum(arr)
                nxt[u,v]=[x/z for x in arr] if z else [0.]*D
            messages=nxt
        scores=[]
        for a in range(D):
            v=1.
            for j in self.nb[i]:v*=messages[j,i][a]
            scores.append(v)
        return scores

    def solve(self,dom=None,policy="lcv",budget=None,scorer=None):
        stats=Counter(nodes=0,arc_revisions=0,pair_revisions=0,model_calls=0)
        capped=False
        start=time.perf_counter()
        def visit(state):
            nonlocal capped
            if budget is not None and stats["nodes"]>=budget:
                capped=True;return None
            stats["nodes"]+=1
            state=self.pc(state,stats) if policy=="pc" else self.ac(state,stats)
            if state is None:return None
            if all(m.bit_count()==1 for m in state):
                v=tuple(next(bits(m)) for m in state)
                assert self.c.valid(v)
                return v
            i=self.variable(state)
            if scorer is not None:
                stats["model_calls"]+=1
                score=scorer(self,state,i)
            elif policy=="bp":
                score=self.bp(state,i)
            else:score=self.lcv(state,i)
            for a in sorted(bits(state[i]),key=lambda a:(-float(score[a]),a)):
                child=list(state);child[i]=1<<a
                v=visit(tuple(child))
                if v is not None:return v
                if capped:return None
            return None
        v=visit(self.c.domains if dom is None else dom)
        return {"status":"solved" if v is not None else "budget" if capped else "refuted",
                "solution":v,**stats,"wall_seconds":time.perf_counter()-start}

    def supports(self,dom):
        """Exact all-solution unary projection via independent complete feasibility checks."""
        masks=[]
        for i,m in enumerate(dom):
            ok=0
            for a in bits(m):
                child=list(dom);child[i]=1<<a
                if self.solve(tuple(child))["solution"] is not None:ok|=1<<a
            masks.append(ok)
        return tuple(masks)


def audit():
    count=Counter(systems=0,worlds=0,feasible_worlds=0,ac_false_negatives=0,pc_false_negatives=0,
                  exact_support_mismatches=0,ring_checks=0)
    for c in generate("audit",4,96,2026112514,allow_unsat=True):
        s=Solver(c)
        worlds=list(itertools.product(range(D),repeat=c.n))
        feasible=[v for v in worlds if c.valid(v)]
        exact=tuple(sum(1<<a for a in range(D) if any(v[i]==a for v in feasible)) for i in range(c.n))
        ac=s.ac(c.domains);pc=s.pc(c.domains)
        got=s.supports(c.domains)
        assert got==exact
        assert not feasible or (ac is not None and pc is not None)
        if feasible:
            assert all(a&b==a for a,b in zip(exact,ac))
            assert all(a&b==a for a,b in zip(exact,pc))
        count["systems"]+=1;count["worlds"]+=len(worlds);count["feasible_worlds"]+=len(feasible)
    for n in (6,8,10,12):
        for c in generate("ring_audit",n,24,2026112520+n):
            if c.family!="parity_ring":continue
            s=Solver(c)
            exact=s.supports(c.domains)
            assert s.ac(c.domains)==c.domains and all(m.bit_count()==2 for m in exact)
            assert s.pc(c.domains)==exact
            count["ring_checks"]+=1
    return dict(count)
