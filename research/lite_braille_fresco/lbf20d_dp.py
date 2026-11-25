"""LBF-20D: finite Bayesian decision trees for monotone premise feasibility.

The outcome of each query is fixed in a latent world. Observing an outcome
updates the posterior, not the underlying world's guards. This is a model
of verifier decisions, not a theorem about Lean tactic behavior.
"""
from __future__ import annotations
import functools
import math
import random
import statistics


def bits(mask):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def minimal(masks):
    out = []
    for m in sorted(set(masks), key=lambda x: (x.bit_count(), x)):
        if not any(n & m == n for n in out):
            out.append(m)
    return tuple(out)


class Problem:
    def __init__(self, masks, components, weights, *, all_queries=True):
        self.n = len(components[0])
        self.full = (1 << self.n) - 1
        self.masks = minimal(masks)
        assert all(0 <= m <= self.full for m in self.masks)
        assert len(weights) == len(components) and abs(sum(weights)-1) < 1e-9
        assert all(w > 0 for w in weights)
        assert all(len(c)==self.n and all(0 < p < 1 for p in c) for c in components)
        self.components = tuple(tuple(c) for c in components)
        self.weights = tuple(weights)
        self.all_queries = all_queries
        self.q = {}
        self.values = {}
        self.posterior_cache = {}
        self.terms_cache = {}
        self._likes = {}

    def remaining(self, s, f):
        key = s, f
        if key not in self.terms_cache:
            self.terms_cache[key] = minimal(m & ~s for m in self.masks if not m & f)
        return self.terms_cache[key]

    def terminal(self, s, f):
        r = self.remaining(s,f)
        return True if 0 in r else False if not r else None

    def _likelihood(self, z, mask, positive):
        key = z, mask, positive
        if key not in self._likes:
            self._likes[key] = math.prod(self.components[z][g] if positive else 1-self.components[z][g]
                                         for g in bits(mask))
        return self._likes[key]

    def posterior(self, s, f):
        assert not s & f and (s|f) <= self.full
        key = s,f
        if key not in self.posterior_cache:
            w = [p*self._likelihood(z,s,True)*self._likelihood(z,f,False)
                 for z,p in enumerate(self.weights)]
            d=sum(w)
            assert d>0
            self.posterior_cache[key]=tuple(sum(w[z]*c[g] for z,c in enumerate(self.components))/d
                                            for g in range(self.n))
        return self.posterior_cache[key]

    def solve(self,s=0,f=0):
        key=s,f
        if key in self.values: return self.values[key]
        r=self.remaining(s,f)
        if not r or 0 in r:
            return 0.,None
        p=self.posterior(s,f)
        u=self.full & ~(s|f)
        if not self.all_queries:
            u=0
            for m in r: u |= m
        qs={}
        for g in bits(u):
            b=1<<g
            qs[g]=1+p[g]*self.solve(s|b,f)[0]+(1-p[g])*self.solve(s,f|b)[0]
        v,g=min((v,g) for g,v in qs.items())
        self.q[key]=qs
        self.values[key]=(v,g)
        return v,g

    def choose_risk(self,s,f,mode="risk",prior=None):
        r=self.remaining(s,f)
        if not r or 0 in r: return None
        p=(prior or self).posterior(s,f)
        if mode=="count":
            mask=min(r,key=lambda m:(m.bit_count(),m))
        else:
            mask=min(r,key=lambda m:(sum(1/max(.05,p[g]) for g in bits(m)),m.bit_count(),m))
        return min(bits(mask),key=lambda g:(p[g],g))

    def expected(self,policy):
        """Exact expectation under this joint law for a deterministic policy."""
        @functools.cache
        def v(s,f):
            if self.terminal(s,f) is not None: return 0.
            g=policy(s,f)
            assert type(g) is int and 0<=g<self.n and not (s|f)&(1<<g)
            b=1<<g
            p=self.posterior(s,f)[g]
            return 1+p*v(s|b,f)+(1-p)*v(s,f|b)
        return v(0,0)

    def exhaustive_expected(self,policy):
        """Independent integration over complete worlds, with explicit traces."""
        cost=mass=success=0.
        for world in range(self.full+1):
            pr=sum(w*math.prod(p if world&(1<<g) else 1-p for g,p in enumerate(c))
                   for w,c in zip(self.weights,self.components))
            s=f=k=0
            while self.terminal(s,f) is None:
                g=policy(s,f); b=1<<g
                assert not (s|f)&b
                if world&b: s|=b
                else: f|=b
                k+=1
                assert k<=self.n
            actual=any(m&world==m for m in self.masks)
            assert self.terminal(s,f)==actual
            cost+=pr*k; mass+=pr; success+=pr*actual
        assert abs(mass-1)<1e-9
        return cost,success


def independent(problem):
    return Problem(problem.masks,[problem.posterior(0,0)],[1.],all_queries=False)


def synthetic_audit():
    rng=random.Random(2026112508)
    rows=[]
    for i in range(128):
        n=6
        masks=minimal(sum(1<<g for g in rng.sample(range(n),rng.randint(1,4))) for _ in range(7))
        a=[rng.choice((.1,.2,.35,.65,.8,.9)) for _ in range(n)]
        b=[1-p if rng.random()<.8 else p for p in a]
        w=rng.choice((.3,.5,.7))
        model=Problem(masks,[a,b],[w,1-w])
        val,action=model.solve()
        exact,success=model.exhaustive_expected(lambda s,f:model.solve(s,f)[1])
        assert abs(val-exact)<1e-9
        ind=independent(model)
        restricted=Problem(masks,[a,b],[w,1-w],all_queries=False)
        row={"joint_opt":val,
             "independent_dp":model.expected(lambda s,f:ind.solve(s,f)[1]),
             "restricted_joint_dp":model.expected(lambda s,f:restricted.solve(s,f)[1]),
             "posterior_risk":model.expected(lambda s,f:model.choose_risk(s,f)),
             "marginal_risk":model.expected(lambda s,f:model.choose_risk(s,f,prior=ind)),
             "success_probability":success}
        assert all(row[k]+1e-9>=val for k in ("independent_dp","restricted_joint_dp","posterior_risk","marginal_risk"))
        rows.append(row)
    keys=tuple(rows[0])
    return {"systems":len(rows),"exact_world_integrations":len(rows),
            "mean":{k:statistics.mean(r[k] for r in rows) for k in keys},
            "strict_joint_advantage":{k:sum(r[k]>r["joint_opt"]+1e-9 for r in rows)
                                      for k in keys if k not in ("joint_opt","success_probability")}}
