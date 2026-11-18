"""LRM-REV-03: finite continuation lower bound and count-state Bayes benchmark.

All arithmetic is exact Fraction arithmetic. The named-edit oracle has access to
all named old bits, while the count-state oracle receives the original total,
correction schedule, accepted replacement values, and disclosed old positions.
Neither oracle has a regime label as input. This is an analytic diagnostic,
not a learned network or a mechanistic intervention result.
"""
from __future__ import annotations

from functools import lru_cache
from fractions import Fraction as F
from itertools import combinations
from math import comb

N, J = 12, 6
PRIORS = (F(9, 20), F(7, 20), F(1, 5))
MATCH = (F(19, 20), F(13, 20), F(3, 20))
HALF = F(1, 2)


def binom(n: int, k: int, p: F) -> F:
    return F(comb(n, k)) * p**k * (1-p)**(n-k) if 0 <= k <= n else F(0)


def posterior_from_weights(weights: tuple[F, ...]) -> tuple[F, ...]:
    total = sum(weights, F(0))
    if total <= 0:
        raise ValueError('impossible observed history')
    return tuple(w / total for w in weights)


@lru_cache(None)
def full_posterior(accepted_ones: int, n: int = N) -> tuple[F, ...]:
    if type(accepted_ones) is not int or type(n) is not int or not (1 <= n <= N and 0 <= accepted_ones <= n):
        raise ValueError('invalid count')
    return posterior_from_weights(tuple(prior * p**accepted_ones * (1-p)**(n-accepted_ones)
                                        for prior, p in zip(PRIORS, MATCH)))


@lru_cache(None)
def initial_count_posterior(old_total: int, n: int = N, j: int = J) -> tuple[F, ...]:
    if type(old_total) is not int or type(n) is not int or type(j) is not int or not (0 <= j <= n <= N and 0 <= old_total <= n):
        raise ValueError('invalid initial count')
    return posterior_from_weights(tuple(
        prior * sum((binom(n-j, valid_ones, p) * binom(j, old_total-valid_ones, HALF)
                     for valid_ones in range(n-j+1)), F(0))
        for prior, p in zip(PRIORS, MATCH)))


@lru_cache(None)
def count_posterior(old_total: int, replacement_ones: int,
                    disclosed_corrected_count: int, disclosed_corrected_ones: int,
                    disclosed_unchanged_count: int, disclosed_unchanged_ones: int,
                    n: int = N, j: int = J) -> tuple[F, ...]:
    """Exact Pr(K | old total, S, Y, disclosed OLD bits), after all Y arrive.

    Disclosed positions may include corrected and unchanged sites. The
    replacement total and disclosed counts are sufficient under the iid law.
    An impossible event raises ValueError. No unobserved individual X values
    are accessed.
    """
    v=(old_total,replacement_ones,disclosed_corrected_count,disclosed_corrected_ones,
       disclosed_unchanged_count,disclosed_unchanged_ones,n,j)
    if any(type(x) is not int for x in v):
        raise ValueError('integer metadata required')
    m,y,dc,xc,du,xu,n,j=v
    if not (0 <= j <= n <= N and 0 <= m <= n and 0 <= y <= j
            and 0 <= xc <= dc <= j and 0 <= xu <= du <= n-j):
        raise ValueError('invalid disclosure counts')
    rc,ru = j-dc, n-j-du
    unobserved = m-xc-xu
    weights=[]
    for prior,p in zip(PRIORS,MATCH):
        unknown_old_likelihood = sum((binom(rc,a,HALF) * binom(ru,unobserved-a,p)
                                      for a in range(rc+1)),F(0))
        weights.append(prior * p**(y+xu) * (1-p)**(j-y+du-xu)
                       * unknown_old_likelihood)
    return posterior_from_weights(tuple(weights))


def named_posterior(old: tuple[int, ...], corrections: tuple[int, ...],
                    replacements: tuple[int, ...]) -> tuple[F, ...]:
    if len(old)!=N or any(type(b) is not int or b not in (0,1) for b in old):
        raise ValueError('requires twelve named old bits')
    if len(corrections)!=J or len(set(corrections))!=J or any(type(i) is not int or not 0<=i<N for i in corrections):
        raise ValueError('requires six distinct corrected slots')
    if len(replacements)!=J or any(type(b) is not int or b not in (0,1) for b in replacements):
        raise ValueError('requires six replacement bits')
    accepted=list(old)
    for i,b in zip(corrections,replacements):
        accepted[i]=b
    return full_posterior(sum(accepted))


@lru_cache(None)
def _partial_named_by_count(unreplaced_old_ones: int, replacement_ones: int,
                            disclosed_corrections: int, n: int = N, j: int = J) -> tuple[F,...]:
    """Full-input Bayes after r distinct REPLACE events, with future S unknown.

    The model has all n named old bits, has learned r positions and their
    accepted replacements, and knows j corrections exist overall. The unseen
    j-r correction positions are integrated out exactly.
    """
    v=(unreplaced_old_ones,replacement_ones,disclosed_corrections,n,j)
    if any(type(x) is not int for x in v):
        raise ValueError('integer metadata required')
    m,y,r,n,j=v
    if not (0<=j<=n<=N and 0<=r<=j and 0<=m<=n-r and 0<=y<=r):
        raise ValueError('invalid partial correction summary')
    h=j-r
    weights=[]
    for prior,p in zip(PRIORS,MATCH):
        unknown_s_likelihood=sum((F(comb(m,a)*comb(n-r-m,h-a),2**h)
                                  * p**(m-a) * (1-p)**(n-j-(m-a))
                                  for a in range(h+1)
                                  if 0<=a<=m and 0<=h-a<=n-r-m),F(0))
        weights.append(prior * p**y * (1-p)**(r-y) * unknown_s_likelihood)
    return posterior_from_weights(tuple(weights))


def partial_named_posterior(old:tuple[int,...], known_corrections:tuple[int,...],
                            replacements:tuple[int,...], n:int=N, j:int=J)->tuple[F,...]:
    """Causal teacher: full named OLD prefix and only currently revealed REPLACE events."""
    if len(old)!=n or any(type(b) is not int or b not in (0,1) for b in old):
        raise ValueError('named old bits required')
    r=len(known_corrections)
    if (len(replacements)!=r or len(set(known_corrections))!=r or r>j
            or any(type(i) is not int or not 0<=i<n for i in known_corrections)
            or any(type(b) is not int or b not in (0,1) for b in replacements)):
        raise ValueError('valid disclosed corrections and replacements required')
    untouched=sum(old)-sum(old[i] for i in known_corrections)
    return _partial_named_by_count(untouched,sum(replacements),r,n,j)


def count_posterior_from_public(old_total: int, corrections: tuple[int,...],
                                replacements: tuple[int,...],
                                disclosures: dict[int,int]) -> tuple[F,...]:
    """Pure public observation interface, with explicit disclosed positions only."""
    if len(corrections)!=J or len(set(corrections))!=J or any(type(i) is not int or not 0<=i<N for i in corrections):
        raise ValueError('requires six distinct corrected slots')
    if len(replacements)!=J or any(type(b) is not int or b not in (0,1) for b in replacements):
        raise ValueError('requires six replacement bits')
    if any(type(i) is not int or not 0<=i<N or type(b) is not int or b not in (0,1)
           for i,b in disclosures.items()):
        raise ValueError('disclosures must be named public old-bit values')
    sc=set(corrections)
    dc=sum(i in sc for i in disclosures)
    xc=sum(b for i,b in disclosures.items() if i in sc)
    du=len(disclosures)-dc
    xu=sum(b for i,b in disclosures.items() if i not in sc)
    return count_posterior(old_total,sum(replacements),dc,xc,du,xu)


def squared_excess(p:tuple[F,...],q:tuple[F,...])->F:
    return sum(((a-b)**2 for a,b in zip(p,q)),F(0))


def count_fiber_bound(n:int=N,j:int=J)->dict:
    if type(n) is not int or type(j) is not int or not (1<=j<n<=N):
        raise ValueError('nontrivial edit horizon required')
    schedules=tuple(sum(1<<i for i in slots) for slots in combinations(range(n),j))
    signatures=set()
    fibers=[0]*(n+1)
    for x in range(1<<n):
        # Zero replacement bits: accepted count after the edit is bit_count(x outside S).
        signature=bytes((x & ~schedule).bit_count() for schedule in schedules)
        signatures.add(signature)
        fibers[x.bit_count()]+=1
    maxfiber=max(fibers)
    return {'n':n,'j':j,'schedules':len(schedules),
            'distinct_continuations':len(signatures),'original_states':1<<n,
            'largest_count_fiber':maxfiber,
            'extra_fixed_width_bits':(maxfiber-1).bit_length(),
            'unconstrained_fixed_width_bits':(len(signatures)-1).bit_length(),
            'fiber_sizes':fibers}


def exact_population_regret(r: int, policy: str, n: int = N, j: int = J) -> F:
    """Population expected squared excess of count-only Bayes at reminder stage r.

    The compressed observer remembers old_total, S and replacement bits, plus
    whatever truthful old bits have appeared in reminder tokens. Its oracle is
    conditional on that observable state. `corrected` reveals r members of S;
    `random` reveals the first r of a uniformly random six-slot set T.
    Random S makes overlap with the first-r T follow a hypergeometric law.
    The loss target is full named-history Bayes, never the hidden regime.
    """
    if (n,j)!=(12,6) or type(r) is not int or not 0<=r<=j or policy not in ('corrected','random'):
        raise ValueError('registered exact-population experiment is n=12,j=6,r=0..6')
    total=F(0)
    if policy=='corrected':
        overlaps=((r,F(1)),)
    else:
        overlaps=((q,F(comb(j,q)*comb(n-j,r-q),comb(n,r)))
                  for q in range(max(0,r-(n-j)),min(r,j)+1))
    for q,overlap_weight in overlaps:
        for prior,p in zip(PRIORS,MATCH):
            for dc1 in range(q+1):
                prob_dc1=binom(q,dc1,HALF)
                for dc_hidden in range(j-q+1):
                    prob_dc_hidden=binom(j-q,dc_hidden,HALF)
                    for du1 in range(r-q+1):
                        prob_du1=binom(r-q,du1,p)
                        for du_hidden in range(n-j-(r-q)+1):
                            prob_du_hidden=binom(n-j-(r-q),du_hidden,p)
                            old_total=dc1+dc_hidden+du1+du_hidden
                            accepted_no_y=du1+du_hidden
                            for y in range(j+1):
                                prob_y=binom(j,y,p)
                                mass=prior*overlap_weight*prob_dc1*prob_dc_hidden*prob_du1*prob_du_hidden*prob_y
                                if mass:
                                    observed=count_posterior(old_total,y,q,dc1,r-q,du1,n,j)
                                    named=full_posterior(accepted_no_y+y,n)
                                    total+=mass*squared_excess(observed,named)
    return total


def exact_population_control_regret(n:int=N,j:int=J)->F:
    """Compatibility name for exact random-control regret after six reminders."""
    return exact_population_regret(j,'random',n,j)

def brute_posterior(old_total:int, correction_slots:tuple[int,...],
                    replacement_ones:int, disclosed:dict[int,int],
                    n:int=N,j:int=J)->tuple[F,...]:
    """Independent exhaustive likelihood for tests, enumerating every old bit vector."""
    if len(correction_slots)!=j or len(set(correction_slots))!=j:
        raise ValueError('bad correction slots')
    s=set(correction_slots)
    weights=[]
    for prior,p in zip(PRIORS,MATCH):
        score=F(0)
        for bits in range(1<<n):
            if bits.bit_count()!=old_total or any(((bits>>i)&1)!=v for i,v in disclosed.items()):
                continue
            prob=F(1)
            for i in range(n):
                b=(bits>>i)&1
                prob*=HALF if i in s else (p if b else (1-p))
            score+=prob
        weights.append(prior*score*p**replacement_ones*(1-p)**(j-replacement_ones))
    return posterior_from_weights(tuple(weights))
