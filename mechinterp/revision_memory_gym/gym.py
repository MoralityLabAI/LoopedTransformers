"""LRM-REV-01: synthetic delayed-evidence recurrence gym.

This is an input-and-score interface, not a pretrained checkpoint assay.
Independent source host decisions are simulated by deterministic public hash
seeds; no authentication, causal mechanism or statistical confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from math import comb
import random

PRIORS=(Q(9,20),Q(7,20),Q(1,5))
MATCH=(Q(19,20),Q(13,20),Q(3,20))
N=12
J_ARMS=(1,3,6)
DOMAIN=b"lrm-rev01/public-engineering/"


def mass(n,k,p):
    if not 0<=k<=n:
        return Q(0)
    return Q(comb(n,k))*p**k*(1-p)**(n-k)


def normalize(weights):
    z=sum(weights,Q(0))
    if z==0:
        raise ValueError("impossible message")
    return tuple(w/z for w in weights)


def oracle(old_count:int,new_count:int,old_revealed:int,r:int,j:int,n:int=N):
    """Bayes posterior for an explicit old-count bottleneck and disclosures.

    The conditioning data are (n, j, S, old_count, replacements, the fixed
    prefix of revealed old bits). A learner that received the named
    old bits has a richer input history; its final Bayes reference is full.
    """
    if not all(type(v)==int for v in (old_count,new_count,old_revealed,r,j,n)):
        raise ValueError("integer metadata required")
    if not(1<=n<=12 and 1<=j<n and 0<=r<=j
           and 0<=old_count<=n and 0<=new_count<=j
           and 0<=old_revealed<=r):
        raise ValueError("invalid stage")
    remaining=old_count-old_revealed
    if remaining<0 or remaining>n-r:
        raise ValueError("impossible old count after disclosure")
    weights=[]
    for pi,t in zip(PRIORS,MATCH):
        accepted_new=mass(j,new_count,t)
        old_valid=sum((mass(n-j,u,t)*mass(j-r,remaining-u,Q(1,2))
                       for u in range(n-j+1)),Q(0))
        weights.append(pi*accepted_new*old_valid)
    return normalize(weights)


def full_posterior(matches:int,n:int=N):
    return normalize(tuple(pi*t**matches*(1-t)**(n-matches)
                          for pi,t in zip(PRIORS,MATCH)))


@dataclass(frozen=True)
class Episode:
    source_id:str
    true_regime:int     # private host label
    old:tuple[int,...]
    correction_slots:tuple[int,...]
    replacements:tuple[int,...]

    def public_encoder_input(self,early:bool):
        """Only host-authorized public info. Regime is never in these messages."""
        metadata={"n":len(self.old),"j":len(self.correction_slots)}
        old=[{"type":"old-match","position":i,"bit":v}
             for i,v in enumerate(self.old)]
        if early:
            return {**metadata,"schedule":list(self.correction_slots),
                    "observations":[event for event in old
                                    if event["position"] not in self.correction_slots]}
        return {**metadata,"schedule":None,"observations":old}

    def correction_messages(self,revealed:int=0):
        if not 0<=revealed<=len(self.correction_slots):
            raise ValueError("out-of-range reveal budget")
        return {"schedule":list(self.correction_slots),
                "accepted":[{"position":i,"bit":v}
                            for i,v in zip(self.correction_slots,self.replacements)],
                "old_disclosures":[{"position":i,"bit":self.old[i]}
                                   for i in self.correction_slots[:revealed]]}

    def reduced_reference(self,revealed:int=0):
        """Bayes for old-count compression plus the declared old disclosures."""
        j=len(self.correction_slots)
        return oracle(sum(self.old),sum(self.replacements),
                      sum(self.old[i] for i in self.correction_slots[:revealed]),
                      revealed,j,len(self.old))

    def reference(self,revealed:int=0):
        """Compatibility alias for reduced_reference; assumes count compression."""
        return self.reduced_reference(revealed)

    def accepted(self):
        bits=list(self.old)
        for i,b in zip(self.correction_slots,self.replacements):
            bits[i]=b
        return tuple(bits)

    def full_reference(self):
        """Input-matched final Bayes after the full old stream and corrections.

        Once S and all replacements are supplied, re-revealing old bits adds
        no information to this full transcript, so this target is fixed in r.
        """
        return full_posterior(sum(self.accepted()),len(self.old))

    def verified_old_total(self,authoritative_old):
        """Simulated external old-history checker for an aggregate v message."""
        if tuple(authoritative_old)!=self.old:
            raise ValueError("archive old transcript mismatch")
        return sum(self.old[i] for i in self.correction_slots)


def make_episode(index:int,j:int=6,n:int=N)->Episode:
    if(type(index)!=int or not 0<=index<1000000
       or type(j)!=int or j not in J_ARMS or type(n)!=int
       or not j<n<=N):
        raise ValueError("registered public engineering domain only")
    seed=sha256(DOMAIN+str(j).encode()+b"/"+str(n).encode()+b"/"+str(index).encode()).digest()
    rng=random.Random(int.from_bytes(seed,"big"))
    draw=rng.randrange(20)
    regime=0 if draw<9 else (1 if draw<16 else 2)
    chosen=tuple(sorted(rng.sample(range(n),j)))
    old=tuple(rng.randrange(2) if i in chosen else
              int(rng.randrange(20)<int(MATCH[regime]*20))
              for i in range(n))
    replacements=tuple(int(rng.randrange(20)<int(MATCH[regime]*20)) for _ in chosen)
    return Episode(seed.hex(),regime,old,chosen,replacements)


def brier(posterior,regime):
    return sum((float(v)-int(k==regime))**2 for k,v in enumerate(posterior))

