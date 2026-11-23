"""OLR-METTA-04: paired dependent proof roots with fallible MeTTa testimony.

These are new synthetic witness claims, not actual MeTTa knowledge. Hyperon
checks source identities and revisions; it cannot certify their truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .data import digest
from .olr_dependent import DependentWorld,generate_pairs
from .metta_budget_bridge import EvidenceClaim,curate_claims


@dataclass(frozen=True)
class DependentEvidence:
    proof: DependentWorld
    mode: str
    claims: tuple[EvidenceClaim,...]
    macro_allowed: bool

    def public(self):
        prepared=curate_claims(self.claims)
        return {"root_id":self.proof.root_id,
                "pair_id":self.proof.pair_id,
                "world":self.proof.world,
                "split":self.proof.split,
                "mode":self.mode,
                "model_features":prepared["features"],
                "claims":prepared["untrusted_claims"],
                "metta_program_sha256":prepared["program_sha256"],
                "admitted_claims":prepared["admitted_ids"],
                "macro_allowed":self.macro_allowed,
                "statement":self.proof.statement(),
                "theorem_sha256":digest(self.proof.statement())}


def collect(split:str,*,seed:int,count:int,low:int,high:int,
            excluded:frozenset[tuple[int,int,int]]=frozenset()):
    roots=generate_pairs(split,seed=seed,count=count,low=low,high=high,
                         excluded=excluded)
    result=[]
    for i,root in enumerate(roots):
        mode=("strong","weak","conflict","stale")[(i//2)%4]
        rng=random.Random("olr-metta04/public-claims/"+root.root_id)
        truth=root.reference_branch()
        if mode=="strong":
            probs=(.97,.94,.91);weights=(3,3,2)
        elif mode=="weak":
            probs=(.60,.54,.50);weights=(1,1,1)
        elif mode=="conflict":
            probs=(.85,.18,.50);weights=(3,3,1)
        else:
            probs=(.93,.83,.74);weights=(3,2,1)
        votes=tuple(truth if rng.random()<p else 1-truth for p in probs)
        claims=(
            EvidenceClaim("c0","s0",1,1-votes[0],3),
            EvidenceClaim("c1","s0",2,votes[0],weights[0]),
            EvidenceClaim("c2","s1",1,votes[1],weights[1]),
            EvidenceClaim("c3","s2",1,1-truth,3),
            EvidenceClaim("c4","s2",1,1-truth,3),
            EvidenceClaim("c5","s3",1,votes[2],weights[2]),
        )
        # Eligibility is a *public source state* independent of the
        # correct branch placement; both members of each pair share it.
        cap=mode in ("strong","conflict")
        result.append(DependentEvidence(root,mode,claims,cap))
    return tuple(result)


def load_world(row:dict)->DependentEvidence:
    """Rebuild source exactly and reject changed lineage/claim observations.

    Used only for native execution and identity checks, not to make a
    model decision from private truth metadata.
    """
    required={"root_id","pair_id","split","world","statement","mode",
              "macro_allowed","claims","model_features","metta_program_sha256"}
    if not required.issubset(row):
        raise ValueError("incomplete registered dependent world")
    from .olr_dependent import STMT
    m=STMT.fullmatch(row["statement"])
    if m is None:
        raise ValueError("registered source outside dependent Lean grammar")
    a,b,c,d,e,f,g,h,i,j=(int(v) for v in m.groups())
    if not (a==d==h and b==e==i and c==f and g==j):
        raise ValueError("dependent proof candidate index or source grammar drifted")
    delta=abs(g-c)
    proof=DependentWorld(row["root_id"],row["pair_id"],row["split"],
                         row["world"],a,b,delta)
    if proof.statement()!=row["statement"]:
        raise ValueError("paired source proof altered")
    claims=tuple(EvidenceClaim(**x) for x in row["claims"])
    return DependentEvidence(proof,row["mode"],claims,row["macro_allowed"])
