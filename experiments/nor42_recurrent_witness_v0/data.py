"""Source-audited, public-only features for NW-01 NOR42/TRM pilot.

The original NOR42 first run is already published. This module keeps its two
worlds together and provides the hidden beta only to the witnessed information
arm. It never uses original grade policy scores for model training or selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import math
import random

SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
GRADE_SHA="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
SOURCE_LENGTH=9059929
GRADE_LENGTH=201956
ORIGINAL_ZIP_SHA="e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d"
ORIGINAL_COMMIT="0a330602c72ec34017178857956e306710f724a1"
ROOT_SEED=202611420201
REPS=tuple(range(100,116))
FAMILIES=("wide_dense_small","wide_regime_blocks")
OPS=("I","N","R","S")
DELAYS=(64,80,96,112)
SPACINGS=(24,32)
AMPLITUDES=(-2,-1,1,2)
INDEX_SAMPLE=(1,2,3,4,8,16,32,64,128,256)
D_REL=(-2,-1,0,1,16,32)
PUBLIC_DIM=10+6+4+4+2+1+16+2+1+4


@dataclass(frozen=True)
class PublicRoot:
    identity: str
    replicate: int
    family: str
    degree: int
    features: tuple[float,...]
    hidden_beta: int


def digest(raw:bytes)->str:
    return sha256(raw).hexdigest()


def validate_archive(path:Path)->list[PublicRoot]:
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    with ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError("invalid archived NOR42 ZIP")
        raw=z.read("nor42_fixture.json")
        graded=z.read("nor42_full.json")
    if len(raw)!=SOURCE_LENGTH or digest(raw)!=SOURCE_SHA:
        raise ValueError("NOR42 first public source SHA mismatch")
    if len(graded)!=GRADE_LENGTH or digest(graded)!=GRADE_SHA:
        raise ValueError("NOR42 first grade SHA mismatch")
    source=json.loads(raw);score=json.loads(graded)
    if source["schema"]!="nor42.paired_grammar_ood_public_only_reps100to115.v1":
        raise ValueError("unexpected original source schema")
    if score["schema"]!="nor42.paired_program_ood_authenticated_witness.v1":
        raise ValueError("unexpected original grade schema")
    if score["public_fixture_sha256"]!=SOURCE_SHA:
        raise ValueError("source/grade association mismatch")
    if not source["public_only"] or not source["planted_and_private_target_absent"]:
        raise ValueError("original source is not public-only")
    case_by_sha={a["full_public_input_sha256"]:a for a in source["cases"]}
    beta_by_sha={a["public_sha256"]:a["hidden_beta"] for a in score["source_records"]}
    if len(case_by_sha)!=64 or len(beta_by_sha)!=64:
        raise ValueError("expected distinct 64 pair roots")
    if set(case_by_sha)!=set(beta_by_sha):
        raise ValueError("first grade does not match public source roster")
    roots=[]
    for source_row in source["cases"]:
        identity=source_row["full_public_input_sha256"]
        if source_row["paired_world_labels"]!=["clean","hidden"]:
            raise ValueError("world pairing changed")
        if source_row["paired_world_prior"]!=[1,2]:
            raise ValueError("world prior changed")
        if source_row["pair_replicate"] not in REPS:
            raise ValueError("unknown source replicate")
        beta=beta_by_sha[identity]
        if beta not in AMPLITUDES:raise ValueError("invalid hidden first coefficient")
        roots.append(PublicRoot(identity,source_row["pair_replicate"],
                    source_row["family"],source_row["degree"],
                    public_features(source_row),beta))
    if len(roots)!=64:
        raise ValueError("invalid unique root count")
    for rep in REPS:
        if {(a.degree,a.family) for a in roots if a.replicate==rep}!={
            (d,f) for d in (512,1024) for f in FAMILIES}:
            raise ValueError("missing a replicate degree/family cell")
    return roots


def _query_features(query:list[int],degree:int)->list[float]:
    q=len(query)-1
    if q!=3*degree//2 or query[0]!=1:
        raise ValueError("query must be normalized degree 3D/2")
    samples=[max(-1.,min(1.,float(query[min(i,q)])/2.))
             for i in INDEX_SAMPLE]
    samples += [max(-1.,min(1.,float(query[min(degree+i,q)])/2.))
                for i in D_REL]
    return samples


def public_features(row:dict)->tuple[float,...]:
    """Never reads a target-world label, native private program or hidden beta."""
    q=row["public_query_input"]
    d=row["degree"]
    family=row["family"]
    head=row["public_header"]
    mask=int(row["original_native_public_mask"])
    if family not in FAMILIES or mask<=0 or mask.bit_count()!=1:
        raise ValueError("invalid NOR42 public feature identity")
    index=mask.bit_length()-1
    coef=_query_features(q,d)
    coef += [float(head["base"]==op) for op in OPS]
    coef += [float(head["delay"]==delta) for delta in DELAYS]
    coef += [float(head["spacing"]==s) for s in SPACINGS]
    coef += [head["amplitude"]/2.0]
    coef += [float(i==index) for i in range(16)]
    coef += [float(family==f) for f in FAMILIES]
    coef += [d/1024.0]
    for regime in range(4):
        indices=[i for i in range(1,len(q))
                 if ((i-1)//64)%4==regime]
        coef.append(sum(q[i]!=0 for i in indices)/len(indices))
    assert len(coef)==PUBLIC_DIM
    assert all(math.isfinite(x) and abs(x)<=1.000001 for x in coef)
    return tuple(float(x) for x in coef)


def synthetic_public_roots(n:int=512,seed:int=ROOT_SEED):
    """Synthetic training roots, structurally separate from all NOR42 IDs."""
    rng=random.Random(seed)
    rows=[]
    for i in range(n):
        d=rng.choice((512,1024))
        q=3*d//2
        fam=rng.choice(FAMILIES)
        public=[1]
        for j in range(1,q+1):
            if fam=="wide_dense_small":
                public.append(rng.choice(AMPLITUDES))
            else:
                regime=((j-1)//64)%4
                coin=rng.random()
                hit=coin<(.90,.04,.16,.25)[regime]
                if regime==2:hit=hit and j%16<4
                public.append(rng.choice(AMPLITUDES) if hit else 0)
        fake={
            "degree":d,"family":fam,"public_query_input":public,
            "public_header":{
                "base":rng.choice(OPS),
                "delay":rng.choice(DELAYS),
                "spacing":rng.choice(SPACINGS),
                "amplitude":rng.choice(AMPLITUDES)},
            "original_native_public_mask":1<<rng.randrange(16)
        }
        rows.append(PublicRoot(f"NW01-synthetic-fit-{i:05d}",-1,fam,d,
                               public_features(fake),rng.choice(AMPLITUDES)))
    return rows


def feature_manifest(roots:list[PublicRoot])->str:
    raw=json.dumps([{
        "id":r.identity,"replicate":r.replicate,"degree":r.degree,
        "family":r.family,"features":r.features
    } for r in roots],separators=(",",":"),sort_keys=True).encode()
    return digest(raw)
