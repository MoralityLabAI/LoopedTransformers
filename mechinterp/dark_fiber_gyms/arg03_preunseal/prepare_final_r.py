"""Generate ONLY a fresh final-R development cohort for ARG-03.

Oracle process only; writes model-visible public rows separately from host
gold labels. Never generates/reads old sealed test.jsonl and never calls
the original cases function (which synthesizes the old test as a side effect).

Cross-cohort duplication against sealed test remains UNKNOWN until a
custodian checks the opaque signature manifests without opening payloads.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse
import json
import random

def public(c):
    data=dict(c.public)
    data["public_sha256"]=c.signature
    return data

FRESH_SEED = 2026111617
PER_PROGRAM = 2

def make_fresh(oracle, arg02, prior_dev_hashes:set[str],
               *, seed:int=FRESH_SEED, per_program:int=PER_PROGRAM):
    if per_program<=0 or seed in (2026111503,2026111504):
        raise ValueError("New development needs a fresh seed")
    families=tuple(oracle.CAL_FAMILIES)
    final_r=[x for x in oracle.PROGRAMS if x[2]=="R"]
    if len(final_r)!=16:
        raise AssertionError("Final-R enumeration changed")
    rng=random.Random(seed)
    shuffled=list(final_r);rng.shuffle(shuffled)
    program_partition={"development_finalR_fit":shuffled[:8],
                       "development_finalR_valid":shuffled[8:]}
    forbidden=set(prior_dev_hashes)
    forbidden.update(c.signature for seq in oracle.roster().values() for c in seq)
    forbidden.update(c.signature for c in arg02.dev_cases())
    forbidden.update(c.signature for c in arg02.shot_cases())
    result={}
    for split_i,(name,progs) in enumerate(program_partition.items()):
        cases=[]
        for family_i,family in enumerate(families):
            for prog_i,prog in enumerate(progs):
                r=random.Random(seed+9973*family_i+1009*split_i+101*prog_i)
                n=0
                for trial in range(1,5000):
                    topo=r.choice(oracle.TOPOLOGIES)
                    c=oracle.make_case(r,split="arg03_"+name,family=family,
                                       topology=topo,graph=prog,
                                       identifier=f"arg03:{name}:{family}:{prog_i:02d}:{trial:04d}")
                    if c.signature in forbidden:
                        continue
                    forbidden.add(c.signature)
                    cases.append(c);n+=1
                    if n==per_program:break
                if n!=per_program:
                    raise RuntimeError(f"Fresh final-R generation failed {family}:{prog}")
        result[name]=tuple(cases)
    assert set(c.signature for c in result["development_finalR_fit"]).isdisjoint(
        c.signature for c in result["development_finalR_valid"])
    return result,program_partition

def export(args)->dict:
    if args.out_dir.exists():
        raise FileExistsError("Refuse to overwrite seeded development outputs")
    from ..arg03_roster import oracle_context  # Oracle-only command; needs NumPy and EvalMath
    oracle,arg02=oracle_context(args.evalmath_root,args.compactification_root)
    prior=json.loads(args.existing_dev_fixture.read_text(encoding="utf-8"))
    if prior.get("split")!="development" or prior.get("n_rows")!=96:
        raise ValueError("Expected only original development hash fixture")
    blocked={x["public_sha256"] for x in prior["rows"]}
    result,parts=make_fresh(oracle,arg02,blocked,seed=args.seed,per_program=args.per_program)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    host=[]
    for split,cases in result.items():
        output=args.out_dir/(split+"_public.jsonl")
        output.write_text("".join(json.dumps(public(c),sort_keys=True,separators=(",",":"))+"\n"
                                  for c in cases),encoding="utf-8")
        for c in cases:
            idx=oracle.PROGRAMS.index(c.source_graph)
            if idx not in range(64) or c.source_graph[2]!="R":
                raise AssertionError("Unexpected source graph")
            host.append({"case_id":c.identifier,"source_sha256":c.signature,
                         "split":split,"family":c.family,"gold_program_index":idx})
    (args.out_dir/"host_only_programs.jsonl").write_text(
        "".join(json.dumps(x,sort_keys=True)+"\n" for x in host),encoding="utf-8")
    assert all(c.source_graph[2]=="R" for pool in result.values() for c in pool)
    assert not {c.signature for pool in result.values() for c in pool} & blocked
    audit={
        "schema":"arg03.finalR.devonly.roster.v1",
        "seed":args.seed,"per_program":args.per_program,
        "counts":{split:len(arr) for split,arr in result.items()},
        "family_counts":dict(Counter(c.family for pool in result.values() for c in pool)),
        "programs_disjoint_across_fit_valid":not bool(set(parts["development_finalR_fit"]) & set(parts["development_finalR_valid"])),
        "source_signatures_disjoint_across_fit_valid":True,
        "prior_development_exclusion":len(blocked),
        "source_development_blob_sha":prior["source_blob_sha"][0],
        "sealed_test_loaded":False,
        "sealed_test_signature_overlap":"UNKNOWN requires independent blinded digest check",
        "model_input_files":[name+"_public.jsonl" for name in result],
        "host_only":"host_only_programs.jsonl",
        "readout_requirement":"model-only tape stores 64 candidate scores and 16x3x4 site logits for each case",
        "policy":"Fit only on development_finalR_fit; validate once on development_finalR_valid in a separate oracle grader."
    }
    (args.out_dir/"audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evalmath-root",type=Path,required=True)
    parser.add_argument("--compactification-root",type=Path,required=True)
    parser.add_argument("--existing-dev-fixture",type=Path,default=Path(__file__).resolve().parent/"data/dev96_compact.json")
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=FRESH_SEED)
    parser.add_argument("--per-program",type=int,default=PER_PROGRAM)
    args=parser.parse_args()
    for p in (args.existing_dev_fixture,args.out_dir):
        if any(part.lower() in {"test","test.jsonl","test_full168"} for part in p.parts):
            raise ValueError("Refuse held-out test paths")
    print(json.dumps(export(args),sort_keys=True))

if __name__=="__main__":
    main()
