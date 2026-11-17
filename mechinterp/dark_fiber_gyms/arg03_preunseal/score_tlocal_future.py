"""Score model-only t-local ARG-03 development continuations via isolated EvalMath.

Only fresh final-R VALID cases. The original 168-case test is sealed. This
assesses the downstream 64-program ranking at t+1 after a matched time-local
intervention, with exact per-case dose and full-logit pre-readout checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ..arg03_roster import exact_mcnemar, oracle_context
from ..core import sha256_file, write_json
from .audit import DEV
from .prepare_final_r import FRESH_SEED, PER_PROGRAM, make_fresh, public
from .score_final_r_future import guard_dev, parse_jsonl, score_one

VALID="development_finalR_valid"


def evaluate(oracle, cases, public_rows:list[dict], tape:list[dict]) -> dict:
    lookup = {case.identifier:case for case in cases}
    pub = {r["case_id"]:r for r in public_rows}
    entries = {r["case_id"]:r for r in tape}
    if (not entries or len(entries)!=len(tape) or len(lookup)!=len(pub)
        or set(pub)!=set(lookup) or set(entries)!=set(lookup)):
        raise ValueError("development validation tape incomplete or duplicate")
    results={}
    for cid in sorted(entries):
        case=lookup[cid]
        rec=entries[cid]
        if (not cid.startswith("arg03:"+VALID+":") or rec.get("source_split")!="development_only"
            or rec["public_sha256"]!=case.signature or pub[cid]!=public(case)
            or rec.get("historical_h2a_unchanged") is not True):
            raise ValueError("source or split mismatch")
        for depth in ("2","4","8"):
            item=rec["depths"][depth]
            if item.get("skipped_reason"):
                results.setdefault(depth,[]).append(None)
                continue
            errors=item["pre_head_logprob_error"]
            dose=item["matched_l2_dose"]
            if (max(errors.values())>1e-6 or dose["relative_error"]>0.005 or
                dose["structured"]<=0 or dose["random_matched"]<=0):
                raise AssertionError("matched-fiber parity/dose failed")
            q=item["next_64_program_scores"]
            data={name:score_one(oracle,case,scores)
                  for name,scores in q.items()}
            data["structured_full_vocab_l2"]=float(item["next_full_vocab_l2"]["structured"])
            data["random_full_vocab_l2"]=float(item["next_full_vocab_l2"]["random_matched"])
            results.setdefault(depth,[]).append(data)
    summary={"schema":"darkfibers.arg03.tlocal_dev_future12.v1",
             "sealed_test_loaded":False,"split":VALID,"n":len(cases),
             "interpretation":"Exploratory development-only, fixed t=2/4/8 H2a; no sealed-test claim",
             "depths":{}}
    for depth,allrows in results.items():
        eligible=[r for r in allrows if r is not None]
        if not eligible:
            summary["depths"][depth]={"eligible":0,"skipped":len(allrows)}
            continue
        out={"eligible":len(eligible),"skipped":len(allrows)-len(eligible)}
        for name in ("clean","structured","random_matched"):
            out[name]={"future_correct_at12":sum(x[name][0] for x in eligible),
                       "accepted_at12":sum(x[name][1] for x in eligible),
                       "top1_future_correct":sum(x[name][2] for x in eligible)}
        out["structured_vs_random_mcnemar"]=exact_mcnemar(
            [x["structured"][0] for x in eligible],
            [x["random_matched"][0] for x in eligible])
        out["structured_vs_clean_mcnemar"]=exact_mcnemar(
            [x["structured"][0] for x in eligible],
            [x["clean"][0] for x in eligible])
        signs=[(r["structured_full_vocab_l2"]>r["random_full_vocab_l2"])
               -(r["structured_full_vocab_l2"]<r["random_full_vocab_l2"])
               for r in eligible]
        out["continuous_structured_wins"]=sum(s>0 for s in signs)
        out["continuous_random_wins"]=sum(s<0 for s in signs)
        out["continuous_ties"]=sum(s==0 for s in signs)
        summary["depths"][depth]=out
    return summary


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evalmath-root",type=Path,required=True)
    parser.add_argument("--compactification-root",type=Path,required=True)
    parser.add_argument("--roster-dir",type=Path,required=True)
    parser.add_argument("--tlocal-tape",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args(argv)
    for path in (a.roster_dir,a.tlocal_tape,a.output):
        guard_dev(path)
    audit=json.loads((a.roster_dir/"audit.json").read_text(encoding="utf-8"))
    if (audit.get("schema")!="arg03.finalR.devonly.roster.v1"
        or audit.get("seed")!=FRESH_SEED or audit.get("per_program")!=PER_PROGRAM
        or audit.get("sealed_test_loaded") is not False):
        raise ValueError("roster not pinned final-R development")
    oracle,arg02=oracle_context(a.evalmath_root,a.compactification_root)
    prior=json.loads(DEV.read_text(encoding="utf-8"))
    banned={r["public_sha256"] for r in prior["rows"]}
    cohorts,_=make_fresh(oracle,arg02,banned,seed=FRESH_SEED,per_program=PER_PROGRAM)
    public_rows=parse_jsonl(a.roster_dir/(VALID+"_public.jsonl"))
    tape=parse_jsonl(a.tlocal_tape)
    report=evaluate(oracle,cohorts[VALID],public_rows,tape)
    report["provenance"]={
        "roster_audit_sha256":sha256_file(a.roster_dir/"audit.json"),
        "public_sha256":sha256_file(a.roster_dir/(VALID+"_public.jsonl")),
        "tlocal_tape_sha256":sha256_file(a.tlocal_tape),
        "oracle_source_sha256":sha256_file(Path(oracle.__file__)),
        "code_sha256":sha256_file(Path(__file__)) }
    write_json(a.output,report)
    print(json.dumps({"split":VALID,"n":len(tape),
                      "eligible":{d:o["eligible"] for d,o in report["depths"].items()},
                      "sealed_test_loaded":False}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
