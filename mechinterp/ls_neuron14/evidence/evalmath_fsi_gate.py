"""Bridge LoopedTransformers per-case JSONL receipts to EvalMath FSI-01.

Both registration and frozen per-case outcomes are hashed into the report.
Install EvalMath or add its src directory to PYTHONPATH; no vendored stats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def score(case_jsonl: Path, registration: Path) -> dict:
    try:
        from spiral_ln.evaluation_finite_sample import evaluate_manifest
    except ImportError as exc:
        raise RuntimeError("EvalMath FSI-01 unavailable; install EvalMath or set PYTHONPATH to EvalMath/src") from exc
    source=case_jsonl.read_bytes()
    registered=registration.read_bytes()
    spec=json.loads(registered)
    if "cases" in spec:
        raise ValueError("registration must not contain scored cases; keep outcomes separately")
    cases=[]
    for lineno,raw in enumerate(source.decode("utf-8").splitlines(),1):
        if not raw.strip():
            continue
        rec=json.loads(raw)
        try:
            cases.append({"case_id":rec["case_id"],"cluster_id":rec["root_id"],
                          "candidate":rec["candidate_correct"],"baseline":rec["control_correct"]})
        except KeyError as exc:
            raise ValueError(f"missing required receipt column at line {lineno}: {exc}") from exc
    spec["cases"]=cases
    report=evaluate_manifest(spec)
    report["registration_sha256"]=hashlib.sha256(registered).hexdigest()
    report["outcomes_sha256"]=hashlib.sha256(source).hexdigest()
    report["input_filenames"]=[case_jsonl.name,registration.name]
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True,help="immutable per-case JSONL outcomes")
    parser.add_argument("--registration",type=Path,required=True,help="pre-data frozen experimental manifest")
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--require-confirmed",action="store_true")
    args=parser.parse_args()
    r=score(args.cases,args.registration)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"{r['experiment_id']}: {r['verdict']} ({r['n_cases']} cases, {r['n_clusters']} roots); {args.report}")
    if args.require_confirmed and r["verdict"] not in ("confirmed","exact_finite_result"):
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
