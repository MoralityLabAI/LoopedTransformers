"""Oracle-isolated final-R DEVELOPMENT grading for new 48+48 Nanbeige cases.

Applies the 12-bias fit solely on new final-R development_fit, evaluates
development_valid on disjoint *program identities*, and independently checks
future-correct@12 via the original frozen EvalMath oracle. This is not the
sealed 168-case ARG-03 test and never reads that test file.

Model inference is in another OS process; this grader receives source-only
model tapes plus a host-only program label manifest.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ..arg03_roster import exact_mcnemar, oracle_context
from ..core import sha256_file, write_json
from .audit import DEV
from . import calibration as CAL
from . import grade_final_r as GR
from .prepare_final_r import make_fresh, public, FRESH_SEED, PER_PROGRAM

FIT = "development_finalR_fit"
VALID = "development_finalR_valid"
BASENAME = {FIT: FIT + "_public.jsonl", VALID: VALID + "_public.jsonl"}


def guard_dev(path: Path) -> None:
    if any(p.lower() in {"test", "test.jsonl", "heldout", "test_full168"}
           for p in path.parts):
        raise ValueError("sealed test path is forbidden for development grader")


def parse_jsonl(path: Path) -> list[dict]:
    guard_dev(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def program_scores_from_record(rec: dict, bias: list[list[float]]) -> list[float]:
    """Apply a training-only bias to the 16x3x4 model-only operator logits."""
    raw = rec.get("loop1_operator_site_logits")
    if raw is None:
        raise ValueError("model tape missing 16x3x4 operator-site logits")
    row = CAL.DevRankRow(
        rec["case_id"], rec["split"], tuple(tuple(tuple(v) for v in ctx) for ctx in raw),
        rec["gold_program_index"],
    )
    return CAL.program_scores(row, bias)


def score_one(oracle, case, scores: list[float]) -> tuple[bool, bool, bool]:
    if len(scores) != 64 or oracle.PROGRAMS != tuple((a,b,c) for a in "INSR"
                                                     for b in "INSR" for c in "INSR"):
        raise ValueError("oracle program enumeration drift")
    order = [oracle.PROGRAMS[i] for i in sorted(range(64), key=lambda i: (-scores[i],i))]
    found = oracle.search(case, order, budget=12)
    quality = oracle.grade(case, found)
    top = oracle.grade(case, oracle.search(case, order[:1], budget=1))
    return bool(quality["future_correct"]), bool(quality["accepted"]), bool(top["future_correct"])


def _is_model_tape_complete(tape: list[dict]) -> bool:
    return bool(tape) and all(all(len(rec.get("baseline", {}).get(str(t), [])) == 64
                                  for t in (1,2,4,8)) and
                              "loop1_operator_site_logits" in rec for rec in tape)


def evaluate(oracle, cohorts: dict, public_rows: dict, host_rows: list[dict],
             tapes: dict) -> dict:
    """Validate source-split identity, fit bias, and score held-out DEV-valid once."""
    if set(cohorts) != {FIT, VALID} or set(public_rows) != {FIT, VALID} or set(tapes) != {FIT, VALID}:
        raise ValueError("expected two fresh development-only partitions")
    for split in (FIT, VALID):
        if any(not c.identifier.startswith("arg03:" + split + ":") for c in cohorts[split]):
            raise ValueError("non-development case in cohort")
        case_index = {c.identifier: c for c in cohorts[split]}
        rows_by_id = {r["case_id"]: r for r in public_rows[split]}
        h_by_id = {h["case_id"]: h for h in host_rows if h["split"] == split}
        t_by_id = {r["case_id"]: r for r in tapes[split]}
        if (len(case_index) != len(cohorts[split]) or len(rows_by_id) != len(cohorts[split])
            or len(h_by_id) != len(cohorts[split]) or len(t_by_id) != len(cohorts[split])
            or set(case_index) != set(rows_by_id) or set(case_index) != set(h_by_id)
            or set(case_index) != set(t_by_id)):
            raise ValueError(f"missing, duplicate or unexpected case in {split}")
        if not _is_model_tape_complete(tapes[split]):
            raise ValueError("model answer tape is incomplete")
        for cid,case in case_index.items():
            if (rows_by_id[cid] != public(case)
                or h_by_id[cid]["source_sha256"] != case.signature
                or h_by_id[cid]["gold_program_index"] != oracle.PROGRAMS.index(case.source_graph)
                or t_by_id[cid]["public_sha256"] != case.signature):
                raise ValueError(f"source identity mismatch in {split}: {cid}")

    rank = GR.run(public_rows[FIT], public_rows[VALID], host_rows,
                  tapes[FIT], tapes[VALID])
    bias = rank["frozen_bias"]["parameters_3x4"]
    out = {
        "schema": "darkfibers.arg03.finalR_dev_future12.v1",
        "sealed_test_loaded": False,
        "source_program_rank": rank,
        "splits": {},
        "decision": "Exploratory dev-only results; original frozen confirmatory test remains sealed.",
    }
    for split in (FIT, VALID):
        rec_by_id = {r["case_id"]: r for r in tapes[split]}
        comparisons: dict[str,list[tuple[bool,bool,bool]]] = {}
        ids = sorted(rec_by_id)
        case_by_id = {c.identifier:c for c in cohorts[split]}
        for cid in ids:
            rec = dict(rec_by_id[cid], split=split,
                       gold_program_index=oracle.PROGRAMS.index(case_by_id[cid].source_graph))
            inputs = {"bias12_devfit": program_scores_from_record(rec, bias)}
            for arm in ("baseline", "open_loop", "equal_compute",
                        "context_swap", "random_context_norm_matched", "temperature_grid"):
                if arm in rec:
                    for depth,scores in rec[arm].items():
                        inputs[f"{arm}@{depth}"] = scores
            for name, scores in inputs.items():
                comparisons.setdefault(name, []).append(score_one(oracle,case_by_id[cid],scores))
        report = {}
        for name,stats in sorted(comparisons.items()):
            if len(stats) != len(ids):
                raise ValueError(f"incomplete arm {name}: {len(stats)} vs {len(ids)}")
            report[name] = {
                "n": len(stats),
                "future_correct_at12": sum(row[0] for row in stats),
                "accepted_at12": sum(row[1] for row in stats),
                "top1_future_correct": sum(row[2] for row in stats),
            }
        # Compare any arm against the model's one-loop baseline within same cohort.
        reference = comparisons["baseline@1"]
        for name,stats in comparisons.items():
            if name == "baseline@1":
                continue
            report[name]["paired_vs_baseline1"] = exact_mcnemar(
                [x[0] for x in stats], [x[0] for x in reference])
        out["splits"][split] = report
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evalmath-root", type=Path, required=True)
    p.add_argument("--compactification-root", type=Path, required=True)
    p.add_argument("--roster-dir", type=Path, required=True,
                   help="Output of prepare_final_r: fresh development only")
    p.add_argument("--fit-tape", type=Path, required=True)
    p.add_argument("--valid-tape", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    for path in (a.roster_dir, a.fit_tape, a.valid_tape, a.out):
        guard_dev(path)
    from ..arg03_roster import manifest_hash
    audit = json.loads((a.roster_dir / "audit.json").read_text(encoding="utf-8"))
    if (audit.get("schema") != "arg03.finalR.devonly.roster.v1"
        or audit.get("seed") != FRESH_SEED or audit.get("per_program") != PER_PROGRAM
        or audit.get("sealed_test_loaded") is not False):
        raise ValueError("fresh development source audit drift")
    oracle, arg02 = oracle_context(a.evalmath_root, a.compactification_root)
    prior = json.loads(DEV.read_text(encoding="utf-8"))
    banned = {r["public_sha256"] for r in prior["rows"]}
    cohorts, _ = make_fresh(oracle, arg02, banned, seed=FRESH_SEED, per_program=PER_PROGRAM)
    public_rows = {split:parse_jsonl(a.roster_dir / filename)
                   for split,filename in BASENAME.items()}
    host = parse_jsonl(a.roster_dir / "host_only_programs.jsonl")
    tapes = {FIT:parse_jsonl(a.fit_tape), VALID:parse_jsonl(a.valid_tape)}
    report = evaluate(oracle, cohorts, public_rows, host, tapes)
    report["provenance"] = {
        "roster_audit_sha256": sha256_file(a.roster_dir/"audit.json"),
        "fit_public_sha256": sha256_file(a.roster_dir/BASENAME[FIT]),
        "valid_public_sha256": sha256_file(a.roster_dir/BASENAME[VALID]),
        "fit_tape_sha256": sha256_file(a.fit_tape),
        "valid_tape_sha256": sha256_file(a.valid_tape),
        "oracle_source_sha256": sha256_file(Path(oracle.__file__)),
        "code_sha256": sha256_file(Path(__file__)),
        "source_dev_fixture_sha256": sha256_file(DEV),
    }
    write_json(a.out, report)
    print(json.dumps({"dev_fit":len(cohorts[FIT]), "dev_valid":len(cohorts[VALID]),
                      "sealed_test_loaded":False,
                      "bias12_future_correct_valid":report["splits"][VALID]["bias12_devfit"]["future_correct_at12"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
