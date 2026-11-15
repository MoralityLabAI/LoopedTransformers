#!/usr/bin/env python3
"""Fresh ALG/ARG-03 roster export and independent EvalMath grade; no GPU required.

The model runner sees only public *.jsonl. Regeneration of hidden cases
occurs here, in the oracle-only process. Test graphs use an operator family
held out from dev (final operator R). All prior EvalMath/ARG-02 signatures
are excluded. Unlike ARG-02, this initial ARG-03 roster is positive-only.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from math import comb

from .core import sha256_file, write_json

DEV_SEED = 2026111503
TEST_SEED = 2026111504
DEV_PER_FAMILY = 32
TEST_PER_FAMILY = 24
OPS = "INSR"


def oracle_context(evalmath_root: Path, compactification_root: Path):
    sys.path.insert(0, str(evalmath_root.resolve()))
    sys.path.insert(0, str(compactification_root.resolve()))
    from spiral_ln import algebraic_reasoning_oracle as oracle
    from research.algernon.algebraic_reasoning_gym.arg02 import splits as arg02_splits
    return oracle, arg02_splits


def manifest_hash(data) -> str:
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cases(oracle, arg02_splits):
    forbidden = {c.signature for seq in oracle.roster().values() for c in seq}
    forbidden |= {c.signature for c in arg02_splits.dev_cases()}
    forbidden |= {c.signature for c in arg02_splits.shot_cases()}
    out = {}
    plans = (
        ("development", DEV_SEED, oracle.CAL_FAMILIES, DEV_PER_FAMILY, False),
        ("test", TEST_SEED, oracle.ALL_FAMILIES, TEST_PER_FAMILY, True),
    )
    for name, seed, families, n_each, held_out in plans:
        group = []
        for family_i, family in enumerate(families):
            rng = random.Random(seed + 7919 * family_i)
            for trial in range(1, n_each * 1000 + 1):
                if len([c for c in group if c.family == family]) >= n_each:
                    break
                topology = rng.choice(oracle.TOPOLOGIES)
                graphs = [p for p in oracle.PROGRAMS if (p[2] == "R") == held_out]
                graph = rng.choice(graphs)
                c = oracle.make_case(rng, split=f"arg03_{name}", family=family, topology=topology,
                                     graph=graph, identifier=f"arg03:{name}:{family}:{trial:05d}")
                if c.signature in forbidden:
                    continue
                forbidden.add(c.signature)
                group.append(c)
            if sum(c.family == family for c in group) < n_each:
                raise RuntimeError(f"insufficient fresh cases: {name}/{family}")
        out[name] = tuple(group)
    if {tuple(c.source_graph) for c in out["development"]} & {tuple(c.source_graph) for c in out["test"]}:
        raise AssertionError("development/test program-graph overlap")
    return out


def public(c) -> dict:
    data = dict(c.public)
    data["public_sha256"] = c.signature
    return data


def prepare(args):
    oracle, split = oracle_context(args.evalmath_root, args.compactification_root)
    roster = cases(oracle, split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, arr in roster.items():
        path = args.out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(public(c), sort_keys=True, separators=(",", ":"))+"\n" for c in arr),
                        encoding="utf-8")
        digests[name] = sha256_file(path)
    audit = {
        "schema": "darkfibers.arg03.new_roster.v1",
        "seeds": {"development": DEV_SEED, "test": TEST_SEED},
        "counts": {key: len(value) for key, value in roster.items()},
        "public_sha256": digests, "model_input_fields": sorted(public(roster["test"][0])),
        "families": {"development": list(oracle.CAL_FAMILIES), "test": list(oracle.ALL_FAMILIES)},
        "held_out_program_rule": "test final operator R; development final operator not-R",
        "forbidden": "all EvalMath oracle.roster signatures and ARG-02 dev/shots",
        "oracle_code_sha256": sha256_file(Path(oracle.__file__)),
        "arg02_splits_sha256": sha256_file(Path(split.__file__)),
        "negative_controls": "not included; prepare a separate registered negative/collision suite",
    }
    write_json(args.out_dir/"audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    gain = sum(x and not y for x,y in zip(a,b))
    loss = sum(y and not x for x,y in zip(a,b))
    n = gain + loss
    p = min(1.0, 2.0 * sum(comb(n,k) for k in range(min(gain,loss)+1)) / (2**n)) if n else 1.0
    return {"gain": gain, "loss": loss, "paired_diff": gain-loss, "exact_two_sided_p": p}


def grade(args):
    oracle, split = oracle_context(args.evalmath_root, args.compactification_root)
    roster = cases(oracle, split)
    audit = json.loads((args.split_dir/"audit.json").read_text())
    expected = {"development": DEV_SEED, "test": TEST_SEED}
    if audit["seeds"] != expected:
        raise AssertionError("split seeds drift")
    if audit["oracle_code_sha256"] != sha256_file(Path(oracle.__file__)):
        raise AssertionError("oracle changed since export")
    if audit["arg02_splits_sha256"] != sha256_file(Path(split.__file__)):
        raise AssertionError("forbidden-signature generator changed since export")
    if audit["public_sha256"][args.split] != sha256_file(args.split_dir/f"{args.split}.jsonl"):
        raise AssertionError("public rows changed")
    ordered = list(oracle.PROGRAMS)
    canonical = [(a,b,c) for a in OPS for b in OPS for c in OPS]
    if ordered != canonical:
        raise AssertionError("program ordering differs from model score order")
    case_map = {c.identifier:c for c in roster[args.split]}
    tape = [json.loads(line) for line in args.tape.read_text().splitlines() if line.strip()]
    if len({r["case_id"] for r in tape}) != len(tape):
        raise ValueError("duplicate model results")
    if args.full and set(r["case_id"] for r in tape) != set(case_map):
        raise ValueError("full held-out evaluation needs every case")
    per_arm = {}
    for rec in tape:
        cid = rec["case_id"]
        c = case_map[cid]
        if rec["public_sha256"] != c.signature:
            raise AssertionError("case identity mismatch")
        for arm in ("baseline", "context_swap"):
            for depth,scores in rec[arm].items():
                if len(scores) != 64:
                    raise ValueError("missing program scores")
                order = [ordered[i] for i in sorted(range(64), key=lambda i:(-scores[i],i))]
                search = oracle.search(c, order, budget=12)
                verified = oracle.grade(c, search)
                per_arm.setdefault((arm,depth), {})[cid] = bool(verified["future_correct"])
    summary = {"schema": "darkfibers.arg03.graded.v1", "split": args.split,
               "test_exploratory": not args.full, "rows_scored": len(tape),
               "tape_sha256": sha256_file(args.tape), "results": {}}
    baseline = per_arm.get(("baseline","1"), {})
    for (arm,depth), row in sorted(per_arm.items()):
        ids = sorted(row)
        info = {"correct": sum(row.values()), "denominator": len(ids)}
        if (arm,depth) != ("baseline","1") and baseline and set(ids)==set(baseline):
            info["vs_baseline1"] = exact_mcnemar([row[c] for c in ids], [baseline[c] for c in ids])
        summary["results"][f"{arm}@{depth}"] = info
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare","grade"))
    p.add_argument("--evalmath-root", type=Path, required=True)
    p.add_argument("--compactification-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, help="Directory for new public split and audit")
    p.add_argument("--split-dir", type=Path, help="Existing audited public split")
    p.add_argument("--split", choices=("development","test"), default="development")
    p.add_argument("--tape", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--full", action="store_true", help="Require complete split; appropriate for final test")
    args = p.parse_args(argv)
    if args.mode == "prepare":
        if args.out_dir is None:
            p.error("prepare requires --out-dir")
        return prepare(args)
    if any(v is None for v in (args.split_dir,args.tape,args.out)):
        p.error("grade requires --split-dir --tape --out")
    return grade(args)


if __name__ == "__main__":
    raise SystemExit(main())
