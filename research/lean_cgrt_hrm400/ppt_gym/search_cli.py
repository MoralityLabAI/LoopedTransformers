"""Run and replay PPT-G03B's bounded proof-tree backtracking on fresh Lean goals.

This is a *search-controller* experiment. All proof outcomes are native Lean
outcomes plus standalone compilation; the untrained model smoke is not an
architecture result. It does not export test labels to training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.lean_repl import LeanRepl

from .cli import checkpoint_load
from .data import canonical, sha256_file
from .live_lean import fingerprint
from .live_search import replay_search, search_episode
from .live_suite import load_suite

SCHEMA = "ppt_gym.backtracking_campaign.v1"
ARMS = ("auto8", "frozen")


def write_json(path: Path, item):
    path.write_text(canonical(item) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def live_repl(timeout):
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise RuntimeError("build the pinned Lean 4.19.0 lgym server first")
    return LeanRepl(timeout=timeout, exe=config.REPL_EXE, env=config.lean_env(),
                    cwd=config.LEAN_DIR)


def run(args):
    if args.split == "train":
        raise ValueError("backtracking pilot is evaluation-only")
    model, checkpoint = checkpoint_load(args.checkpoint)
    cases, source = load_suite(args.suite, args.split, max_cases=args.cases)
    if args.out.exists():
        raise ValueError("output directory already exists")
    started = time.perf_counter()
    with live_repl(args.request_timeout) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("pinned Lean 4.19.0 is required")
        source["lean"] = fingerprint(config.REPL_EXE, config.LEAN_EXE, lean.version)
        records = []
        for case in cases:
            if time.perf_counter() - started >= args.max_seconds:
                break
            for arm in ARMS:
                if time.perf_counter() - started >= args.max_seconds:
                    break
                ep = search_episode(lean, case, model, policy=arm, depth=args.depth,
                                    check_budget=args.checks, rollback_budget=args.rollbacks)
                records.append(ep)

    args.out.mkdir(parents=True)
    by_root = {c.root: c for c in cases}
    proof_items, names = [], {}
    for ep in records:
        if ep["live_kernel"]:
            case = by_root[ep["root"]]
            ident = f"{case.tid}_{ep['policy']}_backtracking"
            name = decl_name(ident)
            if name in names:
                raise ValueError("duplicate standalone proof declaration")
            names[name] = ep
            proof_items.append((ident, case.theorem(), ep["accepted_script"]))
    standalone = None
    if proof_items:
        standalone = certify(proof_items, timeout=args.cert_timeout,
                             keep=args.out / "standalone.lean")
        for name, ep in names.items():
            ep["standalone_certificate"] = standalone["status"].get(name, "missing")
            ep["verified_closed"] = ep["standalone_certificate"] == "passed"

    with (args.out / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for ep in records:
            stream.write(canonical(ep) + "\n")
    write_json(args.out / "certificate.json", {
        "source": standalone, "proofs_submitted": len(proof_items),
        "native_kernel_closures": sum(ep["live_kernel"] for ep in records),
        "standalone_certified": sum(ep["verified_closed"] for ep in records),
    })
    artifacts = {n: sha256_file(args.out / n) for n in ("episodes.jsonl", "certificate.json")}
    if proof_items:
        artifacts["standalone.lean"] = sha256_file(args.out / "standalone.lean")
    score = {}
    for arm in ARMS:
        selected = [x for x in records if x["policy"] == arm]
        score[arm] = {
            "attempts": len(selected), "verified_closures": sum(x["verified_closed"] for x in selected),
            "kernel_closures": sum(x["live_kernel"] for x in selected),
            "tactic_checks": sum(x["tactic_checks"] for x in selected),
            "rollbacks": sum(x["rollbacks"] for x in selected),
            "model_calls": sum(x["model_calls"] for x in selected),
        }
    result = {
        "schema": SCHEMA, "source": source, "split": args.split,
        "model_checkpoint_sha256": sha256_file(args.checkpoint),
        "model_checkpoint_evidence": checkpoint.get("source_evidence_class"),
        "policy_arms": ARMS, "scorecard": score, "check_budget": args.checks,
        "rollback_budget": args.rollbacks, "depth": args.depth,
        "fully_paired_roots": min((x["attempts"] for x in score.values()), default=0),
        "requested_roots": len(cases), "artifacts": artifacts,
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "architecture_promotion_eligible": False,
        "new_proof_environment": "fresh_propositional_horn_not_mathlib",
        "learning": "no DRAM updates or query-label training in this evaluation-only branch",
        "cost_caveat": "backtracking costs host decisions; model FLOPs are not yet matched",
    }
    write_json(args.out / "manifest.json", result)
    return result


def replay(args):
    manifest_path = args.session / "manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("split") == "train":
        raise ValueError("expected evaluation-only backtracking campaign")
    source, info = load_suite(args.suite, m["split"], max_cases=1024)
    if info["suite_manifest_sha256"] != m["source"]["suite_manifest_sha256"]:
        raise ValueError("frozen theorem source changed")
    if (m["source"]["lean"]["repl_bin_sha256"] != sha256_file(config.REPL_EXE)
            or m["source"]["lean"]["lean_bin_sha256"] != sha256_file(config.LEAN_EXE)):
        raise ValueError("Lean binary differs from original run")
    for name, sha in m["artifacts"].items():
        if sha256_file(args.session / name) != sha:
            raise ValueError("backtracking session artifact changed")
    cases = {case.root: case for case in source}
    rows = read_jsonl(args.session / "episodes.jsonl")
    with live_repl(args.request_timeout) as lean:
        if lean.version != m["source"]["lean"]["lean_version"]:
            raise ValueError("replay Lean version changed")
        for ep in rows:
            case = cases.get(ep["root"])
            if case is None or case.lineage != ep["lineage"]:
                raise ValueError("episode belongs to another theorem lineage")
            replay_search(lean, case, ep)
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"replay_passed": True, "episodes": len(rows),
               "tactic_checks_replayed": sum(ep["tactic_checks"] for ep in rows),
               "rollbacks_replayed": sum(ep["rollbacks"] for ep in rows),
               "suite_manifest_sha256": info["suite_manifest_sha256"]}
    write_json(args.out / "replay.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)
    run_p = sp.add_parser("run")
    run_p.add_argument("--suite", type=Path, required=True)
    run_p.add_argument("--checkpoint", type=Path, required=True)
    run_p.add_argument("--split", choices=("calibration", "ordinary_test", "ood_family"),
                       default="calibration")
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--cases", type=int, default=12)
    run_p.add_argument("--checks", type=int, default=8)
    run_p.add_argument("--rollbacks", type=int, default=8)
    run_p.add_argument("--depth", type=int, default=2)
    run_p.add_argument("--request-timeout", type=float, default=30)
    run_p.add_argument("--cert-timeout", type=int, default=120)
    run_p.add_argument("--max-seconds", type=float, default=360)

    audit = sp.add_parser("replay")
    audit.add_argument("--suite", type=Path, required=True)
    audit.add_argument("--session", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--request-timeout", type=float, default=30)
    args = p.parse_args(argv)
    if args.command == "run":
        if not (1 <= args.cases <= 1024 and 1 <= args.checks <= 32
                and 1 <= args.rollbacks <= args.checks and 1 <= args.depth <= 16):
            raise ValueError("bounded cases, tactic checks and rollbacks required")
        if args.max_seconds <= 0:
            raise ValueError("maximum wall time must be positive")
        result = run(args)
    elif args.command == "replay":
        result = replay(args)
    else:
        raise ValueError("unknown command")
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()
