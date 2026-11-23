"""LSA-02F: independent root-seeded random-menu Lean proof baseline.

Each tactic is drawn uniformly from the *current public candidate menu*, after
removing candidates already rejected at the exact Lean state. The reference
oracle never proposes actions. Real Lean supplies every verdict and kernel
certificate; independently compiled standalone scripts and replay attest the
closures. A seeded draw is deterministic per root, visible packet and menu.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.lean_repl import LeanRepl
from .data import canonical, sha256_file
from .live_lean import fingerprint, replay_episode, run_episode
from .live_suite import load_suite
from .native_train import _load_teacher


SCHEMA = "lsa02f.uniform_menu_native.v1"
RANDOM_SEEDS = (17, 31, 47)


def _json(path: Path, value: dict):
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _records(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _root_seed(root: str, fixed_seed: int) -> int:
    if not isinstance(root, str) or not root or type(fixed_seed) is not int:
        raise ValueError("uniform random stream requires a root and integer seed")
    return int.from_bytes(
        hashlib.sha256(canonical(["lsa02f", root, fixed_seed]).encode()).digest()[:4],
        "big",
    )


def run(
    repl, suite: Path, teacher: Path, out: Path, *,
    roots: int = 32, checks: int = 8, seeds: tuple[int, ...] = RANDOM_SEEDS,
    max_seconds: float = 720, cert_timeout: int = 360,
    certified_backend: bool = False, certifier=certify,
) -> dict:
    if out.exists() or roots not in range(1, 1025) or checks not in range(1, 33):
        raise ValueError("fresh output, bounded roots and tactic checks required")
    if (
        not seeds or len(set(seeds)) != len(seeds)
        or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
    ):
        raise ValueError("unique unsigned root-seeded uniform roster required")
    if max_seconds <= 0 or not 1 <= cert_timeout <= 900:
        raise ValueError("invalid source-only compute budget")
    if certified_backend:
        if not isinstance(repl, LeanRepl) or not str(repl.version).startswith("4.19.0"):
            raise ValueError("random-menu evidence requires real Lean 4.19.0")
    source_rows, source = _load_teacher(teacher)
    if (
        source["backend"]["actual_lean_process"] is not True
        or source["source_compiler_passed"] is not True
        or source["query_roots_exposed_to_fit"] != 0
    ):
        raise ValueError("native teacher source is ineligible")
    cases, suite_source = load_suite(suite, "calibration", roots)
    if source["source"]["suite_manifest_sha256"] != suite_source["suite_manifest_sha256"]:
        raise ValueError("teacher and calibration root suites differ")
    source_lineages = {row["lineage"] for row in source_rows}
    if source_lineages & {case.lineage for case in cases}:
        raise ValueError("uniform baseline crosses train/calibration theorem lineages")
    out.mkdir(parents=True)
    started = time.perf_counter()
    episodes = []
    for case in cases:
        for seed in seeds:
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError("random-menu native proof pilot exceeded registered wall cap")
            sampled = _root_seed(case.root, seed)
            episode = run_episode(
                repl, case, None, policy="uniform", checks=checks,
                depth=2, uniform_seed=sampled,
            )
            episode["uniform_seed"] = seed
            episode["root_random_seed"] = sampled
            episode["max_checks"] = checks
            if episode["model_calls"] != 0:
                raise AssertionError("random-menu baseline consulted a neural model")
            episodes.append(episode)

    pending = []
    witness_ids = {}
    by_case = {case.root: case for case in cases}
    for ep in episodes:
        if ep["live_kernel"]:
            case = by_case[ep["root"]]
            name = f"{case.tid}_random_{ep['uniform_seed']}"
            key = decl_name(name)
            if key in witness_ids:
                raise ValueError("duplicate independent uniform proof declaration")
            witness_ids[key] = ep
            pending.append((name, case.theorem(), ep["accepted_script"]))
    receipt = {
        "schema": "lsa02f.random_standalone.v1",
        "proposals": len(episodes),
        "native_kernel_closures": sum(ep["live_kernel"] for ep in episodes),
        "reference_proofs_submitted": len(pending),
        "compiler_audit": None,
    }
    if pending:
        audit = certifier(pending, timeout=cert_timeout, keep=out / "standalone.lean")
        statuses = audit.get("status") if isinstance(audit, dict) else None
        if not isinstance(statuses, dict):
            raise ValueError("native random-menu proof certifier omitted declaration statuses")
        receipt["compiler_audit"] = audit
        for key, ep in witness_ids.items():
            if statuses.get(key) != "passed":
                raise ValueError("random-menu native kernel closure failed standalone compilation")
            ep["standalone_certificate"] = "passed"

    for ep in episodes:
        ep.setdefault("standalone_certificate", None)
        ep["standalone_verified"] = bool(
            certified_backend and ep["live_kernel"] and ep["standalone_certificate"] == "passed"
        )
        if not replay_episode(repl, by_case[ep["root"]], ep):
            raise ValueError("native random-menu action replay failed")
        ep["model_free_replay_passed"] = True

    with (out / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for ep in episodes:
            stream.write(canonical(ep) + "\n")
    observed: dict[str, dict[int, dict]] = defaultdict(dict)
    for ep in episodes:
        if ep["uniform_seed"] in observed[ep["root"]]:
            raise ValueError("random-menu root and seed duplicated")
        observed[ep["root"]][ep["uniform_seed"]] = ep
    if len(observed) != roots or any(set(values) != set(seeds) for values in observed.values()):
        raise ValueError("random-menu evaluation lost independent root-seed pairing")
    root_scores = []
    for root in sorted(observed):
        batch = observed[root]
        root_scores.append({
            "root": root, "lineage": batch[seeds[0]]["lineage"],
            "seed_closed": {
                str(seed): bool(batch[seed]["standalone_verified"]) for seed in seeds
            },
            "mean_seed_closure": sum(batch[seed]["standalone_verified"] for seed in seeds) / len(seeds),
        })
    with (out / "root_scores.jsonl").open("x", encoding="utf-8") as stream:
        for row in root_scores:
            stream.write(canonical(row) + "\n")
    _json(out / "standalone_receipt.json", receipt)
    result = {
        "schema": SCHEMA,
        "source": suite_source,
        "source_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "backend": (
            fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
            if certified_backend else {"actual_lean_process": False, "source": "synthetic_protocol"}
        ),
        "seed_roster": list(seeds),
        "independent_calibration_roots": len(root_scores),
        "native_model_calls": 0,
        "check_budget": checks,
        "complete_episodes": len(episodes),
        "standalone_verified_closures": sum(ep["standalone_verified"] for ep in episodes),
        "native_kernel_closures": sum(ep["live_kernel"] for ep in episodes),
        "uniform_mean_closure_rate": sum(row["mean_seed_closure"] for row in root_scores) / len(root_scores),
        "per_seed_closures": {
            str(seed): sum(observed[root][seed]["standalone_verified"] for root in observed)
            for seed in seeds
        },
        "replayed_native_actions": sum(len(ep["events"]) for ep in episodes),
        "all_replays_passed": True,
        "random_seed_blind_to_verdict_and_reference": True,
        "query_updates": 0,
        "heldout_test_opened": False,
        "scientific_eligible": False,
        "artifacts": {
            name: sha256_file(out / name) for name in (
                "episodes.jsonl", "root_scores.jsonl", "standalone_receipt.json",
            )
        },
    }
    if pending:
        result["artifacts"]["standalone.lean"] = sha256_file(out / "standalone.lean")
    _json(out / "manifest.json", result)
    return result


def replay(repl, suite: Path, session: Path, out: Path, *, certified_backend=False) -> dict:
    if out.exists():
        raise ValueError("audit destination must be fresh")
    report = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA:
        raise ValueError("unknown random-menu session manifest")
    actual_backend = (
        fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        if certified_backend else {"actual_lean_process": False, "source": "synthetic_protocol"}
    )
    if actual_backend != report["backend"]:
        raise ValueError("random-menu Lean binary provenance changed")
    for name, expected in report["artifacts"].items():
        if name != Path(name).name or sha256_file(session / name) != expected:
            raise ValueError("random-menu session checksum differs")
    cases, source = load_suite(suite, "calibration", 1024)
    if source["suite_manifest_sha256"] != report["source"]["suite_manifest_sha256"]:
        raise ValueError("random-menu theorem suite changed after scoring")
    by_root = {case.root: case for case in cases}
    episodes = _records(session / "episodes.jsonl")
    if len(episodes) != report["complete_episodes"]:
        raise ValueError("random-menu episode roster is incomplete")
    for ep in episodes:
        case = by_root.get(ep["root"])
        if not case or ep["lineage"] != case.lineage or ep["split"] != "calibration":
            raise ValueError("random-menu evaluated an unknown theorem lineage")
        if ep["root_random_seed"] != _root_seed(ep["root"], ep["uniform_seed"]):
            raise ValueError("random-menu root seed provenance changed")
        replay_episode(repl, case, ep)
    out.mkdir(parents=True)
    result = {
        "schema": "lsa02f.random_replay.v1",
        "all_replays_passed": True,
        "verified_episodes": len(episodes),
        "native_backend": actual_backend,
        "session_manifest_sha256": sha256_file(session / "manifest.json"),
        "neural_model_used": False,
        "scientific_eligible": False,
    }
    _json(out / "replay.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    runp = cmd.add_parser("run")
    runp.add_argument("--suite", type=Path, required=True)
    runp.add_argument("--teacher", type=Path, required=True)
    runp.add_argument("--out", type=Path, required=True)
    runp.add_argument("--roots", type=int, default=32)
    runp.add_argument("--checks", type=int, default=8)
    runp.add_argument("--seeds", type=int, nargs="+", default=list(RANDOM_SEEDS))
    runp.add_argument("--max-seconds", type=float, default=720)
    runp.add_argument("--cert-timeout", type=int, default=360)
    rep = cmd.add_parser("replay")
    rep.add_argument("--suite", type=Path, required=True)
    rep.add_argument("--session", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if not config.LEAN_EXE.is_file() or not config.REPL_EXE.is_file():
        raise RuntimeError("pinned Lean 4.19 native checker absent")
    with LeanRepl(
        timeout=args.request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        if args.command == "run":
            result = run(
                repl, args.suite, args.teacher, args.out,
                roots=args.roots, checks=args.checks,
                seeds=tuple(args.seeds), max_seconds=args.max_seconds,
                cert_timeout=args.cert_timeout, certified_backend=True,
            )
        else:
            result = replay(repl, args.suite, args.session, args.out,
                            certified_backend=True)
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()
