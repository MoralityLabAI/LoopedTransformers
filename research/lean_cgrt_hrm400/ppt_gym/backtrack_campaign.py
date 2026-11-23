"""LSA-02I native Lean bounded-backtrack calibration and independent controls.

Evidence authority: Lean 4.19 live tactic verdict + terminal kernel; separate
Lean standalone compilation for every closed model/control proof and every
native shortest reference. Independent event-by-event replay uses no model.

Three frozen-policy arms run on identical 32 new calibration roots:
frozen_greedy, frozen_dfs and public auto_dfs. A separate invocation runs
five-seed uniform_dfs without loading model weights. All pay one effort unit
per attempted tactic and one per RECONSIDER, with fixed budget 16.

The reference proof graph is explored only after all frozen/model-free arms
have finished. Training/example source ancestry is authenticated first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.lean_repl import LeanRepl
from .backtrack_search import BUDGET, SCHEMA as EPISODE_SCHEMA, run_episode, replay_episode
from .cli import checkpoint_load
from .data import canonical, sha256_file
from .live_suite import load_suite
from .native_order import (
    FrozenOrder, _checkpoint_provenance, _native_fingerprint,
    _oracle_witness, _weight_fingerprint,
)
from .native_train import _load_teacher


SCHEMA = "lsa02i.native_bounded_backtrack.v1"
REPLAY_SCHEMA = "lsa02i.independent_backtrack_replay.v1"
MODEL_POLICIES = ("frozen_greedy", "frozen_dfs", "auto_dfs")
UNIFORM_SEEDS = (17, 31, 47, 61, 83)
BOOTSTRAPS = 2000


def _write(path: Path, payload: dict) -> None:
    path.write_text(canonical(payload) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bootstrap(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(20261202)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAPS)
    )
    return [means[int(BOOTSTRAPS * .025)], means[int(BOOTSTRAPS * .975)-1]]


def _source(suite: Path, teacher: Path, roots: int):
    frames, meta = _load_teacher(teacher)
    cases, suite_meta = load_suite(suite, "calibration", roots)
    if (
        meta["source"]["suite_manifest_sha256"] != suite_meta["suite_manifest_sha256"]
        or meta.get("source_compiler_passed") is not True
        or meta.get("backend", {}).get("actual_lean_process") is not True
        or meta.get("independent_lineages", 0) < 64
        or {f["lineage"] for f in frames} & {case.lineage for case in cases}
    ):
        raise ValueError("checkpoint teacher and native calibration source have crossed lineages")
    return cases, suite_meta, meta


def _certify_and_replay(
    repl, cases: list, rows: list[dict], out: Path,
    *, oracle: bool, certified_backend: bool, cert_timeout: int,
) -> tuple[list[dict], list[dict], dict]:
    cases_by_root = {case.root: case for case in cases}
    proofs = []
    assigned = {}
    for ep in rows:
        if ep["live_kernel"]:
            case = cases_by_root[ep["root"]]
            suffix = ep["policy"]
            if suffix == "uniform_dfs":
                suffix += f"_{ep['uniform_seed']}"
            name = f"{case.tid}_{suffix}"
            qualified = decl_name(name)
            if qualified in assigned:
                raise ValueError("duplicate accepted proof declaration")
            assigned[qualified] = ep
            proofs.append((name, case.theorem(), ep["accepted_script"]))
    witnesses = []
    if oracle:
        for case in cases:
            # The model-free shortest reference is not available during
            # any live neural or uniform proof choice above.
            reference = _oracle_witness(repl, case, BUDGET)
            qualified = decl_name(f"{case.tid}_reference_oracle")
            if qualified in assigned:
                raise ValueError("oracle proof namespace collides with neural model")
            assigned[qualified] = reference
            proofs.append((f"{case.tid}_reference_oracle", case.theorem(),
                           reference["reference_script"]))
            witnesses.append(reference)
    if not proofs:
        raise ValueError("native calibration produced neither certified nor oracle proof")
    receipt = certify(proofs, timeout=cert_timeout, keep=out / "standalone.lean")
    statuses = receipt.get("status") if isinstance(receipt, dict) else None
    if receipt.get("timed_out") or not isinstance(statuses, dict):
        raise ValueError("native proof compiler did not return declaration verdicts")
    for name, entry in assigned.items():
        if statuses.get(name) != "passed":
            raise ValueError("live Lean closure failed independent standalone compilation")
        entry["standalone_verified"] = bool(certified_backend and
                                          entry.get("live_kernel",True))
    for ep in rows:
        if not ep["live_kernel"]:
            ep["standalone_verified"] = False
        ep["independent_replay_passed"] = bool(
            replay_episode(repl, cases_by_root[ep["root"]], ep)
        )
    if any(not ep["independent_replay_passed"] for ep in rows):
        raise ValueError("native action and reconsideration replay failed")
    return rows, witnesses, receipt


def run(
    repl, suite: Path, teacher: Path, out: Path,
    *, checkpoint_path: Path | None = None,
    control: str = "model",
    roots: int = 32,
    effort: int = BUDGET,
    max_seconds: float = 900,
    cert_timeout: int = 420,
    certified_backend: bool = False,
) -> dict:
    if out.exists() or roots != 32 or effort != BUDGET:
        raise ValueError("registered 32-root and 16-work-unit native calibration required")
    if control not in ("model", "uniform") or (
        (control == "model") != (checkpoint_path is not None)
    ):
        raise ValueError("model and uniform proof-search runners require separate source modes")
    if max_seconds <= 0 or not 1 <= cert_timeout <= 900:
        raise ValueError("bounded compute and certifier budgets required")
    backend = _native_fingerprint(repl, certified_backend)
    cases, source, train_meta = _source(suite, teacher, roots)
    model = None
    checkpoint_sha = None
    provenance = None
    initial_weight_sha = None
    if checkpoint_path is not None:
        model, checkpoint = checkpoint_load(checkpoint_path)
        if (
            model.cfg.plan != "b" or model.cfg.max_bytes != 384
            or checkpoint.get("train_order") != "ab"
            or checkpoint.get("input_view") != "pp96"
            or checkpoint.get("train_steps") != 720
            or checkpoint.get("source_manifest_sha256") != source["suite_manifest_sha256"]
        ):
            raise ValueError("frozen model checkpoint violated registered Plan B source or public input")
        provenance = _checkpoint_provenance(
            checkpoint, None, cases, native_teacher=teacher / "manifest.json",
        )
        if not provenance.get("training_source_verified") or not provenance["source_lineages_disjoint"]:
            raise ValueError("trained Plan B source is unverified or crosses query roots")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        torch.set_num_threads(1)
        checkpoint_sha = sha256_file(checkpoint_path)
        initial_weight_sha = _weight_fingerprint(model)
        adapter = FrozenOrder(model, "ab", view="pp96")
        policies = MODEL_POLICIES
    else:
        adapter = None
        policies = ("uniform_dfs",)
    out.mkdir(parents=True)
    episodes = []
    start = time.perf_counter()
    for case in cases:
        if time.perf_counter() - start > max_seconds:
            raise TimeoutError("native proof-search comparison exceeded registered wall budget")
        for policy in policies:
            repeat = UNIFORM_SEEDS if policy == "uniform_dfs" else (0,)
            for random_seed in repeat:
                ep = run_episode(
                    repl, case, adapter, policy=policy,
                    effort_budget=effort, depth=2, random_seed=random_seed,
                )
                ep["checkpoint_sha256"] = checkpoint_sha
                episodes.append(ep)
    episodes, witnesses, receipt = _certify_and_replay(
        repl, cases, episodes, out, oracle=control == "model",
        certified_backend=certified_backend, cert_timeout=cert_timeout,
    )
    if checkpoint_path is not None and _weight_fingerprint(model) != initial_weight_sha:
        raise ValueError("frozen Plan B weights changed during proof search")
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as f:
        for row in episodes:
            f.write(canonical(row) + "\n")
    by_root: dict[str, dict[str, dict]] = defaultdict(dict)
    for ep in episodes:
        key = ep["policy"] if ep["policy"] != "uniform_dfs" else f"uniform_{ep['uniform_seed']}"
        if key in by_root[ep["root"]]:
            raise ValueError("a native proof evaluation root/policy was repeated")
        by_root[ep["root"]][key] = ep
    if len(by_root) != roots:
        raise ValueError("native proof evaluation lost independent theorem roots")
    witnesses_by_root = {w["root"]: w for w in witnesses}
    root_rows = []
    for case in cases:
        arms = by_root[case.root]
        if set(arms) != (
            set(MODEL_POLICIES) if control == "model"
            else {f"uniform_{s}" for s in UNIFORM_SEEDS}
        ):
            raise ValueError("native root has incomplete proof policy arms")
        row = {
            "root": case.root, "lineage": case.lineage, "family": case.family,
            "policies": {
                name: {
                    "closed": ep["standalone_verified"],
                    "spent": ep["effort_spent"],
                    "tactics": ep["tactic_checks"],
                    "backtracks": ep["backtracks"],
                    "model_calls": ep["model_calls"],
                    "terminal": ep["terminal_status"],
                    "script": ep["accepted_script"] if ep["standalone_verified"] else [],
                }
                for name, ep in arms.items()
            },
        }
        if control == "model":
            w = witnesses_by_root[case.root]
            if not w["standalone_verified"]:
                raise ValueError("native shortest reference did not independently compile")
            row["reference_proof_size"] = w["proof_size"]
            row["attainable_within_effort"] = bool(
                w["fits_model_tactic_budget"] and w["menu_compatible"]
                and w["standalone_verified"]
            )
        root_rows.append(row)
    with (out / "roots.jsonl").open("x", encoding="utf-8") as f:
        for row in root_rows:
            f.write(canonical(row) + "\n")
    counts = {}
    for policy in policies:
        matched = [ep for ep in episodes if ep["policy"] == policy]
        counts[policy] = {
            "episodes": len(matched),
            "native_kernel_closures": sum(ep["live_kernel"] for ep in matched),
            "standalone_certified_closures": sum(ep["standalone_verified"] for ep in matched),
            "sum_tactic_checks": sum(ep["tactic_checks"] for ep in matched),
            "sum_backtracks": sum(ep["backtracks"] for ep in matched),
            "sum_effort_spent": sum(ep["effort_spent"] for ep in matched),
            "sum_model_calls": sum(ep["model_calls"] for ep in matched),
        }
    if control == "model":
        delta = [
            int(row["policies"]["frozen_dfs"]["closed"]) -
            int(row["policies"]["frozen_greedy"]["closed"]) for row in root_rows
        ]
        paired = {
            "dfs_minus_greedy_mean_closure": sum(delta)/len(delta),
            "root_bootstrap95_descriptive": _bootstrap(delta),
            "dfs_wins": delta.count(1),
            "greedy_wins": delta.count(-1),
            "ties": delta.count(0),
        }
    else:
        paired = {
            "model_free_native_random_close_rate": (
                counts["uniform_dfs"]["standalone_certified_closures"] /
                len(episodes)
            ),
            "per_random_seed_closures": {
                str(seed): sum(ep["standalone_verified"] for ep in episodes
                               if ep["uniform_seed"] == seed)
                for seed in UNIFORM_SEEDS
            },
        }
    report = {
        "schema": SCHEMA,
        "status": "complete" if certified_backend else "synthetic_engineering_only",
        "control": control,
        "source": source,
        "backend": backend,
        "native_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_provenance": provenance,
        "frozen_initial_weights_sha256": initial_weight_sha,
        "independent_theorem_roots": len(root_rows),
        "effort_budget": effort,
        "tactic_unit_cost": 1,
        "backtrack_unit_cost": 1,
        "policies": list(policies),
        "model_free_uniform_seeds": list(UNIFORM_SEEDS) if control == "uniform" else [],
        "independent_reference_proofs": len(witnesses),
        "independent_reference_attainable": sum(
            row.get("attainable_within_effort", False) for row in root_rows
        ),
        "standalone_reference_certificates": sum(
            witness["standalone_verified"] for witness in witnesses
        ),
        "scores": counts,
        "paired_effect": paired,
        "action_and_rollback_replay_passed": True,
        "standalone_compiler_passed": True,
        "locked_test_or_ood_opened": False,
        "neural_model_training_updates": 0,
        "scientific_eligible": False,
        "architecture_promotion_eligible": False,
        "reference_oracle_available_only_after_policy_scoring": True,
        "artifacts": {
            name: sha256_file(out / name)
            for name in (
                "episodes.jsonl", "roots.jsonl", "standalone.lean",
            )
        },
    }
    _write(out / "compiler_receipt.json", receipt)
    report["artifacts"]["compiler_receipt.json"] = sha256_file(
        out / "compiler_receipt.json"
    )
    _write(out / "manifest.json", report)
    return report


def replay(repl, suite: Path, session: Path, out: Path,
           *, certified_backend: bool = False) -> dict:
    if out.exists():
        raise ValueError("fresh native replay receipt destination required")
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unknown native bounded-backtrack source manifest")
    if _native_fingerprint(repl, certified_backend) != manifest["backend"]:
        raise ValueError("Lean checker or source binary changed since scoring")
    for filename, sha in manifest["artifacts"].items():
        if filename != Path(filename).name or sha256_file(session / filename) != sha:
            raise ValueError("native backtrack artifact hash changed")
    cases, source = load_suite(suite, "calibration", manifest["independent_theorem_roots"])
    if source["suite_manifest_sha256"] != manifest["source"]["suite_manifest_sha256"]:
        raise ValueError("theorem suite digest changed after native scoring")
    by_root = {case.root: case for case in cases}
    episodes = _rows(session / "episodes.jsonl")
    expected = manifest["independent_theorem_roots"] * (
        len(MODEL_POLICIES) if manifest["control"] == "model"
        else len(UNIFORM_SEEDS)
    )
    if len(episodes) != expected:
        raise ValueError("native replay episodes missing or extra")
    for ep in episodes:
        case = by_root.get(ep["root"])
        if case is None or ep["lineage"] != case.lineage or ep["effort_budget"] != BUDGET:
            raise ValueError("replay episode violates root or effort authorization")
        replay_episode(repl, case, ep)
    out.mkdir(parents=True)
    report = {
        "schema": REPLAY_SCHEMA,
        "native_backend": manifest["backend"],
        "session_manifest_sha256": sha256_file(session / "manifest.json"),
        "independent_roots": len(cases),
        "verified_episodes": len(episodes),
        "all_replays_passed": True,
        "model_calls_during_replay": 0,
        "locked_test_opened": False,
    }
    _write(out / "replay.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runp = commands.add_parser("run")
    runp.add_argument("--suite", type=Path, required=True)
    runp.add_argument("--teacher", type=Path, required=True)
    runp.add_argument("--checkpoint", type=Path)
    runp.add_argument("--out", type=Path, required=True)
    runp.add_argument("--control", choices=("model","uniform"), default="model")
    runp.add_argument("--roots", type=int, default=32)
    runp.add_argument("--effort", type=int, default=BUDGET)
    runp.add_argument("--max-seconds", type=float, default=900)
    runp.add_argument("--cert-timeout", type=int, default=420)
    replayp = commands.add_parser("replay")
    replayp.add_argument("--suite", type=Path, required=True)
    replayp.add_argument("--session", type=Path, required=True)
    replayp.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if not config.LEAN_EXE.is_file() or not config.REPL_EXE.is_file():
        raise RuntimeError("actual Lean 4.19 checker missing")
    with LeanRepl(
        timeout=args.request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        if args.command == "run":
            result = run(
                repl, args.suite, args.teacher, args.out,
                checkpoint_path=args.checkpoint, control=args.control,
                roots=args.roots, effort=args.effort,
                max_seconds=args.max_seconds, cert_timeout=args.cert_timeout,
                certified_backend=True,
            )
        else:
            result = replay(
                repl, args.suite, args.session, args.out,
                certified_backend=True,
            )
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()
