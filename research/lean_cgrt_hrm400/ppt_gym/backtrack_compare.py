"""LSA-02I post-scoring root-paired CPU/backtracking competence audit.

Read native Lean 4.19 kernel + standalone scripts and independently replayed
proof-search logs. Only after every live policy and random control is complete,
open the native Lean action graph to classify dead-end branch attempts. The
model had no access to reference labels or dead-end graph annotations.
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
from lcgrt.extract import explore
from lcgrt.lean_repl import LeanRepl
from .backtrack_campaign import SCHEMA as RUN_SCHEMA, UNIFORM_SEEDS, MODEL_POLICIES
from .data import canonical, sha256_file
from .live_suite import load_suite
from .native_train import _load_teacher


SCHEMA = "lsa02i.native_backtrack_root_comparison.v1"
SEEDS = (173, 179)
DRAWS = 2000


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(t) for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]


def _assert_files(folder: Path, hashes: dict) -> None:
    if not hashes:
        raise ValueError("missing native file custody manifest")
    for name, sha in hashes.items():
        if name != Path(name).name or sha256_file(folder / name) != sha:
            raise ValueError("native proof or data artifact SHA-256 changed: " + name)


def _bootstrap(values: list[float]) -> list[float]:
    n = len(values)
    if n < 2:
        raise ValueError("at least two independent theorem roots required")
    rng = random.Random(20261202)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(DRAWS)
    )
    return [means[int(.025 * DRAWS)], means[int(.975 * DRAWS)-1]]


def _dead_edges(repl, suite: Path, teacher: Path, episodes: list[dict],
                *, roots: int = 32, max_seconds: float = 420) -> list[dict]:
    """Label bad accepted tactics retrospectively using native Lean only."""
    cases, suite_meta = load_suite(suite, "calibration", roots)
    if not isinstance(repl, LeanRepl) or str(repl.version) != "4.19.0":
        raise ValueError("post-scoring dead-branch labels require actual Lean 4.19.0")
    if _json(teacher / "manifest.json")["source"]["suite_manifest_sha256"] != suite_meta["suite_manifest_sha256"]:
        raise ValueError("post-scoring theorem source changed")
    t0 = time.perf_counter()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        grouped[ep["root"]].append(ep)
    if set(grouped) != {case.root for case in cases}:
        raise ValueError("post-scoring model/control episode lineage list changed")
    result = []
    for case in cases:
        if time.perf_counter() - t0 > max_seconds:
            raise TimeoutError("post-scoring native graph audit exceeded CPU budget")
        oracle = explore(repl, case.theorem())
        table, size = oracle["table"], oracle["size"]
        root_rows = []
        for ep in grouped[case.root]:
            dead = overcost = finite = 0
            cycle_rollbacks = 0
            rollback_dead = 0
            dead_script_prefixes = set()
            for event in ep["events"]:
                if event["kind"] == "RECONSIDER":
                    cycle_rollbacks += int(event["reason"] == "cycle")
                    prefix = tuple(event["abandoned_prefix"])
                    if any(prefix[:len(previous)] == previous for previous in dead_script_prefixes):
                        rollback_dead += 1
                    continue
                if event["kind"] != "TACTIC" or not event["lean_accepted"]:
                    continue
                goal = event["active_goal"]
                tactic = event["tactic"]
                children = table.get(goal, {}).get(tactic)
                if children is None:
                    raise ValueError("Lean-audited accepted tactic absent from native proof transition graph")
                costs = [size.get(x) for x in children]
                if any(x is None or not isinstance(x, int) or x < 1 for x in costs):
                    dead += 1
                    dead_script_prefixes.add(tuple(event["script_after"]))
                else:
                    finite += 1
                    if 1 + sum(costs) > ep["effort_budget"] - event["unit"] + 1:
                        overcost += 1
            root_rows.append({
                "policy": ep["policy"],
                "uniform_seed": ep["uniform_seed"],
                "dead_accepted_actions": dead,
                "finite_accepted_actions": finite,
                "overbudget_finite_actions": overcost,
                "cycle_reconsiderations": cycle_rollbacks,
                "dead_branch_reconsiderations": rollback_dead,
                "terminal_proof": ep["standalone_verified"],
                "effort_spent": ep["effort_spent"],
                "backtracks": ep["backtracks"],
            })
        result.append({
            "root": case.root,
            "lineage": case.lineage,
            "native_reference_proof_size": len(oracle["proof"]["script"]),
            "native_reference_kernel": oracle["proof"]["repl_kernel"],
            "policy_audit": root_rows,
        })
    return result


def compare(
    suite: Path, teacher: Path, fits: Path,
    probes: Path, uniform: Path, out: Path,
    *, certified_repl=None,
) -> dict:
    if out.exists():
        raise ValueError("fresh paired-root backtracking report directory required")
    _, train_meta = _load_teacher(teacher)
    teacher_sha = sha256_file(teacher / "manifest.json")
    training = _json(fits / "manifest.json")
    if not (
        training.get("schema") == "lsa02i.same_worker_frozen_policy_fit.v1"
        and training.get("teacher_manifest_sha256") == teacher_sha
        and training.get("view_roster") == ["pp96"]
        and training.get("seed_roster") == list(SEEDS)
        and training.get("same_worker") is True
        and training.get("all_duplicate_fits_reproduced") is True
        and training.get("fully_matched_optimizer_samples") is True
        and training.get("single_torch_thread") is True
        and training.get("torch_deterministic") is True
        and training.get("mkldnn_enabled") is False
        and training.get("optimizer_steps") == 720
        and len(training.get("cells", [])) == 2
    ):
        raise ValueError("same-worker Plan B model source or seed roster is not attested")

    cells = []
    root_sets = []
    root_policy = {}
    all_model_episodes = []
    for seed in SEEDS:
        key = f"{seed}-pp96"
        cell = next(x for x in training["cells"] if x["seed"] == seed)
        ckpt = fits / "fits" / key / "checkpoint.pt"
        if cell["checkpoint_sha256"] != sha256_file(ckpt):
            raise ValueError("frozen native model checkpoint hash differs")
        directory = probes / str(seed)
        native = _json(directory / "manifest.json")
        replay = _json(probes / f"{seed}-replay" / "replay.json")
        _assert_files(directory, native.get("artifacts"))
        if not (
            native.get("schema") == RUN_SCHEMA
            and native.get("control") == "model"
            and native.get("status") == "complete"
            and native.get("independent_theorem_roots") == 32
            and native.get("effort_budget") == 16
            and native.get("checkpoint_sha256") == cell["checkpoint_sha256"]
            and native.get("native_teacher_sha256") == teacher_sha
            and native.get("checkpoint_provenance", {}).get("training_source_verified") is True
            and native.get("checkpoint_provenance", {}).get("source_lineages_disjoint") is True
            and native.get("backend", {}).get("actual_lean_process") is True
            and native.get("standalone_reference_certificates") == 32
            and native.get("independent_reference_proofs") == 32
            and native.get("policies") == list(MODEL_POLICIES)
            and native.get("action_and_rollback_replay_passed") is True
            and native.get("neural_model_training_updates") == 0
            and replay.get("schema") == "lsa02i.independent_backtrack_replay.v1"
            and replay.get("all_replays_passed") is True
            and replay.get("verified_episodes") == 96
            and replay.get("session_manifest_sha256") == sha256_file(directory / "manifest.json")
        ):
            raise ValueError("native model proof, standalone, source or model-free replay custody invalid")
        rows = _rows(directory / "roots.jsonl")
        if len(rows) != 32 or len({r["lineage"] for r in rows}) != 32:
            raise ValueError("same-seed native model evaluation has repeated or missing theorem roots")
        root_sets.append({r["root"] for r in rows})
        root_policy[seed] = {r["root"]: r for r in rows}
        all_model_episodes.extend(_rows(directory / "episodes.jsonl"))
        cells.append({
            "seed": seed,
            "checkpoint_sha256": cell["checkpoint_sha256"],
            "native_greedy_closures": native["scores"]["frozen_greedy"]["standalone_certified_closures"],
            "native_dfs_closures": native["scores"]["frozen_dfs"]["standalone_certified_closures"],
            "native_auto_dfs_closures": native["scores"]["auto_dfs"]["standalone_certified_closures"],
            "greedy_total_work_units": native["scores"]["frozen_greedy"]["sum_effort_spent"],
            "dfs_total_work_units": native["scores"]["frozen_dfs"]["sum_effort_spent"],
            "dfs_total_backtracks": native["scores"]["frozen_dfs"]["sum_backtracks"],
            "dfs_total_tactic_checks": native["scores"]["frozen_dfs"]["sum_tactic_checks"],
            "auto_total_work_units": native["scores"]["auto_dfs"]["sum_effort_spent"],
            "auto_total_backtracks": native["scores"]["auto_dfs"]["sum_backtracks"],
        })
    if root_sets[0] != root_sets[1]:
        raise ValueError("frozen model seed cohorts used different theorem roots")

    control = _json(uniform / "manifest.json")
    random_replay = _json(uniform / "replay" / "replay.json")
    _assert_files(uniform, control.get("artifacts"))
    if not (
        control.get("schema") == RUN_SCHEMA
        and control.get("control") == "uniform"
        and control.get("status") == "complete"
        and control.get("independent_theorem_roots") == 32
        and control.get("effort_budget") == 16
        and control.get("model_free_uniform_seeds") == list(UNIFORM_SEEDS)
        and control.get("native_teacher_sha256") == teacher_sha
        and control.get("backend", {}).get("actual_lean_process") is True
        and control.get("scores", {}).get("uniform_dfs", {}).get("sum_model_calls") == 0
        and control.get("action_and_rollback_replay_passed") is True
        and random_replay.get("all_replays_passed") is True
        and random_replay.get("verified_episodes") == 160
        and random_replay.get("session_manifest_sha256") == sha256_file(uniform / "manifest.json")
    ):
        raise ValueError("independent uniform bounded-DFS control source or native Lean custody invalid")
    uniform_rows = _rows(uniform / "roots.jsonl")
    if len(uniform_rows) != 32 or {r["root"] for r in uniform_rows} != root_sets[0]:
        raise ValueError("uniform DFS used a different theorem root cohort")
    uniform_by_root = {r["root"]: r for r in uniform_rows}
    random_episodes = _rows(uniform / "episodes.jsonl")
    if len(random_episodes) != 160 or len(all_model_episodes) != 2*96:
        raise ValueError("native proof policy episode census incomplete")

    root_comparison = []
    for root in sorted(root_sets[0]):
        first = root_policy[SEEDS[0]][root]
        second = root_policy[SEEDS[1]][root]
        if not (
            first["lineage"] == second["lineage"]
            and first["reference_proof_size"] == second["reference_proof_size"]
            and first["attainable_within_effort"] == second["attainable_within_effort"]
        ):
            raise ValueError("frozen model theorem ancestry or attainable reference changed")
        uniform_row = uniform_by_root[root]
        if uniform_row["lineage"] != first["lineage"]:
            raise ValueError("uniform control did not score same theorem lineage")
        random_success = [
            int(uniform_row["policies"][f"uniform_{seed}"]["closed"])
            for seed in UNIFORM_SEEDS
        ]
        measurements = []
        for seed in SEEDS:
            rows = root_policy[seed][root]["policies"]
            measurements.append({
                "seed": seed,
                "greedy": rows["frozen_greedy"],
                "dfs": rows["frozen_dfs"],
                "auto_dfs": rows["auto_dfs"],
            })
        root_comparison.append({
            "root": root, "lineage": first["lineage"],
            "reference_proof_size": first["reference_proof_size"],
            "reference_attainable": first["attainable_within_effort"],
            "random_uniform_dfs_closure": sum(random_success) / len(random_success),
            "random_uniform_successes": random_success,
            "model_seed_results": measurements,
            "dfs_minus_greedy": sum(
                int(x["dfs"]["closed"]) - int(x["greedy"]["closed"])
                for x in measurements
            )/len(measurements),
            "dfs_minus_auto": sum(
                int(x["dfs"]["closed"]) - int(x["auto_dfs"]["closed"])
                for x in measurements
            )/len(measurements),
        })
    deltas = [r["dfs_minus_greedy"] for r in root_comparison]
    random_rate = sum(r["random_uniform_dfs_closure"] for r in root_comparison)/32
    scores = {
        policy: sum(
            int(sample[policy_name]["closed"])
            for row in root_comparison
            for sample in row["model_seed_results"]
        )/(32*len(SEEDS))
        for policy, policy_name in (
            ("greedy","greedy"),("dfs","dfs"),("auto_dfs","auto_dfs")
        )
    }
    score_differences = {
        "dfs_minus_greedy_mean": sum(deltas)/len(deltas),
        "dfs_minus_greedy_root_bootstrap95_descriptive": _bootstrap(deltas),
        "dfs_minus_auto_mean": sum(r["dfs_minus_auto"] for r in root_comparison)/32,
        "dfs_minus_auto_root_bootstrap95_descriptive": _bootstrap(
            [r["dfs_minus_auto"] for r in root_comparison],
        ),
        "dfs_minus_uniform_random": scores["dfs"] - random_rate,
    }
    if certified_repl is not None:
        oracle_receipt = _dead_edges(
            certified_repl, suite, teacher,
            all_model_episodes + random_episodes,
        )
    else:
        raise ValueError("post-scoring native Lean oracle process is required")
    if {r["root"] for r in oracle_receipt} != root_sets[0]:
        raise ValueError("post-scoring native accepted/dead branch labels differ from verified root roster")
    root_dead = {
        r["root"]: r for r in oracle_receipt
    }
    for row in root_comparison:
        row["postscore_branch_audit"] = root_dead[row["root"]]
    all_model_backtracks = sum(
        ep["backtracks"] for ep in all_model_episodes if ep["policy"] == "frozen_dfs"
    )
    validated = (
        scores["dfs"] >= .40 and
        scores["dfs"] - random_rate >= .10
    )
    development = (
        scores["dfs"] - scores["greedy"] >= .05 and
        all(
            x["native_dfs_closures"] >= x["native_greedy_closures"]
            for x in cells
        )
    )
    report = {
        "schema": SCHEMA,
        "source_teacher_manifest_sha256": teacher_sha,
        "trainer_manifest_sha256": sha256_file(fits / "manifest.json"),
        "uniform_manifest_sha256": sha256_file(uniform / "manifest.json"),
        "source_lineages": train_meta["independent_lineages"],
        "source_train_frames": train_meta["train_examples"],
        "independent_calibration_roots": len(root_comparison),
        "unique_theorem_root_unit": True,
        "model_seed_roster": list(SEEDS),
        "uniform_seed_roster": list(UNIFORM_SEEDS),
        "work_budget_per_attempt": 16,
        "tactic_work_cost": 1,
        "reconsider_work_cost": 1,
        "frozen_model_close_rate": scores,
        "random_uniform_dfs_native_close_rate": random_rate,
        "native_reference_attainable": sum(row["reference_attainable"] for row in root_comparison),
        "paired_effects": score_differences,
        "total_frozen_model_dfs_backtracks": all_model_backtracks,
        "cells": cells,
        "root_results": root_comparison,
        "registered_development_screen_passed": development,
        "competence_preflight_passed": validated,
        "formal_competence_gate": (
            "CANDIDATE_REQUIRES_NEW_SEALED_CONFIRMATION" if validated
            else "BLOCKED_INCOMPETENT_CHECKPOINT"
        ),
        "postscore_oracle_blind_during_inference": True,
        "calibration_model_parameter_updates": 0,
        "locked_test_opened": False,
        "scientific_eligible": False,
        "architecture_promotion_eligible": False,
        "research_caveat": (
            "Finite native randomized control; repeated model seeds share 32 theorem roots. "
            "Exact same-worker training required, cross-worker numerical drift unmeasured. "
            "No live oracle branching labels were available to any model."
        ),
    }
    out.mkdir(parents=True)
    with (out / "postscore_dead_branch_graph.jsonl").open("x", encoding="utf-8") as handle:
        for row in oracle_receipt:
            handle.write(canonical(row) + "\n")
    report["postscore_native_graph_sha256"] = sha256_file(
        out / "postscore_dead_branch_graph.jsonl"
    )
    (out / "matched_backtracking.json").write_text(
        canonical(report) + "\n", encoding="utf-8",
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--uniform", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--request-timeout", type=int, default=60)
    args = p.parse_args(argv)
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise RuntimeError("native Lean 4.19 checker executable absent")
    torch.set_num_threads(1)
    with LeanRepl(
        timeout=args.request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        r = compare(args.suite, args.teacher, args.fits, args.probes,
                    args.uniform, args.out, certified_repl=repl)
    print(canonical({
        "native_closures": r["frozen_model_close_rate"],
        "uniform_native": r["random_uniform_dfs_native_close_rate"],
        "reference_attainable": r["native_reference_attainable"],
        "native_dfs_backtracks": r["total_frozen_model_dfs_backtracks"],
        "paired_effects": r["paired_effects"],
        "dev_screen": r["registered_development_screen_passed"],
        "competence_gate": r["formal_competence_gate"],
    }))
    return r


if __name__ == "__main__":
    main()
