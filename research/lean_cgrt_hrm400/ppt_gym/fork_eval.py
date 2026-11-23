"""LSA-02G native Lean read-only fork calibration and proof-competence audit.

First collect the *new calibration shard's* reference trajectory and shortest
continuation labels using native Lean, after all frozen model evaluations.
Independently compile every reference proof. Then score saved checkpoints on
those identical reference-path fork states, with no optimizer or model-guided
Lean calls. Combine with whole-proof native Lean and random-menu receipts.
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
from lcgrt.extract import explore
from lcgrt.lean_repl import LeanRepl
from .cli import checkpoint_load
from .data import canonical, sha256_file
from .live_lean import fingerprint
from .live_suite import load_suite
from .native_train import _load_teacher, _snapshot_trajectory
from .typed_state import encode_public_packet, schema_for_view
from .uniform_baseline import SCHEMA as RANDOM_SCHEMA


SCHEMA = "lsa02g.native_fork_competence.v1"
SEEDS = (149, 151)
MODES = ("plain", "fork4", "sham4")
VIEW = "pp96"
RANDOM_SEEDS = (17, 31, 47, 61, 83)
BOOTSTRAPS = 2000


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _write(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _hash_files(directory: Path, claims: dict[str, str]) -> None:
    if not claims:
        raise ValueError("missing original native artifact receipts")
    for filename, checksum in claims.items():
        if filename != Path(filename).name or sha256_file(directory / filename) != checksum:
            raise ValueError("archived native artifact hash differs: " + filename)


def _bootstrap(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(20261128)
    n = len(values)
    samples = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAPS)
    )
    return [samples[int(.025 * BOOTSTRAPS)], samples[int(.975 * BOOTSTRAPS) - 1]]


def collect(repl, suite: Path, teacher: Path, out: Path, *,
            roots: int = 32, max_seconds: float = 720,
            cert_timeout: int = 360, certified_backend: bool = False) -> dict:
    if out.exists() or roots != 32 or max_seconds <= 0:
        raise ValueError("fresh output and registered 32-root developmental roster required")
    if certified_backend and (
        not isinstance(repl, LeanRepl) or str(repl.version) != "4.19.0"
    ):
        raise ValueError("calibration fork labels need pinned native Lean 4.19.0")
    source_frames, train_meta = _load_teacher(teacher)
    cases, suite_meta = load_suite(suite, "calibration", roots)
    if train_meta["source"]["suite_manifest_sha256"] != suite_meta["suite_manifest_sha256"]:
        raise ValueError("training teacher and calibration theorem suite differ")
    if {r["lineage"] for r in source_frames} & {c.lineage for c in cases}:
        raise ValueError("calibration fork labels overlap native training roots")
    began = time.perf_counter()
    training_lineage_count = train_meta["independent_lineages"]
    reference = []
    eligible = []
    census = []
    for case in cases:
        if time.perf_counter() - began > max_seconds:
            raise TimeoutError("native Lean fork reference extraction exceeded cap")
        oracle = explore(repl, case.theorem())
        frames = _snapshot_trajectory(repl, case, oracle)
        if not frames:
            raise ValueError("native Lean yielded no checked reference frames")
        script = oracle["proof"]["script"]
        reference.append((case.tid, case.theorem(), script))
        within_budget = len(script) <= 8
        forks = [row for row in frames if len(row["labels"]) >= 2 and
                 0 < sum(row["labels"]) < len(row["labels"])]
        if within_budget:
            eligible.extend(forks)
        census.append({
            "root": case.root, "lineage": case.lineage,
            "family": case.family,
            "reference_tactics": len(script),
            "within_model_tactic_budget": within_budget,
            "all_reference_frames": len(frames),
            "all_discriminating_forks": len(forks),
            "eligible_discriminating_forks": len(forks) if within_budget else 0,
        })
    out.mkdir(parents=True)
    receipt = certify(
        reference, timeout=cert_timeout, keep=out / "standalone.lean",
    )
    if not isinstance(receipt, dict) or any(
        receipt.get("status", {}).get(decl_name(tid)) != "passed"
        for tid, _, _ in reference
    ):
        raise ValueError("independently compiled calibration reference proof failed")
    with (out / "fork_frames.jsonl").open("x", encoding="utf-8") as writer:
        for frame in eligible:
            writer.write(canonical(frame) + "\n")
    with (out / "calibration_roots.jsonl").open("x", encoding="utf-8") as writer:
        for row in census:
            writer.write(canonical(row) + "\n")
    _write(out / "independent_compiler.json", receipt)
    report = {
        "schema": "lsa02g.fork_reference_corpus.v1",
        "teacher_sha256": sha256_file(teacher / "manifest.json"),
        "suite_manifest_sha256": suite_meta["suite_manifest_sha256"],
        "backend": (
            fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
            if certified_backend else {"actual_lean_process": False}
        ),
        "split": "calibration",
        "independent_train_lineages": training_lineage_count,
        "independent_calibration_roots": len(census),
        "independently_compiled_reference_proofs": len(reference),
        "reference_proofs_within_budget": sum(c["within_model_tactic_budget"] for c in census),
        "eligible_reference_path_fork_frames": len(eligible),
        "roots_with_eligible_forks": sum(c["eligible_discriminating_forks"] > 0 for c in census),
        "reference_path_not_model_visited": True,
        "labels_collected_after_model_native_scoring": True,
        "model_parameter_updates": 0,
        "heldout_opened": False,
        "artifacts": {
            f: sha256_file(out / f)
            for f in (
                "fork_frames.jsonl", "calibration_roots.jsonl",
                "standalone.lean", "independent_compiler.json",
            )
        },
    }
    _write(out / "manifest.json", report)
    return report


def _fork_scores(model, frames: list[dict]) -> dict[str, dict]:
    by_root: dict[str, list[dict]] = defaultdict(list)
    with torch.no_grad():
        for frame in frames:
            labels = frame["labels"]
            if not (len(labels) > 1 and 0 < sum(labels) < len(labels)):
                raise ValueError("calibration fork corpus contains a forced or ambiguous action")
            text = encode_public_packet(
                frame["goal"], frame["candidates"],
                view=VIEW, max_bytes=model.cfg.max_bytes,
            )
            outcome = model(
                text, frame["candidates"], depth=2,
                checks_remaining=frame["checks_remaining"],
                middle_order="ab",
            )[-1]["logits"]
            chosen = int(outcome.argmax())
            by_root[frame["root"]].append({
                "correct": int(labels[chosen] == 1),
                "random": sum(labels) / len(labels),
                "choice": frame["candidates"][chosen],
            })
    return {
        root: {
            "forks": len(items),
            "correct": sum(r["correct"] for r in items),
            "accuracy": sum(r["correct"] for r in items) / len(items),
            "random_menu_optimal_accuracy": sum(r["random"] for r in items) / len(items),
        }
        for root, items in by_root.items()
    }


def aggregate(teacher: Path, fits: Path, probes: Path, uniform: Path,
              reference: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("fresh paired result destination required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    _, teacher_meta = _load_teacher(teacher)
    teacher_sha = sha256_file(teacher / "manifest.json")
    training = _json(fits / "manifest.json")
    oracle_meta = _json(reference / "manifest.json")
    _hash_files(reference, oracle_meta["artifacts"])
    frames = _records(reference / "fork_frames.jsonl")
    census = _records(reference / "calibration_roots.jsonl")
    if not (
        oracle_meta.get("schema") == "lsa02g.fork_reference_corpus.v1"
        and oracle_meta.get("teacher_sha256") == teacher_sha
        and oracle_meta.get("split") == "calibration"
        and oracle_meta.get("backend", {}).get("actual_lean_process") is True
        and oracle_meta.get("independent_calibration_roots") == 32
        and oracle_meta.get("independently_compiled_reference_proofs") == 32
        and oracle_meta.get("eligible_reference_path_fork_frames") == len(frames)
        and oracle_meta.get("labels_collected_after_model_native_scoring") is True
        and oracle_meta.get("model_parameter_updates") == 0
        and len(census) == 32
        and len({r["lineage"] for r in census}) == 32
    ):
        raise ValueError("Lean calibration fork oracle evidence is incomplete")
    if not (
        training.get("schema") == "lsa02g.matched_fork_emphasis_cpu.v1"
        and training.get("teacher_manifest_sha256") == teacher_sha
        and training.get("source_manifest_sha256") == teacher_meta["source"]["suite_manifest_sha256"]
        and training.get("seed_roster") == list(SEEDS)
        and training.get("mode_roster") == list(MODES)
        and training.get("input_view") == VIEW
        and training.get("same_worker") is True
        and training.get("all_duplicate_fits_reproduced") is True
        and training.get("fully_matched_optimizer_samples") is True
        and training.get("same_seed_initial_weights") is True
        and training.get("optimizer_steps") == 720
        and training.get("single_torch_thread") is True
        and training.get("torch_deterministic") is True
        and training.get("mkldnn_enabled") is False
        and len(training["cells"]) == 6
    ):
        raise ValueError("matched CPU fork/emphasis training source contract differs")
    cell_table = {(row["seed"], row["mode"]): row for row in training["cells"]}
    if len(cell_table) != 6:
        raise ValueError("duplicated or absent matched source training cell")
    frozen = {}
    root_sets = []
    report_cells = []
    for seed in SEEDS:
        init = set()
        for mode in MODES:
            key = f"{seed}-{mode}"
            item = cell_table[(seed, mode)]
            fit_dir = fits / "fits" / key
            checkpoint_path = fit_dir / "checkpoint.pt"
            fit_receipt = _json(fit_dir / "manifest.json")
            checkpoint_sha = sha256_file(checkpoint_path)
            model, checkpoint = checkpoint_load(checkpoint_path)
            init.add(item["initial_weights_sha256"])
            if not (
                checkpoint_sha == item["checkpoint_sha256"] == fit_receipt["checkpoint_sha256"]
                and checkpoint.get("seed") == seed
                and checkpoint.get("train_order") == "ab"
                and checkpoint.get("weight_mode") == mode
                and checkpoint.get("input_view") == VIEW
                and checkpoint.get("native_teacher_sha256") == teacher_sha
                and checkpoint.get("source_manifest_sha256") == teacher_meta["source"]["suite_manifest_sha256"]
                and model.cfg.max_bytes == 384
                and model.parameter_count() == 57475
                and fit_receipt.get("optimizer_steps") == 720
                and fit_receipt.get("weight_mode") == mode
            ):
                raise ValueError("fork checkpoint training-source or view identity changed")
            base = probes / key
            proof_dir = base / "lsa02g-proof"
            native = _json(proof_dir / "manifest.json")
            replay = _json(base / "lsa02g-replay" / "replay.json")
            _hash_files(proof_dir, native["artifacts"])
            if not (
                native.get("schema") == "lsa02b.native_block_order.v1"
                and native.get("status") == "complete"
                and native.get("checkpoint_sha256") == checkpoint_sha
                and native.get("input_view") == VIEW
                and native.get("paired_roots") == 32
                and native.get("split") == "calibration"
                and native.get("standalone_reference_certificates") == 32
                and native.get("backend", {}).get("actual_lean_process") is True
                and native.get("training_provenance", {}).get("source_lineages_disjoint") is True
                and native.get("training_provenance", {}).get("training_source_verified") is True
                and native.get("training_provenance", {}).get("native_teacher_manifest_sha256") == teacher_sha
                and native.get("query_label_training_updates") == 0
                and replay.get("replay_passed") is True
                and replay.get("independent_replay_episodes") == 96
                and replay.get("session_manifest_sha256") == sha256_file(proof_dir / "manifest.json")
            ):
                raise ValueError("native root proof or independent replay source changed")
            proof_roots = _records(proof_dir / "paired_roots.jsonl")
            if len(proof_roots) != 32 or len({r["lineage"] for r in proof_roots}) != 32:
                raise ValueError("missing native same-root proof records")
            root_sets.append({r["root"] for r in proof_roots})
            scored = _fork_scores(model.eval(), frames)
            frozen[key] = {"roots": {r["root"]: r for r in proof_roots}, "forks": scored}
            report_cells.append({
                "seed": seed, "mode": mode,
                "native_full_proofs": native["scores"]["ab"]["standalone_verified_closures"],
                "native_BA_proofs": native["scores"]["ba"]["standalone_verified_closures"],
                "native_auto8_proofs": native["scores"]["auto8"]["standalone_verified_closures"],
                "native_attainable": native["reference_reachable_within_budget"],
                "reference_path_fork_roots": len(scored),
                "reference_path_fork_frames": sum(s["forks"] for s in scored.values()),
                "root_mean_fork_correct": (
                    sum(s["accuracy"] for s in scored.values()) / len(scored) if scored else None
                ),
                "root_mean_uniform_optimal_fork": (
                    sum(s["random_menu_optimal_accuracy"] for s in scored.values()) / len(scored) if scored else None
                ),
                "train_boosted_source_frames": item["boosted_source_frames"],
                "train_boosted_overlap_with_forks": item["boosted_overlap_with_forks"],
                "train_optimizer_fork_draws": item["optimizer_draws_on_forks"],
                "train_optimizer_boosted_draws": item["optimizer_draws_on_boosted_frames"],
                "checkpoint_sha256": checkpoint_sha,
            })
        if len(init) != 1:
            raise ValueError("same-seed matched modes did not start from same Plan B tensors")
    if any(s != root_sets[0] for s in root_sets[1:]):
        raise ValueError("different calibration root rosters across emphasis conditions")
    if root_sets[0] != {r["root"] for r in census}:
        raise ValueError("reference fork and native proof evaluation cohorts differ")

    uniform_meta = _json(uniform / "manifest.json")
    uniform_replay = _json(uniform / "replay" / "replay.json")
    _hash_files(uniform, uniform_meta["artifacts"])
    random_roots = _records(uniform / "root_scores.jsonl")
    if not (
        uniform_meta.get("schema") == RANDOM_SCHEMA
        and uniform_meta.get("source_teacher_sha256") == teacher_sha
        and uniform_meta.get("independent_calibration_roots") == 32
        and uniform_meta.get("seed_roster") == list(RANDOM_SEEDS)
        and uniform_meta.get("native_model_calls") == 0
        and uniform_meta.get("all_replays_passed") is True
        and uniform_meta.get("backend", {}).get("actual_lean_process") is True
        and uniform_replay.get("all_replays_passed") is True
        and uniform_replay.get("verified_episodes") == 32 * len(RANDOM_SEEDS)
        and uniform_replay.get("session_manifest_sha256") == sha256_file(uniform / "manifest.json")
        and {r["root"] for r in random_roots} == root_sets[0]
    ):
        raise ValueError("native independently replayed uniform-menu reference is invalid")
    root_random = {r["root"]: r["mean_seed_closure"] for r in random_roots}

    root_results = []
    for row in sorted(census, key=lambda r: r["root"]):
        root = row["root"]
        evidence = []
        for seed in SEEDS:
            ref = frozen[f"{seed}-plain"]["roots"][root]
            observed = {}
            fork = {}
            for mode in MODES:
                entry = frozen[f"{seed}-{mode}"]["roots"][root]
                if (
                    entry["lineage"] != ref["lineage"]
                    or entry["reference_budget_attainable"] != ref["reference_budget_attainable"]
                    or entry["auto8_certified"] != ref["auto8_certified"]
                ):
                    raise ValueError("same-root native reference proofs differ across arms")
                if bool(row["within_model_tactic_budget"]) != bool(entry["reference_budget_attainable"]):
                    raise ValueError("independent reference path budget contradicts native proof receipt")
                observed[mode] = int(entry["ab_certified"])
                if root in frozen[f"{seed}-{mode}"]["forks"]:
                    fork[mode] = frozen[f"{seed}-{mode}"]["forks"][root]["accuracy"]
            evidence.append({"seed": seed, "native_proofs": observed, "reference_forks": fork})
        diff_proof = sum(
            ep["native_proofs"]["fork4"] - ep["native_proofs"]["plain"] for ep in evidence
        ) / len(evidence)
        sham_delta = sum(
            ep["native_proofs"]["fork4"] - ep["native_proofs"]["sham4"] for ep in evidence
        ) / len(evidence)
        if row["eligible_discriminating_forks"]:
            if any(set(ep["reference_forks"]) != set(MODES) for ep in evidence):
                raise ValueError("missing root-paired reference fork predictions")
            fork_diff = sum(
                ep["reference_forks"]["fork4"] - ep["reference_forks"]["plain"]
                for ep in evidence
            ) / len(evidence)
            fork_sham = sum(
                ep["reference_forks"]["fork4"] - ep["reference_forks"]["sham4"]
                for ep in evidence
            ) / len(evidence)
        else:
            fork_diff = fork_sham = None
        root_results.append({
            "root": root, "lineage": row["lineage"],
            "reference_budget_attainable": row["within_model_tactic_budget"],
            "eligible_reference_forks": row["eligible_discriminating_forks"],
            "uniform_native_root_close_rate": root_random[root],
            "seed_outcomes": evidence,
            "fork4_minus_plain_proof": diff_proof,
            "fork4_minus_sham4_proof": sham_delta,
            "fork4_minus_plain_fork_accuracy": fork_diff,
            "fork4_minus_sham4_fork_accuracy": fork_sham,
        })
    def contrast(name, subset=None):
        values = [r[name] for r in (subset if subset is not None else root_results)
                  if r[name] is not None]
        return {"mean": sum(values) / len(values) if values else None,
                "root_bootstrap95_descriptive": _bootstrap(values),
                "independent_roots": len(values)}
    effect = {
        "fork4_minus_plain_full_proof": contrast("fork4_minus_plain_proof"),
        "fork4_minus_sham4_full_proof": contrast("fork4_minus_sham4_proof"),
        "fork4_minus_plain_reference_forks": contrast("fork4_minus_plain_fork_accuracy"),
        "fork4_minus_sham4_reference_forks": contrast("fork4_minus_sham4_fork_accuracy"),
    }
    closure_rates = {
        mode: sum(cell["native_full_proofs"] for cell in report_cells if cell["mode"] == mode)
        / (32 * len(SEEDS))
        for mode in MODES
    }
    random_rate = uniform_meta["uniform_mean_closure_rate"]
    best = max(closure_rates.values())
    screen = (
        effect["fork4_minus_plain_reference_forks"]["mean"] is not None
        and effect["fork4_minus_sham4_reference_forks"]["mean"] is not None
        and effect["fork4_minus_plain_reference_forks"]["mean"] >= .05
        and effect["fork4_minus_sham4_reference_forks"]["mean"] >= .05
        and effect["fork4_minus_plain_full_proof"]["mean"] >= 0
        and effect["fork4_minus_sham4_full_proof"]["mean"] >= 0
    )
    competence = best >= .4 and best - random_rate >= .10
    out.mkdir(parents=True)
    report = {
        "schema": SCHEMA,
        "teacher_sha256": teacher_sha,
        "training_manifest_sha256": sha256_file(fits / "manifest.json"),
        "calibration_reference_sha256": sha256_file(reference / "manifest.json"),
        "uniform_manifest_sha256": sha256_file(uniform / "manifest.json"),
        "independent_training_roots": teacher_meta["independent_lineages"],
        "training_frames": teacher_meta["train_examples"],
        "independent_calibration_roots": len(census),
        "independent_calibration_roots_with_eligible_reference_forks": oracle_meta["roots_with_eligible_forks"],
        "eligible_calibration_reference_forks": len(frames),
        "trained_input_view": VIEW,
        "modes": list(MODES),
        "training_seeds": list(SEEDS),
        "uniform_seeds": list(RANDOM_SEEDS),
        "native_model_close_rate_by_mode": closure_rates,
        "native_uniform_close_rate": random_rate,
        "native_uniform_per_seed": uniform_meta["per_seed_closures"],
        "native_reference_attainable_roots": sum(row["within_model_tactic_budget"] for row in census),
        "effect": effect,
        "predeclared_development_screen_passed": screen,
        "native_competence_gate_met": competence,
        "formal_competence_gate": (
            "CANDIDATE_REQUIRES_FRESH_SEALED_CONFIRMATION"
            if competence else "BLOCKED_INCOMPETENT_CHECKPOINT"
        ),
        "cells": report_cells,
        "root_results": root_results,
        "heldout_or_ood_opened": False,
        "source_only_calibration_prediction_without_model_updates": True,
        "scientific_eligible": False,
        "architecture_promotion_eligible": False,
        "interpretation": (
            "Root-paired development test of training fork weighting and shuffled sham. "
            "Reference-state tactic accuracy uses independently compiled native Lean "
            "shortest continuation labels collected after all model proof episodes."
        ),
    }
    _write(out / "fork_competence.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--uniform", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--request-timeout", type=int, default=60)
    args = p.parse_args(argv)
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise RuntimeError("native Lean 4.19 binaries unavailable")
    with LeanRepl(
        timeout=args.request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        collect(repl, args.suite, args.teacher, args.reference,
                certified_backend=True)
    result = aggregate(
        args.teacher, args.fits, args.probes, args.uniform,
        args.reference, args.out,
    )
    print(canonical({
        "independent_calibration_roots": result["independent_calibration_roots"],
        "fork_root_count": result["independent_calibration_roots_with_eligible_reference_forks"],
        "reference_fork_frames": result["eligible_calibration_reference_forks"],
        "native_model_close_rates": result["native_model_close_rate_by_mode"],
        "native_uniform": result["native_uniform_close_rate"],
        "effects": result["effect"],
        "screen": result["predeclared_development_screen_passed"],
        "competence": result["formal_competence_gate"],
    }))
    return result


if __name__ == "__main__":
    main()
