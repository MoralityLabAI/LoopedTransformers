"""LSA-02H: post-scoring Lean reference and off-reference calibration.

This job MUST depend on all native proof evaluation jobs in GitHub Actions.
Its fresh calibration-state labels are never available to source fitting.
Native Lean certifies complete reference and recoverable alternate scripts.
Frozen Plan B models are scored on the same reference-path and offpath states;
all statistics are paired by unique theorem lineage, not tactic frame count.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

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
from .offpath_source import _collect_case
from .offpath_train import SCHEMA as TRAIN_SCHEMA, SEEDS, ARMS, VIEW, _load_offpath
from .uniform_baseline import SCHEMA as RANDOM_SCHEMA
from .typed_state import encode_public_packet


SCHEMA = "lsa02h.native_offreference_calibration.v1"
BOOTSTRAPS = 2000


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _hash_artifacts(directory: Path, manifest: dict) -> None:
    if not manifest:
        raise ValueError("missing Lean compiler artifact custody")
    for filename, checksum in manifest.items():
        if filename != Path(filename).name or sha256_file(directory / filename) != checksum:
            raise ValueError("native corpus or replay artifact hash differs: " + filename)


def _bootstrap(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(20261130)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(BOOTSTRAPS)
    )
    return [means[int(BOOTSTRAPS * .025)], means[int(BOOTSTRAPS * .975)-1]]


def collect(repl, suite: Path, teacher: Path, out: Path, *,
            roots: int = 32, cert_timeout: int = 420,
            certified_backend: bool = False) -> dict:
    if out.exists() or roots != 32:
        raise ValueError("fresh calibration reference with 32 independent roots required")
    train_rows, teacher_meta = _load_teacher(teacher)
    cases, suite_meta = load_suite(suite, "calibration", roots)
    if teacher_meta["source"]["suite_manifest_sha256"] != suite_meta["suite_manifest_sha256"]:
        raise ValueError("calibration reference changed training theorem suite")
    if {r["lineage"] for r in train_rows} & {c.lineage for c in cases}:
        raise ValueError("new calibration roots overlap original training source")
    if certified_backend and (
        not isinstance(repl, LeanRepl) or repl.version != "4.19.0"
    ):
        raise ValueError("need pinned real Lean 4.19.0 for reference state authority")
    all_refs, all_off, census, proofs = [], [], [], []
    for case in cases:
        oracle = explore(repl, case.theorem())
        ref = _snapshot_trajectory(repl, case, oracle)
        off, alt_proofs, source_counts = _collect_case(repl, case, oracle, ref)
        ref_frames = [
            f for f in ref
            if len(oracle["proof"]["script"]) <= 8 and len(f["labels"]) > 1
            and 0 < sum(f["labels"]) < len(f["labels"])
        ]
        all_refs.extend(ref_frames)
        for candidate in off:
            all_off.extend(candidate["correction_states"])
        proofs.append((case.tid, case.theorem(), oracle["proof"]["script"]))
        proofs.extend(alt_proofs)
        census.append({
            "root": case.root,
            "lineage": case.lineage,
            "reference_size": len(oracle["proof"]["script"]),
            "within_budget": len(oracle["proof"]["script"]) <= 8,
            "reference_forks": len(ref_frames),
            "offpath_correction_states": sum(len(a["correction_states"]) for a in off),
            "offpath_discriminating_forks": sum(
                len(r["labels"]) > 1 and 0 < sum(r["labels"]) < len(r["labels"])
                for a in off for r in a["correction_states"]
            ),
            "offpath_full_proofs": len(alt_proofs),
            "source_alternatives_within_budget": source_counts["candidate_within_eight_checks"],
        })
    out.mkdir(parents=True)
    receipt = certify(proofs, timeout=cert_timeout,
                      keep=out / "standalone.lean")
    if receipt.get("timed_out") or any(
        receipt.get("status", {}).get(decl_name(tid)) != "passed"
        for tid, _, _ in proofs
    ):
        raise ValueError("calibration reference or recoverable alternate proof failed standalone compilation")
    for name, rows in (("reference_forks.jsonl", all_refs),
                       ("offreference_frames.jsonl", all_off),
                       ("calibration_census.jsonl", census)):
        with (out / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
    _write(out / "compiler_receipt.json", receipt)
    report = {
        "schema": "lsa02h.lean_calibration_state_evidence.v1",
        "suite_manifest_sha256": suite_meta["suite_manifest_sha256"],
        "teacher_manifest_sha256": sha256_file(teacher / "manifest.json"),
        "backend": (
            fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
            if certified_backend else {"actual_lean_process": False}
        ),
        "independent_calibration_roots": len(census),
        "independently_compiled_reference_roots": len(cases),
        "independently_compiled_recoverable_deviations": sum(r["offpath_full_proofs"] for r in census),
        "reference_fork_frames": len(all_refs),
        "reference_fork_roots": sum(r["reference_forks"] > 0 for r in census),
        "offreference_correction_frames": len(all_off),
        "offreference_correction_roots": sum(r["offpath_correction_states"] > 0 for r in census),
        "offreference_discriminating_fork_frames": sum(r["offpath_discriminating_forks"] for r in census),
        "offreference_discriminating_fork_roots": sum(r["offpath_discriminating_forks"] > 0 for r in census),
        "source_labels_produced_after_native_model_runs": True,
        "calibration_optimizer_updates": 0,
        "locked_test_opened": False,
        "artifacts": {name: sha256_file(out / name) for name in (
            "reference_forks.jsonl", "offreference_frames.jsonl",
            "calibration_census.jsonl", "standalone.lean", "compiler_receipt.json",
        )},
    }
    _write(out / "manifest.json", report)
    return report


def _score_forks(model, rows: list[dict], *, require_fork: bool) -> dict[str, dict]:
    by_root = defaultdict(list)
    with torch.no_grad():
        for row in rows:
            labels = row["labels"]
            fork = len(labels) > 1 and 0 < sum(labels) < len(labels)
            if require_fork and not fork:
                continue
            if len(labels) == 0 or not any(labels):
                raise ValueError("calibration state omitted a shortest continuation witness")
            goal = model(
                encode_public_packet(
                    row["goal"], row["candidates"],
                    view=VIEW, max_bytes=model.cfg.max_bytes,
                ),
                row["candidates"], depth=2,
                checks_remaining=row["checks_remaining"], middle_order="ab",
            )[-1]["logits"]
            chosen = int(goal.argmax())
            by_root[row["root"]].append({
                "correct": int(labels[chosen] == 1),
                "uniform": sum(labels) / len(labels),
                "fork": fork,
            })
    return {
        root: {
            "states": len(values),
            "correct": sum(x["correct"] for x in values),
            "accuracy": sum(x["correct"] for x in values)/len(values),
            "uniform_optimal_accuracy": sum(x["uniform"] for x in values)/len(values),
        } for root, values in by_root.items()
    }


def aggregate(teacher: Path, supplement: Path, fits: Path, probes: Path,
              random_pack: Path, cal_source: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("fresh calibration aggregate output is required")
    original, teacher_meta = _load_teacher(teacher)
    off_pool, off_meta = _load_offpath(
        teacher, supplement, original, teacher_meta,
    )
    teacher_sha = sha256_file(teacher / "manifest.json")
    off_sha = sha256_file(supplement / "manifest.json")
    trained = _read(fits / "manifest.json")
    calibration = _read(cal_source / "manifest.json")
    _hash_artifacts(cal_source, calibration["artifacts"])
    cal_census = _rows(cal_source / "calibration_census.jsonl")
    ref_frames = _rows(cal_source / "reference_forks.jsonl")
    off_frames = _rows(cal_source / "offreference_frames.jsonl")
    if not (
        calibration.get("schema") == "lsa02h.lean_calibration_state_evidence.v1"
        and calibration.get("suite_manifest_sha256") == teacher_meta["source"]["suite_manifest_sha256"]
        and calibration.get("teacher_manifest_sha256") == teacher_sha
        and calibration.get("backend", {}).get("actual_lean_process") is True
        and calibration.get("independent_calibration_roots") == 32
        and calibration.get("independently_compiled_reference_roots") == 32
        and calibration.get("source_labels_produced_after_native_model_runs") is True
        and calibration.get("calibration_optimizer_updates") == 0
        and calibration.get("locked_test_opened") is False
        and len(cal_census) == 32
        and len({r["lineage"] for r in cal_census}) == 32
        and len(ref_frames) == calibration["reference_fork_frames"]
        and len(off_frames) == calibration["offreference_correction_frames"]
    ):
        raise ValueError("post-proof calibration reference/alternate state evidence invalid")
    if {r["lineage"] for r in original} & {r["lineage"] for r in cal_census}:
        raise ValueError("native training and calibration lineage overlap")
    if not (
        trained.get("schema") == TRAIN_SCHEMA
        and trained.get("teacher_manifest_sha256") == teacher_sha
        and trained.get("offreference_manifest_sha256") == off_sha
        and trained.get("suite_manifest_sha256") == teacher_meta["source"]["suite_manifest_sha256"]
        and trained.get("training_lineages") == teacher_meta["independent_lineages"]
        and trained.get("train_novel_offreference_states") == sum(len(r) for r in off_pool.values())
        and trained.get("seed_roster") == list(SEEDS)
        and trained.get("arm_roster") == list(ARMS)
        and trained.get("view") == VIEW
        and trained.get("optimizer_steps_per_cell") == 720
        and trained.get("all_exact_duplicate_fits_passed") is True
        and trained.get("same_seed_identical_source_draws") is True
        and trained.get("same_seed_bitwise_initial_weights") is True
        and trained.get("one_cpu_worker") is True
        and trained.get("torch_threads") == 1
        and trained.get("torch_deterministic_algorithms") is True
        and trained.get("mkldnn_enabled") is False
        and trained.get("native_calibration_queried_during_fit") is False
        and trained.get("test_or_ood_opened") is False
        and len(trained["cells"]) == len(SEEDS) * len(ARMS)
    ):
        raise ValueError("offreference same-host fit evidence differs from source registry")
    training_cells = {(c["seed"], c["arm"]): c for c in trained["cells"]}
    all_roots = []
    cells = []
    frozen = {}
    torch.set_num_threads(1)
    for seed in SEEDS:
        initial = set()
        for arm in ARMS:
            item = training_cells[(seed,arm)]
            key = f"{seed}-{arm}"
            folder = fits / "fits" / key
            path = folder / "checkpoint.pt"
            cp_sha = sha256_file(path)
            model, saved = checkpoint_load(path)
            if not (
                item["checkpoint_sha256"] == cp_sha
                and saved.get("seed") == seed
                and saved.get("train_order") == "ab"
                and saved.get("offreference_training_mode") == arm
                and saved.get("offreference_source_manifest_sha256") == off_sha
                and saved.get("native_teacher_sha256") == teacher_sha
                and saved.get("input_view") == VIEW
                and model.parameter_count() == 57475
                and model.cfg.max_bytes == 384
                and item["optimizer_steps"] == 720
                and item["query_label_optimizer_updates"] == 0
            ):
                raise ValueError("checkpoint differs from the registered offreference source/model")
            initial.add(item["initial_weights_sha256"])
            evaldir = probes / key
            proofdir = evaldir / "lsa02h-proof"
            receipt = _read(proofdir / "manifest.json")
            replay = _read(evaldir / "lsa02h-replay" / "replay.json")
            _hash_artifacts(proofdir, receipt["artifacts"])
            if not (
                receipt.get("schema") == "lsa02b.native_block_order.v1"
                and receipt.get("status") == "complete"
                and receipt.get("checkpoint_sha256") == cp_sha
                and receipt.get("input_view") == VIEW
                and receipt.get("paired_roots") == 32
                and receipt.get("standalone_reference_certificates") == 32
                and receipt.get("training_provenance", {}).get("offreference_source_verified") is True
                and receipt.get("training_provenance", {}).get("offreference_manifest_sha256") == off_sha
                and receipt.get("training_provenance", {}).get("source_lineages_disjoint") is True
                and receipt.get("backend", {}).get("actual_lean_process") is True
                and receipt.get("query_label_training_updates") == 0
                and replay.get("replay_passed") is True
                and replay.get("independent_replay_episodes") == 96
                and replay.get("session_manifest_sha256") == sha256_file(proofdir / "manifest.json")
            ):
                raise ValueError("offreference native proof and replay custody mismatch")
            roots = _rows(proofdir / "paired_roots.jsonl")
            if len(roots) != 32 or len({r["lineage"] for r in roots}) != 32:
                raise ValueError("native proof evaluation repeated theorem roots")
            all_roots.append({r["root"] for r in roots})
            frozen[key] = {
                "proofs": {r["root"]:r for r in roots},
                "off": _score_forks(model.eval(), off_frames, require_fork=False),
                "off_forks": _score_forks(model.eval(), off_frames, require_fork=True),
                "reference": _score_forks(model.eval(), ref_frames, require_fork=True),
            }
            cells.append({
                "seed": seed, "arm": arm,
                "full_proofs": receipt["scores"]["ab"]["standalone_verified_closures"],
                "native_attainable": receipt["reference_reachable_within_budget"],
                "auto8_proofs": receipt["scores"]["auto8"]["standalone_verified_closures"],
                "reference_fork_roots": len(frozen[key]["reference"]),
                "reference_fork_root_mean_accuracy": (
                    sum(s["accuracy"] for s in frozen[key]["reference"].values()) /
                    len(frozen[key]["reference"]) if frozen[key]["reference"] else None
                ),
                "offpath_state_roots": len(frozen[key]["off"]),
                "offpath_state_root_mean_accuracy": (
                    sum(s["accuracy"] for s in frozen[key]["off"].values()) /
                    len(frozen[key]["off"]) if frozen[key]["off"] else None
                ),
                "offpath_fork_roots": len(frozen[key]["off_forks"]),
                "offpath_fork_root_mean_accuracy": (
                    sum(s["accuracy"] for s in frozen[key]["off_forks"].values()) /
                    len(frozen[key]["off_forks"]) if frozen[key]["off_forks"] else None
                ),
                "offreference_optimizer_draws": item["actual_offreference_optimizer_draws"],
                "sham_optimizer_draws": item["actual_sham_optimizer_draws"],
                "checkpoint_sha256": cp_sha,
            })
        if len(initial) != 1:
            raise ValueError("offreference same-seed initial weights differ")
    if any(roots != all_roots[0] for roots in all_roots[1:]):
        raise ValueError("native proof cohorts differ by training arm")
    if {r["root"] for r in cal_census} != all_roots[0]:
        raise ValueError("native reference correction cohort differs from evaluated proof roots")
    uniform = _read(random_pack / "manifest.json")
    uniform_replay = _read(random_pack / "replay" / "replay.json")
    _hash_artifacts(random_pack, uniform["artifacts"])
    if not (
        uniform.get("schema") == "lsa02f.uniform_menu_native.v1"
        and uniform.get("source_teacher_sha256") == teacher_sha
        and uniform.get("independent_calibration_roots") == 32
        and uniform.get("seed_roster") == [17,31,47,61,83]
        and uniform.get("native_model_calls") == 0
        and uniform.get("all_replays_passed") is True
        and uniform.get("backend", {}).get("actual_lean_process") is True
        and uniform_replay.get("all_replays_passed") is True
        and uniform_replay.get("verified_episodes") == 160
        and uniform_replay.get("session_manifest_sha256") == sha256_file(random_pack / "manifest.json")
    ):
        raise ValueError("independent uniform native reference lacks verified custody")
    random_roots = _rows(random_pack / "root_scores.jsonl")
    if len(random_roots) != 32 or {r["root"] for r in random_roots} != all_roots[0]:
        raise ValueError("random-menu native theorem roster differs from neural cohort")
    random_by_root = {r["root"]:r["mean_seed_closure"] for r in random_roots}

    root_table = []
    for row in cal_census:
        root = row["root"]
        seed_views = []
        for seed in SEEDS:
            base = frozen[f"{seed}-reference"]["proofs"][root]
            for arm in ARMS:
                x = frozen[f"{seed}-{arm}"]["proofs"][root]
                if (
                    x["lineage"] != base["lineage"]
                    or x["reference_budget_attainable"] != base["reference_budget_attainable"]
                    or x["auto8_certified"] != base["auto8_certified"]
                ):
                    raise ValueError("same-root native proof authority differs among arms")
            seed_views.append({
                "seed": seed,
                "proofs": {a:int(frozen[f"{seed}-{a}"]["proofs"][root]["ab_certified"]) for a in ARMS},
                "reference_forks": {
                    a:frozen[f"{seed}-{a}"]["reference"][root]["accuracy"] for a in ARMS
                } if row["reference_forks"] else {},
                "offpath_states": {
                    a:frozen[f"{seed}-{a}"]["off"][root]["accuracy"] for a in ARMS
                } if row["offpath_correction_states"] else {},
                "offpath_forks": {
                    a:frozen[f"{seed}-{a}"]["off_forks"][root]["accuracy"] for a in ARMS
                } if row["offpath_discriminating_forks"] else {},
            })
        root_table.append({
            "root": root, "lineage": row["lineage"],
            "random_mean": random_by_root[root],
            "reference_within_budget": row["within_budget"],
            "calibration_offreference_states": row["offpath_correction_states"],
            "calibration_offreference_forks": row["offpath_discriminating_forks"],
            "seed_views": seed_views,
        })

    def effect(field: str, arm: str = "offpath", control: str = "reference"):
        deltas=[]
        for row in root_table:
            sample=[]
            for entry in row["seed_views"]:
                vals=entry[field]
                if vals:
                    sample.append(vals[arm] - vals[control])
            if sample:
                deltas.append(sum(sample)/len(sample))
        return {
            "root_count": len(deltas),
            "mean": sum(deltas)/len(deltas) if deltas else None,
            "descriptive_root_bootstrap95": _bootstrap(deltas),
        }

    proof_rate = {
        arm: sum(c["full_proofs"] for c in cells if c["arm"] == arm)/(32*len(SEEDS))
        for arm in ARMS
    }
    off_count = calibration["offreference_discriminating_fork_frames"]
    off_roots = calibration["offreference_discriminating_fork_roots"]
    effects = {
        "offpath_minus_reference_full_proof": effect("proofs"),
        "offpath_minus_sham_full_proof": effect("proofs", control="sham"),
        "offpath_minus_reference_on_offpath_forks": effect("offpath_forks"),
        "offpath_minus_sham_on_offpath_forks": effect("offpath_forks", control="sham"),
        "offpath_minus_reference_on_offpath_all": effect("offpath_states"),
        "offpath_minus_reference_on_reference_forks": effect("reference_forks"),
    }
    eligible = off_count >= 16 and off_roots >= 8
    screen = bool(
        eligible
        and effects["offpath_minus_reference_on_offpath_forks"]["mean"] is not None
        and effects["offpath_minus_sham_on_offpath_forks"]["mean"] is not None
        and effects["offpath_minus_reference_on_offpath_forks"]["mean"] >= 0.05
        and effects["offpath_minus_sham_on_offpath_forks"]["mean"] >= 0.05
        and effects["offpath_minus_reference_full_proof"]["mean"] >= 0
        and effects["offpath_minus_sham_full_proof"]["mean"] >= 0
    )
    random_rate=uniform["uniform_mean_closure_rate"]
    competence=max(proof_rate.values())>=.40 and max(proof_rate.values())-random_rate>=.10
    report = {
        "schema": SCHEMA,
        "teacher_sha256": teacher_sha,
        "offreference_source_sha256": off_sha,
        "native_calibration_source_sha256": sha256_file(cal_source / "manifest.json"),
        "uniform_manifest_sha256": sha256_file(random_pack / "manifest.json"),
        "training_manifest_sha256": sha256_file(fits / "manifest.json"),
        "train_roots": teacher_meta["independent_lineages"],
        "train_reference_frames": len(original),
        "train_offreference_states": sum(len(r) for r in off_pool.values()),
        "train_offreference_roots": len(off_pool),
        "calibration_unique_roots": len(cal_census),
        "calibration_reference_fork_roots": calibration["reference_fork_roots"],
        "calibration_reference_forks": calibration["reference_fork_frames"],
        "calibration_offreference_roots": calibration["offreference_correction_roots"],
        "calibration_offreference_states": len(off_frames),
        "calibration_offreference_fork_roots": off_roots,
        "calibration_offreference_forks": off_count,
        "offpath_screen_eligible": eligible,
        "model_seeds": list(SEEDS), "arms":list(ARMS), "view":VIEW,
        "native_close_rates": proof_rate,
        "native_uniform_mean_rate": random_rate,
        "random_seeds":uniform["seed_roster"],
        "native_reference_attainable":sum(r["within_budget"] for r in cal_census),
        "effects": effects,
        "predeclared_development_screen_passed": screen,
        "formal_competence_floor_met": competence,
        "formal_competence_gate": (
            "NEEDS_NEW_SEALED_CONFIRMATION" if competence
            else "BLOCKED_INCOMPETENT_CHECKPOINT"
        ),
        "cells":cells, "root_results":root_table,
        "calibration_labels_collected_after_all_native_model_scoring": True,
        "calibration_optimizer_updates":0,
        "locked_test_or_ood_opened":False,
        "scientific_eligible":False,
        "architecture_promotion_eligible":False,
        "caveat": "Native Lean recoverable off-reference training; all correction-state statistics are root-clustered. Sparse fork coverage fails closed.",
    }
    out.mkdir(parents=True)
    _write(out / "matched_offreference.json", report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--supplement",type=Path,required=True)
    p.add_argument("--fits",type=Path,required=True)
    p.add_argument("--probes",type=Path,required=True)
    p.add_argument("--uniform",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--request-timeout",type=int,default=60)
    a=p.parse_args(argv)
    with LeanRepl(timeout=a.request_timeout,exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR,env=config.lean_env()) as repl:
        ref=collect(repl,a.suite,a.teacher,a.reference,certified_backend=True)
    result=aggregate(a.teacher,a.supplement,a.fits,a.probes,
                     a.uniform,a.reference,a.out)
    print(canonical({
        "calibration_roots":result["calibration_unique_roots"],
        "offreference_states":result["calibration_offreference_states"],
        "offreference_forks":result["calibration_offreference_forks"],
        "offreference_fork_roots":result["calibration_offreference_fork_roots"],
        "root_paired_proof_rates":result["native_close_rates"],
        "uniform_native_rate":result["native_uniform_mean_rate"],
        "screen_eligible":result["offpath_screen_eligible"],
        "screen":result["predeclared_development_screen_passed"],
        "effects":result["effects"],
    }))
    return result


if __name__=="__main__":
    main()
