"""LSA-02J: native Lean 16-work-unit branch utility versus controls.

All search decisions use public Lean state + frozen learned tensors. The
independently compiled reference graph is queried only *after* all model
and no-model policies have finished. Each policy's action/backtrack journal
is independently replayed through the actual Lean 4.19 process.

The strong public_horn_dfs control is an explicit classical proof-search
algorithm over the *visible* local hypothesis types, independent of any
learned model, training target or native hidden oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl

from .backtrack_search import BUDGET, run_episode, replay_episode
from .backtrack_campaign import _certify_and_replay, _native_fingerprint, _source
from .branch_utility import BranchUtility, SCHEMA as HEAD_SCHEMA
from .branch_utility_train import SCHEMA as TRAIN_SCHEMA, SEEDS, VIEW, MODES
from .cli import checkpoint_load
from .data import canonical, sha256_file
from .native_order import FrozenOrder, _checkpoint_provenance, _weight_fingerprint


SCHEMA = "lsa02j.native_utility_branch_dfs.v1"
REPLAY_SCHEMA = "lsa02j.verified_branch_replay.v1"
POLICIES = ("frozen_dfs", "utility_dfs", "sham_utility_dfs", "public_horn_dfs")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, obj: dict) -> None:
    path.write_text(canonical(obj) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_head(path: Path, *, seed: int, mode: str, backbone: Path,
               training: dict, label_manifest: Path) -> BranchUtility:
    meta = torch.load(path, map_location="cpu", weights_only=True)
    if not (
        meta.get("schema") == HEAD_SCHEMA
        and meta.get("mode") == mode and meta.get("seed") == seed
        and meta.get("backbone_checkpoint_sha256") == sha256_file(backbone)
        and meta.get("native_teacher_sha256") == training["native_teacher_sha256"]
        and meta.get("viability_source_sha256") == sha256_file(label_manifest)
        and meta.get("optimizer_steps") == 480
        and meta.get("architecture") == {"width":32,"public_features":8,"hidden":24}
    ):
        raise ValueError("learned branch utility head is not attested to frozen train-only source")
    expected = next((x for x in training["cells"]
                     if x["seed"] == seed and x["mode"] == mode), None)
    if expected is None or expected["head_sha256"] != sha256_file(path) or not expected["duplicate_bitwise"]:
        raise ValueError("branch head does not match duplicate verified train receipt")
    head = BranchUtility(width=32)
    head.load_state_dict(meta["weights"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def run(
    repl, suite: Path, teacher: Path, label_manifest: Path,
    trained: Path, checkpoint: Path, out: Path,
    *, seed: int, effort: int = BUDGET,
    roots: int = 32, max_seconds: float = 900,
    cert_timeout: int = 420, certified_backend: bool = False,
) -> dict:
    if out.exists() or seed not in SEEDS or roots != 32 or effort != BUDGET:
        raise ValueError("registered two-seed 32-root 16-unit native utility run required")
    if max_seconds <= 0 or not 1 <= cert_timeout <= 900:
        raise ValueError("invalid CPU native effort or compiler budget")
    cases, source, teacher_meta = _source(suite, teacher, roots)
    training = _json(trained / "manifest.json")
    if not (
        training.get("schema") == TRAIN_SCHEMA
        and training.get("native_teacher_sha256") == sha256_file(teacher / "manifest.json")
        and training.get("branch_labels_sha256") == sha256_file(label_manifest)
        and training.get("suite_manifest_sha256") == source["suite_manifest_sha256"]
        and training.get("backbone_seed_roster") == list(SEEDS)
        and training.get("head_mode_roster") == list(MODES)
        and training.get("all_duplicate_fits_bitwise") is True
        and training.get("same_worker") is True
        and training.get("frozen_backbone") is True
        and training.get("native_calibration_updates") == 0
        and training.get("test_or_ood_opened") is False
    ):
        raise ValueError("native source-trained probe and frozen backbone provenance failed")
    selected_backbone = next(
        (x for x in _json(trained / "backbones" / "manifest.json")["cells"]
         if x["seed"] == seed), None
    )
    if selected_backbone is None or selected_backbone["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("frozen Plan B checkpoint has not passed same-worker repeat")
    model, check = checkpoint_load(checkpoint)
    if (
        model.cfg.plan != "b" or model.cfg.width != 32
        or model.cfg.max_bytes != 384 or model.parameter_count() != 57475
        or check.get("seed") != seed or check.get("input_view") != VIEW
        or check.get("train_order") != "ab" or check.get("train_steps") != 720
    ):
        raise ValueError("registered pretrain Plan B architecture or source identity changed")
    provenance = _checkpoint_provenance(
        check, None, cases, native_teacher=teacher / "manifest.json",
    )
    if not provenance["training_source_verified"] or not provenance["source_lineages_disjoint"]:
        raise ValueError("native frozen training roots overlap calibration theorem lineage")
    if _json(label_manifest)["split"] != "train":
        raise ValueError("branch viability source is not train-only")
    heads = {
        "utility_dfs": _load_head(
            trained / "heads" / f"{seed}-viability" / "head.pt",
            seed=seed, mode="viability", backbone=checkpoint,
            training=training, label_manifest=label_manifest,
        ),
        "sham_utility_dfs": _load_head(
            trained / "heads" / f"{seed}-shuffled_sham" / "head.pt",
            seed=seed, mode="shuffled_sham", backbone=checkpoint,
            training=training, label_manifest=label_manifest,
        ),
    }
    backend = _native_fingerprint(repl, certified_backend)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(1)
    adapter = FrozenOrder(model, "ab", view=VIEW)
    original = _weight_fingerprint(model)
    out.mkdir(parents=True)
    events = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for case in cases:
            if time.perf_counter() - t0 > max_seconds:
                raise TimeoutError("native ranked DFS case evaluation exceeded CPU budget")
            for policy in POLICIES:
                ep = run_episode(
                    repl, case, None if policy == "public_horn_dfs" else adapter,
                    policy=policy, effort_budget=effort, depth=2,
                    utility=heads.get(policy),
                )
                ep["checkpoint_sha256"] = sha256_file(checkpoint)
                ep["utility_head_sha256"] = (
                    sha256_file(
                        trained / "heads" /
                        f"{seed}-{'viability' if policy=='utility_dfs' else 'shuffled_sham'}" /
                        "head.pt"
                    ) if policy in heads else None
                )
                events.append(ep)
    # The reference oracle is *never* available until after all 128 attempts.
    episodes, witnesses, receipt = _certify_and_replay(
        repl, cases, events, out, oracle=True,
        certified_backend=certified_backend,
        cert_timeout=cert_timeout,
    )
    if original != _weight_fingerprint(model):
        raise ValueError("native branch-utility inference updated frozen Plan B model")
    by_root = {}
    for ep in episodes:
        by_root.setdefault(ep["root"], {})[ep["policy"]] = ep
    if len(by_root) != roots or any(set(x) != set(POLICIES) for x in by_root.values()):
        raise ValueError("native model/policy proof roster is missing roots")
    oracle = {w["root"]: w for w in witnesses}
    scores = {}
    for policy in POLICIES:
        values = [e for e in episodes if e["policy"] == policy]
        scores[policy] = {
            "episodes": len(values),
            "native_kernel_closures": sum(e["live_kernel"] for e in values),
            "standalone_certified_closures": sum(e["standalone_verified"] for e in values),
            "work_units": sum(e["effort_spent"] for e in values),
            "tactic_attempts": sum(e["tactic_checks"] for e in values),
            "backtracks": sum(e["backtracks"] for e in values),
            "model_calls": sum(e["model_calls"] for e in values),
        }
    roots_report = []
    for case in cases:
        ref = oracle[case.root]
        if not ref["standalone_verified"]:
            raise ValueError("native oracle proof failed standalone Lean compilation")
        roots_report.append({
            "root": case.root, "lineage": case.lineage,
            "reference_proof_size": ref["proof_size"],
            "reference_attainable": (
                ref["menu_compatible"] and ref["fits_model_tactic_budget"]
            ),
            "policies": {
                p: {
                    "closed": by_root[case.root][p]["standalone_verified"],
                    "effort": by_root[case.root][p]["effort_spent"],
                    "backtracks": by_root[case.root][p]["backtracks"],
                    "model_calls": by_root[case.root][p]["model_calls"],
                } for p in POLICIES
            },
        })
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as writer:
        for ep in episodes:
            writer.write(canonical(ep) + "\n")
    with (out / "roots.jsonl").open("x", encoding="utf-8") as writer:
        for row in roots_report:
            writer.write(canonical(row) + "\n")
    _write(out / "compiler_receipt.json", receipt)
    manifest = {
        "schema": SCHEMA,
        "status": "complete" if certified_backend else "synthetic_only",
        "backend": backend,
        "source": source,
        "seed": seed,
        "control": "model",
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest_sha256": sha256_file(trained / "manifest.json"),
        "native_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "branch_source_sha256": sha256_file(label_manifest),
        "utility_head_sha256": sha256_file(trained / "heads" / f"{seed}-viability" / "head.pt"),
        "sham_head_sha256": sha256_file(trained / "heads" / f"{seed}-shuffled_sham" / "head.pt"),
        "checkpoint_provenance": provenance,
        "independent_theorem_roots": len(roots_report),
        "standalone_reference_certificates": sum(w["standalone_verified"] for w in witnesses),
        "independent_reference_attainable": sum(r["reference_attainable"] for r in roots_report),
        "effort_budget": effort,
        "tactic_work_cost": 1,
        "backtrack_work_cost": 1,
        "policy_roster": list(POLICIES),
        "scores": scores,
        "all_native_episode_replays_passed": True,
        "reference_oracle_available_only_postscoring": True,
        "frozen_backbone": True,
        "frozen_utility_heads": True,
        "native_calibration_optimizer_steps": 0,
        "locked_test_opened": False,
        "scientific_eligible": False,
        "architecture_promotion_eligible": False,
        "artifacts": {
            n: sha256_file(out / n)
            for n in ("episodes.jsonl", "roots.jsonl", "compiler_receipt.json",
                      "standalone.lean")
        },
    }
    _write(out / "manifest.json", manifest)
    return manifest


def replay(repl, suite: Path, session: Path, out: Path,
           *, certified_backend=False):
    if out.exists():
        raise ValueError("fresh replay output required")
    manifest = _json(session / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("unknown/incomplete branch-utility native session")
    if manifest["backend"] != _native_fingerprint(repl, certified_backend):
        raise ValueError("native replay Lean toolchain hash mismatch")
    for name, digest in manifest["artifacts"].items():
        if name != Path(name).name or sha256_file(session / name) != digest:
            raise ValueError("native branch-utility episode artifact was modified")
    cases, source = load_suite(suite, "calibration", manifest["independent_theorem_roots"])
    if source["suite_manifest_sha256"] != manifest["source"]["suite_manifest_sha256"]:
        raise ValueError("frozen calibration theorem source changed")
    by_root = {case.root: case for case in cases}
    ep = _rows(session / "episodes.jsonl")
    if len(ep) != len(cases)*len(POLICIES):
        raise ValueError("independent replay episode census differs")
    for row in ep:
        case = by_root.get(row["root"])
        if case is None or row["lineage"] != case.lineage:
            raise ValueError("replay encountered unknown theorem lineage")
        replay_episode(repl, case, row)
    out.mkdir(parents=True)
    receipt = {
        "schema": "lsa02j.native_independent_replay.v1",
        "session_manifest_sha256": sha256_file(session / "manifest.json"),
        "backend": manifest["backend"],
        "verified_episodes": len(ep),
        "all_replays_passed": True,
        "model_calls_during_replay": 0,
    }
    _write(out / "replay.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run")
    for name in ("suite","teacher","branch-label-manifest","trained","checkpoint","out"):
        r.add_argument("--"+name, type=Path, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--roots", type=int, default=32)
    r.add_argument("--effort", type=int, default=BUDGET)
    r.add_argument("--max-seconds", type=float, default=900)
    r.add_argument("--cert-timeout", type=int, default=420)
    rep = commands.add_parser("replay")
    for name in ("suite","session","out"):
        rep.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--request-timeout", type=int, default=60)
    args = p.parse_args(argv)
    if not config.LEAN_EXE.is_file() or not config.REPL_EXE.is_file():
        raise RuntimeError("expected pinned native Lean 4.19 verifier executable")
    with LeanRepl(timeout=args.request_timeout, exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR, env=config.lean_env()) as repl:
        if args.command == "run":
            report = run(
                repl, args.suite, args.teacher, args.branch_label_manifest,
                args.trained, args.checkpoint, args.out,
                seed=args.seed, effort=args.effort, roots=args.roots,
                max_seconds=args.max_seconds, cert_timeout=args.cert_timeout,
                certified_backend=True,
            )
        else:
            report = replay(repl, args.suite, args.session, args.out,
                            certified_backend=True)
    print(canonical({
        "schema": report["schema"],
        "roots": report.get("independent_theorem_roots"),
        "seed": report.get("seed"),
        "scores": report.get("scores"),
        "all_replays": report.get("all_native_episode_replays_passed"),
    }))
    return report


if __name__ == "__main__":
    main()
