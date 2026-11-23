"""LSA-02B native AB/BA proof-order study.

A frozen PPT-B checkpoint is evaluated on the same independently generated
Horn theorem roots with both block orders. Lean alone owns tactic outcomes.
Every closure is submitted for standalone kernel compilation and all action
traces are independently replayed without model calls.

This is an exploratory, *checkpoint-conditional* study. It does not authorize
a population-level architecture comparison or use historical archive outcomes
as training labels. The FakeLean harness is explicitly non-scientific.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path

import torch

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.extract import explore
from lcgrt.lean_repl import LeanRepl
from .cli import checkpoint_load
from .data import canonical, ensure_no_overlap, load_checked_pack, sha256_file
from .live_lean import candidate_menu, fingerprint, replay_episode, run_episode
from .live_suite import load_suite
from .typed_state import encode_public_packet, SCHEMA as TYPED_STATE_SCHEMA, VIEWS
from .lbf_hybrid import SCHEMA as LBF_HYBRID_SCHEMA


SCHEMA = "lsa02b.native_block_order.v1"
ORDERS = ("ab", "ba", "auto8")
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261123


class FrozenOrder:
    """No copied parameters, gradients, adapters, mutable fast state or data labels."""

    def __init__(self, model, order: str, view: str = "raw"):
        if model.cfg.plan != "b" or order not in ("ab", "ba") or view not in VIEWS:
            raise ValueError("native order requires a PPT-B model, ab|ba and a valid input view")
        self.model = model
        self.cfg = model.cfg
        self.order = order
        self.view = view

    def __call__(self, goal, candidates, **kwargs):
        text = encode_public_packet(
            goal, candidates, view=self.view, max_bytes=self.cfg.max_bytes,
        )
        return self.model(text, candidates, middle_order=self.order, **kwargs)


def _json(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _weight_fingerprint(model) -> str:
    """Hash frozen parameter names, shapes, values and buffers using only PyTorch."""
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _interval(deltas: list[int]) -> list[float] | None:
    if len(deltas) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    size = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(BOOTSTRAPS)
    )
    return [means[int(BOOTSTRAPS * 0.025)], means[int(BOOTSTRAPS * 0.975) - 1]]


def _checkpoint_provenance(checkpoint: dict, training_pack: Path | None, cases,
                           native_teacher: Path | None = None) -> dict:
    """Keep prior training ancestry independent of native query proof outcomes."""
    result = {
        "checkpoint_evidence_class": checkpoint.get("source_evidence_class", "unknown"),
        "training_pack_provided": training_pack is not None,
        "training_source_verified": False,
        "source_lineages_disjoint": None,
        "train_steps": checkpoint.get("train_steps"),
        "train_order": checkpoint.get("train_order", "ab"),
    }
    if native_teacher is not None:
        if training_pack is not None:
            raise ValueError("choose either checked-pack provenance or native teacher")
        teacher = json.loads(native_teacher.read_text(encoding="utf-8"))
        if (
            teacher.get("schema") != "lsa02b.native_teacher.v1"
            or teacher.get("split") != "train"
            or teacher.get("evidence_class") != "lean_kernel_verified_reference_path"
            or teacher.get("backend", {}).get("actual_lean_process") is not True
            or teacher.get("source_compiler_passed") is not True
            or teacher.get("query_roots_exposed_to_fit") != 0
            or checkpoint.get("native_teacher_sha256") != sha256_file(native_teacher)
            or checkpoint.get("source_manifest_sha256") !=
               teacher.get("source", {}).get("suite_manifest_sha256")
            or checkpoint.get("source_evidence_class") != teacher.get("evidence_class")
            or checkpoint.get("train_steps", 0) <= 0
        ):
            raise ValueError("native teacher provenance does not match checkpoint")
        for filename, expected in teacher.get("artifacts", {}).items():
            if filename != Path(filename).name or sha256_file(native_teacher.parent / filename) != expected:
                raise ValueError("native teacher artifact hash differs from source")
        if "train_census.jsonl" not in teacher.get("artifacts", {}):
            raise ValueError("teacher source has no theorem-lineage census")
        census = _rows(native_teacher.parent / "train_census.jsonl")
        lineages = [row["lineage"] for row in census]
        if len(set(lineages)) != teacher["independent_lineages"]:
            raise ValueError("teacher train lineage census is inconsistent")
        if set(lineages) & {c.lineage for c in cases}:
            raise ValueError("native training and query lineages overlap")
        result["training_source_verified"] = True
        result["source_lineages_disjoint"] = True
        result["training_source_manifest_sha256"] = teacher["source"]["suite_manifest_sha256"]
        result["native_teacher_manifest_sha256"] = sha256_file(native_teacher)
        result["training_lineages"] = len(lineages)
        result["training_evidence"] = "native Lean shortest-continuation labels"
        return result
    if training_pack is None:
        return result
    source_cases, meta = load_checked_pack(training_pack, "train", 20000)
    if checkpoint.get("source_manifest_sha256") != meta["manifest_sha256"]:
        raise ValueError("training pack manifest differs from trained checkpoint")
    if checkpoint.get("source_evidence_class") != meta["evidence_class"]:
        raise ValueError("checkpoint training evidence differs from training pack")
    if not source_cases:
        raise ValueError("training pack has no training roots")
    ensure_no_overlap(source_cases, cases)
    result["training_source_verified"] = bool(meta["scientific_eligible"])
    result["source_lineages_disjoint"] = True
    result["training_source_manifest_sha256"] = meta["manifest_sha256"]
    result["training_lineages"] = len({c.lineage for c in source_cases})
    return result


def _oracle_witness(repl, case, check_budget: int) -> dict:
    """Native exact proof witness, evaluated only after the neural arms are frozen.

    The full Lean-reported AND/OR graph supplies the shortest reference script.
    Recheck that *every* tactic of that script exists in the model's public
    candidate grammar. Reference search costs are kept distinct from inference.
    """
    reference = explore(repl, case.theorem())
    script = reference["proof"]["script"]
    if not script or not reference["proof"]["repl_kernel"]["ok"]:
        raise ValueError("native generated theorem has no certified proof witness")
    repl.reset()
    state = repl.init(case.statement, pp=True)
    menu_compatible = True
    for tactic in script:
        menu_compatible = menu_compatible and tactic in candidate_menu(repl, state)
        result = repl.run(state["sid"], tactic)
        if not result.get("ok"):
            raise ValueError("native reference proof failed under public menu replay")
        state = result["state"]
    if state.get("goals"):
        raise ValueError("native reference proof left open goals")
    cert = repl.kernel(state["sid"])
    if not isinstance(cert, dict) or not cert.get("ok") or cert.get("axioms") != []:
        raise ValueError("native structural reference failed terminal kernel")
    return {
        "root": case.root,
        "lineage": case.lineage,
        "proof_size": len(script),
        "menu_compatible": menu_compatible,
        "fits_model_tactic_budget": len(script) <= check_budget,
        "reference_script": script,
        "native_repl_kernel": True,
        "standalone_certificate": None,
        "standalone_verified": False,
    }


def _native_fingerprint(repl, certified_backend: bool) -> dict:
    if certified_backend:
        if not isinstance(repl, LeanRepl) or not str(repl.version).startswith("4.19.0"):
            raise ValueError("certified mode requires the actual pinned Lean 4.19.0 process")
        return fingerprint(config.REPL_EXE, config.LEAN_EXE, str(repl.version))
    return {
        "lean_version": str(getattr(repl, "version", "unknown")),
        "source": "simulated protocol test",
        "actual_lean_process": False,
        "repl_bin_sha256": None,
        "lean_bin_sha256": None,
    }


def run(
    repl,
    suite: Path,
    checkpoint_path: Path,
    out: Path,
    *,
    split: str = "calibration",
    roots: int = 8,
    checks: int = 8,
    depth: int = 2,
    max_seconds: float = 360.0,
    cert_timeout: int = 120,
    training_pack: Path | None = None,
    native_teacher: Path | None = None,
    certified_backend: bool = False,
    certifier=certify,
) -> dict:
    if out.exists() or roots not in range(1, 1025):
        raise ValueError("fresh output directory and 1..1024 roots required")
    if split not in ("calibration", "ordinary_test", "ood_family"):
        raise ValueError("unrecognized generated Lean query split")
    if split != "calibration":
        raise ValueError("development-only runner: locked and OOD splits require a new sealed gate")
    if checks not in range(1, 33) or depth not in range(1, 17):
        raise ValueError("invalid tactic or recurrence budget")
    if max_seconds <= 0 or cert_timeout not in range(1, 901):
        raise ValueError("positive bounded compute and certificate time required")
    backend = _native_fingerprint(repl, certified_backend)
    model, checkpoint = checkpoint_load(checkpoint_path)
    if model.cfg.plan != "b":
        raise ValueError("native block ordering requires a two-block Plan B checkpoint")
    if checkpoint.get("train_order", "ab") not in ("ab", "ba"):
        raise ValueError("unsupported training order in checkpoint")
    input_view = checkpoint.get("input_view", "raw")
    if input_view not in VIEWS or (
        (input_view == "typed" and checkpoint.get("input_view_schema") != TYPED_STATE_SCHEMA)
        or (input_view == "hybrid" and checkpoint.get("input_view_schema") != LBF_HYBRID_SCHEMA)
    ):
        raise ValueError("checkpoint public-state view is unrecognized")
    torch.set_num_threads(1)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    selected, source = load_suite(suite, split, roots)
    training, _ = load_suite(suite, "train", 1024)
    if {c.lineage for c in training} & {c.lineage for c in selected}:
        raise ValueError("training suite lineages overlap queried Lean theorem roots")
    provenance = _checkpoint_provenance(checkpoint, training_pack, selected, native_teacher)
    checkpoint_hash = sha256_file(checkpoint_path)
    initial_weights = _weight_fingerprint(model)
    out.mkdir(parents=True)
    begun = time.perf_counter()
    observations: list[dict] = []
    case_by_root = {case.root: case for case in selected}
    exhausted = False
    with torch.no_grad():
        for case_index, case in enumerate(selected):
            if time.perf_counter() - begun >= max_seconds:
                exhausted = True
                break
            # Alternate arm order to reduce a systematic host-cache/time bias.
            arms = ("ab", "ba", "auto8") if case_index % 2 == 0 else ("ba", "ab", "auto8")
            for order in arms:
                if time.perf_counter() - begun >= max_seconds:
                    exhausted = True
                    break
                adapter = FrozenOrder(
                    model, order if order != "auto8" else "ab", view=input_view,
                )
                ep = run_episode(
                    repl, case, adapter,
                    policy="auto8" if order == "auto8" else "frozen",
                    checks=checks, depth=depth,
                )
                ep["order"] = order
                ep["max_checks"] = checks
                ep["checkpoint_sha256"] = checkpoint_hash
                observations.append(ep)
            if exhausted:
                break

    # Only fully paired roots enter scores or certificates.
    counts: dict[str, int] = {}
    for episode in observations:
        counts[episode["root"]] = counts.get(episode["root"], 0) + 1
    paired = {root for root, count in counts.items() if count == len(ORDERS)}
    complete = [ep for ep in observations if ep["root"] in paired]
    proofs = []
    named = {}
    for ep in complete:
        if ep["live_kernel"]:
            case = case_by_root[ep["root"]]
            name = f"{case.tid}_{ep['order']}"
            qualified = decl_name(name)
            if qualified in named:
                raise ValueError("duplicate standalone proof declaration")
            named[qualified] = ep
            proofs.append((name, case.theorem(), ep["accepted_script"]))
    # Native structural ceiling is never exposed to the model or its action
    # selector. Compute only after all three arms have finished scoring.
    references: list[dict] = []
    oracle_named: dict[str, dict] = {}
    if certified_backend:
        for case in selected:
            if case.root not in paired:
                continue
            if time.perf_counter() - begun >= max_seconds:
                exhausted = True
                break
            witness = _oracle_witness(repl, case, checks)
            name = f"{case.tid}_oracle"
            qualified = decl_name(name)
            if qualified in named or qualified in oracle_named:
                raise ValueError("duplicate reference declaration")
            oracle_named[qualified] = witness
            references.append(witness)
            proofs.append((name, case.theorem(), witness["reference_script"]))
    receipt = {
        "authoritative_only_with_real_lean": certified_backend,
        "proofs_submitted": len(proofs),
        "standalone_passed": 0,
        "compiler_audit": None,
    }
    if proofs:
        audit = certifier(proofs, timeout=cert_timeout, keep=out / "standalone.lean")
        if not isinstance(audit, dict) or not isinstance(audit.get("status"), dict):
            raise ValueError("standalone certifier omitted per-declaration status")
        receipt["compiler_audit"] = audit
        for name, ep in named.items():
            ep["standalone_certificate"] = audit["status"].get(name, "missing")
        for name, witness in oracle_named.items():
            witness["standalone_certificate"] = audit["status"].get(name, "missing")
            witness["standalone_verified"] = witness["standalone_certificate"] == "passed"
            if not witness["standalone_verified"]:
                raise ValueError("model-free native reference failed standalone Lean compilation")
    for ep in complete:
        ep.setdefault("standalone_certificate", None)
        ep["standalone_verified"] = bool(
            certified_backend and ep["live_kernel"] and ep["standalone_certificate"] == "passed"
        )
        ep["replay_passed"] = bool(replay_episode(repl, case_by_root[ep["root"]], ep))
        if certified_backend and ep["live_kernel"] and not ep["standalone_verified"]:
            raise ValueError("a native Lean kernel closure failed standalone certification")
    receipt["standalone_passed"] = sum(ep["standalone_verified"] for ep in complete)
    receipt["reference_proofs_passed"] = sum(w["standalone_verified"] for w in references)
    receipt["reference_proofs_submitted"] = len(references)
    _json(out / "standalone_receipt.json", receipt)

    final_weights = _weight_fingerprint(model)
    if final_weights != initial_weights:
        raise ValueError("frozen backbone weights changed during native order intervention")
    episode_path = out / "paired_episodes.jsonl"
    with episode_path.open("x", encoding="utf-8") as handle:
        for ep in complete:
            handle.write(canonical(ep) + "\n")
    by_root = {}
    for ep in complete:
        by_root.setdefault(ep["root"], {})[ep["order"]] = ep
    deltas = []
    root_rows = []
    reference_by_root = {w["root"]: w for w in references}
    for root, entries in sorted(by_root.items()):
        if set(entries) != set(ORDERS):
            raise ValueError("missing paired arm")
        ab, ba = entries["ab"], entries["ba"]
        delta = int(ab["standalone_verified"]) - int(ba["standalone_verified"])
        deltas.append(delta)
        root_rows.append({
            "root": root, "lineage": ab["lineage"], "family": ab["family"],
            "ab_certified": ab["standalone_verified"],
            "ba_certified": ba["standalone_verified"],
            "auto8_certified": entries["auto8"]["standalone_verified"],
            "delta": delta,
            "reference_proof_size": (
                reference_by_root[root]["proof_size"] if root in reference_by_root else None
            ),
            "reference_budget_attainable": (
                reference_by_root[root]["fits_model_tactic_budget"]
                and reference_by_root[root]["menu_compatible"]
                and reference_by_root[root]["standalone_verified"]
                if root in reference_by_root else None
            ),
            "ab_checks": ab["attempts"], "ba_checks": ba["attempts"],
            "auto8_checks": entries["auto8"]["attempts"],
            "ab_tactics": ab["accepted_script"], "ba_tactics": ba["accepted_script"],
            "changed_first_tactic": (
                ab["events"][0]["tactic"] != ba["events"][0]["tactic"]
                if ab["events"] and ba["events"] else None
            ),
        })
    root_path = out / "paired_roots.jsonl"
    with root_path.open("x", encoding="utf-8") as handle:
        for row in root_rows:
            handle.write(canonical(row) + "\n")
    denominator = len(root_rows)
    scores = {
        order: {
            "kernel_closures": sum(ep["live_kernel"] for ep in complete if ep["order"] == order),
            "standalone_verified_closures": sum(
                ep["standalone_verified"] for ep in complete if ep["order"] == order
            ),
            "tactic_checks": sum(ep["attempts"] for ep in complete if ep["order"] == order),
            "model_calls": sum(ep["model_calls"] for ep in complete if ep["order"] == order),
        }
        for order in ORDERS
    }
    native_complete = bool(certified_backend and not exhausted and denominator == len(selected)
                           and len(references) == denominator)
    report = {
        "schema": SCHEMA,
        "status": "complete" if native_complete else "engineering_only_or_partial",
        "source": source,
        "backend": backend,
        "checkpoint_sha256": checkpoint_hash,
        "input_view": input_view,
        "input_view_schema": TYPED_STATE_SCHEMA if input_view == "typed" else LBF_HYBRID_SCHEMA if input_view == "hybrid" else "raw-legacy-json",
        "frozen_weights_sha256": final_weights,
        "training_provenance": provenance,
        "requested_roots": len(selected),
        "paired_roots": denominator,
        "split": split,
        "check_budget": checks,
        "depth": depth,
        "ab_ba_intervention": "same_model_same_weights_same_passes",
        "root_alternating_order": True,
        "scores": scores,
        "independent_reference_proofs": len(references),
        "standalone_reference_certificates": sum(w["standalone_verified"] for w in references),
        "reference_reachable_within_budget": sum(
            w["standalone_verified"] and w["fits_model_tactic_budget"]
            and w["menu_compatible"] for w in references
        ),
        "mean_delta_ab_minus_ba": (sum(deltas) / denominator if denominator else None),
        "bootstrap_95_descriptive": _interval(deltas),
        "ab_wins": deltas.count(1),
        "ba_wins": deltas.count(-1),
        "ties": deltas.count(0),
        "first_tactic_disagreements": sum(r["changed_first_tactic"] is True for r in root_rows),
        "query_label_training_updates": 0,
        "new_native_lean_calls": sum(ep["attempts"] for ep in complete),
        "timed_out": exhausted,
        "independent_standalone_compile": bool(proofs),
        "scientific_eligible": False,
        "architecture_promotion_eligible": False,
        "interpretation": (
            "Native Lean outcomes and standalone proof compilation are ground truth for this "
            "frozen checkpoint. Source ancestry, checkpoint competence, matched learned AB/BA "
            "seeds and FSI-01 power must be established before a general architecture claim."
        ),
        "artifacts": {
            "paired_episodes.jsonl": sha256_file(episode_path),
            "paired_roots.jsonl": sha256_file(root_path),
            "standalone_receipt.json": sha256_file(out / "standalone_receipt.json"),
        },
    }
    if proofs:
        report["artifacts"]["standalone.lean"] = sha256_file(out / "standalone.lean")
    _json(out / "manifest.json", report)
    return report


def replay(repl, suite: Path, session: Path, out: Path, *, certified_backend=False) -> dict:
    if out.exists():
        raise ValueError("replay output directory must be fresh")
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected native order schema")
    if manifest["backend"]["actual_lean_process"] != certified_backend:
        raise ValueError("backend provenance differs from original session")
    if certified_backend:
        actual = _native_fingerprint(repl, True)
        for field in ("lean_version", "repl_bin_sha256", "lean_bin_sha256"):
            if manifest["backend"][field] != actual[field]:
                raise ValueError("native Lean replay binary changed")
    source, context = load_suite(suite, manifest["split"], 1024)
    if context["suite_manifest_sha256"] != manifest["source"]["suite_manifest_sha256"]:
        raise ValueError("native suite manifest changed after scoring")
    by_root = {c.root: c for c in source}
    for filename, expected in manifest["artifacts"].items():
        if sha256_file(session / filename) != expected:
            raise ValueError(f"native order artifact hash changed: {filename}")
    episodes = _rows(session / "paired_episodes.jsonl")
    if len(episodes) != 3 * manifest["paired_roots"]:
        raise ValueError("incomplete paired root event set")
    counts = {}
    for ep in episodes:
        case = by_root.get(ep["root"])
        if not case or ep["lineage"] != case.lineage or ep["split"] != case.split:
            raise ValueError("unknown or mismatched native theorem lineage")
        if ep["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError("native checkpoint identity changed in episode")
        replay_episode(repl, case, ep)
        counts.setdefault(ep["root"], []).append(ep["order"])
    if any(sorted(arms) != sorted(ORDERS) for arms in counts.values()):
        raise ValueError("root arms lost pairing")
    result = {
        "schema": "lsa02b.native_replay.v1",
        "replay_passed": True,
        "independent_replay_episodes": len(episodes),
        "paired_roots": len(counts),
        "backend": _native_fingerprint(repl, certified_backend),
        "session_manifest_sha256": sha256_file(session / "manifest.json"),
        "model_used_in_replay": False,
        "scientific_eligible": False,
    }
    out.mkdir(parents=True)
    _json(out / "replay.json", result)
    return result


def _repl(timeout: int):
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise RuntimeError("pinned Lean/lgym missing; build research/lean_cgrt_hrm400/lean")
    return LeanRepl(timeout=timeout, exe=config.REPL_EXE,
                    cwd=config.LEAN_DIR, env=config.lean_env())


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="Evaluate a frozen PPT-B on fresh native Lean roots")
    run_cmd.add_argument("--suite", type=Path, required=True)
    run_cmd.add_argument("--checkpoint", type=Path, required=True)
    run_cmd.add_argument("--training-pack", type=Path)
    run_cmd.add_argument("--native-teacher", type=Path, help="frozen native teacher manifest.json")
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--split", choices=("calibration", "ordinary_test", "ood_family"),
                         default="calibration")
    run_cmd.add_argument("--roots", type=int, default=8)
    run_cmd.add_argument("--checks", type=int, default=8)
    run_cmd.add_argument("--depth", type=int, default=2)
    run_cmd.add_argument("--max-seconds", type=float, default=360.0)
    run_cmd.add_argument("--cert-timeout", type=int, default=120)
    replay_cmd = commands.add_parser("replay", help="Independently replay every native Lean action")
    replay_cmd.add_argument("--suite", type=Path, required=True)
    replay_cmd.add_argument("--session", type=Path, required=True)
    replay_cmd.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    with _repl(args.request_timeout) as repl:
        if args.command == "run":
            result = run(
                repl, args.suite, args.checkpoint, args.out,
                split=args.split, roots=args.roots, checks=args.checks,
                depth=args.depth, max_seconds=args.max_seconds,
                cert_timeout=args.cert_timeout,
                training_pack=args.training_pack, native_teacher=args.native_teacher,
                certified_backend=True,
            )
        else:
            result = replay(repl, args.suite, args.session, args.out,
                            certified_backend=True)
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()
