"""PPT-G03 fresh proof-search campaign and independent Lean-certificate replay.

The support split supplies feedback for DRAM fast heads. Query split never
updates weights. Policy arms share the *same* Lean tactic-call allowance.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import torch

from lcgrt.certify import certify, decl_name
from .core import FastHead, PlasticProofTransformer, RetrievalControl
from .data import canonical, sha256_file
from .live_lean import replay_episode, run_episode
from .accretion import state_hash
from .live_suite import LiveCase, load_suite

SCHEMA = "ppt_gym.live_campaign.v1"
POLICIES = ("auto8", "auto4_then_ppt", "frozen", "dram", "retrieval", "sham")


def _save_json(path: Path, body):
    path.write_text(canonical(body) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("x", encoding="utf-8") as f:
        for item in rows:
            f.write(canonical(item) + "\n")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def run_campaign(
    repl, model: PlasticProofTransformer, support: list[LiveCase],
    queries: list[LiveCase], *, out: Path, checks: int = 8, depth: int = 3,
    fast_lr: float = 1.0, score_scale: float = 2.0,
    max_seconds: float = 360.0,
    checkpoint: Path | None = None, source: dict | None = None,
    certified_backend: bool = False,
    certifier: Callable = certify, cert_timeout: int = 120,
    allow_partial: bool = True,
):
    if not support or not queries or out.exists():
        raise ValueError("fresh output and nonempty support/query cohorts required")
    if (any(c.split != "train" for c in support)
            or len({c.lineage for c in support + queries}) != len(support + queries)):
        raise ValueError("train-only support and globally disjoint theorem lineages required")
    if not 1 <= checks <= 32 or not 1 <= depth <= 16 or max_seconds <= 0:
        raise ValueError("bounded proof steps, recurrence depth and campaign wall clock required")
    if not 0 < fast_lr <= 20 or not 0 <= score_scale <= 20:
        raise ValueError("invalid fast-weight budget")
    if certified_backend and not str(repl.version).startswith("4.19.0"):
        raise ValueError("PPT-G03 requires pinned Lean 4.19.0")
    out.mkdir(parents=True)
    model.eval()
    memory: dict[str, tuple[FastHead, RetrievalControl, FastHead]] = {}
    start = time.perf_counter()
    support_rows, query_rows = [], []
    aborted = False

    def members(family):
        if family not in memory:
            d = model.cfg.width
            memory[family] = (FastHead(d, fast_lr), RetrievalControl(d), FastHead(d, fast_lr))
        return memory[family]

    for case in support:
        if time.perf_counter() - start >= max_seconds:
            aborted = True
            break
        fast, retrieval, sham = members(case.family)
        ep = run_episode(repl, case, model, policy="frozen",
                         checks=checks, depth=depth, fast=fast, memory=retrieval,
                         sham=sham, scale=score_scale, collect_support=True)
        support_rows.append(ep)

    # Query labels never influence fast weights, retrieval, or the base model.
    if not aborted:
        for case in queries:
            if time.perf_counter() - start >= max_seconds:
                aborted = True
                break
            fast, retrieval, sham = members(case.family)
            for arm in POLICIES:
                if time.perf_counter() - start >= max_seconds:
                    aborted = True
                    break
                ep = run_episode(repl, case, model, policy=arm,
                                 checks=checks, depth=depth,
                                 fast=fast, memory=retrieval, sham=sham,
                                 scale=score_scale)
                query_rows.append(ep)
            if aborted:
                break

    if aborted and not allow_partial:
        raise RuntimeError("campaign exceeded its declared wall-time cap")

    # Second independent Lean check: compile the recorded proof scripts as
    # standalone declarations, with an axiom audit. This certificate is the
    # record of final proof success, beyond the live REPL's kernel check.
    cases_by_root = {c.root: c for c in support + queries}
    episodes = support_rows + query_rows
    proof_items = []
    names = {}
    for ep in episodes:
        if ep["live_kernel"]:
            case = cases_by_root[ep["root"]]
            ident = f"{case.tid}_{ep['policy']}"
            name = decl_name(ident)
            if name in names:
                raise ValueError("duplicate certificate declaration name")
            names[name] = ep
            proof_items.append((ident, case.theorem(), ep["accepted_script"]))
    certificate = None
    if proof_items:
        certificate = certifier(proof_items, timeout=cert_timeout, keep=out / "standalone.lean")
        state = certificate.get("status") if isinstance(certificate, dict) else None
        if not isinstance(state, dict):
            raise ValueError("standalone Lean certification returned no per-theorem status")
        for name, ep in names.items():
            ep["standalone_certificate"] = state.get(name, "missing")
    for ep in episodes:
        ep["max_checks"] = checks
        ep["verified_closed"] = bool(ep["live_kernel"] and ep["standalone_certificate"] == "passed")
    if any(ep["live_kernel"] and not ep["verified_closed"] for ep in support_rows):
        raise ValueError("support learning used a proof failing independent standalone certification")

    _write_jsonl(out / "support.jsonl", support_rows)
    _write_jsonl(out / "queries.jsonl", query_rows)
    _save_json(out / "standalone_receipt.json", {
        "compiler_audit": certificate,
        "certificate_source_sha256": sha256_file(out / "standalone.lean") if proof_items else None,
        "proofs_submitted": len(proof_items),
        "standalone_passed": sum(ep["verified_closed"] for ep in episodes),
        "authoritative_only_with_real_lean": certified_backend,
    })

    arm_scores = {}
    for arm in POLICIES:
        selected = [r for r in query_rows if r["policy"] == arm]
        arm_scores[arm] = {
            "attempted_roots": len(selected),
            "kernel_closures": sum(ep["live_kernel"] for ep in selected),
            "standalone_verified_closures": sum(ep["verified_closed"] for ep in selected),
            "tactic_checks": sum(ep["attempts"] for ep in selected),
            "model_calls": sum(ep["model_calls"] for ep in selected),
            "kernel_calls": sum(ep["kernel_calls"] for ep in selected),
            "accepted_nonterminal": sum(s["status"] == "accepted_nonterminal"
                                        for ep in selected for s in ep["events"]),
            "rejected": sum(s["status"] == "rejected"
                            for ep in selected for s in ep["events"]),
            "censored": sum(ep["status"] not in ("kernel_closed", "budget_exhausted", "no_candidates")
                            for ep in selected),
        }
    artifacts = {}
    for name in ("support.jsonl", "queries.jsonl", "standalone_receipt.json"):
        artifacts[name] = {"sha256": sha256_file(out / name)}
    if proof_items:
        artifacts["standalone.lean"] = {"sha256": sha256_file(out / "standalone.lean")}
    report = {
        "schema": SCHEMA, "plan": model.cfg.plan, "source": source,
        "backend": ("live_lean_4_19_kernel_plus_standalone"
                    if certified_backend else "simulated_protocol"),
        "actual_lean_process": certified_backend, "checkpoint_sha256":
            sha256_file(checkpoint) if checkpoint else None,
        "checkpoint_evidence": source.get("checkpoint_evidence") if source else "unspecified",
        "support_roots": len(support_rows), "query_roots": len(queries),
        "fully_evaluated_query_roots": min((
            arm_scores[arm]["attempted_roots"] for arm in POLICIES
        ), default=0),
        "query_split": queries[0].split, "check_budget": checks, "depth": depth,
        "online_query_updates": 0,
        "support_definitive_updates": {family: state[0].updates for family, state in memory.items()},
        "fast_weight_hashes": {
            fam: state_hash(state[0])
            for fam, state in memory.items()},
        "arms": arm_scores, "certification": {
            "repl_kernel": True, "standalone_source": bool(proof_items),
            "source_sha256": certificate.get("source_sha256") if certificate else None,
        },
        "artifacts": artifacts, "aborted_due_to_wall_limit": aborted,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "architecture_promotion_eligible": False,
        "claim": "live Lean Horn proof benchmark" if certified_backend else "simulated protocol test",
        "caveat": "generator-family overlap with earlier LC-GRT public corpora is unchecked; "
                  "matched check counts do not match inference FLOPs",
    }
    _save_json(out / "manifest.json", report)
    return report


def audit_campaign(repl, suite: Path, session: Path):
    """Independent second pass of every recorded Lean action; no model needed."""
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unrecognized live campaign manifest")
    for filename, info in manifest["artifacts"].items():
        if sha256_file(session / filename) != info["sha256"]:
            raise ValueError("live campaign artifact hash changed: " + filename)
    support = _read_jsonl(session / "support.jsonl")
    queries = _read_jsonl(session / "queries.jsonl")
    groups = {}
    for split in ("train", manifest["query_split"]):
        selected, _ = load_suite(suite, split, max_cases=1024)
        for case in selected:
            groups[case.root] = case
    count = 0
    for ep in support + queries:
        case = groups.get(ep["root"])
        if not case or case.lineage != ep["lineage"] or case.split != ep["split"]:
            raise ValueError("episode root or lineage is absent from frozen suite")
        replay_episode(repl, case, ep)
        count += 1
    return {"replay_passed": True, "episodes": count,
            "source_manifest_sha256": manifest["source"]["suite_manifest_sha256"],
            "new_lean_replay_actions": sum(len(ep["events"]) for ep in support + queries),
            "live_verifier_version": repl.version}
