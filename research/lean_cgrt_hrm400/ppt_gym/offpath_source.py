"""LSA-02H: Lean-authoritative *recoverable* off-reference proof trajectories.

For frozen TRAIN theorems, sample at most two legal alternative tactics
that depart from the native shortest reference script. The candidate must
lead to a finite Lean-explored continuation that fits the remaining eight
tactic checks. Replay the entire alternate proof from the theorem root with
real Lean, check its terminal kernel receipt and compile every full script
as a standalone axiom-free declaration.

Only correction states *after the deviation* can become training records.
They carry the same public packet / public action-menu / exact shortest
continuation label contract as the existing native teacher. A reference
path state is not counted as novel off-reference supervision.

Unlike a model rollout, this controlled source selection has fixed public
tactic ordering and no dependence on model predictions or calibration data.
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
from lcgrt.extract import explore, goal_key
from lcgrt.lean_repl import LeanRepl
from .data import canonical, sha256_file
from .live_lean import candidate_menu, fingerprint, public_state, state_input
from .live_suite import load_suite
from .native_train import (
    CHECKS, EVIDENCE, FRAME_SCHEMA, _load_teacher,
    _reference_optimal_actions, _snapshot_trajectory,
)


SCHEMA = "lsa02h.recoverable_offreference_train.v1"
ROW_SCHEMA = "lsa02h.offreference_correction_frame.v1"
MAX_DEVIATIONS_PER_ROOT = 2


def _json(path: Path, payload: dict) -> None:
    path.write_text(canonical(payload) + "\n", encoding="utf-8")


def _records(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _replay_from_root(repl, case, script: list[str]) -> tuple[dict, list[dict]]:
    repl.reset()
    state = repl.init(case.statement, pp=True)
    frames = []
    for tactic in script:
        if not state["goals"]:
            raise ValueError("off-reference script continues after terminal closure")
        frames.append(public_state(repl, state))
        if tactic not in candidate_menu(repl, state):
            raise ValueError("off-reference proof uses tactic absent from public grammar")
        step = repl.run(state["sid"], tactic)
        if step.get("ok") is not True:
            raise ValueError("Lean rejected an off-reference proof tactic")
        state = step["state"]
    return state, frames


def _minimum_finite_script(table: dict, sizes: dict, key: str) -> list[str]:
    """Construct native shortest script, ignoring Lean-reported dead edges.

    The original extractor's optimal_script assumes all action children have
    finite size. The source graph deliberately includes accepted tactics that
    lead to unprovable decoy goals, represented by None. Filter their edges
    before scoring, then insist the selected tactic equals the exact native
    finite size. Strict size decrease prevents cycles.
    """
    size = sizes.get(key)
    if not isinstance(size, int) or size < 1:
        raise ValueError("offreference goal is not Lean-provable")
    candidates = []
    for tactic, child in table.get(key, {}).items():
        costs = [sizes.get(g) for g in child]
        if any(not isinstance(s, int) or s < 1 for s in costs):
            continue
        total = 1 + sum(costs)
        if total == size:
            candidates.append((tactic, child))
    if not candidates:
        raise ValueError("native finite shortest proof has no matching admissible tactic")
    tactic, children = min(candidates, key=lambda item: item[0])
    script = [tactic]
    for child in children:
        if sizes[child] >= size:
            raise ValueError("finite native proof graph contains a nondecreasing cycle")
        script.extend(_minimum_finite_script(table, sizes, child))
    if len(script) != size:
        raise ValueError("native minimum completion length differs from reconstructed script")
    return script


def _completion_script(oracle: dict, pending: list[dict]) -> list[str] | None:
    script = []
    for goal in pending:
        key = goal_key(goal)
        size = oracle["size"].get(key)
        if not isinstance(size, int) or size < 1:
            return None
        script.extend(_minimum_finite_script(oracle["table"], oracle["size"], key))
    return script


def _collect_case(repl, case, oracle: dict, reference_frames: list[dict],
                  cap: int = MAX_DEVIATIONS_PER_ROOT) -> tuple[list[dict], list[tuple], dict]:
    if not 1 <= cap <= 8:
        raise ValueError("off-reference source exploration cap outside registry")
    reference_actions = oracle["proof"]["script"]
    original = {row["public_state_sha256"] for row in reference_frames}
    census = {
        "root": case.root,
        "lineage": case.lineage,
        "reference_size": len(reference_actions),
        "alternative_public_menu_proposals": 0,
        "alternatives_accepted_in_lean_graph": 0,
        "alternative_finite_continuations": 0,
        "candidate_within_eight_checks": 0,
        "deviations_compiled_pending": 0,
        "offreference_training_frames": 0,
        "unreachable_or_over_budget": 0,
    }
    discoveries: list[dict] = []
    reference_proofs: list[tuple] = []
    repl.reset()
    current = repl.init(case.statement, pp=True)
    prefix: list[str] = []
    for pos, selected in enumerate(reference_actions):
        if current["goals"] == []:
            raise ValueError("native reference terminated before expected step")
        front = goal_key(current["goals"][0])
        menu = candidate_menu(repl, current)
        if selected not in menu:
            raise ValueError("reference tactic outside public menu")
        table = oracle["table"].get(front, {})
        alternatives = sorted(t for t in menu if t != selected)
        for tactic in alternatives:
            census["alternative_public_menu_proposals"] += 1
            if tactic not in table:
                continue
            census["alternatives_accepted_in_lean_graph"] += 1
            children = table[tactic]
            if any(oracle["size"].get(k) is None for k in children):
                census["unreachable_or_over_budget"] += 1
                continue
            census["alternative_finite_continuations"] += 1
            # Reconstruct the complete alternate state *with Lean*, rather
            # than projecting a one-goal graph cost onto a multi-goal proof.
            base_state, _ = _replay_from_root(repl, case, prefix + [tactic])
            extra = _completion_script(oracle, base_state["goals"])
            if extra is None or pos + 1 + len(extra) > CHECKS:
                census["unreachable_or_over_budget"] += 1
                continue
            census["candidate_within_eight_checks"] += 1
            if len(discoveries) >= cap:
                continue
            if not extra:
                # An alternate terminal proof has no *new decision state*
                # and cannot provide the intended off-reference training.
                continue
            full = prefix + [tactic] + extra
            seen = []
            repl.reset()
            state = repl.init(case.statement, pp=True)
            accepted = []
            new_frames = []
            for step_number, action in enumerate(full):
                if not state["goals"]:
                    raise ValueError("off-reference path has premature terminal proof")
                packet, public_sha = state_input(
                    repl, state, (), max(1, CHECKS - step_number),
                )
                if step_number >= pos + 1 and public_sha not in original:
                    current_front = goal_key(state["goals"][0])
                    observed_menu = candidate_menu(repl, state)
                    labels = _reference_optimal_actions(
                        current_front, observed_menu, oracle["table"], oracle["size"],
                    )
                    if labels[observed_menu.index(action)] != 1:
                        raise ValueError("off-reference correction did not follow verified shortest continuation")
                    new_frames.append({
                        "schema": ROW_SCHEMA,
                        "evidence": EVIDENCE,
                        "root": case.root,
                        "lineage": case.lineage,
                        "family": case.family,
                        "goal": packet,
                        "public_state_sha256": public_sha,
                        "candidates": list(observed_menu),
                        "labels": list(labels),
                        "checks_remaining": max(1, CHECKS - step_number),
                        "step": len(reference_frames) + pos * 100 + step_number,
                        "source_prefix_size": pos,
                        "deviation_tactic": tactic,
                        "deviation_pos": pos,
                        "correction_index": step_number - (pos + 1),
                        "remaining_proof_steps": len(full) - step_number,
                        "path_length": len(full),
                        "reference_root_size": len(reference_actions),
                        "label_semantics": "native_Lean_shortest_continuation_after_legal_recoverable_deviation",
                        "source_kind": "off_reference",
                    })
                outcome = repl.run(state["sid"], action)
                if outcome.get("ok") is not True:
                    raise ValueError("native Lean rejected replayed alternate proof")
                accepted.append(action)
                state = outcome["state"]
            if state["goals"]:
                raise ValueError("native alternate proof left unfinished goals")
            cert = repl.kernel(state["sid"])
            if cert.get("ok") is not True or cert.get("axioms") != []:
                raise ValueError("native off-reference alternate proof failed terminal kernel")
            if not new_frames:
                continue
            signature = hashlib.sha256(canonical(full).encode()).hexdigest()[:16]
            proof_id = f"{case.tid}_off_{signature}"
            discoveries.append({
                "proof_id": proof_id,
                "script": full,
                "deviation_pos": pos,
                "deviation_tactic": tactic,
                "correction_states": new_frames,
            })
            reference_proofs.append((proof_id, case.theorem(), full))
            census["deviations_compiled_pending"] += 1
            census["offreference_training_frames"] += len(new_frames)
        # Multiple alternative replays reset the proof server. Restore the
        # canonical shortest prefix before stepping to its next state.
        nxt, _ = _replay_from_root(repl, case, prefix + [selected])
        current = nxt
        prefix.append(selected)
    if current["goals"]:
        raise ValueError("canonical reference proof no longer closes")
    return discoveries, reference_proofs, census


def collect(suite: Path, teacher: Path, out: Path, *, roots: int = 64,
            cap_per_root: int = MAX_DEVIATIONS_PER_ROOT,
            max_seconds: float = 900.0, request_timeout: int = 60,
            cert_timeout: int = 420) -> dict:
    if out.exists() or roots not in range(1, 1025) or max_seconds <= 0:
        raise ValueError("fresh offreference output and bounded source roots required")
    native_frames, teacher_meta = _load_teacher(teacher)
    cases, source = load_suite(suite, "train", roots)
    if source["suite_manifest_sha256"] != teacher_meta["source"]["suite_manifest_sha256"]:
        raise ValueError("teacher and off-reference source generator differ")
    if teacher_meta["independent_lineages"] != roots:
        raise ValueError("native training teacher source root census differs")
    per_root_reference = defaultdict(list)
    for frame in native_frames:
        per_root_reference[frame["root"]].append(frame)
    began = time.perf_counter()
    out.mkdir(parents=True)
    all_source_rows = []
    evidence = []
    census = []
    proof_records = []
    with LeanRepl(
        timeout=request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        if str(repl.version) != "4.19.0":
            raise ValueError("native offreference extraction needs Lean 4.19.0")
        backend = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        for case in cases:
            if time.perf_counter() - began > max_seconds:
                raise TimeoutError("native off-reference corpus source collection timed out")
            oracle = explore(repl, case.theorem())
            reference = _snapshot_trajectory(repl, case, oracle)
            expected = per_root_reference[case.root]
            if [(r["goal"], r["candidates"], r["labels"]) for r in reference] != [
                (r["goal"], r["candidates"], r["labels"]) for r in expected
            ]:
                raise ValueError("off-reference generator changed frozen native reference teacher")
            alternatives, proofs, root_report = _collect_case(
                repl, case, oracle, reference, cap=cap_per_root,
            )
            census.append(root_report)
            proof_records.extend(proofs)
            evidence.extend({
                "root": case.root,
                "lineage": case.lineage,
                "proof_id": x["proof_id"],
                "script": x["script"],
                "deviation_pos": x["deviation_pos"],
                "deviation_tactic": x["deviation_tactic"],
                "correction_states": len(x["correction_states"]),
            } for x in alternatives)
            all_source_rows.extend(
                frame for x in alternatives for frame in x["correction_states"]
            )
    receipt = {"status": {}, "passed": 0, "timed_out": False}
    if proof_records:
        receipt = certify(proof_records, timeout=cert_timeout, keep=out / "standalone.lean")
        if receipt.get("timed_out") or any(
            receipt.get("status", {}).get(decl_name(tid)) != "passed"
            for tid, _, _ in proof_records
        ):
            raise ValueError("independent Lean compilation rejected off-reference correction path")
    # Deduplicate public state+menu+remaining budget within a theorem. This
    # preserves individual theorem as statistical unit and avoids repeated
    # corrections with different original prefixes counting as more examples.
    by_identity: dict[tuple, dict] = {}
    for row in all_source_rows:
        key = (
            row["root"], row["public_state_sha256"],
            row["checks_remaining"], tuple(row["candidates"]),
        )
        if key in by_identity:
            if by_identity[key]["labels"] != row["labels"]:
                raise ValueError("same public correction state has conflicting shortest labels")
            continue
        by_identity[key] = row
    unique = list(by_identity.values())
    for index, row in enumerate(unique):
        row["step"] = 1000 + index
        row["source_proof_compiled"] = True
    with (out / "offreference_frames.jsonl").open("x", encoding="utf-8") as writer:
        for row in unique:
            writer.write(canonical(row) + "\n")
    with (out / "source_census.jsonl").open("x", encoding="utf-8") as writer:
        for row in census:
            writer.write(canonical(row) + "\n")
    with (out / "alternate_proofs.jsonl").open("x", encoding="utf-8") as writer:
        for item in evidence:
            writer.write(canonical(item) + "\n")
    _json(out / "compiler_receipt.json", receipt)
    artifacts = ["offreference_frames.jsonl", "source_census.jsonl",
                 "alternate_proofs.jsonl", "compiler_receipt.json"]
    if proof_records:
        artifacts.append("standalone.lean")
    result = {
        "schema": SCHEMA,
        "source_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "suite_manifest_sha256": source["suite_manifest_sha256"],
        "backend": backend,
        "train_split_only": True,
        "training_lineages": len(cases),
        "roots_with_recoverable_deviations": len({
            r["root"] for r in unique
        }),
        "novel_offreference_training_states": len(unique),
        "all_candidate_offreference_states": len(all_source_rows),
        "certified_alternate_full_proofs": len(proof_records),
        "max_deviations_per_train_root": cap_per_root,
        "alternative_public_menu_proposals": sum(c["alternative_public_menu_proposals"] for c in census),
        "alternatives_accepted_in_lean_graph": sum(c["alternatives_accepted_in_lean_graph"] for c in census),
        "alternative_finite_continuations": sum(c["alternative_finite_continuations"] for c in census),
        "candidate_within_eight_checks": sum(c["candidate_within_eight_checks"] for c in census),
        "unreachable_or_over_budget": sum(c["unreachable_or_over_budget"] for c in census),
        "independent_compiler_passed": True,
        "model_calls_during_collection": 0,
        "locked_test_opened": False,
        "scientific_eligible": False,
        "claim": "native Lean witnessed recoverable legal deviations; source-only coverage not model competence",
        "artifacts": {f: sha256_file(out / f) for f in artifacts},
    }
    _json(out / "manifest.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roots", type=int, default=64)
    p.add_argument("--cap-per-root", type=int, default=2)
    p.add_argument("--max-seconds", type=float, default=900)
    p.add_argument("--request-timeout", type=int, default=60)
    p.add_argument("--cert-timeout", type=int, default=420)
    args = p.parse_args(argv)
    result = collect(args.suite, args.teacher, args.out,
                     roots=args.roots, cap_per_root=args.cap_per_root,
                     max_seconds=args.max_seconds, request_timeout=args.request_timeout,
                     cert_timeout=args.cert_timeout)
    print(canonical({
        "roots": result["training_lineages"],
        "novel_states": result["novel_offreference_training_states"],
        "roots_with_deviations": result["roots_with_recoverable_deviations"],
        "alternate_proofs": result["certified_alternate_full_proofs"],
        "candidates": result["candidate_within_eight_checks"],
        "unreachable_or_over_budget": result["unreachable_or_over_budget"],
        "source_manifest_sha256": result["source_teacher_sha256"],
    }))
    return result


if __name__ == "__main__":
    main()
