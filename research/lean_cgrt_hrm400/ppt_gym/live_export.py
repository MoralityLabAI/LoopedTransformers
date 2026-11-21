"""Train-only DataGoblin-style exports from independently replayed PPT-G03 proofs.

The exporter owns no proof labels. All success claims require a native Lean
kernel closure, a standalone compiled proof, and a separate live replay receipt.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import canonical, digest, sha256_file
from .live_suite import load_suite

SCHEMA = "ppt_gym.live_export.v1"
STREAMS = ("sft", "outcomes", "transitions", "repairs", "preferences")
CAP = 4


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _outcome(status: str, verified: bool):
    if status == "rejected":
        return {"accepted": False, "closed": False, "status": status}
    if status == "accepted_nonterminal":
        return {"accepted": True, "closed": False, "status": status}
    if status == "kernel_closed" and verified:
        return {"accepted": True, "closed": True, "status": "certified_closure"}
    return None


def export(session: Path, suite: Path, replay_receipt: Path, out: Path):
    report = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "ppt_gym.live_campaign.v1"
            or report.get("actual_lean_process") is not True
            or report.get("source", {}).get("lean", {}).get("actual_lean_process") is not True):
        raise ValueError("training export requires native live Lean, not simulated protocol")
    if report["query_split"] == "train":
        raise ValueError("training export requires a held-out query split")
    for name, item in report["artifacts"].items():
        if name != Path(name).name or sha256_file(session / name) != item["sha256"]:
            raise ValueError(f"live source artifact changed: {name}")
    replay = json.loads(replay_receipt.read_text(encoding="utf-8"))
    support = list(rows(session / "support.jsonl"))
    query = list(rows(session / "queries.jsonl"))
    if (replay.get("replay_passed") is not True or
            replay.get("source_manifest_sha256") != report["source"]["suite_manifest_sha256"] or
            replay.get("episodes") != len(support) + len(query) or
            replay.get("live_verifier_version") != report["source"]["lean"]["lean_version"]):
        raise ValueError("export requires matching independently replayed native Lean actions")
    selected, suite_meta = load_suite(suite, "train", max_cases=1024)
    if suite_meta["suite_manifest_sha256"] != report["source"]["suite_manifest_sha256"]:
        raise ValueError("frozen source theorem suite changed")
    registered = {case.root: case for case in selected}
    proposals: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    custody = Counter()
    for ep in support:
        if ep["split"] != "train" or ep["policy"] != "frozen":
            raise ValueError("support export is contaminated by evaluation policies")
        case = registered.get(ep["root"])
        if case is None or case.lineage != ep["lineage"]:
            raise ValueError("support episode root/lineage differs from frozen source")
        if ep["verified_closed"] != (ep["live_kernel"] and ep["standalone_certificate"] == "passed"):
            raise ValueError("source certificate verdict mismatch")
        history = []
        first = None
        earlier_failures = []
        positive = []
        for step in ep["events"]:
            input_packet = {"theorem": case.statement, "state": step["input"],
                            "candidate_menu": step["candidate_menu"],
                            "history": history.copy(), "checks_remaining":
                                ep["max_checks"] - step["attempt"] + 1}
            if first is None:
                first = input_packet
            observed = _outcome(step["status"], ep["verified_closed"])
            meta = {
                "root": ep["lineage"], "source_root": ep["root"], "split": "train",
                "family": "lean_proof", "subfamily": case.family,
                "episode_policy": ep["policy"], "visit": step["attempt"],
                "evidence_class": "native_lean_kernel_and_standalone_replayed",
                "source_statement_sha256": ep["statement_sha256"],
                "public_packet_sha256": step["public_packet_sha256"],
                "original_tactic_sha256": digest(step["tactic"]),
                "source_session_sha256": sha256_file(session / "manifest.json"),
                "source_replay_sha256": sha256_file(replay_receipt),
                "source_kernel_check": bool(ep["verified_closed"]),
            }
            if observed is not None:
                # Each trained prediction only sees the state before THIS check.
                outcome = {"input": {**input_packet, "chosen_tactic": step["tactic"]},
                           "target": observed}
                proposals[ep["root"]].append((3, "outcomes", {**meta, **outcome}))
                transition = {"input": input_packet,
                              "target": {"tactic": step["tactic"], "result": observed,
                                         "successor_goals": step.get("state_after")}}
                proposals[ep["root"]].append((4, "transitions", {**meta, **transition}))
                if observed["closed"]:
                    positive.append(step)
                if step["status"] == "rejected":
                    earlier_failures.append(step)
                if positive and earlier_failures:
                    same_state = [f for f in earlier_failures
                                  if f["state_before_sha256"] == step["state_before_sha256"]]
                    if same_state and step["status"] == "kernel_closed":
                        bad = same_state[-1]
                        pair = {
                            "input": input_packet,
                            "target": {"chosen": step["tactic"], "rejected": bad["tactic"],
                                       "verifier": "native_lean_certified"},
                        }
                        proposals[ep["root"]].append((2, "preferences", {**meta, **pair}))
            history.append({"tactic": step["tactic"], "status": step["status"],
                            "error": step.get("error"),
                            "next_state_sha256": step["state_after_sha256"]})
        if ep["verified_closed"] and first is not None:
            proof = {
                "root": ep["lineage"], "source_root": ep["root"], "split": "train",
                "family": "lean_proof", "subfamily": case.family,
                "source_statement_sha256": ep["statement_sha256"],
                "source_session_sha256": sha256_file(session / "manifest.json"),
                "source_replay_sha256": sha256_file(replay_receipt),
                "evidence_class": "native_lean_kernel_and_standalone_replayed",
                "input": first,
                "target": {"tactic_script": ep["accepted_script"]},
            }
            proposals[ep["root"]].append((0, "sft", proof))
            if earlier_failures:
                proposals[ep["root"]].append((1, "repairs", {
                    **proof, "input": {**first, "failed_tactics": [
                        {"tactic": s["tactic"], "status": s["status"]}
                        for s in earlier_failures[:4]]},
                }))
        custody["support_roots"] += 1
        custody["live_kernel_closed"] += int(ep["live_kernel"])
        custody["standalone_certified_closed"] += int(ep["verified_closed"])
        custody["rejected_tactics"] += len(earlier_failures)

    # Single cap over *all* supervision views of one lineage. Historical
    # source traces remain intact after the export drops repeated views.
    out.mkdir(parents=True, exist_ok=False)
    chosen: dict[str, list[dict]] = {kind: [] for kind in STREAMS}
    dropped = Counter()
    for root in sorted(proposals):
        candidates = sorted(proposals[root], key=lambda r: (r[0], r[1], digest(r[2])))
        for index, (_, kind, item) in enumerate(candidates):
            if index >= CAP:
                dropped[kind] += 1
                continue
            item = {**item, "id": digest([root, kind, item["input"], item["target"]]),
                    "loss": {"boundary": "response_only", "field": "target",
                             "eligible": True}}
            chosen[kind].append(item)
    artifacts = {}
    for kind, records in chosen.items():
        path = out / f"{kind}.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for item in records:
                f.write(canonical(item) + "\n")
        artifacts[path.name] = {"rows": len(records), "sha256": sha256_file(path)}
    result = {
        "schema": SCHEMA, "split": "train", "support_roots": custody["support_roots"],
        "counts": dict(custody), "max_presentations_per_lineage": CAP,
        "dropped_due_to_cap": dict(dropped), "streams": artifacts,
        "source_manifest_sha256": sha256_file(session / "manifest.json"),
        "source_suite_sha256": suite_meta["suite_manifest_sha256"],
        "independent_replay_sha256": sha256_file(replay_receipt),
        "source_checkpoint_evidence": report["source"].get("checkpoint_evidence"),
        "query_labels_exported": 0, "new_lean_calls": 0,
        "source_kind": "generated_horn_problems_independently_lean_certified",
        "benchmark_ancestry_status": "external_dataset_overlap_unchecked",
        "training_admission_status": "pending_external_overlap_and_source_rights_audit",
    }
    (out / "manifest.json").write_text(canonical(result) + "\n", encoding="utf-8")
    return result
