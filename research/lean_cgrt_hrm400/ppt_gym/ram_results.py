"""Learning curves, paired outcomes and replay-gated local DataGoblin exports."""
from __future__ import annotations

import csv
import gzip
import json
import statistics
from collections import Counter, defaultdict

import torch

from . import ram_curriculum, ram_store
from .data import canonical, digest, sha256_file
from .ram_learning import credit


def _json(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _head(db):
    head = db.execute("SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    return head[0] if head else None


def _check_curriculum(db, curriculum):
    manifest = ram_store.read_manifest(db)
    if manifest["curriculum_sha256"] != sha256_file(curriculum):
        raise ValueError("curriculum digest differs from recorded session")
    return manifest


def report(db, curriculum, output):
    manifest = _check_curriculum(db, curriculum)
    output.mkdir(parents=True, exist_ok=False)
    curves, pass_cost, proof_cost = defaultdict(Counter), defaultdict(list), defaultdict(list)
    paired = defaultdict(dict)
    support = Counter()
    training_wall = 0.0
    ledger_count = 0
    with gzip.open(output / "episode-ledger.jsonl.gz", "wt", encoding="utf-8", newline="\n") as trace:
        for row in ram_store.ledger(db):
            trace.write(canonical(row) + "\n")
            ledger_count += 1
            kind = row["task"]["kind"]
            if kind == "fit":
                fit = row["encoder_fit"]
                training_wall += fit["wall_seconds"] if fit else 0.0
            if "episode" not in row:
                continue
            task, ep = row["task"], row["episode"]
            proof_cost[(kind, ep["policy"])].append(ep)
            for call in ep["forward_calls"]:
                for item in call["passes"]:
                    pass_cost[(ep["policy"], ep["depth"], item["pass"])].append(item)
            if kind == "support":
                support["roots_consumed"] += 1
                support["roots_with_definitive_credit"] += bool(row["updates"])
                support["certified_proofs"] += ep["verified_closed"]
                support["weight_updates"] += sum(update["before"] != update["after"] for update in row["updates"])
                support.update(update["credit"] for update in row["updates"])
                support["lean_tactic_calls"] += ep["tactic_checks"]
                training_wall += ep["wall_seconds"] + ep["certificate_seconds"] + row["update_wall_seconds"]
            if kind != "query":
                continue
            key = (ep["split"], task["family"], task["round"], task["dose"],
                   row["verified_support_roots"], ep["policy"], ep["depth"])
            counts = curves[key]
            passed = ep["verified_closed"] and ep["within_cpu_budget"]
            counts.update(roots=1, certified=int(passed), tactic_checks=ep["tactic_checks"],
                          estimated_macs=ep["estimated_macs"], rollbacks=ep["rollbacks"])
            pair_key = (ep["root"], task["family"], task["round"], task["dose"])
            paired[pair_key][(ep["policy"], ep["depth"])] = int(passed)

    curve_keys = ("split", "support_family", "round", "scheduled_support_roots",
                  "definitive_support_roots", "arm", "depth")
    with (output / "learning-curves.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (*curve_keys, "roots", "certified", "certified_rate", "tactic_checks", "rollbacks", "estimated_macs")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, count in sorted(curves.items()):
            writer.writerow({**dict(zip(curve_keys, key)), **count,
                             "certified_rate": count["certified"] / count["roots"]})
    comparisons = {}
    for depth in manifest["options"]["depths"]:
        for baseline in ("retrieval", "frozen", "shuffled", "horn"):
            outcomes = Counter()
            for key, arms in paired.items():
                if key[2] != 0 or key[3] != manifest["options"]["support"]:
                    continue
                alt = (baseline, 1 if baseline == "horn" else depth)
                if ("sgd", depth) in arms and alt in arms:
                    outcomes[arms[("sgd", depth)] - arms[alt]] += 1
            comparisons[f"sgd@{depth}_vs_{baseline}"] = {"wins": outcomes[1], "ties": outcomes[0],
                                                         "losses": outcomes[-1]}
    result = {
        "manifest": manifest, "ledger_head": _head(db), "entries": ledger_count,
        "complete": ledger_count == manifest["scheduled_entries"],
        "support": dict(support), "training_wall_seconds": training_wall,
        "definitive_actions_per_training_second": support["weight_updates"] / training_wall if training_wall else 0,
        "paired_final_dose": comparisons,
        "pass_costs": [
            {"arm": arm, "depth": depth, "pass": k, "calls": len(items),
             "mean_cpu_ns": statistics.mean(row["cpu_ns"] for row in items),
             "median_wall_ns": statistics.median(row["wall_ns"] for row in items)}
            for (arm, depth, k), items in sorted(pass_cost.items())
        ],
        "proof_costs": [
            {"kind": kind, "arm": arm, "roots": len(items),
             "mean_lean_rpc_ms": statistics.mean(ep["verification_seconds"] for ep in items) * 1000,
             "mean_compiler_ms": statistics.mean(ep["certificate_seconds"] for ep in items) * 1000,
             "mean_search_cpu_seconds": statistics.mean(ep["search_cpu_seconds"] for ep in items),
             "proofs_outside_cpu_budget": sum(not ep["within_cpu_budget"] for ep in items)}
            for (kind, arm), items in sorted(proof_cost.items())
        ],
        "scope": "generated Horn theorem pilot; source-family cohorts and repeated roots are descriptive",
    }
    _json(output / "summary.json", result)
    return {k: result[k] for k in ("entries", "complete", "support", "paired_final_dose")}


def export(db, curriculum, replay_path, output):
    manifest = _check_curriculum(db, curriculum)
    _, cohorts = ram_curriculum.load(curriculum)
    train = {case.root: case for group in ("warmup", "support") for case in cohorts[group]}
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if (replay["ledger_head"] != _head(db) or replay["curriculum_sha256"] != manifest["curriculum_sha256"]
            or replay["verifier"] != manifest["verifier"] or replay["fast_weights_replayed"] is not True):
        raise ValueError("export requires a matching native verifier and update replay")
    witnesses = {item["seq"]: item for item in replay["verified_episodes"]}
    output.mkdir(parents=True, exist_ok=False)
    streams = {name: [] for name in ("sft", "transitions", "outcomes", "repairs", "preferences")}
    seen, roots = set(), set()
    with gzip.open(output / "train-experience.jsonl.gz", "wt", encoding="utf-8", newline="\n") as experience:
        for row in ram_store.ledger(db):
            if row["task"]["kind"] not in ("warmup", "support"):
                continue
            ep = row["episode"]
            case = train[row["task"]["root"]]
            witness = witnesses.get(row["seq"])
            if (not witness or not witness["replay_passed"] or witness["episode_sha256"] != digest(ep)
                    or case.root in roots or ep["split"] != "train"):
                raise ValueError("export contains an unregistered, repeated or unreplayed training root")
            roots.add(case.root)
            experience.write(canonical({**row, "replay_receipt": witness}) + "\n")
            credits = dict(credit(ep))
            indexed = [(i, step) for i, step in enumerate(ep["events"]) if step["kind"] == "tactic"]
            if not indexed:
                continue
            first = indexed[0][1]
            base = {"theorem": case.statement, "state": first["input"]}
            meta = {"root": case.lineage, "source_root": case.root, "split": "train",
                    "family": "lean_proof", "subfamily": case.family,
                    "proof_certificate_sha256": witness["certificate_sha256"],
                    "loss": {"boundary": "response_only", "field": "target", "eligible": False,
                             "admission": "pending_ancestry_and_rights_audit"}}

            def add(stream, packet, target):
                key = digest([stream, case.lineage, packet, target])
                if key not in seen:
                    streams[stream].append({**meta, "id": key, "input": packet, "target": target})
                    seen.add(key)

            losers = defaultdict(list)
            for index, step in indexed:
                if index in credits and credits[index][0] == 0:
                    losers[step["state_before_sha256"]].append(step["tactic"])
            for index, step in indexed:
                if step["status"] == "certificate_rejected" or (
                        step["status"] == "kernel_closed" and not ep["verified_closed"]):
                    continue
                packet = {"theorem": case.statement, "state": step["input"], "menu": step["candidate_menu"]}
                outcome = {"status": step["status"], "proof_credit": credits.get(index)}
                add("outcomes", {**packet, "candidate": step["tactic"]}, outcome)
                add("transitions", packet, {"tactic": step["tactic"], **outcome,
                                            "successor": step.get("state_after")})
                if index in credits and credits[index][0] == 1:
                    for loser in losers[step["state_before_sha256"]]:
                        if loser != step["tactic"]:
                            add("preferences", packet, {"chosen": step["tactic"], "rejected": loser})
            if ep["verified_closed"]:
                proof = {"tactic_script": ep["accepted_script"]}
                add("sft", base, proof)
                failed = [{"state_sha256": state, "tactics": tactics} for state, tactics in losers.items()]
                if failed:
                    add("repairs", {**base, "failed": failed}, proof)

    artifacts = {}
    for kind, records in streams.items():
        path = output / f"{kind}.jsonl"
        path.write_text("".join(canonical(record) + "\n" for record in records), encoding="utf-8")
        artifacts[kind] = {"rows": len(records), "sha256": sha256_file(path)}
    state = ram_store.checkpoint(db)
    torch.save({"schema": "ppt_gym.ram_specialist.v1", "architecture": manifest["architecture"],
                "backbone": ram_store.load_backbone(db, state["model_sha"]), "state": state,
                "source_manifest": manifest, "ledger_head": replay["ledger_head"]}, output / "specialist.pt")
    result = {"schema": "ppt_gym.ram_datagoblin_export.v1", "streams": artifacts, "train_roots": len(roots),
              "query_rows": 0, "ledger_head": replay["ledger_head"], "admission": "quarantined_for_review",
              "experience_sha256": sha256_file(output / "train-experience.jsonl.gz"),
              "specialist_sha256": sha256_file(output / "specialist.pt")}
    _json(output / "manifest.json", result)
    return result
