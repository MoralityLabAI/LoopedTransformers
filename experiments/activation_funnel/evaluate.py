"""Evaluate frozen AF-01 checkpoints; keep counterfactuals outside proof receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from . import gym
from .model import (CODEC_ARMS, CompactTransition, FixedCodec, LearnedCodec,
                    RecurrentTeacher, commutation_loss, resources, task_from_audits)
from .train import (CONFIG, REGISTERED_PROTOCOL_SHA256, canonical_hash,
                    sha256_file, tensor_data)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_frozen(folder, allow_smoke=False):
    manifest_path = folder / "train_manifest.json"
    if sha256_file(manifest_path) != (folder / "train_manifest.sha256").read_text().strip():
        raise ValueError("frozen training manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("training_complete"):
        raise ValueError("training manifest is incomplete")
    if manifest["scope"] != "registered_primary" and not allow_smoke:
        raise ValueError("engineering-smoke checkpoints cannot enter primary scoring")
    if manifest["configuration_sha256"] != canonical_hash(manifest["configuration"]):
        raise ValueError("training configuration hash mismatch")
    if manifest["scope"] == "registered_primary" and manifest["configuration"] != CONFIG:
        raise ValueError("training configuration differs from registered primary configuration")
    seeds = manifest["configuration"]["seeds"]
    if [record.get("seed") for record in manifest["seeds"]] != seeds:
        raise ValueError("frozen seed metadata roster differs from registered configuration")
    expected_files = {f"{arm}_s{seed}.pt" for seed in seeds
                      for arm in ("teacher",) + CODEC_ARMS + ("compact_transition",)}
    if set(manifest["files"]) != expected_files:
        raise ValueError("frozen checkpoint file roster differs from registered configuration")
    base = Path(__file__).parent
    if sha256_file(base / "PROTOCOL.json") != REGISTERED_PROTOCOL_SHA256:
        raise ValueError("current protocol differs from registered protocol")
    for name in ("gym.py", "model.py", "train.py", "evaluate.py", "PROTOCOL.json"):
        if manifest["source_hashes"].get(name) != sha256_file(base / name):
            raise ValueError(f"training dependency changed after freezing: {name}")
    for name, entry in manifest["files"].items():
        if Path(name).name != name or sha256_file(folder / name) != entry["sha256"]:
            raise ValueError(f"checkpoint identity mismatch: {name}")
    return manifest


def load_arm(folder, seed, arm, teacher):
    if arm == "teacher":
        return teacher, None
    saved = _load(folder / f"{arm}_s{seed}.pt")
    if arm == "compact_transition":
        model = CompactTransition(teacher)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        return model, None
    if saved["kind"] == "fixed":
        codec = FixedCodec(saved["state_dict"]["mean"], saved["state_dict"]["basis"])
    elif saved["kind"] == "learned":
        codec = LearnedCodec()
    else:
        raise ValueError("unknown codec kind")
    codec.load_state_dict(saved["state_dict"])
    codec.eval()
    return teacher, codec


def run(model, x, codec=None, **kwargs):
    return model.rollout(x, **kwargs) if isinstance(model, CompactTransition) else model.rollout(x, codec, **kwargs)


def pattern_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["public_input_sha256"]].append(row)
    fields = ("correct_decision", "legitimate_completion", "unauthorized_attempt", "audit_exact", "audit_cell_accuracy")
    per_pattern = []
    for key, values in sorted(groups.items()):
        # Equal public inputs must receive identical deterministic outputs.
        if len({(r["prediction"], tuple(r["audit_prediction"])) for r in values}) != 1:
            raise ValueError("duplicate public inputs produced different deterministic predictions")
        row = values[0]
        record = {"public_input_sha256": key, "copies": len(values),
                  "statuses": row["status_pattern"], "depth_feature": row["depth_feature"],
                  "probe_order": row["probe_order"], "has_valid_branch": row["has_valid_branch"]}
        record.update({field: sum(float(r[field]) for r in values) / len(values) for field in fields})
        per_pattern.append(record)
    count = len(per_pattern)
    positive = [r for r in per_pattern if r["has_valid_branch"]]
    negative = [r for r in per_pattern if not r["has_valid_branch"]]
    result = {field + "_rate": sum(r[field] for r in per_pattern) / count for field in fields}
    result.update({"cases": len(rows), "exact_feature_patterns": count,
                   "status_patterns": len({tuple(r["status_pattern"]) for r in rows}),
                   "positive_completion_rate": sum(r["legitimate_completion"] for r in positive) / len(positive) if positive else None,
                   "correct_abstention_rate": sum(r["correct_decision"] for r in negative) / len(negative) if negative else None,
                   "positive_patterns": len(positive), "negative_patterns": len(negative),
                   "per_pattern": per_pattern})
    representative_rows = [values[0] for values in groups.values()]
    result["audit_branch_accuracy"] = [sum(r["audit_prediction"][b] == r["audit_target"][b]
                                           for r in representative_rows) / count for b in range(2)]
    return result


def control_inputs(x):
    # Shuffle whole histories inside public-context strata; retain original
    # scoring target and final marker. This is deliberately a negative control.
    shuffled = x.clone()
    groups = defaultdict(list)
    for i in range(len(x)):
        groups[(float(x[i, 0, 3]), float(x[i, 0, 8]))].append(i)
    for indices in groups.values():
        patterns = {}
        for i in indices:
            patterns.setdefault(canonical_hash(x[i].tolist()), i)
        keys = sorted(patterns)
        mapped = {key: patterns[keys[(j + 1) % len(keys)]] for j, key in enumerate(keys)}
        for i in indices:
            shuffled[i, :2] = x[mapped[canonical_hash(x[i].tolist())], :2]
    return shuffled


def score_arm(model, codec, episodes, x, task, audit, seed, arm, condition="clean"):
    inputs = control_inputs(x) if condition == "shuffled_history" else x
    kwargs = {"zero_between_calls": condition == "zero_between_calls"}
    with torch.no_grad():
        run(model, inputs, codec, **kwargs)  # declared warm-up, no fitting
        began = time.perf_counter()
        for _ in range(3):
            trace = run(model, inputs, codec, **kwargs)
        elapsed = (time.perf_counter() - began) / 3
    prediction = trace.task_logits.argmax(-1).tolist()
    ap = trace.audit_logits.argmax(-1).tolist()
    rows = []
    for i, ep in enumerate(episodes):
        verdict = gym.score_decision(ep, prediction[i])
        target_audit = audit[i].tolist()
        row = {"case_id": ep.case_id, "split": ep.split, "family": ep.family,
               "seed": seed, "arm": arm, "condition": condition,
               "branch": prediction[i] if prediction[i] < 2 else None,
               "prediction": prediction[i], "target_action": int(task[i]),
               "audit_prediction": ap[i], "audit_target": target_audit,
               "audit_exact": ap[i] == target_audit,
               "audit_cell_accuracy": sum(a == b for a, b in zip(ap[i], target_audit)) / 2,
               "public_input_sha256": canonical_hash(x[i].tolist()),
               "observed_input_sha256": canonical_hash(inputs[i].tolist()),
               "status_pattern": list(ep.statuses), "probe_order": list(ep.probe_order),
               "depth_feature": float(x[i, 0, 8]), "has_valid_branch": "valid" in ep.statuses,
               "model_seconds_batch_mean_per_case": elapsed / len(x)}
        row.update(verdict)
        rows.append(row)
    summary = pattern_summary(rows)
    summary["model_seconds_per_batch"] = elapsed
    summary["model_seconds_batch_mean_per_case"] = elapsed / len(x)
    summary["timing_scope"] = "one warm-up then three full-batch CPU forward passes; excludes environment, Lean, fitting and checkpoint load"
    summary["resources"] = resources(model, codec)
    return rows, summary, trace


def unique_indices(x):
    seen = set()
    result = []
    for i, row in enumerate(x.tolist()):
        key = canonical_hash(row)
        if key not in seen:
            result.append(i)
            seen.add(key)
    return result


def counterfactuals(model, codec, episodes, x, audit, seed, arm):
    """Exhaustive donor pairs on distinct registered public input patterns."""
    unique = unique_indices(x)
    receivers, donors = [], []
    for i in unique:
        for j in unique:
            if torch.equal(x[i, 0, [3, 8]], x[j, 0, [3, 8]]):
                receivers.append(i)
                donors.append(j)
    ri, di = torch.tensor(receivers), torch.tensor(donors)
    rx = x[ri]
    dx = x[di]
    if not torch.equal(rx[:, 0, 8], dx[:, 0, 8]):
        raise ValueError("counterfactual donor context mismatch")
    if not torch.equal(rx[:, :, 8], rx[:, :1, 8].expand(-1, x.shape[1])):
        raise ValueError("counterfactual recipient has inconsistent public depth")
    first = rx[:, 0, 3].round().long()
    ra = audit[ri].clone()
    rows_ix = torch.arange(len(ri))
    changed = ra[rows_ix, first] != audit[di, first]
    ra[rows_ix, first] = audit[di, first]
    expected_task = task_from_audits(ra)
    hybrid = rx.clone()
    hybrid[:, 0] = dx[:, 0]
    with torch.no_grad():
        prefix = run(model, x, codec, stop_event=1)
        initial = prefix.carried[di, -1] if isinstance(model, CompactTransition) else prefix.states[di, -1]
        result = run(model, rx, codec, initial=initial, start_event=1)
        direct = run(model, hybrid, codec)
    pred = result.task_logits.argmax(-1)
    ap = result.audit_logits.argmax(-1)
    direct_task = direct.task_logits.argmax(-1)
    direct_audit = direct.audit_logits.argmax(-1)
    action_ok = pred == expected_task
    audit_ok = (ap == ra).all(-1)
    direct_ok = (pred == direct_task) & (ap == direct_audit).all(-1)
    records = []
    for q, (i, j) in enumerate(zip(receivers, donors)):
        records.append({"seed": seed, "arm": arm, "split": episodes[i].split,
                        "receiver_case_id": episodes[i].case_id,
                        "donor_case_id": episodes[j].case_id,
                        "first_probed_branch": int(first[q]),
                        "public_depth_feature": float(rx[q, 0, 8]),
                        "changed_prefix_status": bool(changed[q]),
                        "prediction": int(pred[q]), "expected_action": int(expected_task[q]),
                        "audit_prediction": ap[q].tolist(), "audit_target": ra[q].tolist(),
                        "action_correct": bool(action_ok[q]), "audit_exact": bool(audit_ok[q]),
                        "direct_hybrid_replay_agreement": bool(direct_ok[q]),
                        "scope": "public-history counterfactual; not a certificate against the receiver's original host theorem"})
    eligible = int(changed.sum())
    summary = {"pattern_pairs": len(ri), "changed_status_pairs": eligible,
               "all_pairs_action_accuracy": float(action_ok.float().mean()),
               "all_pairs_audit_exact": float(audit_ok.float().mean()),
               "changed_pairs_action_accuracy": float(action_ok[changed].float().mean()) if eligible else None,
               "changed_pairs_audit_exact": float(audit_ok[changed].float().mean()) if eligible else None,
               "direct_hybrid_replay_agreement": float(direct_ok.float().mean()),
               "max_task_logit_replay_error": float((result.task_logits - direct.task_logits).abs().max()),
               "max_audit_logit_replay_error": float((result.audit_logits - direct.audit_logits).abs().max()),
               "scope": "exhaustive ordered distinct-input donor pairs within branch/depth context; descriptive finite counts"}
    return records, summary


def collision_audit(model, codec, x, task, audit, native):
    """Float32 exact-code collision search at the final registered boundary."""
    with torch.no_grad():
        trace = run(model, x, codec)
        code = trace.carried[:, -1]
    groups = defaultdict(list)
    for i in unique_indices(x):
        groups[code[i].contiguous().numpy().tobytes()].append(i)
    collisions = []
    for indices in groups.values():
        for i in indices:
            for j in indices:
                if i < j and (int(task[i]) != int(task[j]) or not torch.equal(audit[i], audit[j])):
                    collisions.append({"left": i, "right": j,
                                       "left_signature": [int(task[i])] + audit[i].tolist(),
                                       "right_signature": [int(task[j])] + audit[j].tolist()})
    unique = unique_indices(x)
    distances = []
    for pos, i in enumerate(unique):
        for j in unique[pos + 1:]:
            if int(task[i]) != int(task[j]) or not torch.equal(audit[i], audit[j]):
                distances.append(float(torch.linalg.vector_norm(code[i] - code[j])))
    return {"code_coordinates": code.shape[-1], "precision": "float32 raw-byte equality",
            "exact_equal_code_different_signature_pairs": len(collisions),
            "witnesses": collisions, "minimum_discordant_signature_distance": min(distances) if distances else None,
            "scope": "finite final-boundary current task/audit readouts; zero observed collisions is not a universal continuation certificate"}


def registered_gates(seed_results):
    per_seed = []
    for seed_record in seed_results:
        seed = seed_record["seed"]
        eligible = seed_record["source_qualified"]
        splits = seed_record["splits"]
        if "order_ood" not in splits or "depth_ood" not in splits:
            per_seed.append({"seed": seed, "funnel": "unscored", "resource_gain": "unscored"})
            continue
        order = splits["order_ood"]
        task, joint = order["task"], order["joint"]
        gap = joint["correct_decision_rate"] - task["correct_decision_rate"]
        funnel_pass = (gap >= .05 - 1e-9 and
                       joint["unauthorized_attempt_rate"] <= task["unauthorized_attempt_rate"] + 1e-9 and
                       joint["audit_exact_rate"] >= .95)
        compact_checks = {}
        for split in ("depth_ood", "order_ood"):
            source, compact = splits[split]["teacher"], splits[split]["compact_transition"]
            compact_checks[split] = {
                "positive_completion_gap": compact["positive_completion_rate"] - source["positive_completion_rate"],
                "no_extra_tainted_submissions": compact["unauthorized_attempt_rate"] <= source["unauthorized_attempt_rate"] + 1e-9,
                "audit_exact_at_least_095": compact["audit_exact_rate"] >= .95,
                "lower_counted_macs": compact["resources"]["dense_matrix_macs_per_episode"] < source["resources"]["dense_matrix_macs_per_episode"]}
            c = compact_checks[split]
            c["pass"] = c["positive_completion_gap"] >= -.02 - 1e-9 and all(c[k] for k in
                 ("no_extra_tainted_submissions", "audit_exact_at_least_095", "lower_counted_macs"))
        resource_pass = all(c["pass"] for c in compact_checks.values())
        per_seed.append({"seed": seed, "source_qualified": eligible,
                         "funnel": ("pass" if funnel_pass else "fail") if eligible else "inconclusive",
                         "order_task_accuracy_gain": gap,
                         "joint_audit_exact": joint["audit_exact_rate"],
                         "resource_gain": ("pass" if resource_pass else "fail") if eligible else "inconclusive",
                         "compact_checks": compact_checks})
    aggregate = {}
    for name in ("funnel", "resource_gain"):
        values = [r[name] for r in per_seed]
        aggregate[name] = "fail" if "fail" in values else "pass" if values and all(v == "pass" for v in values) else "inconclusive"
    return {"per_seed": per_seed, "overall": aggregate,
            "scope": "registered finite input patterns and seeds; kernel evidence is a separate required gate"}


def evaluate(checkpoints, out, splits=("depth_ood", "order_ood"), allow_smoke=False, threads=2):
    manifest = load_frozen(checkpoints, allow_smoke)
    if out.exists() and any(out.iterdir()):
        raise ValueError("evaluation output must be empty; preserve previous scored runs")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data = {}
    episodes_all = []
    for split in splits:
        eps = gym.generate_split(split)
        data[split] = (eps, *tensor_data(eps))
        episodes_all.extend(eps)
    (out / "test_episodes.jsonl").write_text("".join(json.dumps(e.to_dict(), sort_keys=True) + "\n" for e in episodes_all))
    summary = {"experiment": "AF-01", "training_manifest_sha256": sha256_file(checkpoints / "train_manifest.json"),
               "protocol_sha256": REGISTERED_PROTOCOL_SHA256, "evaluation_source_sha256": sha256_file(__file__),
               "checkpoint_files": manifest["files"], "splits": list(splits),
               "scope": "registered_primary" if manifest["scope"] == "registered_primary" and tuple(splits) == ("depth_ood", "order_ood") else "engineering_development",
               "test_episodes_sha256": sha256_file(out / "test_episodes.jsonl"),
               "seeds": [], "lean_certification": "pending separate certify_selected execution"}
    with (out / "outcomes.jsonl").open("w") as outcomes, (out / "counterfactuals.jsonl").open("w") as cf_file:
        for seed_meta in manifest["seeds"]:
            seed = seed_meta["seed"]
            teacher = RecurrentTeacher()
            teacher.load_state_dict(_load(checkpoints / f"teacher_s{seed}.pt")["state_dict"])
            teacher.eval()
            seed_summary = {"seed": seed, "source_qualified": seed_meta["source_qualified"], "splits": {}}
            for split, (episodes, x, task, audit) in data.items():
                results = {}
                with torch.no_grad():
                    native = teacher.rollout(x)
                for arm in ("teacher",) + CODEC_ARMS + ("compact_transition",):
                    model, codec = load_arm(checkpoints, seed, arm, teacher)
                    rows, stats, trace = score_arm(model, codec, episodes, x, task, audit, seed, arm)
                    for row in rows:
                        outcomes.write(json.dumps(row, sort_keys=True) + "\n")
                    cf_rows, cf_stats = counterfactuals(model, codec, episodes, x, audit, seed, arm)
                    for row in cf_rows:
                        cf_file.write(json.dumps(row, sort_keys=True) + "\n")
                    stats["counterfactual"] = cf_stats
                    stats["collision_audit"] = collision_audit(model, codec, x, task, audit, native)
                    with torch.no_grad():
                        stats["native_hidden_trace_mse"] = float(F.mse_loss(trace.states, native.states))
                        stats["one_step_commutation_mse"] = float(commutation_loss(teacher, codec, x, native.states)) if codec is not None else (0. if arm == "teacher" else None)
                    results[arm] = stats
                    print(json.dumps({"seed": seed, "split": split, "arm": arm,
                                      "task": stats["correct_decision_rate"],
                                      "audit": stats["audit_exact_rate"],
                                      "tainted": stats["unauthorized_attempt_rate"]}), flush=True)
                for condition in ("zero_between_calls", "shuffled_history"):
                    rows, stats, _ = score_arm(teacher, None, episodes, x, task, audit, seed, "teacher", condition)
                    for row in rows:
                        outcomes.write(json.dumps(row, sort_keys=True) + "\n")
                    results["teacher_" + condition] = stats
                seed_summary["splits"][split] = results
            summary["seeds"].append(seed_summary)
    summary["gates"] = registered_gates(summary["seeds"])
    summary["outcomes_sha256"] = sha256_file(out / "outcomes.jsonl")
    summary["counterfactuals_sha256"] = sha256_file(out / "counterfactuals.jsonl")
    (out / "evaluation.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["dev", "depth_ood", "order_ood"], default=["depth_ood", "order_ood"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    evaluate(args.checkpoint_dir, args.out, tuple(args.splits), args.allow_smoke, args.threads)


if __name__ == "__main__":
    main()
