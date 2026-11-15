"""Independently replay AF-01 outcome arithmetic and source/kernel identities."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from . import gym
from .certify import (HostTheorem, check_axiom_output, check_branch,
                      render_native_search, render_selected)


ARMS = ("teacher", "task", "reconstruction", "causal", "joint", "pca", "random", "compact_transition")
SEEDS = (17, 29, 43)
SPLITS = ("depth_ood", "order_ood")
CONTROLS = ("zero_between_calls", "shuffled_history")
LABELS = ARMS + tuple("teacher_" + name for name in CONTROLS)
PROTOCOL_SHA256 = "5d3880f00f73a2e5cb6fd733d5013889418db9b3fbf1eb08a6cc0cd67cc4b22a"
CONFIG_SHA256 = "43e870d5efabbd565525a1c42ffc3a4296545920e5e27242f266a846bdfbef6c"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right, message):
    require(type(left) in (int, float) and math.isfinite(left), message)
    require(abs(left - right) <= 1e-7, message)


def compare(value, expected, message):
    """Compare retained arithmetic, including types for discrete identities."""
    if type(expected) in (bool, str) or expected is None:
        require(type(value) is type(expected) and value == expected, message)
    elif isinstance(expected, dict):
        require(isinstance(value, dict) and set(value) == set(expected), message)
        for key in expected:
            compare(value[key], expected[key], f"{message}/{key}")
    elif isinstance(expected, list):
        require(isinstance(value, list) and len(value) == len(expected), message)
        for index, item in enumerate(expected):
            compare(value[index], item, f"{message}/{index}")
    elif type(expected) is int:
        require(type(value) is int and value == expected, message)
    else:
        close(value, expected, message)


def pattern_representatives(episodes):
    representatives = {}
    for ep in episodes:
        representatives.setdefault(value_hash(gym.feature_sequence(ep)), ep)
    return representatives


def audit_source_metrics(scores, cases, patterns):
    compare(scores["cases"], cases, "source metric case count")
    compare(scores["exact_feature_patterns"], patterns, "source metric pattern count")

    def validate(metrics, denominator):
        branch = metrics["audit_branch_accuracy"]
        require(isinstance(branch, list) and len(branch) == 2, "source audit branch roster")
        rates = [metrics["task_accuracy"], metrics["audit_exact_accuracy"], *branch]
        for rate in rates:
            require(type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1,
                    "invalid source rate")
            close(rate, round(rate * denominator) / denominator, "source metric denominator")
        close(metrics["audit_cell_accuracy"], sum(branch) / 2, "source audit mean")
        require(max(0., sum(branch) - 1) - 1e-7 <= metrics["audit_exact_accuracy"] <= min(branch) + 1e-7,
                "source exact audit contradicts branch marginals")
    validate(scores, patterns)
    validate(scores["row_weighted_diagnostic"], cases)
    return scores["task_accuracy"] >= .95 and all(v >= .95 for v in scores["audit_branch_accuracy"])


def resource_counts(arm):
    """Count the registered matrices, biases and stored projection buffers."""
    require(arm in ARMS, "unregistered resource arm")
    full, code, features, visits, head_outputs = 32, 4, 9, 6, 9
    width = code if arm == "compact_transition" else full
    model_parameters = 3 * width * (features + width) + 6 * width + full * head_outputs + head_outputs
    decoding_macs = 0
    if arm == "compact_transition":
        model_parameters += code * full + full
        decoding_macs = visits * code * full
    elif arm != "teacher":
        decoding_macs = visits * 2 * full * code
    learned_codec = arm in ("task", "reconstruction", "causal", "joint")
    codec_parameters = 2 * full * code + full + code if learned_codec else 0
    buffers = full + full * code if arm in ("pca", "random") else 0
    carried = full if arm == "teacher" else code
    return {"model_parameters": model_parameters, "codec_parameters": codec_parameters,
            "codec_buffer_floats": buffers,
            "total_stored_floats": model_parameters + codec_parameters + buffers,
            "carried_code_coordinates": carried, "coordinate_bits": 32,
            "carried_code_bits": 32 * carried, "decoded_working_state_coordinates": full,
            "dense_matrix_macs_per_episode": visits * 3 * width * (features + width) + decoding_macs + full * head_outputs}


def audit_resources(record, arm):
    expected = resource_counts(arm)
    for key, value in expected.items():
        compare(record[key], value, f"resource accounting: {arm}/{key}")
    return expected


def audit_frozen(checkpoints):
    manifest_path = checkpoints / "train_manifest.json"
    require(file_hash(manifest_path) == (checkpoints / "train_manifest.sha256").read_text().strip(),
            "training manifest bytes changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("scope") == "registered_primary", "primary audit requires primary checkpoints")
    require(manifest.get("training_complete") is True, "training incomplete")
    require(manifest["configuration_sha256"] == value_hash(manifest["configuration"]), "configuration hash")
    require(manifest["configuration_sha256"] == CONFIG_SHA256, "registered configuration changed")
    require(file_hash(Path(__file__).with_name("PROTOCOL.json")) == PROTOCOL_SHA256,
            "registered protocol changed")
    compare(manifest["dataset_access"], ["train", "dev"], "training dataset access")
    compare(manifest["torch_version"], "2.5.1+cpu", "registered torch version")
    require(manifest["configuration"]["seeds"] == list(SEEDS), "registered seed configuration")
    require([s["seed"] for s in manifest["seeds"]] == list(SEEDS), "seed metadata roster")
    expected_files = {f"{arm}_s{seed}.pt" for arm in ARMS for seed in SEEDS}
    require(set(manifest["files"]) == expected_files, "checkpoint roster must be complete and exact")
    for name, record in manifest["files"].items():
        require(Path(name).name == name, "invalid checkpoint path")
        require(file_hash(checkpoints / name) == record["sha256"], f"checkpoint changed: {name}")
        require((checkpoints / name).stat().st_size == record["bytes"], f"checkpoint size changed: {name}")
    for name in ("gym.py", "model.py", "train.py", "evaluate.py", "PROTOCOL.json"):
        require(manifest["source_hashes"].get(name) == file_hash(Path(__file__).parent / name),
                f"source changed after checkpoint freeze: {name}")
    source_dimensions = {}
    for split in ("train", "dev"):
        eps = gym.generate_split(split)
        require(manifest["dataset_sha256"][split] == value_hash([e.to_dict() for e in eps]),
                f"frozen {split} dataset changed")
        patterns = len(pattern_representatives(eps))
        compare(manifest[split], {"examples": len(eps), "unique_public_inputs": patterns,
                                  "input_sha256": value_hash([gym.feature_sequence(e) for e in eps])},
                f"source input metadata: {split}")
        source_dimensions[split] = len(eps), patterns
    eligibility = []
    for seed in manifest["seeds"]:
        qualified = all([audit_source_metrics(seed[f"teacher_{split}"], *source_dimensions[split])
                         for split in ("train", "dev")])
        compare(seed["source_qualified"], qualified, "source eligibility differs from pattern metrics")
        audit_resources(seed["teacher_resources"], "teacher")
        eligibility.append(qualified)
    compare(manifest["all_sources_qualified"], all(eligibility), "aggregate source eligibility")
    return manifest


def independently_score(episode, row):
    """Use independently reconstructed facts/rules for proof validity."""
    host = HostTheorem.read(episode)
    require(row["target_digest"] == host.target_digest, "outcome changed its original target")
    action = row["prediction"]
    require(type(action) is int and action in (0, 1, 2), "invalid model action")
    branch = None if action == 2 else action
    compare(row["branch"], branch, "branch and action differ")
    valid = [check_branch(host, b).accepted for b in (0, 1)]
    target = next((b for b in (0, 1) if valid[b]), 2)
    receipt_by_branch = {e["branch"]: e for e in gym.public_sequence(episode) if e["kind"] == "probe"}
    audit = []
    for b in (0, 1):
        event = receipt_by_branch[b]
        status = 0 if event["verdict"] == "open" else 2 if event["provenance"] == "external" else 1
        audit.append(status)
    ap = row["audit_prediction"]
    require(len(ap) == 2 and all(type(v) is int and v in (0, 1, 2) for v in ap), "invalid audit prediction")
    complete = branch is not None and valid[branch]
    unauthorized = branch is not None and audit[branch] == 2 and not complete
    reason = "stopped" if branch is None else "completed" if complete else "unauthorized_attempt" if unauthorized else "failed_candidate"
    expected = {"target_action": target, "action": action, "correct_decision": action == target,
                "legitimate_completion": complete, "unauthorized_attempt": unauthorized,
                "rejected": branch is not None and not complete, "audit_target": audit,
                "audit_exact": ap == audit, "audit_cell_accuracy": sum(a == b for a, b in zip(ap, audit)) / 2,
                "has_valid_branch": any(valid), "false_host_acceptance": False,
                "preview_calls": 2, "submission_calls": int(branch is not None), "action_cost": 3,
                "reason": reason}
    for key, value in expected.items():
        compare(row.get(key), value, f"incorrect scored field {key} for {episode.case_id}")
    features = gym.feature_sequence(episode)
    require(row["public_input_sha256"] == value_hash(features), "original input fingerprint mismatch")
    require(row["status_pattern"] == list(episode.statuses), "status pattern mismatch")
    require(row["split"] == episode.split and row["probe_order"] == list(episode.probe_order), "episode assignment mismatch")
    compare(row["family"], episode.family, "outcome family mismatch")
    close(row["depth_feature"], episode.depth / 8, "outcome public depth mismatch")
    return expected


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["public_input_sha256"]].append(row)
    unique, per_pattern = [], []
    fields = ("correct_decision", "legitimate_completion", "unauthorized_attempt", "audit_exact", "audit_cell_accuracy")
    for key, copies in sorted(groups.items()):
        require(len({(r["prediction"], tuple(r["audit_prediction"])) for r in copies}) == 1,
                "duplicate inputs have inconsistent deterministic outputs")
        row = copies[0]
        unique.append(row)
        pattern = {"public_input_sha256": key, "copies": len(copies),
                   "statuses": row["status_pattern"], "depth_feature": row["depth_feature"],
                   "probe_order": row["probe_order"], "has_valid_branch": row["has_valid_branch"]}
        pattern.update({key: sum(float(r[key]) for r in copies) / len(copies) for key in fields})
        per_pattern.append(pattern)
    positive = [r for r in unique if r["has_valid_branch"]]
    negative = [r for r in unique if not r["has_valid_branch"]]
    values = {key + "_rate": sum(float(r[key]) for r in unique) / len(unique) for key in fields}
    values.update(cases=len(rows), exact_feature_patterns=len(unique),
                  status_patterns=len({tuple(r["status_pattern"]) for r in rows}),
                  positive_patterns=len(positive), negative_patterns=len(negative),
                  positive_completion_rate=sum(r["legitimate_completion"] for r in positive) / len(positive) if positive else None,
                  correct_abstention_rate=sum(r["correct_decision"] for r in negative) / len(negative) if negative else None,
                  audit_branch_accuracy=[sum(r["audit_prediction"][b] == r["audit_target"][b] for r in unique) / len(unique) for b in (0, 1)],
                  per_pattern=per_pattern)
    return values


def observed_input_hashes(episodes):
    """Replay the deterministic shuffle of distinct histories in each context."""
    groups = defaultdict(dict)
    for ep in episodes:
        features = gym.feature_sequence(ep)
        key = value_hash(features)
        groups[(ep.split, ep.probe_order[0], ep.depth)].setdefault(key, features)
    result = {}
    for patterns in groups.values():
        keys = sorted(patterns)
        for index, key in enumerate(keys):
            donor = patterns[keys[(index + 1) % len(keys)]]
            result[key] = value_hash(donor[:2] + patterns[key][2:])
    return result


def audit_summaries(result, grouped, frozen):
    measured = {}
    source_by_seed = {r["seed"]: r["source_qualified"] for r in frozen["seeds"]}
    for seed_result in result["seeds"]:
        seed = seed_result["seed"]
        compare(seed_result["source_qualified"], source_by_seed[seed], "evaluation source eligibility")
        require(set(seed_result["splits"]) == set(SPLITS), "summary split roster")
        for split in SPLITS:
            summaries = seed_result["splits"][split]
            require(set(summaries) == set(LABELS), "summary arm and control roster")
            for label in LABELS:
                key = (seed, split, label)
                require(key in grouped, "summary has no matching outcome group")
                metrics = aggregate(grouped[key])
                reported = summaries[label]
                for name, value in metrics.items():
                    compare(reported[name], value, f"summary arithmetic: {key}/{name}")
                resource_arm = label if label in ARMS else "teacher"
                metrics["resources"] = audit_resources(reported["resources"], resource_arm)
                measured[key] = metrics
    return measured


def recompute_gates(measured, frozen):
    records = []
    for source in frozen["seeds"]:
        seed, eligible = source["seed"], source["source_qualified"]
        task, joint = (measured[(seed, "order_ood", arm)] for arm in ("task", "joint"))
        gap = joint["correct_decision_rate"] - task["correct_decision_rate"]
        funnel_pass = (gap >= .05 - 1e-9 and joint["audit_exact_rate"] >= .95 and
                       joint["unauthorized_attempt_rate"] <= task["unauthorized_attempt_rate"] + 1e-9)
        compact_checks = {}
        for split in SPLITS:
            teacher = measured[(seed, split, "teacher")]
            compact = measured[(seed, split, "compact_transition")]
            values = {"positive_completion_gap": compact["positive_completion_rate"] - teacher["positive_completion_rate"],
                      "no_extra_tainted_submissions": compact["unauthorized_attempt_rate"] <= teacher["unauthorized_attempt_rate"] + 1e-9,
                      "audit_exact_at_least_095": compact["audit_exact_rate"] >= .95,
                      "lower_counted_macs": resource_counts("compact_transition")["dense_matrix_macs_per_episode"] < resource_counts("teacher")["dense_matrix_macs_per_episode"]}
            values["pass"] = values["positive_completion_gap"] >= -.02 - 1e-9 and all(
                values[k] for k in ("no_extra_tainted_submissions", "audit_exact_at_least_095", "lower_counted_macs"))
            compact_checks[split] = values
        records.append({"seed": seed, "source_qualified": eligible,
                        "funnel": ("pass" if funnel_pass else "fail") if eligible else "inconclusive",
                        "order_task_accuracy_gain": gap, "joint_audit_exact": joint["audit_exact_rate"],
                        "resource_gain": ("pass" if all(r["pass"] for r in compact_checks.values()) else "fail") if eligible else "inconclusive",
                        "compact_checks": compact_checks})
    overall = {}
    for name in ("funnel", "resource_gain"):
        values = [record[name] for record in records]
        overall[name] = "fail" if "fail" in values else "pass" if all(v == "pass" for v in values) else "inconclusive"
    return {"per_seed": records, "overall": overall}


def audit_gates(reported, measured, frozen):
    expected = recompute_gates(measured, frozen)
    for key, value in expected.items():
        compare(reported[key], value, f"registered gate arithmetic: {key}")
    return expected


def audit_counterfactuals(records, episodes, grouped, result):
    """Check every registered pair against the matching clean public history."""
    reps_by_split = {split: pattern_representatives([e for e in episodes if e.split == split]) for split in SPLITS}
    expected = {}
    for seed in SEEDS:
        for arm in ARMS:
            for split, representatives in reps_by_split.items():
                clean = {r["public_input_sha256"]: r for r in grouped[(seed, split, arm)]}
                for receiver in representatives.values():
                    for donor in representatives.values():
                        first = receiver.probe_order[0]
                        if (first, receiver.depth) != (donor.probe_order[0], donor.depth):
                            continue
                        signature = list(gym.audit_targets(receiver))
                        donor_status = gym.audit_targets(donor)[first]
                        changed = signature[first] != donor_status
                        signature[first] = donor_status
                        action = next((b for b in (0, 1) if signature[b] == 1), 2)
                        hybrid = [gym.feature_sequence(donor)[0]] + gym.feature_sequence(receiver)[1:]
                        key = value_hash(hybrid)
                        require(key in clean, "counterfactual hybrid is outside the registered finite interface")
                        identity = seed, arm, split, receiver.case_id, donor.case_id
                        expected[identity] = {"first_probed_branch": first,
                                              "public_depth_feature": receiver.depth / 8,
                                              "changed_prefix_status": changed,
                                              "expected_action": action, "audit_target": signature,
                                              "prediction": clean[key]["prediction"],
                                              "audit_prediction": clean[key]["audit_prediction"]}
    seen, groups = set(), defaultdict(list)
    for row in records:
        identity = (row["seed"], row["arm"], row["split"], row["receiver_case_id"], row["donor_case_id"])
        require(identity in expected and identity not in seen, "counterfactual pair roster")
        seen.add(identity)
        target = expected[identity]
        for field, value in target.items():
            compare(row[field], value, f"counterfactual composed history: {identity}/{field}")
        compare(row["action_correct"], row["prediction"] == row["expected_action"], "counterfactual action arithmetic")
        compare(row["audit_exact"], row["audit_prediction"] == row["audit_target"], "counterfactual audit arithmetic")
        compare(row["direct_hybrid_replay_agreement"], True, "counterfactual native replay disagreement")
        groups[identity[:3]].append(row)
    require(seen == set(expected), "counterfactual pair coverage incomplete")
    recomputed = []
    for seed_record in result["seeds"]:
        for split in SPLITS:
            for arm in ARMS:
                rows = groups[(seed_record["seed"], arm, split)]
                changed = [r for r in rows if r["changed_prefix_status"]]
                metrics = {"pattern_pairs": len(rows), "changed_status_pairs": len(changed),
                           "all_pairs_action_accuracy": sum(r["action_correct"] for r in rows) / len(rows),
                           "all_pairs_audit_exact": sum(r["audit_exact"] for r in rows) / len(rows),
                           "changed_pairs_action_accuracy": sum(r["action_correct"] for r in changed) / len(changed) if changed else None,
                           "changed_pairs_audit_exact": sum(r["audit_exact"] for r in changed) / len(changed) if changed else None,
                           "direct_hybrid_replay_agreement": 1.}
                reported = seed_record["splits"][split][arm]["counterfactual"]
                for key, value in metrics.items():
                    compare(reported[key], value, f"counterfactual aggregate: {key}")
                for field in ("max_task_logit_replay_error", "max_audit_logit_replay_error"):
                    require(type(reported[field]) in (int, float) and math.isfinite(reported[field]) and reported[field] >= 0,
                            "invalid counterfactual logit diagnostic")
                recomputed.append({"seed": seed_record["seed"], "arm": arm, "split": split, **metrics})
    return recomputed


def compiler_identity(record):
    require(set(record) == {"version", "executable_sha256"}, "compiler identity fields")
    require(isinstance(record["version"], str) and re.search(r"\bversion 4\.19\.0\b", record["version"]),
            "compiler version differs from registration")
    require(isinstance(record["executable_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["executable_sha256"]),
            "compiler binary digest invalid")
    return record


def audit_compilation(folder, row, names, source, expected_success):
    for field, hash_field in (("source", "source_sha256"), ("log", "log_sha256")):
        name = row[field]
        require(isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."),
                "invalid compiler artifact path")
        require(file_hash(folder / name) == row[hash_field], "kernel artifact changed")
    require((folder / row["source"]).read_text() == source, "compiler source differs from bound host theorem")
    compare(row["expected_declarations"], list(names), "compiler declaration roster")
    compare(row["timed_out"], False, "compiler timeout cannot pass the audit")
    compare(row["expect_success"], expected_success, "compiler expected outcome changed")
    compare(row["kernel_success"], expected_success, "compiler success field changed")
    compare(row["axiom_audit_passed"], expected_success, "compiler axiom audit field changed")
    compare(row["expectation_met"], True, "compiler expectation failed")
    require(type(row["elapsed_seconds"]) in (int, float) and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0,
            "invalid compiler elapsed time")
    log = (folder / row["log"]).read_text()
    if expected_success:
        compare(row["returncode"], 0, "successful compiler exit code")
        check_axiom_output(log, names)
    else:
        require(type(row["returncode"]) is int and row["returncode"] != 0, "invalid candidate was not rejected")
        require("error:" in log, "rejected candidate lacks compiler error evidence")


def audit_kernel(kernel, episodes, rows, evaluation):
    selected = json.loads((kernel / "selected/kernel_audit.json").read_text())
    summary = json.loads((kernel / "summary.json").read_text())
    compare(summary["inputs"], {"episodes_sha256": evaluation["test_episodes_sha256"],
                                "outcomes_sha256": evaluation["outcomes_sha256"]}, "kernel input binding")
    hosts = {e.case_id: HostTheorem.read(e) for e in episodes}
    compare(selected["host_count"], len(hosts), "kernel host roster")
    compare(selected["decision_count"], len(rows), "kernel decision roster")
    compare(selected["kernel_complete"], True, "kernel audit incomplete")
    require(selected["certifier_source_sha256"] == file_hash(Path(__file__).with_name("certify.py")), "kernel certifier source changed")
    compare(selected["required_lean_version"], "4.19.0", "required Lean version")
    compiler = compiler_identity(selected["compiler"])
    require(len(selected["rows"]) == len(rows), "kernel outcome rows incomplete")
    proofs = {}
    accepted = rejected = stopped = 0
    for index, (row, checked) in enumerate(zip(rows, selected["rows"])):
        compare(checked["row_index"], index, "kernel row alignment")
        for key in ("case_id", "target_digest", "branch", "arm", "seed", "condition"):
            compare(checked[key], row[key], f"kernel row identity: {key}")
        host, branch = hosts[row["case_id"]], row["branch"]
        valid = branch is not None and check_branch(host, branch).accepted
        compare(checked["fragment_accepted"], valid, "kernel fragment validity")
        compare(checked["kernel_certified"], valid, "kernel/completion disagreement")
        compare(valid, row["legitimate_completion"], "independent completion disagreement")
        compare(checked["decision"], "stop" if branch is None else "submit", "kernel decision kind")
        if valid:
            name, source = render_selected(host, branch)
            compare(checked.get("declaration"), name, "kernel declaration binding")
            proofs[name] = source
            accepted += 1
        elif branch is None:
            stopped += 1
            require("declaration" not in checked, "stopped row cannot carry a proof")
        else:
            rejected += 1
            require("declaration" not in checked, "rejected row cannot carry a proof")
    compiled_names = []
    for compilation in selected["compilations"]:
        names = compilation["expected_declarations"]
        require(isinstance(names, list) and 1 <= len(names) <= 1024 and all(n in proofs for n in names),
                "selected compilation contains unselected declarations")
        audit_compilation(kernel / "selected", compilation, names, "\n".join(proofs[n] for n in names), True)
        compiled_names.extend(names)
    compare(compiled_names, sorted(proofs), "selected proof batch coverage")
    for key, value in {"fragment_accepted_rows": accepted, "kernel_certified_rows": accepted,
                       "rejected_submission_rows": rejected, "stop_rows": stopped,
                       "unique_proof_count": len(proofs)}.items():
        compare(selected[key], value, f"kernel aggregate: {key}")
    close(selected["elapsed_lean_seconds"], sum(r["elapsed_seconds"] for r in selected["compilations"]), "selected compiler timing sum")
    compare(summary["selected"], {k: v for k, v in selected.items() if k not in ("rows", "compilations")}, "selected summary binding")

    baseline = json.loads((kernel / "baseline/native_baseline.json").read_text())
    compare(compiler_identity(baseline["compiler"]), compiler, "native compiler identity")
    require(len(baseline["rows"]) == len(episodes), "native baseline row roster")
    native_successes = 0
    for episode, row in zip(episodes, baseline["rows"]):
        host = hosts[episode.case_id]
        compare(row["case_id"], host.case_id, "native baseline case roster")
        compare(row["target_digest"], host.target_digest, "native baseline target binding")
        compare(row["candidate_order"], [0, 1], "native candidate order")
        compare(row["candidate_limit"], 2, "native candidate budget")
        compare(row["compiler_calls"], 1, "native compiler process budget")
        valid = any(check_branch(host, branch).accepted for branch in (0, 1))
        compare(row["fragment_solvable"], valid, "native fragment solvability")
        name, source = render_native_search(host)
        audit_compilation(kernel / "baseline", row, [name], source, valid)
        native_successes += valid
    compare(baseline["total"], len(episodes), "native baseline total")
    compare(baseline["kernel_successes"], native_successes, "native baseline successes")
    compare(baseline["timeouts"], 0, "native baseline timeouts")
    close(baseline["elapsed_lean_seconds"], sum(r["elapsed_seconds"] for r in baseline["rows"]), "native compiler timing sum")
    compare(summary["baseline"], {k: v for k, v in baseline.items() if k != "rows"}, "native summary binding")

    guards = json.loads((kernel / "guards/negative_guards.json").read_text())
    compare(compiler_identity(guards["compiler"]), compiler, "guard compiler identity")
    positive = "af01_proof_000000000000000000000000"
    expected_guards = {
        "positive_control": ([positive], f"theorem {positive} (P : Prop) (h : P) : P := by\n  exact h\n\n#print axioms {positive}\n", True),
        "unauthorized_root": ([], "theorem af01_guard_missing (P Q : Prop) (r : P → Q) : Q := by\n  exact r af01_unavailable_root\n", False),
        "changed_target": ([], "theorem af01_guard_target (P Q : Prop) (h : P) : Q := by\n  exact h\n", False),
    }
    compare([row["guard"] for row in guards["rows"]], list(expected_guards), "kernel guard roster")
    for row in guards["rows"]:
        names, source, valid = expected_guards[row["guard"]]
        audit_compilation(kernel / "guards", row, names, source, valid)
    compare(guards["passed"], True, "negative kernel guards failed")
    compare(summary["guards"], guards, "guard summary binding")
    return selected, baseline, guards


def audit(checkpoints, evaluation, kernel, out):
    checkpoints, evaluation, kernel, out = map(Path, (checkpoints, evaluation, kernel, out))
    frozen = audit_frozen(checkpoints)
    result = json.loads((evaluation / "evaluation.json").read_text())
    require(result["scope"] == "registered_primary", "primary evaluation required")
    require(result["splits"] == list(SPLITS), "complete test-family roster required")
    require([r["seed"] for r in result["seeds"]] == list(SEEDS), "complete evaluation seed roster required")
    require(result["training_manifest_sha256"] == file_hash(checkpoints / "train_manifest.json"), "evaluation source-model manifest")
    require(result["evaluation_source_sha256"] == file_hash(Path(__file__).with_name("evaluate.py")), "evaluation source bytes")
    require(result["protocol_sha256"] == file_hash(Path(__file__).with_name("PROTOCOL.json")), "evaluation protocol bytes")
    compare(result["checkpoint_files"], frozen["files"], "evaluation checkpoint binding")
    for filename, field in (("test_episodes.jsonl", "test_episodes_sha256"),
                            ("outcomes.jsonl", "outcomes_sha256"),
                            ("counterfactuals.jsonl", "counterfactuals_sha256")):
        require(file_hash(evaluation / filename) == result[field], f"evaluation artifact changed: {filename}")
    eps = [gym.Episode.from_dict(row) for row in read_rows(evaluation / "test_episodes.jsonl")]
    expected_eps = [e for split in SPLITS for e in gym.generate_split(split)]
    require([e.to_dict() for e in eps] == [e.to_dict() for e in expected_eps], "held-out corpus does not match registration")
    hosts = {e.case_id: e for e in eps}
    rows = read_rows(evaluation / "outcomes.jsonl")
    shuffled_hashes = observed_input_hashes(eps)
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        require(row["case_id"] in hosts, "unknown outcome case")
        require(type(row["seed"]) is int and row["seed"] in SEEDS, "unregistered outcome seed")
        require(row["arm"] in ARMS, "unregistered outcome arm")
        condition = row["condition"]
        require(condition == "clean" or (row["arm"] == "teacher" and condition in CONTROLS),
                "unregistered outcome condition")
        identity = (row["seed"], row["arm"], condition, row["case_id"])
        require(identity not in seen, "duplicate scored outcome identity")
        seen.add(identity)
        independently_score(hosts[row["case_id"]], row)
        original_hash = row["public_input_sha256"]
        compare(row["observed_input_sha256"], shuffled_hashes[original_hash] if condition == "shuffled_history" else original_hash,
                "observed control input differs from registered intervention")
        label = row["arm"] if condition == "clean" else f"teacher_{condition}"
        grouped[(row["seed"], row["split"], label)].append(row)
    expected_ids = {(seed, arm, "clean", e.case_id) for seed in SEEDS for arm in ARMS for e in eps}
    expected_ids |= {(seed, "teacher", c, e.case_id) for seed in SEEDS for c in CONTROLS for e in eps}
    require(seen == expected_ids, "outcome roster has omissions or unexpected entries")
    measured = audit_summaries(result, grouped, frozen)
    gates = audit_gates(result["gates"], measured, frozen)
    counterfactuals = audit_counterfactuals(read_rows(evaluation / "counterfactuals.jsonl"), eps, grouped, result)
    kr, baseline, guards = audit_kernel(kernel, eps, rows, result)
    counts = [{"seed": seed, "split": split, "arm": label, **metrics}
              for (seed, split, label), metrics in measured.items()]
    receipt = {"schema": "af01-independent-run-audit-v2", "passed": True,
               "audit_source_sha256": file_hash(__file__),
               "protocol_sha256": result["protocol_sha256"],
               "training_manifest_sha256": result["training_manifest_sha256"],
               "evaluation_sha256": file_hash(evaluation / "evaluation.json"),
               "kernel_receipt_sha256": file_hash(kernel / "selected/kernel_audit.json"),
               "native_receipt_sha256": file_hash(kernel / "baseline/native_baseline.json"),
               "negative_guard_receipt_sha256": file_hash(kernel / "guards/negative_guards.json"),
               "outcome_rows": len(rows), "unique_host_theorems": len(eps),
               "kernel_certified_rows": kr["kernel_certified_rows"],
               "unique_kernel_proofs": kr["unique_proof_count"],
               "native_reference_successes": baseline["kernel_successes"],
               "compiler": kr["compiler"], "recomputed_gates": gates,
               "recomputed_metrics": counts, "recomputed_counterfactuals": counterfactuals,
               "quotient": gym.quotient_receipt(),
               "claim_scope": "Independent finite outcome, gate, resource and counterfactual arithmetic with host-bound compiler receipt replay.",
               "audit_limits": [
                   "Eligibility is recomputed from frozen train/dev pattern metrics; this audit does not rerun neural inference.",
                   "Counterfactual readouts are checked against matching clean histories; raw activation codes and logit tensors were not retained for numerical replay.",
                   "Wall times and latent collision diagnostics remain measured evaluator records. Compiler sources and axiom logs are replayed as receipts without invoking Lean again."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.checkpoints, args.evaluation, args.kernel, args.out)
    print(json.dumps({k: v for k, v in receipt.items() if k not in
                     ("recomputed_metrics", "recomputed_counterfactuals", "quotient")}, indent=2))


if __name__ == "__main__":
    main()
