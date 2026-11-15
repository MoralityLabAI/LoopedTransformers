"""Independent AF-02 tensor replay, finite outcome arithmetic and Lean receipts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from experiments.activation_funnel import audit_run as kernel_receipts
from experiments.activation_funnel.certify import HostTheorem, check_branch
from . import gym, inverse
from .replay import (NumpyCodec, NumpyModel, PROTOCOL_SHA256, assert_close,
                     file_hash, load_arrays, patch_code, require, resource_counts,
                     rollout)


ARCHITECTURES = ("gru", "native")
ARMS = ("teacher", "task", "joint")
SEEDS = (17, 29, 43)
SPLITS = ("depth_context", "delay2", "delay4")
CONTROLS = ("zero_between_calls", "address_erased")
LABELS = ARMS + tuple("teacher_" + name for name in CONTROLS)
PATCHES = ("whole", "self", "addressed_pair", "other_pair", "goal_only")
TRACE_FIELDS = ("before", "raw", "states", "carried", "task_logits", "audit_logits")
CONFIG_SHA256 = "74811acc4e850896e586041ebdd9ca9d362e860fd95c75206e886ddef06b2ecc"
KERNEL_REPLAY_SHA256 = "6b5dc700097bf41bb8ad8c87317a3c68a7fb2b9c5a080c2cb754e941253ccf44"


def plain(value):
    return json.loads(json.dumps(value))


def compare(actual, expected, label):
    kernel_receipts.compare(plain(actual), plain(expected), label)


def numeric(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual), label)
    require(abs(actual - expected) <= 1e-6 + 1e-6 * abs(expected), label)


def replayed_numeric(actual, expected, label):
    require(type(actual) in (int, float), label)
    assert_close(np.asarray(actual), np.asarray(expected), label)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def safe_path(folder, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in Path(relative).parts), "invalid artifact path")
    path = folder / relative
    require(path.resolve().is_relative_to(folder.resolve()), "artifact escaped its directory")
    return path


def verify_file(folder, name, record):
    path = safe_path(folder, name)
    require(file_hash(path) == record["sha256"] and path.stat().st_size == record["bytes"],
            f"artifact changed: {name}")


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    for name, array in arrays.items():
        require(array.dtype.kind in "fiubUS", f"unsupported tensor/archive dtype: {name}")
        if array.dtype.kind == "f":
            require(np.isfinite(array).all(), f"nonfinite retained tensor: {name}")
    return arrays


def exact_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.array_equal(actual, expected), label)


def prediction_fields(case, row, *, outcome=False):
    compare(row["case_id"], case.case_id, "case identity")
    for key, value in {"canonical_id": case.canonical_id, "split": case.split,
                       "target_digest": case.target_digest, "context_digest": case.context_digest,
                       "probe_order": case.probe_order, "schedule": case.schedule,
                       "depth": case.depth, "status_pattern": case.statuses,
                       "public_input_sha256": gym.public_input_hash(case)}.items():
        compare(row[key], value, f"case binding: {key}")
    task = np.asarray(row["task_logits"])
    audit = np.asarray(row["audit_logits"])
    require(task.shape == (3,) and audit.shape == (2, 3) and np.isfinite(task).all() and np.isfinite(audit).all(),
            "invalid raw model logits")
    prediction, audit_prediction = int(np.argmax(task)), np.argmax(audit, axis=-1).tolist()
    compare(row["prediction"], prediction, "raw task argmax")
    compare(row["audit_prediction"], audit_prediction, "raw audit argmax")
    compare(row["branch"], None if prediction == 2 else prediction, "selected branch")
    host = HostTheorem.read(case)
    supported = [check_branch(host, b).accepted for b in (0, 1)]
    target = next((b for b in (0, 1) if supported[b]), 2)
    protected = list(gym.audit_targets(case))
    compare(row["target_action"], target, "independent target action")
    compare(row["audit_target"], protected, "protected target")
    if outcome:
        submitted = prediction != 2
        completed = submitted and supported[prediction]
        unauthorized = submitted and protected[prediction] == 2 and not completed
        reason = "stopped" if not submitted else "completed" if completed else "unauthorized_attempt" if unauthorized else "failed_candidate"
        expected = {"action": prediction, "family": case.family, "correct_decision": prediction == target,
                    "legitimate_completion": completed, "unauthorized_attempt": unauthorized,
                    "rejected": submitted and not completed, "reason": reason,
                    "audit_exact": audit_prediction == protected,
                    "audit_cell_accuracy": mean(a == b for a, b in zip(audit_prediction, protected)),
                    "has_valid_branch": any(supported), "false_host_acceptance": False,
                    "preview_calls": 2, "wait_calls": case.schedule.count("W"),
                    "submission_calls": int(submitted), "initial_budget": case.budget,
                    "action_cost": case.budget, "remaining_budget": 0}
        for key, value in expected.items():
            compare(row[key], value, "host outcome arithmetic: " + key)


def prediction_summary(rows):
    def metrics(selected):
        require(bool(selected), "empty prediction stratum")
        return {"cases": len(selected), "exact_feature_patterns": len({r["public_input_sha256"] for r in selected}),
                "task_accuracy": mean(r["prediction"] == r["target_action"] for r in selected),
                "audit_branch_accuracy": [mean(r["audit_prediction"][b] == r["audit_target"][b] for r in selected) for b in (0, 1)],
                "audit_exact_accuracy": mean(r["audit_prediction"] == r["audit_target"] for r in selected)}
    result = metrics(rows)
    require(result["cases"] == result["exact_feature_patterns"], "duplicate canonical source predictions")
    result["by_order"] = {",".join(map(str, order)): metrics([r for r in rows if r["probe_order"] == list(order)]) for order in gym.ORDERS}
    return result


def source_eligibility(train, dev):
    return all(stratum["task_accuracy"] >= .95 and min(stratum["audit_branch_accuracy"]) >= .95
               for metrics in (train, dev) for stratum in metrics["by_order"].values())


def check_resources(record, architecture, arm, events):
    counts = resource_counts(architecture, arm, events)
    expected = {"model_parameters": counts["source_parameters"], "model_buffer_floats": counts["source_buffer_floats"],
                "codec_parameters": counts["codec_parameters"], "codec_buffer_floats": 0,
                "total_stored_floats": counts["total_stored_floats"],
                "source_state_coordinates": counts["decoded_working_coordinates"],
                "carried_code_coordinates": counts["carried_code_coordinates"], "coordinate_bits": 32,
                "carried_code_bits": counts["carried_code_bits"],
                "decoded_working_state_coordinates": counts["decoded_working_coordinates"],
                "real_events": events, "recurrent_visits": counts["visits"],
                "dense_matrix_macs_per_episode": counts["dense_matrix_macs"],
                "writer_dense_macs_per_visit": 288 if architecture == "native" else 0}
    for key, value in expected.items():
        compare(record[key], value, "resource accounting: " + key)
    return counts


def audit_frozen(checkpoints):
    path = checkpoints / "train_manifest.json"
    require(file_hash(path) == (checkpoints / "train_manifest.sha256").read_text().strip(), "training manifest hash")
    frozen = read_json(path)
    require(frozen["experiment"] == "AF-02" and frozen["scope"] == "registered_primary" and frozen["training_complete"] is True,
            "primary frozen training required")
    require(frozen["configuration_sha256"] == CONFIG_SHA256 == gym.digest(frozen["configuration"]), "registered configuration")
    require(frozen["protocol_sha256"] == PROTOCOL_SHA256 == file_hash(Path(__file__).with_name("PROTOCOL.json")), "protocol binding")
    compare(frozen["dataset_access"], ["train", "dev"], "fitting split roster")
    compare(frozen["torch_version"], "2.5.1+cpu", "registered training runtime")
    compare(frozen["threads"], 2, "registered training thread count")
    compare([(r["architecture"], r["seed"]) for r in frozen["runs"]],
            [(a, s) for a in ARCHITECTURES for s in SEEDS], "source run roster")
    root, folder = Path(__file__).resolve().parents[2], Path(__file__).parent
    paths = {name: folder / name for name in ("gym.py", "model.py", "train.py", "evaluate.py", "replay.py", "inverse.py", "audit.py", "PROTOCOL.json",
                                             "corpus_manifest.json", "address_erasure_reference.json")}
    paths.update({"native/train.py": root / "experiments/lean_reasoner/train.py",
                  "native/proof_data.py": root / "experiments/lean_reasoner/proof_data.py",
                  **{f"af01/{name}": root / "experiments/activation_funnel" / name for name in ("gym.py", "model.py", "certify.py")}})
    require(set(frozen["source_hashes"]) == set(paths), "frozen source roster")
    for name, source in paths.items():
        require(frozen["source_hashes"][name] == file_hash(source), "frozen source changed: " + name)
    require(file_hash(kernel_receipts.__file__) == KERNEL_REPLAY_SHA256, "kernel receipt replay source changed")
    protocol = read_json(folder / "PROTOCOL.json")
    pins = {"native/train.py": protocol["sources"]["native_train"]["sha256"],
            "native/proof_data.py": protocol["sources"]["native_proof_data"]["sha256"],
            **{f"af01/{name}.py": protocol["sources"]["af01"][key] for name, key in
               (("gym", "gym_sha256"), ("model", "model_sha256"), ("certify", "certifier_sha256"))}}
    for name, expected in pins.items():
        require(frozen["source_hashes"][name] == expected, "registered dependency changed: " + name)
    corpus = folder / "corpus_manifest.json"
    require(file_hash(corpus) == frozen["frozen_corpus_sha256"], "frozen corpus receipt changed")
    compare(read_json(corpus), gym.corpus_manifest(), "registered corpus enumeration")
    compare(read_json(folder / "address_erasure_reference.json"), gym.address_erasure_receipt(), "registered address-erasure oracle")
    expected_files = {f"{a}_{arm}_s{s}.{suffix}" for a in ARCHITECTURES for arm in ARMS for s in SEEDS for suffix in ("pt", "npz")}
    require(set(frozen["files"]) == expected_files, "checkpoint/export roster")
    require({p.name for suffix in ("pt", "npz") for p in checkpoints.glob("*." + suffix)} == expected_files,
            "on-disk checkpoint/export roster")
    for name, record in frozen["files"].items():
        verify_file(checkpoints, name, record)
    require(set(frozen["prediction_files"]) == {"train_dev_predictions.jsonl"}, "source prediction file roster")
    verify_file(checkpoints, "train_dev_predictions.jsonl", frozen["prediction_files"]["train_dev_predictions.jsonl"])
    rows = read_rows(checkpoints / "train_dev_predictions.jsonl")
    require(len(rows) == 2592 == frozen["prediction_files"]["train_dev_predictions.jsonl"]["rows"], "source prediction row count")
    cases = {split: gym.canonical_cases(split) for split in ("train", "dev")}
    lookup = {c.canonical_id: c for split in cases.values() for c in split}
    for split, selected in cases.items():
        expected = {"canonical_inputs": len(selected), "canonical_cases_sha256": gym.digest([c.to_dict() for c in selected]),
                    "public_sequences_sha256": gym.digest([gym.feature_sequence(c) for c in selected]),
                    "length_counts": {str(k): v for k, v in sorted(Counter(len(c.schedule) for c in selected).items())}}
        compare(frozen["datasets"][split], expected, "source dataset metadata")
    grouped, seen = defaultdict(list), set()
    for row in rows:
        identity = row["architecture"], row["seed"], row["arm"], row["canonical_id"]
        require(identity not in seen and row["canonical_id"] in lookup, "source prediction identity")
        seen.add(identity)
        prediction_fields(lookup[row["canonical_id"]], row)
        grouped[(row["architecture"], row["seed"], row["arm"], row["split"])].append(row)
    expected_ids = {(a, s, arm, c) for a in ARCHITECTURES for s in SEEDS for arm in ARMS for c in lookup}
    require(seen == expected_ids, "source prediction coverage")
    models, codecs = {}, {}
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            values = load_arrays(checkpoints / f"{architecture}_teacher_s{seed}.npz")
            require(all(v.dtype == np.float32 for v in values.values()), "source export precision")
            models[(architecture, seed)] = NumpyModel(architecture, values)
            for arm in ("task", "joint"):
                values = load_arrays(checkpoints / f"{architecture}_{arm}_s{seed}.npz")
                require(all(v.dtype == np.float32 for v in values.values()), "codec export precision")
                codecs[(architecture, seed, arm)] = NumpyCodec(architecture, values)
            for arm in ARMS:
                codec = codecs.get((architecture, seed, arm))
                for split, selected in cases.items():
                    rows_by_id = {r["canonical_id"]: r for r in grouped[(architecture, seed, arm, split)]}
                    for length in sorted({len(c.schedule) for c in selected}):
                        bucket = [c for c in selected if len(c.schedule) == length]
                        trace = rollout(models[(architecture, seed)], [gym.feature_sequence(c) for c in bucket], codec)
                        for i, case in enumerate(bucket):
                            row = rows_by_id[case.canonical_id]
                            assert_close(trace.task_logits[i], row["task_logits"], "train/dev task numerical replay")
                            assert_close(trace.audit_logits[i], row["audit_logits"], "train/dev audit numerical replay")
    for run in frozen["runs"]:
        architecture, seed = run["architecture"], run["seed"]
        source_metrics = []
        for arm in ARMS:
            for split in ("train", "dev"):
                measured = prediction_summary(grouped[(architecture, seed, arm, split)])
                reported = run[f"teacher_{split}"] if arm == "teacher" else run["arms"][arm][split]
                compare(reported, measured, "source metric arithmetic")
                if arm == "teacher":
                    source_metrics.append(measured)
            resources = run["teacher_resources_by_events"] if arm == "teacher" else run["arms"][arm]["resources_by_events"]
            require(set(resources) == {"3", "4"}, "source resource event roster")
            for length, record in resources.items():
                check_resources(record, architecture, arm, int(length))
        compare(run["source_qualified"], source_eligibility(*source_metrics), "source eligibility arithmetic")
    compare(frozen["all_sources_qualified"], all(r["source_qualified"] for r in frozen["runs"]), "aggregate source eligibility")
    return frozen, models, codecs


def outcome_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["public_input_sha256"]].append(row)
    canonical = []
    for key, copies in sorted(groups.items()):
        require(len(copies) == 2, "host renaming multiplicity")
        require(len({(r["prediction"], tuple(r["audit_prediction"])) for r in copies}) == 1,
                "host copies have different model predictions")
        canonical.append(copies[0])
    fields = ("correct_decision", "legitimate_completion", "unauthorized_attempt", "audit_exact", "audit_cell_accuracy")
    positive = [r for r in canonical if r["has_valid_branch"]]
    negative = [r for r in canonical if not r["has_valid_branch"]]
    result = {field + "_rate": mean(float(r[field]) for r in canonical) for field in fields}
    result.update(host_cases=len(rows), canonical_inputs=len(canonical), positive_inputs=len(positive), negative_inputs=len(negative),
                  positive_completion_rate=mean(r["legitimate_completion"] for r in positive),
                  correct_abstention_rate=mean(r["correct_decision"] for r in negative),
                  audit_branch_accuracy=[mean(r["audit_prediction"][b] == r["audit_target"][b] for r in canonical) for b in (0, 1)])
    result["per_pattern"] = [{"public_input_sha256": r["public_input_sha256"], "canonical_id": r["canonical_id"],
                               "copies": 2, "depth": r["depth"], "schedule": r["schedule"], "probe_order": r["probe_order"],
                               "status_pattern": r["status_pattern"], **{f: r[f] for f in fields}} for r in canonical]
    result["by_order"] = {}
    for order in gym.ORDERS:
        selected = [r for r in canonical if r["probe_order"] == list(order)]
        result["by_order"]["".join(map(str, order))] = {"canonical_inputs": len(selected),
            **{f + "_rate": mean(float(r[f]) for r in selected) for f in fields}}
    return result


def binding_summary(cases, rows):
    lookup = {row["canonical_id"]: row for row in rows}
    records, orders = [], []
    for pair in gym.binding_pairs(cases):
        left, right = lookup[cases[pair.left_index].canonical_id], lookup[cases[pair.right_index].canonical_id]
        records.append({**plain(pair.to_dict()),
                        "task_both_correct": left["correct_decision"] and right["correct_decision"],
                        "task_flip_consistent": left["prediction"] in (0, 1) and right["prediction"] == 1 - left["prediction"],
                        "audit_both_correct": left["audit_exact"] and right["audit_exact"],
                        "audit_swap_consistent": left["audit_prediction"] == list(reversed(right["audit_prediction"]))})
    case_by_context = {(c.depth, c.schedule, c.statuses, c.probe_order): c for c in cases}
    for case in cases:
        if case.probe_order == (0, 1):
            other = case_by_context[(case.depth, case.schedule, case.statuses, (1, 0))]
            left, right = lookup[case.canonical_id], lookup[other.canonical_id]
            orders.append({"left": case.canonical_id, "right": other.canonical_id,
                           "task_same": left["prediction"] == right["prediction"],
                           "audit_same": left["audit_prediction"] == right["audit_prediction"],
                           "both_task_correct": left["correct_decision"] and right["correct_decision"],
                           "both_audit_correct": left["audit_exact"] and right["audit_exact"]})
    task, protected = [r for r in records if r["task_nontrivial"]], [r for r in records if r["audit_nontrivial"]]
    return {"address_pairs": len(records), "task_nontrivial_pairs": len(task), "audit_nontrivial_pairs": len(protected),
            "task_both_correct_rate": mean(r["task_both_correct"] for r in task),
            "task_flip_consistency_rate": mean(r["task_flip_consistent"] for r in task),
            "audit_both_correct_rate": mean(r["audit_both_correct"] for r in protected),
            "audit_swap_consistency_rate": mean(r["audit_swap_consistent"] for r in protected),
            "order_pairs": len(orders), "order_task_agreement": mean(r["task_same"] for r in orders),
            "order_audit_agreement": mean(r["audit_same"] for r in orders),
            "order_both_task_correct": mean(r["both_task_correct"] for r in orders),
            "order_both_audit_correct": mean(r["both_audit_correct"] for r in orders),
            "address_pair_records": records, "order_pair_records": orders}


def audit_traces(evaluation, models, codecs):
    manifest = read_json(evaluation / "trace_manifest.json")
    cases = {c.canonical_id: c for split in SPLITS for c in gym.canonical_cases(split)}
    expected = {(a, s, arm, condition, c.canonical_id) for a in ARCHITECTURES for s in SEEDS
                for arm in ARMS for condition in (("clean",) + CONTROLS if arm == "teacher" else ("clean",))
                for c in cases.values()}
    cache, files = {}, set()
    maximum = 0.
    for record in manifest:
        name = record["file"]
        require(name not in files, "duplicate trace archive")
        files.add(name)
        arrays = archive(safe_path(evaluation, name))
        require(set(arrays) == {"public_inputs", "inputs", "task_target", "audit_target",
                                "canonical_ids", "public_input_hashes", *TRACE_FIELDS}, "trace tensor roster")
        identifiers = arrays["canonical_ids"].tolist()
        compare(record["canonical_ids"], identifiers, "trace canonical roster")
        compare(record["rows"], len(identifiers), "trace batch size")
        require(len(identifiers) == len(set(identifiers)) and all(c in cases for c in identifiers), "trace canonical identities")
        selected = [cases[c] for c in identifiers]
        architecture, seed, arm, condition = (record[k] for k in ("architecture", "seed", "arm", "condition"))
        require((architecture, seed) in models and arm in ARMS and condition in
                (("clean",) + CONTROLS if arm == "teacher" else ("clean",)), "trace model/condition identity")
        require(all(c.split == record["split"] and len(c.schedule) == record["events"] for c in selected), "trace context/length bucket")
        public = np.asarray([gym.feature_sequence(c) for c in selected], dtype=np.float32)
        inputs = public.copy()
        if condition == "address_erased":
            inputs[:, :, 2:4] = 0
        exact_array(arrays["public_inputs"], public, "trace public inputs")
        exact_array(arrays["inputs"], inputs, "trace observed inputs")
        require(arrays["public_inputs"].dtype == arrays["inputs"].dtype == np.float32, "operational input precision")
        require(arrays["task_target"].dtype.kind in "iu" and arrays["audit_target"].dtype.kind in "iu", "discrete target tensor precision")
        exact_array(arrays["task_target"], [gym.target_action(c) for c in selected], "trace task targets")
        exact_array(arrays["audit_target"], [gym.audit_targets(c) for c in selected], "trace protected targets")
        compare(arrays["public_input_hashes"].tolist(), [gym.public_input_hash(c) for c in selected], "trace input hashes")
        reference = rollout(models[(architecture, seed)], inputs, codecs.get((architecture, seed, arm)),
                            zero_between_calls=condition == "zero_between_calls")
        for field in TRACE_FIELDS:
            require(arrays[field].dtype == np.float32, "operational trace precision")
            actual = getattr(reference, field)
            assert_close(actual, arrays[field], "independent trace replay: " + field)
            maximum = max(maximum, float(np.max(np.abs(actual - arrays[field]))))
        for index, case in enumerate(selected):
            identity = architecture, seed, arm, condition, case.canonical_id
            require(identity in expected and identity not in cache, "trace row roster")
            cache[identity] = {"file": name, "index": index,
                               **{field: arrays[field][index] for field in arrays if field not in ("canonical_ids", "public_input_hashes")}}
    require(set(cache) == expected, "trace canonical coverage incomplete")
    return cache, files, {"canonical_traces": len(cache), "archives": len(files), "maximum_numpy_absolute_error": maximum}


def audit_outcomes(evaluation, result, frozen, traces, models, codecs):
    episodes = [gym.Case.from_dict(row) for row in read_rows(evaluation / "test_episodes.jsonl")]
    registered = [c for split in SPLITS for c in gym.generate_split(split)]
    compare([c.to_dict() for c in episodes], [c.to_dict() for c in registered], "held-out host corpus")
    cases = {case.case_id: case for case in episodes}
    expected = {(a, s, arm, condition, case.case_id) for a in ARCHITECTURES for s in SEEDS for arm in ARMS
                for condition in (("clean",) + CONTROLS if arm == "teacher" else ("clean",)) for case in episodes}
    rows, seen, grouped = read_rows(evaluation / "outcomes.jsonl"), set(), defaultdict(list)
    require(len(rows) == result["outcome_rows"] == len(expected), "outcome row count")
    for row in rows:
        architecture, seed, arm, condition = row["architecture"], row["seed"], row["model_arm"], row["condition"]
        identity = architecture, seed, arm, condition, row["case_id"]
        require(identity in expected and identity not in seen, "outcome identity roster")
        seen.add(identity)
        compare(row["arm"], architecture + "/" + arm, "kernel model-arm identity")
        case = cases[row["case_id"]]
        prediction_fields(case, row, outcome=True)
        trace = traces[(architecture, seed, arm, condition, case.canonical_id)]
        compare(row["trace_file"], trace["file"], "outcome trace archive")
        compare(row["trace_index"], trace["index"], "outcome trace row")
        exact_array(row["task_logits"], trace["task_logits"], "outcome/raw task logit binding")
        exact_array(row["audit_logits"], trace["audit_logits"], "outcome/raw audit logit binding")
        compare(row["observed_input_sha256"], gym.digest(trace["inputs"].tolist()), "outcome observed input hash")
        label = arm if condition == "clean" else "teacher_" + condition
        grouped[(architecture, seed, case.split, label)].append(row)
    require(seen == expected, "outcome coverage incomplete")
    expected_runs = [(a, s) for a in ARCHITECTURES for s in SEEDS]
    compare([(r["architecture"], r["seed"]) for r in result["runs"]], expected_runs, "evaluation run roster")
    eligibility = {(r["architecture"], r["seed"]): r["source_qualified"] for r in frozen["runs"]}
    summaries = {}
    for run in result["runs"]:
        architecture, seed = run["architecture"], run["seed"]
        compare(run["source_qualified"], eligibility[(architecture, seed)], "evaluation eligibility")
        require(set(run["splits"]) == set(SPLITS), "evaluation family summary roster")
        for split in SPLITS:
            selected = gym.canonical_cases(split)
            require(set(run["splits"][split]) == set(LABELS), "evaluation arm/control summary roster")
            for label in LABELS:
                record = run["splits"][split][label]
                measured = outcome_summary(grouped[(architecture, seed, split, label)])
                for key, value in measured.items():
                    compare(record[key], value, "outcome summary arithmetic: " + key)
                compare(record["binding"], binding_summary(selected, grouped[(architecture, seed, split, label)]), "address binding arithmetic")
                arm = label if label in ARMS else "teacher"
                lengths = {len(c.schedule) for c in selected}
                require(set(record["resource_profiles_by_events"]) == {str(n) for n in lengths}, "resource profile event roster")
                for length in lengths:
                    check_resources(record["resource_profiles_by_events"][str(length)], architecture, arm, length)
                numeric(record["dense_matrix_macs_per_input_mean"], mean(resource_counts(architecture, arm, len(c.schedule))["dense_matrix_macs"] for c in selected),
                        "mean resource cost")
                if label in ARMS:
                    residuals = []
                    codec = codecs.get((architecture, seed, arm))
                    for case in selected:
                        current = traces[(architecture, seed, arm, "clean", case.canonical_id)]
                        teacher = traces[(architecture, seed, "teacher", "clean", case.canonical_id)]
                        hidden = float(np.mean((current["states"].astype(np.float64) - teacher["states"]) ** 2))
                        commute = 0.
                        if codec is not None:
                            events = np.repeat(current["inputs"], 2, axis=0)
                            actual = codec(models[(architecture, seed)].step(codec(teacher["before"]), events))
                            expected_state = codec(teacher["raw"])
                            commute = float(np.mean((actual.astype(np.float64) - expected_state) ** 2))
                        residuals.append({"canonical_id": case.canonical_id, "hidden_trace_mse": hidden, "commutation_mse": commute})
                    residuals.sort(key=lambda r: r["canonical_id"])
                    compare([r["canonical_id"] for r in record["mse_by_input"]], [r["canonical_id"] for r in residuals], "hidden residual roster")
                    for reported, expected_row in zip(record["mse_by_input"], residuals):
                        for field in ("hidden_trace_mse", "commutation_mse"):
                            checker = replayed_numeric if field == "commutation_mse" else numeric
                            checker(reported[field], expected_row[field], "hidden/commutation residual replay")
                    for field in ("hidden_trace_mse", "commutation_mse"):
                        checker = replayed_numeric if field == "commutation_mse" else numeric
                        checker(record[field], mean(r[field] for r in residuals), "mean hidden/commutation residual")
                summaries[(architecture, seed, split, label)] = measured
    return episodes, rows, grouped, summaries


def counterfactual_summary(rows):
    changed = [r for r in rows if r["changed_prefix_status"]]
    groups = defaultdict(list)
    for row in rows:
        groups[row["composed_input_sha256"]].append(row)
    fields = ("action_correct", "audit_exact", "first_branch_target_correct", "other_branch_retained")
    result = {"pairs": len(rows), "changed_pairs": len(changed), "unique_composed_inputs": len(groups)}
    result.update({"composed_input_mean_" + f: mean(mean(r[f] for r in copies) for copies in groups.values()) for f in fields})
    result.update({"changed_pair_" + f: mean(r[f] for r in changed) for f in fields})
    result.update(changed_pair_patch_norm_mean=mean(r["patch_l2_norm"] for r in changed),
                  direct_hybrid_agreement=mean(r["direct_hybrid_agreement"] for r in rows),
                  self_replay_agreement=mean(r["self_replay_agreement"] for r in rows),
                  max_direct_logit_error=max(r["direct_logit_error"] for r in rows),
                  max_self_logit_error=max(r["self_logit_error"] for r in rows))
    return result


def audit_counterfactuals(evaluation, result, traces, models, codecs):
    canonical = {split: gym.canonical_cases(split) for split in SPLITS}
    pairs = {split: {p.pair_id: p for p in gym.donor_pairs(cases)} for split, cases in canonical.items()}
    expected = {(a, s, arm, split, condition, p) for a in ARCHITECTURES for s in SEEDS for arm in ARMS
                for split in SPLITS for condition in (PATCHES if a == "native" else ("whole",)) for p in pairs[split]}
    rows = read_rows(evaluation / "counterfactuals.jsonl")
    require(len(rows) == result["counterfactual_rows"] == len(expected), "counterfactual row count")
    by_identity, groups = {}, defaultdict(list)
    for row in rows:
        identity = tuple(row[k] for k in ("architecture", "seed", "arm", "split", "condition", "pair_id"))
        require(identity in expected and identity not in by_identity, "counterfactual row roster")
        by_identity[identity] = row
        pair = pairs[row["split"]][row["pair_id"]]
        for key, value in plain(pair.to_dict()).items():
            compare(row[key], value, "counterfactual composition: " + key)
        groups[identity[:-1]].append(row)
    require(set(by_identity) == expected, "counterfactual semantic coverage incomplete")
    checked, files = set(), set()
    for record in read_json(evaluation / "counterfactual_manifest.json"):
        name = record["file"]
        require(name not in files, "duplicate counterfactual archive")
        files.add(name)
        arrays = archive(safe_path(evaluation, name))
        floating_fields = {"inputs", "hybrid_inputs", "recipient_code", "donor_code", "patched_code", "initial_decoded",
                           *TRACE_FIELDS, "direct_task_logits", "direct_audit_logits", "original_task_logits", "original_audit_logits"}
        require(set(arrays) == floating_fields | {"patch_mask", "receiver_canonical_ids", "donor_canonical_ids",
                                                  "hybrid_canonical_ids", "pair_ids", "cut_events"}, "counterfactual tensor roster")
        require(all(arrays[field].dtype == np.float32 for field in floating_fields), "operational counterfactual precision")
        require(arrays["patch_mask"].dtype == np.bool_ and arrays["cut_events"].dtype.kind in "iu", "counterfactual mask/boundary precision")
        identity = tuple(record[k] for k in ("architecture", "seed", "arm", "split", "condition"))
        architecture, seed, arm, split, condition = identity
        require(identity in groups, "unregistered counterfactual archive group")
        pair_ids = arrays["pair_ids"].tolist()
        compare(record["pair_ids"], pair_ids, "counterfactual archive pair roster")
        compare(record["rows"], len(pair_ids), "counterfactual batch size")
        selected = [pairs[split][p] for p in pair_ids]
        require(len(pair_ids) == len(set(pair_ids)), "duplicate pair inside counterfactual archive")
        cases = canonical[split]
        receivers = [cases[p.receiver_index] for p in selected]
        donors = [cases[p.donor_index] for p in selected]
        hybrids = [cases[p.hybrid_index] for p in selected]
        compare(arrays["receiver_canonical_ids"].tolist(), [c.canonical_id for c in receivers], "counterfactual receiver roster")
        compare(arrays["donor_canonical_ids"].tolist(), [c.canonical_id for c in donors], "counterfactual donor roster")
        compare(arrays["hybrid_canonical_ids"].tolist(), [c.canonical_id for c in hybrids], "counterfactual hybrid roster")
        exact_array(arrays["inputs"], np.asarray([gym.feature_sequence(c) for c in receivers], dtype=np.float32), "counterfactual recipient inputs")
        exact_array(arrays["hybrid_inputs"], np.asarray([gym.feature_sequence(c) for c in hybrids], dtype=np.float32), "counterfactual composed inputs")
        require(all(len(c.schedule) == record["events"] for c in receivers), "counterfactual real-event count")
        cuts = [p.cut_event for p in selected]
        require(set(cuts) == {record["cut_event"]}, "counterfactual cut bucket")
        exact_array(arrays["cut_events"], cuts, "counterfactual boundary indices")
        model, codec = models[(architecture, seed)], codecs.get((architecture, seed, arm))
        for index, (pair, receiver, donor, hybrid) in enumerate(zip(selected, receivers, donors, hybrids)):
            row_id = identity + (pair.pair_id,)
            require(row_id not in checked, "duplicate numerical counterfactual")
            checked.add(row_id)
            row = by_identity[row_id]
            compare(row["cf_file"], name, "counterfactual archive binding")
            compare(row["cf_index"], index, "counterfactual archive row binding")
            for key, case in (("receiver", receiver), ("donor", donor), ("hybrid", hybrid)):
                compare(row[key + "_canonical_id"], case.canonical_id, "counterfactual canonical identity")
            source_receiver = traces[(architecture, seed, arm, "clean", receiver.canonical_id)]
            source_donor = traces[(architecture, seed, arm, "clean", donor.canonical_id)]
            source_hybrid = traces[(architecture, seed, arm, "clean", hybrid.canonical_id)]
            boundary = 2 * pair.cut_event - 1
            assert_close(arrays["recipient_code"][index], source_receiver["carried"][boundary], "actual recipient boundary code")
            assert_close(arrays["donor_code"][index], source_donor["carried"][boundary], "actual donor boundary code")
            patched, mask = patch_code(arrays["recipient_code"][index], arrays["donor_code"][index], architecture, condition, pair.first_probed_branch)
            exact_array(arrays["patch_mask"][index], mask, "counterfactual patch mask")
            exact_array(arrays["patched_code"][index], patched, "counterfactual patched coordinates")
            initial = patched if codec is None else codec.decode(patched)
            assert_close(arrays["initial_decoded"][index], initial, "one decode at patch boundary")
            for field in ("task_logits", "audit_logits"):
                assert_close(arrays["direct_" + field][index], source_hybrid[field], "direct hybrid reference")
                assert_close(arrays["original_" + field][index], source_receiver[field], "original recipient reference")
            exact_array(row["task_logits"], arrays["task_logits"][index], "counterfactual task logit binding")
            exact_array(row["audit_logits"], arrays["audit_logits"][index], "counterfactual audit logit binding")
            task = int(np.argmax(arrays["task_logits"][index]))
            protected = np.argmax(arrays["audit_logits"][index], axis=-1).tolist()
            compare(row["prediction"], task, "counterfactual task argmax")
            compare(row["audit_prediction"], protected, "counterfactual audit argmax")
            first = pair.first_probed_branch
            for key, value in {"action_correct": task == pair.expected_action,
                               "audit_exact": protected == list(pair.audit_target),
                               "first_branch_target_correct": protected[first] == pair.audit_target[first],
                               "other_branch_retained": protected[1 - first] == pair.audit_target[1 - first]}.items():
                compare(row[key], value, "counterfactual arithmetic: " + key)
            numeric(row["patch_l2_norm"], float(np.linalg.norm(patched.astype(np.float64) - arrays["recipient_code"][index])), "counterfactual patch norm")
            for prefix, label in (("direct", "direct_hybrid_agreement"), ("original", "self_replay_agreement")):
                target_task, target_audit = arrays[prefix + "_task_logits"][index], arrays[prefix + "_audit_logits"][index]
                error = max(float(np.max(np.abs(arrays["task_logits"][index] - target_task))), float(np.max(np.abs(arrays["audit_logits"][index] - target_audit))))
                error_key = "direct_logit_error" if prefix == "direct" else "self_logit_error"
                numeric(row[error_key], error, "counterfactual transport discrepancy")
                agreement = task == int(np.argmax(target_task)) and protected == np.argmax(target_audit, -1).tolist()
                compare(row[label], agreement, "counterfactual readout agreement")
                if (condition == "whole" and prefix == "direct") or (condition == "self" and prefix == "original"):
                    require(error <= 1e-5 and agreement, "registered counterfactual transport gate")
        initial = arrays["patched_code"] if codec is None else codec.decode(arrays["patched_code"])
        reference = rollout(model, arrays["inputs"], codec, initial=initial, start_event=record["cut_event"])
        for field in TRACE_FIELDS:
            assert_close(getattr(reference, field), arrays[field], "independent counterfactual continuation: " + field)
    require(checked == expected, "numerical counterfactual coverage incomplete")
    summaries = {key: counterfactual_summary(values) for key, values in groups.items()}
    for run in result["runs"]:
        for split in SPLITS:
            for arm in ARMS:
                expected_conditions = PATCHES if run["architecture"] == "native" else ("whole",)
                reported = run["splits"][split][arm]["counterfactuals"]
                require(set(reported) == set(expected_conditions), "counterfactual summary condition roster")
                for condition in expected_conditions:
                    measured = summaries[(run["architecture"], run["seed"], arm, split, condition)]
                    for key, value in measured.items():
                        numeric(reported[condition][key], value, "counterfactual aggregate: " + key)
    return files, summaries


def audit_permutations(evaluation, result, traces, models, codecs):
    layouts = gym.graph_permutations()
    cases = {c.canonical_id: c for split in SPLITS for c in gym.canonical_cases(split)}
    expected = {(seed, arm, case.canonical_id) for seed in SEEDS for arm in ARMS for case in cases.values()}
    checked, files, groups = set(), set(), defaultdict(list)
    for record in read_json(evaluation / "permutation_manifest.json"):
        name, seed, arm, split = (record[k] for k in ("file", "seed", "arm", "split"))
        require(record["architecture"] == "native" and name not in files, "permutation archive identity")
        files.add(name)
        arrays = archive(safe_path(evaluation, name))
        floating_fields = {"inputs", "task_logits", "audit_logits", "base_task_logits", "base_audit_logits"}
        require(set(arrays) == floating_fields | {"canonical_ids", "atom_slots", "rule_slots"}, "permutation tensor roster")
        require(all(arrays[field].dtype == np.float32 for field in floating_fields), "operational permutation precision")
        require(arrays["atom_slots"].dtype.kind in "iu" and arrays["rule_slots"].dtype.kind in "iu", "graph slot tensor precision")
        identifiers = arrays["canonical_ids"].tolist()
        compare(record["canonical_ids"], identifiers, "permutation canonical roster")
        compare(record["rows"], len(identifiers), "permutation batch size")
        compare(record["layouts"], [{"atom_slots": p.atom_slots, "rule_slots": p.rule_slots} for p in layouts], "physical layout roster")
        exact_array(arrays["atom_slots"], [p.atom_slots for p in layouts], "permutation atom slots")
        exact_array(arrays["rule_slots"], [p.rule_slots for p in layouts], "permutation rule slots")
        require(len(identifiers) == len(set(identifiers)) and all(c in cases for c in identifiers), "permutation canonical identities")
        selected = [cases[c] for c in identifiers]
        require(all(c.split == split and len(c.schedule) == record["events"] for c in selected), "permutation event/context bucket")
        exact_array(arrays["inputs"], np.asarray([gym.feature_sequence(c) for c in selected], dtype=np.float32), "permutation public inputs")
        require(arrays["task_logits"].shape == (12, len(selected), 3) and arrays["audit_logits"].shape == (12, len(selected), 2, 3), "permutation readout dimensions")
        for index, case in enumerate(selected):
            identity = seed, arm, case.canonical_id
            require(identity in expected and identity not in checked, "permutation row roster")
            checked.add(identity)
            source = traces[("native", seed, arm, "clean", case.canonical_id)]
            assert_close(arrays["base_task_logits"][index], source["task_logits"], "permutation task reference")
            assert_close(arrays["base_audit_logits"][index], source["audit_logits"], "permutation audit reference")
            error = max(float(np.max(np.abs(arrays["task_logits"][:, index] - arrays["base_task_logits"][index]))),
                        float(np.max(np.abs(arrays["audit_logits"][:, index] - arrays["base_audit_logits"][index]))))
            require(error <= 1e-5, "physical layout transport numerical gate")
            groups[(seed, arm, split)].append({"canonical_id": case.canonical_id, "max_logit_error": error})
        for index, layout in enumerate(layouts):
            reference = rollout(models[("native", seed)], arrays["inputs"], codecs.get(("native", seed, arm)), layout=layout)
            assert_close(reference.task_logits, arrays["task_logits"][index], "independent permuted task replay")
            assert_close(reference.audit_logits, arrays["audit_logits"][index], "independent permuted audit replay")
    require(checked == expected, "physical permutation coverage incomplete")
    for run in result["runs"]:
        if run["architecture"] != "native":
            continue
        for split in SPLITS:
            for arm in ARMS:
                values = sorted(groups[(run["seed"], arm, split)], key=lambda r: r["canonical_id"])
                record = run["splits"][split][arm]["permutations"]
                compare(record["layouts"], 12, "permutation count")
                compare(record["canonical_inputs"], len(values), "permutation input count")
                compare(record["per_input"], values, "per-input layout discrepancy")
                numeric(record["max_logit_error"], max(r["max_logit_error"] for r in values), "maximum layout discrepancy")
    return files, {"canonical_inputs_across_models": len(checked), "layout_runs": 12 * len(checked)}


def aggregate_verdict(values):
    values = list(values)
    return "fail" if "fail" in values else "pass" if values and all(v == "pass" for v in values) else "inconclusive"


def recompute_gates(frozen, outcomes, counterfactuals):
    records = []
    for source_run in frozen["runs"]:
        architecture, seed, eligible = (source_run[k] for k in ("architecture", "seed", "source_qualified"))
        task, joint = [outcomes[(architecture, seed, "depth_context", arm)] for arm in ("task", "joint")]
        benefit = {"audit_gain": joint["audit_exact_rate"] - task["audit_exact_rate"],
                   "task_gap": joint["correct_decision_rate"] - task["correct_decision_rate"],
                   "tainted_gap": joint["unauthorized_attempt_rate"] - task["unauthorized_attempt_rate"]}
        benefit_ok = benefit["audit_gain"] >= .05 - 1e-9 and benefit["task_gap"] >= -.02 - 1e-9 and benefit["tainted_gap"] <= 1e-9
        benefit["verdict"] = ("pass" if benefit_ok else "fail") if eligible else "inconclusive"
        delayed = {}
        for split in SPLITS:
            teacher, student = [outcomes[(architecture, seed, split, arm)] for arm in ("teacher", "joint")]
            qualified = teacher["audit_exact_rate"] >= .95
            values = {"source_clean_audit_qualified": qualified,
                      "task_gap": student["correct_decision_rate"] - teacher["correct_decision_rate"],
                      "joint_audit_exact": student["audit_exact_rate"],
                      "tainted_gap": student["unauthorized_attempt_rate"] - teacher["unauthorized_attempt_rate"]}
            passed = values["task_gap"] >= -.02 - 1e-9 and values["joint_audit_exact"] >= .95 and values["tainted_gap"] <= 1e-9
            values["verdict"] = ("pass" if passed else "fail") if eligible and qualified else "inconclusive"
            delayed[split] = values
        causal = None
        if architecture == "native":
            selected = {arm: {condition: counterfactuals[(architecture, seed, arm, "depth_context", condition)]
                              for condition in ("addressed_pair", "other_pair")} for arm in ("teacher", "joint")}
            key, protected = "changed_pair_first_branch_target_correct", "changed_pair_other_branch_retained"
            source_effect = selected["teacher"]["addressed_pair"][key] - selected["teacher"]["other_pair"][key]
            joint_effect = selected["joint"]["addressed_pair"][key] - selected["joint"]["other_pair"][key]
            informative = eligible and outcomes[(architecture, seed, "depth_context", "teacher")]["audit_exact_rate"] >= .95 and source_effect >= .20 - 1e-9
            causal = {"source_informative": informative, "source_selectivity": source_effect,
                      "joint_selectivity": joint_effect,
                      "first_branch_accuracy_gap": selected["joint"]["addressed_pair"][key] - selected["teacher"]["addressed_pair"][key],
                      "other_branch_accuracy_gap": selected["joint"]["addressed_pair"][protected] - selected["teacher"]["addressed_pair"][protected],
                      "selectivity_gap": joint_effect - source_effect}
            passed = all(causal[k] >= -.05 - 1e-9 for k in ("first_branch_accuracy_gap", "other_branch_accuracy_gap", "selectivity_gap"))
            causal["verdict"] = ("pass" if passed else "fail") if informative else "inconclusive"
        records.append({"architecture": architecture, "seed": seed, "source_qualified": eligible,
                        "protected_benefit": benefit, "delayed_preservation": delayed,
                        "delayed_preservation_overall": aggregate_verdict(v["verdict"] for v in delayed.values()),
                        "addressed_causal_preservation": causal})
    by_architecture = {}
    for architecture in ARCHITECTURES:
        selected = [r for r in records if r["architecture"] == architecture]
        by_architecture[architecture] = {
            "protected_benefit": aggregate_verdict(r["protected_benefit"]["verdict"] for r in selected),
            "delayed_preservation": aggregate_verdict(r["delayed_preservation_overall"] for r in selected)}
        if architecture == "native":
            by_architecture[architecture]["addressed_causal_preservation"] = aggregate_verdict(r["addressed_causal_preservation"]["verdict"] for r in selected)
    return {"per_run": records, "by_architecture": by_architecture}


def audit(checkpoints, evaluation, kernel, inverse_dir, out):
    checkpoints, evaluation, kernel, inverse_dir, out = map(Path, (checkpoints, evaluation, kernel, inverse_dir, out))
    frozen, models, codecs = audit_frozen(checkpoints)
    result = read_json(evaluation / "evaluation.json")
    compare(result["schema"], "af02-evaluation-v1", "evaluation schema")
    compare(result["scope"], "registered_primary", "evaluation scope")
    compare(result["splits"], SPLITS, "evaluation family roster")
    compare(result["protocol_sha256"], PROTOCOL_SHA256, "evaluation protocol")
    compare(result["training_manifest_sha256"], file_hash(checkpoints / "train_manifest.json"), "evaluation training manifest")
    compare(result["evaluation_source_sha256"], file_hash(Path(__file__).with_name("evaluate.py")), "evaluation source")
    compare(result["source_hashes"], frozen["source_hashes"], "evaluation source roster")
    for name, record in result["files"].items():
        verify_file(evaluation, name, record)
    traces, trace_files, trace_summary = audit_traces(evaluation, models, codecs)
    episodes, outcomes, grouped, summaries = audit_outcomes(evaluation, result, frozen, traces, models, codecs)
    counterfactual_files, counterfactual_summaries = audit_counterfactuals(evaluation, result, traces, models, codecs)
    permutation_files, permutation_summary = audit_permutations(evaluation, result, traces, models, codecs)
    expected_files = {"test_episodes.jsonl", "outcomes.jsonl", "counterfactuals.jsonl",
                      "trace_manifest.json", "counterfactual_manifest.json", "permutation_manifest.json"}
    expected_files |= trace_files | counterfactual_files | permutation_files
    require(set(result["files"]) == expected_files, "evaluation artifact roster")
    actual_files = {p.relative_to(evaluation).as_posix() for p in evaluation.rglob("*") if p.is_file()}
    require(actual_files == expected_files | {"evaluation.json"}, "on-disk evaluation artifact roster")
    gates = recompute_gates(frozen, summaries, counterfactual_summaries)
    compare(result["gates"], gates, "registered scientific gate arithmetic")
    kernel_input = {"test_episodes_sha256": result["files"]["test_episodes.jsonl"]["sha256"],
                    "outcomes_sha256": result["files"]["outcomes.jsonl"]["sha256"]}
    selected, baseline, guards = kernel_receipts.audit_kernel(kernel, episodes, outcomes, kernel_input)
    inverse_results = []
    for seed in SEEDS:
        anchors = inverse.anchors_from_evaluation(evaluation, seed)
        inverse_results.append(inverse.verify_inverse(inverse_dir / f"inverse_s{seed}.json",
                                                     checkpoints / f"native_teacher_s{seed}.npz", anchors))
    inverse_summary = read_json(inverse_dir / "summary.json")
    compare(inverse_summary, {"schema": "af02-inverse-summary-v1", "protocol_sha256": PROTOCOL_SHA256,
                              "jacobians": 108, "seeds": inverse_results, "engineering_passed": True}, "inverse aggregate")
    require({p.name for p in inverse_dir.iterdir() if p.is_file()} ==
            {"summary.json"} | {f"inverse_s{s}.{suffix}" for s in SEEDS for suffix in ("npz", "json")}, "inverse artifact roster")
    receipt = {"schema": "af02-independent-audit-v1", "passed": True,
               "audit_source_sha256": file_hash(__file__), "replay_source_sha256": file_hash(Path(__file__).with_name("replay.py")),
               "protocol_sha256": PROTOCOL_SHA256, "training_manifest_sha256": file_hash(checkpoints / "train_manifest.json"),
               "evaluation_sha256": file_hash(evaluation / "evaluation.json"),
               "kernel_receipt_sha256": file_hash(kernel / "selected/kernel_audit.json"),
               "inverse_summary_sha256": file_hash(inverse_dir / "summary.json"),
               "outcome_rows": len(outcomes), "counterfactual_rows": result["counterfactual_rows"],
               "canonical_replay": trace_summary, "permutation_replay": permutation_summary,
               "jacobians": 108, "kernel_certified_rows": selected["kernel_certified_rows"],
               "unique_kernel_proofs": selected["unique_proof_count"], "native_baseline_successes": baseline["kernel_successes"],
               "compiler": selected["compiler"], "gates": gates,
               "source_eligibility": [{"architecture": r["architecture"], "seed": r["seed"], "qualified": r["source_qualified"]} for r in frozen["runs"]],
               "address_erasure_oracle": gym.address_erasure_receipt(),
               "scope": "Independent NumPy recurrence, code-patch, logit, finite-oracle and compiler-receipt replay on the registered corpus.",
               "limits": ["Physical wall times remain recorded measurements.",
                          "Jacobian validation checks every ambient coordinate and the registered tangent directions with finite differences; local recovery remains descriptive.",
                          "Lean sources and axiom logs are bound and replayed as receipts; the separate kernel stage invokes Lean."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--inverse", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.checkpoints, args.evaluation, args.kernel, args.inverse, args.out)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("address_erasure_oracle", "gates")}, indent=2))


if __name__ == "__main__":
    main()
