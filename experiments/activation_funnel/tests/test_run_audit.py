"""Pure-Python receipt mutations; synthetic fixtures never claim Lean execution."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.activation_funnel import audit_run as A, certify as C, gym


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def perfect_metrics(cases, patterns):
    values = {"task_accuracy": 1., "audit_cell_accuracy": 1.,
              "audit_branch_accuracy": [1., 1.], "audit_exact_accuracy": 1.}
    return {**values, "cases": cases, "exact_feature_patterns": patterns,
            "row_weighted_diagnostic": copy.deepcopy(values)}


def outcome(ep, prediction=None, *, arm="teacher", condition="clean"):
    prediction = gym.target_action(ep) if prediction is None else prediction
    audit = list(gym.audit_targets(ep))
    row = {"case_id": ep.case_id, "split": ep.split, "family": ep.family,
           "seed": 17, "arm": arm, "condition": condition,
           "branch": None if prediction == 2 else prediction, "prediction": prediction,
           "target_action": gym.target_action(ep), "audit_prediction": audit,
           "audit_target": audit, "audit_exact": True, "audit_cell_accuracy": 1.,
           "public_input_sha256": A.value_hash(gym.feature_sequence(ep)),
           "observed_input_sha256": A.value_hash(gym.feature_sequence(ep)),
           "status_pattern": list(ep.statuses), "probe_order": list(ep.probe_order),
           "depth_feature": ep.depth / 8, "has_valid_branch": "valid" in ep.statuses}
    row.update(gym.score_decision(ep, prediction))
    return row


class ReceiptOnlyCompilerFixture:
    """Fabricate compiler-shaped records solely to test receipt rejection paths."""
    version_output = "Lean (version 4.19.0, synthetic unit-test receipt)"
    executable_sha256 = "0" * 64

    def run(self, path, names, timeout, expect_success=True):
        log = path.with_suffix(".lean.log")
        log.write_text("".join(f"'{n}' does not depend on any axioms\n" for n in names)
                       if expect_success else "fixture:1:0: error: synthetic rejection receipt\n")
        return {"source": path.name, "source_sha256": A.file_hash(path),
                "log": log.name, "log_sha256": A.file_hash(log),
                "returncode": 0 if expect_success else 1, "timed_out": False,
                "elapsed_seconds": .001, "expected_declarations": list(names),
                "axiom_audit_passed": expect_success, "kernel_success": expect_success,
                "expect_success": expect_success, "expectation_met": True, "error": None}


class RunAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def frozen_fixture(self):
        folder = self.folder / "checkpoints"
        folder.mkdir()
        source = Path(A.__file__).parent
        module = ast.parse((source / "train.py").read_text())
        config = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
        manifest = {"scope": "registered_primary", "training_complete": True,
                    "configuration": config, "configuration_sha256": A.value_hash(config),
                    "dataset_access": ["train", "dev"], "torch_version": "2.5.1+cpu",
                    "files": {}, "seeds": [], "all_sources_qualified": True,
                    "dataset_sha256": {}, "source_hashes": {name: A.file_hash(source / name)
                    for name in ("gym.py", "model.py", "train.py", "evaluate.py", "PROTOCOL.json")}}
        dimensions = {}
        for split in ("train", "dev"):
            eps = gym.generate_split(split)
            patterns = len(A.pattern_representatives(eps))
            dimensions[split] = len(eps), patterns
            manifest[split] = {"examples": len(eps), "unique_public_inputs": patterns,
                               "input_sha256": A.value_hash([gym.feature_sequence(e) for e in eps])}
            manifest["dataset_sha256"][split] = A.value_hash([e.to_dict() for e in eps])
        for seed in A.SEEDS:
            manifest["seeds"].append({"seed": seed, "source_qualified": True,
                                      "teacher_train": perfect_metrics(*dimensions["train"]),
                                      "teacher_dev": perfect_metrics(*dimensions["dev"]),
                                      "teacher_resources": A.resource_counts("teacher")})
            for arm in A.ARMS:
                name = f"{arm}_s{seed}.pt"
                (folder / name).write_bytes(b"synthetic hash fixture; never loaded as a model")
                manifest["files"][name] = {"sha256": A.file_hash(folder / name),
                                          "bytes": (folder / name).stat().st_size}

        def save():
            write_json(folder / "train_manifest.json", manifest)
            (folder / "train_manifest.sha256").write_text(A.file_hash(folder / "train_manifest.json"))
        save()
        return folder, manifest, save

    def summaries_fixture(self):
        eps = gym.generate_split("dev")
        grouped, summaries = {}, {}
        for label in A.LABELS:
            arm = label if label in A.ARMS else "teacher"
            condition = "clean" if label in A.ARMS else label.removeprefix("teacher_")
            rows = [outcome(e, arm=arm, condition=condition) for e in eps]
            grouped[(17, "dev", label)] = rows
            summaries[label] = {**A.aggregate(rows), "resources": A.resource_counts(arm)}
        result = {"seeds": [{"seed": 17, "source_qualified": True, "splits": {"dev": summaries}}]}
        frozen = {"seeds": [{"seed": 17, "source_qualified": True}]}
        return eps, grouped, result, frozen

    def kernel_fixture(self):
        folder = self.folder / "kernel"
        eps = [gym.make_episode(("valid", "tainted"), case_id="a"),
               gym.make_episode(("dead", "tainted"), case_id="b")]
        rows = [outcome(eps[0], 0), outcome(eps[1], 1), outcome(eps[1], 2)]
        with mock.patch.object(C.LeanCompiler, "discover", return_value=ReceiptOnlyCompilerFixture()):
            selected = C.certify_selected(eps, rows, folder / "selected")
            baseline = C.native_baseline(eps, folder / "baseline")
            guards = C.negative_guards(folder / "guards")
        evaluation = {"test_episodes_sha256": "1" * 64, "outcomes_sha256": "2" * 64}
        summary = {"selected": {k: v for k, v in selected.items() if k not in ("rows", "compilations")},
                   "baseline": {k: v for k, v in baseline.items() if k != "rows"}, "guards": guards,
                   "inputs": {"episodes_sha256": evaluation["test_episodes_sha256"],
                              "outcomes_sha256": evaluation["outcomes_sha256"]}}
        write_json(folder / "summary.json", summary)
        return folder, eps, rows, evaluation

    def test_rehashed_source_qualification_cannot_override_pattern_metrics(self):
        folder, manifest, save = self.frozen_fixture()
        self.assertTrue(A.audit_frozen(folder)["all_sources_qualified"])
        scores = manifest["seeds"][0]["teacher_train"]
        scores["task_accuracy"] = 17 / 18
        scores["row_weighted_diagnostic"]["task_accuracy"] = 140 / 144
        save()
        with self.assertRaisesRegex(ValueError, "source eligibility"):
            A.audit_frozen(folder)

    def test_source_metrics_reject_impossible_denominators_and_audit_means(self):
        scores = perfect_metrics(144, 18)
        scores["task_accuracy"] = .99
        with self.assertRaisesRegex(ValueError, "denominator"):
            A.audit_source_metrics(scores, 144, 18)
        scores = perfect_metrics(144, 18)
        scores["audit_cell_accuracy"] = .5
        with self.assertRaisesRegex(ValueError, "audit mean"):
            A.audit_source_metrics(scores, 144, 18)

    def test_summary_arm_omission_and_source_flag_are_rejected(self):
        _, grouped, result, frozen = self.summaries_fixture()
        with mock.patch.object(A, "SPLITS", ("dev",)):
            A.audit_summaries(result, grouped, frozen)
            omitted = copy.deepcopy(result)
            del omitted["seeds"][0]["splits"]["dev"]["joint"]
            with self.assertRaisesRegex(ValueError, "summary arm"):
                A.audit_summaries(omitted, grouped, frozen)
            result["seeds"][0]["source_qualified"] = False
            with self.assertRaisesRegex(ValueError, "source eligibility"):
                A.audit_summaries(result, grouped, frozen)

    def test_forged_pattern_breakdown_and_compute_claim_are_rejected(self):
        _, grouped, result, frozen = self.summaries_fixture()
        with mock.patch.object(A, "SPLITS", ("dev",)):
            forged = copy.deepcopy(result)
            forged["seeds"][0]["splits"]["dev"]["teacher"]["per_pattern"][0]["copies"] += 1
            with self.assertRaisesRegex(ValueError, "summary arithmetic"):
                A.audit_summaries(forged, grouped, frozen)
            result["seeds"][0]["splits"]["dev"]["compact_transition"]["resources"]["dense_matrix_macs_per_episode"] = 1
            with self.assertRaisesRegex(ValueError, "resource accounting"):
                A.audit_summaries(result, grouped, frozen)

    def test_registered_gate_recomputes_failure_and_ineligibility(self):
        frozen = {"seeds": [{"seed": 17, "source_qualified": True}]}
        metrics = {"correct_decision_rate": 1., "audit_exact_rate": 1.,
                   "unauthorized_attempt_rate": 0., "positive_completion_rate": 1.}
        measured = {(17, split, arm): copy.deepcopy(metrics) for split in A.SPLITS for arm in A.ARMS}
        expected = A.recompute_gates(measured, frozen)
        self.assertEqual(expected["overall"], {"funnel": "fail", "resource_gain": "pass"})
        A.audit_gates(expected, measured, frozen)
        forged = copy.deepcopy(expected)
        forged["overall"]["funnel"] = "pass"
        with self.assertRaisesRegex(ValueError, "registered gate"):
            A.audit_gates(forged, measured, frozen)
        frozen["seeds"][0]["source_qualified"] = False
        self.assertEqual(A.recompute_gates(measured, frozen)["overall"],
                         {"funnel": "inconclusive", "resource_gain": "inconclusive"})

    def test_independent_resource_counts_charge_codec_and_decoding(self):
        self.assertEqual(A.resource_counts("teacher")["dense_matrix_macs_per_episode"], 23904)
        self.assertEqual(A.resource_counts("joint")["dense_matrix_macs_per_episode"], 25440)
        self.assertEqual(A.resource_counts("compact_transition")["dense_matrix_macs_per_episode"], 1992)
        self.assertEqual(A.resource_counts("compact_transition")["model_parameters"], 637)
        self.assertEqual(A.resource_counts("pca")["codec_buffer_floats"], 160)

    def test_outcome_target_forgery_and_boolean_action_are_rejected(self):
        ep = gym.make_episode(("valid", "tainted"))
        row = outcome(ep)
        A.independently_score(ep, row)
        forged = dict(row, target_digest="0" * 64)
        with self.assertRaisesRegex(ValueError, "original target"):
            A.independently_score(ep, forged)
        with self.assertRaisesRegex(ValueError, "model action"):
            A.independently_score(ep, dict(row, prediction=False))

    def test_counterfactual_pair_coverage_labels_and_native_readouts(self):
        eps, grouped, result, _ = self.summaries_fixture()
        representatives = A.pattern_representatives(eps)
        records = []
        for receiver in representatives.values():
            for donor in representatives.values():
                signature = list(gym.audit_targets(receiver))
                changed = signature[0] != gym.audit_targets(donor)[0]
                signature[0] = gym.audit_targets(donor)[0]
                action = next((b for b in (0, 1) if signature[b] == 1), 2)
                records.append({"seed": 17, "arm": "teacher", "split": "dev",
                                "receiver_case_id": receiver.case_id, "donor_case_id": donor.case_id,
                                "first_probed_branch": 0, "public_depth_feature": 3 / 8,
                                "changed_prefix_status": changed, "prediction": action,
                                "expected_action": action, "audit_prediction": signature,
                                "audit_target": signature, "action_correct": True, "audit_exact": True,
                                "direct_hybrid_replay_agreement": True})
        result["seeds"][0]["splits"]["dev"]["teacher"]["counterfactual"] = {
            "pattern_pairs": 81, "changed_status_pairs": 54,
            "all_pairs_action_accuracy": 1., "all_pairs_audit_exact": 1.,
            "changed_pairs_action_accuracy": 1., "changed_pairs_audit_exact": 1.,
            "direct_hybrid_replay_agreement": 1., "max_task_logit_replay_error": 0.,
            "max_audit_logit_replay_error": 0.}
        with mock.patch.object(A, "SEEDS", (17,)), mock.patch.object(A, "ARMS", ("teacher",)), mock.patch.object(A, "SPLITS", ("dev",)):
            A.audit_counterfactuals(records, eps, grouped, result)
            with self.assertRaisesRegex(ValueError, "coverage incomplete"):
                A.audit_counterfactuals(records[:-1], eps, grouped, result)
            for field, replacement in (("audit_target", [2, 2]), ("prediction", 9),
                                       ("public_depth_feature", .5), ("changed_prefix_status", True)):
                forged = copy.deepcopy(records)
                forged[0][field] = replacement
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "composed history"):
                    A.audit_counterfactuals(forged, eps, grouped, result)

    def test_kernel_proof_batch_omission_cannot_leave_certified_rows(self):
        folder, eps, rows, result = self.kernel_fixture()
        A.audit_kernel(folder, eps, rows, result)
        path = folder / "selected/kernel_audit.json"
        receipt = json.loads(path.read_text())
        receipt["compilations"] = []
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "proof batch coverage"):
            A.audit_kernel(folder, eps, rows, result)

    def test_native_case_substitution_is_rejected_with_unchanged_totals(self):
        folder, eps, rows, result = self.kernel_fixture()
        path = folder / "baseline/native_baseline.json"
        receipt = json.loads(path.read_text())
        receipt["rows"][1]["case_id"] = "substituted"
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "native baseline case roster"):
            A.audit_kernel(folder, eps, rows, result)

    def test_guard_omission_and_deleted_error_log_fail_closed(self):
        folder, eps, rows, result = self.kernel_fixture()
        path = folder / "guards/negative_guards.json"
        original = json.loads(path.read_text())
        omitted = copy.deepcopy(original)
        omitted["rows"].pop()
        write_json(path, omitted)
        with self.assertRaisesRegex(ValueError, "guard roster"):
            A.audit_kernel(folder, eps, rows, result)
        write_json(path, original)
        (folder / "guards" / original["rows"][1]["log"]).unlink()
        with self.assertRaises(FileNotFoundError):
            A.audit_kernel(folder, eps, rows, result)

    def test_rehashed_compiler_source_cannot_change_original_target(self):
        folder, eps, rows, result = self.kernel_fixture()
        path = folder / "selected/kernel_audit.json"
        receipt = json.loads(path.read_text())
        compilation = receipt["compilations"][0]
        source = folder / "selected" / compilation["source"]
        source.write_text(source.read_text().replace(" := by", " := by\n  sorry\n--"))
        compilation["source_sha256"] = A.file_hash(source)
        write_json(path, receipt)
        with self.assertRaisesRegex(ValueError, "bound host theorem"):
            A.audit_kernel(folder, eps, rows, result)

    def test_rehashed_axiom_log_cannot_keep_clean_audit_flag(self):
        folder, eps, rows, result = self.kernel_fixture()
        path = folder / "selected/kernel_audit.json"
        receipt = json.loads(path.read_text())
        compilation = receipt["compilations"][0]
        log = folder / "selected" / compilation["log"]
        name = compilation["expected_declarations"][0]
        log.write_text(f"'{name}' depends on axioms: [sorryAx]\n")
        compilation["log_sha256"] = A.file_hash(log)
        write_json(path, receipt)
        with self.assertRaises(C.KernelAuditError):
            A.audit_kernel(folder, eps, rows, result)


if __name__ == "__main__":
    unittest.main()
