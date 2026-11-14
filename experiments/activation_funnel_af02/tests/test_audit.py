"""Finite arithmetic and saved-tensor mutation fixtures; no optimization runs."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.activation_funnel_af02 import audit as A, gym, replay

try:
    import torch
except ImportError:
    torch = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


class ArithmeticTests(unittest.TestCase):
    def test_order_specific_source_failure_cannot_hide_in_an_average(self):
        perfect = {"by_order": {order: {"task_accuracy": 1., "audit_branch_accuracy": [1., 1.]}
                                for order in ("0,1", "1,0")}}
        self.assertTrue(A.source_eligibility(perfect, perfect))
        bad = copy.deepcopy(perfect)
        bad["by_order"]["1,0"]["audit_branch_accuracy"][1] = .94
        self.assertFalse(A.source_eligibility(perfect, bad))

    def test_composed_history_weighting_ignores_duplicate_multiplicity(self):
        rows = []
        for key, correct, copies in (("a", True, 9), ("b", False, 1)):
            for _ in range(copies):
                rows.append({"composed_input_sha256": key, "changed_prefix_status": True,
                             "action_correct": correct, "audit_exact": correct,
                             "first_branch_target_correct": correct, "other_branch_retained": correct,
                             "patch_l2_norm": 0., "direct_hybrid_agreement": True,
                             "self_replay_agreement": False, "direct_logit_error": 0., "self_logit_error": 1.})
        result = A.counterfactual_summary(rows)
        self.assertEqual(result["composed_input_mean_action_correct"], .5)
        self.assertEqual(result["changed_pair_action_correct"], .9)

    def gate_fixture(self):
        frozen = {"runs": [{"architecture": a, "seed": s, "source_qualified": True}
                            for a in A.ARCHITECTURES for s in A.SEEDS]}
        outcomes, counterfactuals = {}, {}
        for a in A.ARCHITECTURES:
            for seed in A.SEEDS:
                for split in A.SPLITS:
                    for arm in A.ARMS:
                        outcomes[(a, seed, split, arm)] = {
                            "correct_decision_rate": 1., "audit_exact_rate": .5 if arm == "task" else 1.,
                            "unauthorized_attempt_rate": 0.}
                if a == "native":
                    for arm in ("teacher", "joint"):
                        for condition in ("addressed_pair", "other_pair"):
                            counterfactuals[(a, seed, arm, "depth_context", condition)] = {
                                "changed_pair_first_branch_target_correct": .8 if condition == "addressed_pair" else .4,
                                "changed_pair_other_branch_retained": .9}
        return frozen, outcomes, counterfactuals

    def test_scientific_gates_fail_or_become_inconclusive_at_registered_conditions(self):
        frozen, outcomes, cf = self.gate_fixture()
        result = A.recompute_gates(frozen, outcomes, cf)
        self.assertEqual(result["by_architecture"]["native"]["addressed_causal_preservation"], "pass")
        cf[("native", 17, "joint", "depth_context", "addressed_pair")]["changed_pair_first_branch_target_correct"] = .6
        result = A.recompute_gates(frozen, outcomes, cf)
        self.assertEqual(result["by_architecture"]["native"]["addressed_causal_preservation"], "fail")
        outcomes[("native", 17, "depth_context", "teacher")]["audit_exact_rate"] = .9
        result = A.recompute_gates(frozen, outcomes, cf)
        native17 = next(r for r in result["per_run"] if (r["architecture"], r["seed"]) == ("native", 17))
        self.assertEqual(native17["addressed_causal_preservation"]["verdict"], "inconclusive")
        self.assertEqual(native17["delayed_preservation"]["depth_context"]["verdict"], "inconclusive")
        forged = copy.deepcopy(result)
        forged["by_architecture"]["native"]["addressed_causal_preservation"] = "pass"
        with self.assertRaisesRegex(ValueError, "forged gate"):
            A.compare(forged, result, "forged gate")

    def test_artifact_paths_and_nonfinite_scalar_receipts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for path in ("../outside", "/outside", "x/../../outside"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    A.safe_path(folder, path)
        with self.assertRaises(ValueError):
            A.numeric(float("nan"), 0., "bad numerical summary")


@unittest.skipIf(torch is None, "Torch is required to create untrained saved-state fixtures")
class SavedTensorAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.activation_funnel_af02 import evaluate
        from experiments.activation_funnel_af02.model import make_model
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        torch.set_num_threads(1)
        torch.manual_seed(94)
        cls.source = make_model("native")
        cls.weights = {k: v.detach().numpy().copy() for k, v in cls.source.state_dict().items()}
        cls.models = {("native", 17): replay.NumpyModel("native", cls.weights)}
        cls.cases = gym.canonical_cases("dev")
        traces, native = [], {}
        cls.rows, cls.stats = evaluate.score_arm(cls.source, None, cls.cases, gym.generate_split("dev"),
            "native", "teacher", 17, "clean", cls.folder, traces, native)
        write_json(cls.folder / "trace_manifest.json", traces)
        cf_manifest = []
        cls.cf_rows, cf_summary = evaluate.counterfactuals(cls.source, None, cls.cases, "native", "teacher", 17,
                                                          cls.folder, cf_manifest)
        write_json(cls.folder / "counterfactual_manifest.json", cf_manifest)
        (cls.folder / "counterfactuals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cls.cf_rows))
        perm_manifest = []
        perm_summary = evaluate.permutations(cls.source, None, cls.cases, "native", "teacher", 17, cls.folder, perm_manifest)
        write_json(cls.folder / "permutation_manifest.json", perm_manifest)
        cls.stats["counterfactuals"] = cf_summary
        cls.stats["permutations"] = perm_summary
        cls.result = {"counterfactual_rows": len(cls.cf_rows),
                      "runs": [{"architecture": "native", "seed": 17,
                                "splits": {"dev": {"teacher": cls.stats}}}]}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for key, value in (("ARCHITECTURES", ("native",)), ("SEEDS", (17,)),
                           ("ARMS", ("teacher",)), ("SPLITS", ("dev",)), ("CONTROLS", ())):
            stack.enter_context(mock.patch.object(A, key, value))
        self.traces, _, _ = A.audit_traces(self.folder, self.models, {})

    def test_complete_saved_native_trace_patch_and_layout_replay(self):
        A.audit_counterfactuals(self.folder, self.result, self.traces, self.models, {})
        _, summary = A.audit_permutations(self.folder, self.result, self.traces, self.models, {})
        self.assertEqual(summary["layout_runs"], 36 * 12)

    def test_raw_prediction_target_and_context_mutations_fail(self):
        row = copy.deepcopy(self.rows[0])
        case = gym.generate_split("dev")[0]
        A.prediction_fields(case, row, outcome=True)
        for field, value in (("context_digest", "0" * 64), ("prediction", True),
                             ("action_cost", case.budget - 1)):
            bad = dict(row)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.prediction_fields(case, bad, outcome=True)

    def test_rewritten_trace_tensor_is_rejected_by_independent_replay(self):
        manifest = A.read_json(self.folder / "trace_manifest.json")
        path = self.folder / manifest[0]["file"]
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        arrays = A.archive(path)
        arrays["states"][0, 1, 0] += 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "independent trace replay"):
            A.audit_traces(self.folder, self.models, {})

    def test_counterfactual_omission_and_composed_label_forgery_fail(self):
        path = self.folder / "counterfactuals.jsonl"
        original = path.read_text()
        self.addCleanup(path.write_text, original)
        path.write_text("".join(json.dumps(r) + "\n" for r in self.cf_rows[:-1]))
        with self.assertRaisesRegex(ValueError, "row count"):
            A.audit_counterfactuals(self.folder, self.result, self.traces, self.models, {})
        rows = copy.deepcopy(self.cf_rows)
        rows[0]["audit_target"] = [9, 9]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "counterfactual composition"):
            A.audit_counterfactuals(self.folder, self.result, self.traces, self.models, {})

    def test_actual_code_patch_forgery_fails_even_with_retained_logit_rows(self):
        manifest = A.read_json(self.folder / "counterfactual_manifest.json")
        path = self.folder / manifest[0]["file"]
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        arrays = A.archive(path)
        arrays["patched_code"][0, 0] += 1
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "patched coordinates"):
            A.audit_counterfactuals(self.folder, self.result, self.traces, self.models, {})

    def test_layout_roster_substitution_fails_with_same_output_count(self):
        manifest = A.read_json(self.folder / "permutation_manifest.json")
        path = self.folder / manifest[0]["file"]
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        arrays = A.archive(path)
        arrays["atom_slots"][-1] = arrays["atom_slots"][0]
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "permutation atom slots"):
            A.audit_permutations(self.folder, self.result, self.traces, self.models, {})

    def test_rehashed_operational_precision_changes_are_rejected(self):
        for manifest_name, field, check, message in (
                ("counterfactual_manifest.json", "carried", A.audit_counterfactuals, "counterfactual precision"),
                ("permutation_manifest.json", "task_logits", A.audit_permutations, "permutation precision")):
            manifest = A.read_json(self.folder / manifest_name)
            path = self.folder / manifest[0]["file"]
            original = path.read_bytes()
            arrays = A.archive(path)
            arrays[field] = arrays[field].astype(np.float64)
            np.savez_compressed(path, **arrays)
            try:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                    check(self.folder, self.result, self.traces, self.models, {})
            finally:
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
