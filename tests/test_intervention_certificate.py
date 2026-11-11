"""MC-02 exact and native integration tests; all neural fixtures are untrained."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_gym.analysis.intervention_certificate import (
    Counterfactual, exact_controls, score_continuation, select_pairs, validate_manifest,
)
from research_gym.analysis.mechanistic_certificate import oracle
from research_gym.scripts.loopscope_intervention_certificate import (
    execute_pair, make_interventions, run, summarize,
)


def panel():
    graphs = list(itertools.permutations(range(3)))
    starts = [i % 3 for i in range(len(graphs))]
    visits = [1, 2, 4]
    endpoints = [[oracle(g, s, v)[-1] for g, s in zip(graphs, starts)] for v in visits]
    return graphs, starts, endpoints, visits


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        example = exact_controls()["example"]["claim"]
        self.claim = Counterfactual(tuple(example["receiver_graph"]), tuple(example["donor_graph"]),
            example["receiver_cursor"], example["donor_cursor"], example["horizon"])

    def test_exhaustive_graph_controls(self):
        result = exact_controls()
        for key in ("challenges", "correct_accepted", "donor_copies_rejected", "ignored_interventions_rejected"):
            self.assertEqual(result[key], 144)
        self.assertEqual(result["trained_neural_runs"], 0)
        self.assertEqual(result["donor_copy_correct_final_answer"], 60)

    def test_true_counterfactual_certificate(self):
        paths = self.claim.paths()
        result = score_continuation(self.claim, self.claim.donor_cursor, paths["counterfactual"][1:])
        self.assertTrue(result["lift_matches"])
        self.assertTrue(result["counterfactual_trace_match"])
        self.assertFalse(result["cryptographic_source_binding"])

    def test_donor_copy_rejected_at_first_step(self):
        result = score_continuation(self.claim, self.claim.donor_cursor, self.claim.paths()["donor_baseline"][1:])
        self.assertFalse(result["counterfactual_trace_match"])
        self.assertEqual(result["first_wrong_future_offset"], 1)
        self.assertTrue(result["first_future_matches_donor"])

    def test_ignored_patch_rejected_at_first_step(self):
        result = score_continuation(self.claim, self.claim.donor_cursor, self.claim.paths()["receiver_baseline"][1:])
        self.assertFalse(result["counterfactual_trace_match"])
        self.assertEqual(result["first_wrong_future_offset"], 1)

    def test_wrong_lift_rejected_even_with_correct_future(self):
        result = score_continuation(self.claim, self.claim.receiver_cursor, self.claim.paths()["counterfactual"][1:])
        self.assertTrue(result["counterfactual_future_match"])
        self.assertFalse(result["counterfactual_trace_match"])
        self.assertFalse(result["lift_matches"])

    def test_every_intermediate_label_is_constrained(self):
        good = self.claim.paths()["counterfactual"][1:]
        for t in range(len(good)):
            wrong = list(good)
            wrong[t] = (wrong[t] + 1) % 3
            self.assertFalse(score_continuation(self.claim, self.claim.donor_cursor, wrong)["counterfactual_trace_match"])

    def test_statement_uses_receiver_graph(self):
        self.assertEqual(self.claim.statement().graph, self.claim.receiver_graph)
        self.assertEqual(self.claim.statement().start, self.claim.donor_cursor)

    def test_equal_cursor_is_ineligible(self):
        claim = Counterfactual((1, 2, 0), (2, 0, 1), 0, 0, 3)
        self.assertFalse(claim.discriminating())

    def test_same_graph_is_ineligible(self):
        self.assertFalse(Counterfactual((1, 2, 0), (1, 2, 0), 0, 1, 3).discriminating())

    def test_invalid_graph_and_labels(self):
        for args in (((0, 0), (0, 1), 0, 1, 3), ((0, 1), (0, 1, 2), 0, 1, 3),
                     ((0, 1), (1, 0), True, 1, 3), ((0, 1), (1, 0), 0, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Counterfactual(*args)

    def test_invalid_future_shape_and_types(self):
        good = self.claim.paths()["counterfactual"][1:]
        for bad in (good[:-1], (True,) + good[1:], (-1,) + good[1:]):
            with self.assertRaises(ValueError):
                score_continuation(self.claim, self.claim.donor_cursor, bad)


class SelectionTests(unittest.TestCase):
    def select(self, **kwargs):
        return select_pairs(*panel(), horizon=3, max_pairs=kwargs.get("max_pairs", 12), seed=kwargs.get("seed", 5))

    def test_deterministic_and_cap(self):
        a, b = self.select(), self.select()
        self.assertEqual(a, b)
        self.assertLessEqual(len(a["selected_pairs"]), 12)
        validate_manifest(a)

    def test_unique_receiver_visit_slots(self):
        pairs = self.select()["selected_pairs"]
        keys = [(row["visit"], row["receiver"]) for row in pairs]
        self.assertEqual(len(keys), len(set(keys)))

    def test_all_selected_claims_are_discriminating(self):
        for row in self.select()["selected_pairs"]:
            self.assertTrue(row["claim"]["first_step_discriminating"])
            self.assertNotEqual(row["receiver"], row["donor"])

    def test_manifest_rejects_modified_target(self):
        selected = self.select()
        selected["selected_pairs"][0]["claim"]["donor_cursor"] ^= 1
        with self.assertRaises(ValueError):
            validate_manifest(selected)

    def test_current_errors_are_excluded(self):
        graphs, starts, labels, visits = panel()
        wrong = [[(v + 1) % 3 for v in row] for row in labels]
        result = select_pairs(graphs, starts, wrong, visits, horizon=3, max_pairs=12, seed=5)
        self.assertEqual(result["endpoint_correct_slots"], 0)
        self.assertEqual(result["selected_pairs"], [])

    def test_no_future_output_argument_exists(self):
        with self.assertRaises(TypeError):
            select_pairs(*panel(), horizon=3, max_pairs=12, seed=5, future_predictions=[])

    def test_validation_rejects_bad_visits_and_shapes(self):
        graphs, starts, labels, visits = panel()
        for bad in ([1, 1, 2], [4, 2, 1], [0, 1, 2]):
            with self.assertRaises(ValueError):
                select_pairs(graphs, starts, labels, bad, horizon=3, max_pairs=12, seed=5)
        with self.assertRaises(ValueError):
            select_pairs(graphs, starts, [labels[0][:-1]] * 3, visits, horizon=3, max_pairs=12, seed=5)

    def test_zero_coverage_is_undefined(self):
        summary = summarize([], status="MEASURED_CHECKPOINT", visits=[1])
        self.assertIsNone(summary["value"])
        self.assertEqual(summary["status"], "NO_ELIGIBLE_PAIRS")

    def test_smoke_never_has_scientific_summary(self):
        self.assertIsNone(summarize([], status="UNTRAINED_SMOKE", visits=[1]))

    def test_item_aggregation_not_raw_pair_mean(self):
        def row(receiver, visit, hit):
            return {"receiver": receiver, "visit": visit, "eligible": True,
                    "arms": [{"name": "cursor_transfer", "counterfactual_trace_match": hit},
                             {"name": "random_cursor_0", "counterfactual_future_match": False}]}
        rows = [row(0, 1, True), row(0, 2, True), row(1, 1, False)]
        self.assertEqual(summarize(rows, status="MEASURED_CHECKPOINT", visits=[1,2])["value"], 0.5)


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        import torch
        from research_gym.scripts import loopscope_semantic_echo as native
        from research_gym.scripts.loopscope_toy import make_batch
        self.torch, self.native = torch, native
        self.model, _ = native.load_model(None, smoke=True, seed=71, device="cpu", dtype="float32")
        self.tokens, _ = make_batch(4, 4, 123, 7, depth=None)
        with torch.inference_mode():
            self.embedding, self.states = native.native_trace(self.model, self.tokens, 7)

    def test_full_state_parity(self):
        with self.torch.inference_mode():
            result = self.native.verify_native_trace(self.model, self.tokens, self.states, rtol=2e-5, atol=2e-6)
        self.assertTrue(result["passed"])
        self.assertEqual(result["own_head_label_mismatches"], 0)

    def test_cursor_patch_preserves_receiver_context(self):
        base, donor = self.states[2, :1], self.states[2, 1:2]
        names, patched = make_interventions(base, donor, seed=9, random_controls=4)
        for i, name in enumerate(names):
            if name != "native_donor_replay":
                self.assertTrue(self.torch.equal(patched[i:i+1, :-1], base[:, :-1]))
        self.assertTrue(self.torch.equal(patched[:1, -1], donor[:, -1]))

    def test_random_and_gauge_norms_match(self):
        base, donor = self.states[2, :1], self.states[2, 1:2]
        names, patched = make_interventions(base, donor, seed=9, random_controls=4)
        target = float(self.torch.linalg.vector_norm(donor[:, -1] - base[:, -1]))
        for i, name in enumerate(names):
            if name.startswith("random_cursor_") or name == "cursor_constant_shift":
                self.assertAlmostEqual(float(self.torch.linalg.vector_norm(patched[i:i+1] - base)), target, places=5)

    def test_constant_shift_erased_by_native_injection(self):
        base, donor = self.states[2, :1], self.states[2, 1:2]
        names, patched = make_interventions(base, donor, seed=9, random_controls=2)
        with self.torch.inference_mode():
            i = names.index("cursor_constant_shift")
            self.assertTrue(self.torch.allclose(self.model.inject(patched[i:i+1]), self.model.inject(base), atol=2e-6, rtol=2e-5))

    def test_native_pair_replay_controls(self):
        graphs = []
        for tokens in self.tokens.tolist():
            g = [0] * 4
            for src, dst in zip(tokens[:8:2], tokens[1:8:2]):
                g[src] = dst
            graphs.append(tuple(g))
        claim = Counterfactual(graphs[0], graphs[1], oracle(graphs[0], int(self.tokens[0,-1]), 2)[-1],
                               oracle(graphs[1], int(self.tokens[1,-1]), 2)[-1], 3)
        row = {"receiver":0,"donor":1,"visit":2,"eligible":False,"claim":claim.description()}
        with self.torch.inference_mode():
            result, arrays = execute_pair(self.model, self.native, self.states, self.embedding,
                                          row, horizon=3, seed=17, random_controls=2)
        for name in ("sham", "native_donor_replay"):
            self.assertTrue(result["native_parity"][name]["passed"])
            self.assertTrue(result["native_parity"][name]["logits"]["passed"])
        self.assertEqual(arrays["resumed"].shape[:2], (3, 6))

    def test_nonfinite_patch_rejected(self):
        base = self.states[2, :1].clone()
        base[0,0,0] = float("nan")
        with self.assertRaises(ValueError):
            make_interventions(base, self.states[2, 1:2], seed=9, random_controls=2)

    def test_smoke_executes_and_retains_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(smoke=True, checkpoint=None, expected_checkpoint_sha256=None,
                items=12, visits=[1,2,4], horizon=3, max_pairs=2, random_controls=2,
                seed=940211, cpu_threads=1, device="cpu", out=Path(d)/"run")
            result = run(args)
            self.assertIsNone(result["scientific_summary"])
            self.assertEqual(result["receipt"]["trained_checkpoints_evaluated"],0)
            self.assertTrue(result["receipt"]["native_parity"]["passed"])
            validate_manifest(json.loads((args.out/"selection.json").read_text()))
            self.assertTrue((args.out/"replays.npz").exists())
            with self.assertRaises(ValueError):
                run(args)

    def test_checkpoint_pin_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"checkpoint.pt"
            path.write_bytes(b"invalid")
            args = argparse.Namespace(smoke=False, checkpoint=path, expected_checkpoint_sha256=None,
                items=4, visits=[1], horizon=2, max_pairs=2, random_controls=2,
                seed=1, cpu_threads=1, device="cpu", out=Path(d)/"out")
            with self.assertRaises(ValueError):
                run(args)
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
