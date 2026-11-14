"""Untrained development fixtures for the independent numerical boundary."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.activation_funnel_af02 import gym, inverse, replay

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for source-equation fixtures")
class ReplayTests(unittest.TestCase):
    def setUp(self):
        from experiments.activation_funnel_af02.model import make_codec, make_model
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.make_codec, self.make_model = make_codec, make_model
        cases = [gym.make_case(("valid", "tainted"), schedule="PWPG"),
                 gym.make_case(("dead", "valid"), schedule="PWPG", probe_order=(1, 0))]
        self.events = np.asarray([gym.feature_sequence(c) for c in cases], dtype=np.float32)

    def exports(self, module):
        return {key: tensor.detach().numpy().copy() for key, tensor in module.state_dict().items()}

    def check_trace(self, actual, expected):
        for field in ("before", "raw", "states", "carried", "task_logits", "audit_logits"):
            replay.assert_close(getattr(actual, field), getattr(expected, field).detach().numpy(), field)

    def test_independent_recurrence_and_actual_codes_for_both_architectures(self):
        for architecture in ("gru", "native"):
            source, codec = self.make_model(architecture), self.make_codec(architecture)
            reference = replay.NumpyModel(architecture, self.exports(source))
            numpy_codec = replay.NumpyCodec(architecture, self.exports(codec))
            for compressed in (False, True):
                with self.subTest(architecture=architecture, compressed=compressed), torch.no_grad():
                    actual = replay.rollout(reference, self.events, numpy_codec if compressed else None)
                    expected = source.rollout(torch.from_numpy(self.events), codec if compressed else None)
                    self.check_trace(actual, expected)

    def test_zero_reset_and_saved_boundary_resume(self):
        for architecture in ("gru", "native"):
            source, codec = self.make_model(architecture), self.make_codec(architecture)
            reference = replay.NumpyModel(architecture, self.exports(source))
            numpy_codec = replay.NumpyCodec(architecture, self.exports(codec))
            with torch.no_grad():
                expected = source.rollout(torch.from_numpy(self.events), codec, zero_between_calls=True)
                self.check_trace(replay.rollout(reference, self.events, numpy_codec, zero_between_calls=True), expected)
                prefix = source.rollout(torch.from_numpy(self.events), codec, stop_event=1)
                resumed = source.rollout(torch.from_numpy(self.events), codec, initial=prefix.states[:, -1], start_event=1)
            actual = replay.rollout(reference, self.events, numpy_codec,
                                    initial=prefix.states[:, -1].detach().numpy(), start_event=1)
            self.check_trace(actual, resumed)

    def test_all_physical_layouts_preserve_native_semantics(self):
        source, codec = self.make_model("native"), self.make_codec("native")
        reference = replay.NumpyModel("native", self.exports(source))
        numpy_codec = replay.NumpyCodec("native", self.exports(codec))
        with torch.no_grad():
            for layout in gym.graph_permutations():
                expected = source.rollout(torch.from_numpy(self.events), codec, layout=layout)
                actual = replay.rollout(reference, self.events, numpy_codec, layout=layout)
                self.check_trace(actual, expected)

    def test_parameter_roster_shapes_and_graph_fail_closed(self):
        arrays = self.exports(self.make_model("native"))
        extra = dict(arrays, unauthorized=np.zeros(1))
        with self.assertRaisesRegex(ValueError, "roster"):
            replay.NumpyModel("native", extra)
        bad = copy.deepcopy(arrays)
        bad["prem"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "premise graph"):
            replay.NumpyModel("native", bad)

    def test_resource_counts_match_registered_parameter_exports(self):
        for architecture in ("gru", "native"):
            source, codec = self.make_model(architecture), self.make_codec(architecture)
            profile = replay.resource_counts(architecture, "joint", 4)
            self.assertEqual(profile["source_parameters"], sum(p.numel() for p in source.parameters()))
            self.assertEqual(profile["codec_parameters"], sum(p.numel() for p in codec.parameters()))
            self.assertEqual(profile["source_buffer_floats"], sum(p.numel() for p in source.buffers()))

    def test_local_jacobian_tangent_and_gauge_against_source(self):
        source = self.make_model("native").double()
        arrays = self.exports(source)
        reference = replay.NumpyModel("native", arrays)
        events = self.events.astype(np.float64)
        with torch.no_grad():
            trace = source.rollout(torch.from_numpy(events[:, :2]))
        h = trace.states[0, -1].numpy()
        goal = np.asarray([0., 1., 0., 0., 0., 0., 0., 0., .25])
        event = torch.from_numpy(goal[None])
        jacobian = torch.autograd.functional.jacobian(lambda v: source.one_step(v[None], event)[0],
                                                      torch.from_numpy(h), vectorize=True).detach().numpy()
        basis = inverse.tangent_basis(arrays)
        self.assertEqual(basis.shape, (80, 75))
        replay.assert_close(basis.T @ basis, np.eye(75), "orthonormal tangent", atol=1e-12, rtol=1e-12)
        _, directions = inverse.seeded_directions(17, "fixture", basis)
        for direction in directions:
            finite = (reference.step(h + 1e-5 * direction, goal) - reference.step(h - 1e-5 * direction, goal)) / 2e-5
            replay.assert_close(finite, jacobian @ direction, "independent Jacobian", atol=1e-7, rtol=1e-4)
        full_fd = inverse.finite_difference_jacobian(reference, h, goal, 1)
        replay.assert_close(full_fd, jacobian, "every ambient Jacobian entry", atol=1e-7, rtol=1e-4)
        _, _, vh = np.linalg.svd(directions, full_matrices=True)
        forged = jacobian + .01 * np.outer(np.eye(80)[0], vh[-1])
        replay.assert_close(directions @ forged.T, directions @ jacobian.T, "four directions miss this mutation", atol=1e-10, rtol=1e-10)
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            replay.assert_close(full_fd, forged, "full ambient mutation guard", atol=1e-7, rtol=1e-4)
        gauge = inverse._gauge_arrays(reference, h, goal, 2)
        for kind in ("atom", "rule"):
            replay.assert_close(gauge[kind + "_normalized"], gauge[kind + "_shifted_normalized"],
                                "constant gauge", atol=1e-10, rtol=0)
        arrays["cell.na.weight"][0] = 0
        self.assertIsNone(inverse.tangent_basis(arrays))


class BoundaryTests(unittest.TestCase):
    def test_patch_uses_actual_carried_coordinates_for_each_token(self):
        for width in (20, 80):
            receiver, donor = np.zeros(width), np.arange(width) + 1.
            patched, mask = replay.patch_code(receiver, donor, "native", "addressed_pair", 1)
            token_width = width // 5
            expected = np.zeros(width, dtype=bool)
            expected[token_width:2 * token_width] = True
            expected[4 * token_width:] = True
            self.assertTrue(np.array_equal(mask, expected))
            self.assertTrue(np.array_equal(patched, donor * expected))
            unchanged, empty = replay.patch_code(receiver, donor, "native", "self", 1)
            self.assertFalse(empty.any())
            self.assertTrue(np.array_equal(unchanged, receiver))
        with self.assertRaisesRegex(ValueError, "native patch"):
            replay.patch_code(np.zeros(4), np.ones(4), "gru", "addressed_pair", 0)

    def test_numeric_mismatch_and_nonfinite_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            replay.assert_close([0.], [1.], "forged state")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            replay.assert_close([np.nan], [np.nan], "bad state")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            np.savez(path, weight=np.asarray([np.inf]))
            with self.assertRaisesRegex(ValueError, "numeric export"):
                replay.load_arrays(path)


@unittest.skipIf(torch is None, "Torch is required for the untrained Jacobian receipt fixture")
class InverseReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.activation_funnel_af02.model import make_model
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        torch.set_num_threads(2)
        torch.manual_seed(101)
        cls.model = make_model("native")
        cls.source = cls.folder / "native_teacher_s17.npz"
        np.savez(cls.source, **{k: v.detach().numpy() for k, v in cls.model.state_dict().items()})
        cases = inverse._expected_anchors()
        events = np.asarray([gym.feature_sequence(c) for c in cases], dtype=np.float32)
        with torch.no_grad():
            trace = cls.model.rollout(torch.from_numpy(events))
        cls.anchors = [{"canonical_id": c.canonical_id, "public_input_sha256": gym.public_input_hash(c),
                        "state": trace.states[i, 3].numpy().copy(), "event": events[i, 2].copy(),
                        "source_trace": {"file": "unit_fixture.npz", "index": i, "sha256": "0" * 64}}
                       for i, c in enumerate(cases)]
        cls.output = cls.folder / "inverse"
        inverse.run_inverse(cls.model, cls.source, cls.anchors, cls.output, 17)
        cls.receipt_path = cls.output / "inverse_s17.json"
        cls.original = cls.receipt_path.read_text()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def tearDown(self):
        self.receipt_path.write_text(self.original)

    def test_retained_inverse_round_trip_replays_all_arrays(self):
        checked = inverse.verify_inverse(self.receipt_path, self.source, self.anchors)
        self.assertEqual(checked["jacobians"], 36)
        self.assertTrue(checked["verified"])

    def test_inverse_summary_and_source_trace_mutations_fail_closed(self):
        for field, value, message in (("finite_difference_max_absolute_error", 1., "finite-difference summary"),
                                      ("full_finite_difference_max_absolute_error", 1., "full finite-difference summary"),
                                      ("source_trace", {"file": "forged.npz"}, "source trace reference")):
            receipt = json.loads(self.original)
            receipt["records"][0][field] = value
            self.receipt_path.write_text(json.dumps(receipt))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                inverse.verify_inverse(self.receipt_path, self.source, self.anchors)

    def test_anchor_reader_binds_full_history_before_goal(self):
        evaluation = self.folder / "anchor_reader"
        evaluation.mkdir(exist_ok=True)
        cases = inverse._expected_anchors()
        events = np.asarray([gym.feature_sequence(c) for c in cases], dtype=np.float32)
        states = np.zeros((18, 6, 80), dtype=np.float32)
        for i, anchor in enumerate(self.anchors):
            states[i, 3] = anchor["state"]
        np.savez(evaluation / "trace.npz", inputs=events, states=states,
                 canonical_ids=np.asarray([c.canonical_id for c in cases]))
        (evaluation / "trace_manifest.json").write_text(json.dumps([
            {"architecture": "native", "arm": "teacher", "seed": 17, "split": "depth_context",
             "condition": "clean", "file": "trace.npz"}]))
        self.assertEqual(len(inverse.anchors_from_evaluation(evaluation, 17)), 18)
        events[0, 0, 6] = 1 - events[0, 0, 6]
        np.savez(evaluation / "trace.npz", inputs=events, states=states,
                 canonical_ids=np.asarray([c.canonical_id for c in cases]))
        with self.assertRaisesRegex(ValueError, "full public input"):
            inverse.anchors_from_evaluation(evaluation, 17)


if __name__ == "__main__":
    unittest.main()
