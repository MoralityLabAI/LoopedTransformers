"""Offline control and root-level analysis contracts, using synthetic inputs."""

import unittest
from unittest.mock import patch

import numpy as np

from research_gym.inkling_routing_01.core import TARGET_COORD, WIDTH
from research_gym.inkling_routing_02.native import CalibrationTrace, development_roots, edit_vector
from research_gym.inkling_routing_02.reader_calibration import capture, replay_reader
from research_gym.inkling_routing_02.receipts import describe_endpoint, labels, per_endpoint, root_grid


class ControlTests(unittest.TestCase):
    def test_realized_bfloat16_controls(self):
        import torch

        original = torch.tensor(np.random.default_rng(11).normal(size=WIDTH), dtype=torch.bfloat16)
        original[TARGET_COORD] = 20
        for preserve in (False, True):
            source = original.float().numpy()
            edited = edit_vector(source, 777, preserve)
            np.testing.assert_array_equal(edited, edit_vector(source, 777, preserve))
            after = torch.tensor(edited, dtype=torch.bfloat16).float().numpy()
            self.assertEqual(after[TARGET_COORD], source[TARGET_COORD])
            self.assertLess(abs(np.linalg.norm(after - source) / 20 - 1), 0.01)
            self.assertGreater(np.count_nonzero(after - source), WIDTH // 2)
            if preserve:
                self.assertLess(abs(np.linalg.norm(after) / np.linalg.norm(source) - 1), 0.01)

    def test_infeasible_rotation_does_not_silently_change_dose(self):
        source = np.zeros(WIDTH)
        source[TARGET_COORD] = 10
        source[0] = 1
        with self.assertRaisesRegex(ValueError, "infeasible"):
            edit_vector(source, 777, True)

    def test_all_calibration_lanes_verify_and_restore(self):
        import torch

        trace = CalibrationTrace(None, 2, development_roots(None)[0])
        bank3 = torch.ones((len(trace.lanes), 2, WIDTH), dtype=torch.bfloat16)
        bank3[:, -1, TARGET_COORD] = 20
        bank2 = bank3.unsqueeze(-2).clone()
        trace.writers = [bank3[0, -1].float().numpy()] * 3
        module = type("Module", (), {"query": torch.zeros(12, 4, WIDTH)})()
        for call in range(24):
            patch_result = trace.before_reader(module, (bank2, bank3, call % 12))
            if call == 6:
                self.assertIs(patch_result[0], bank2)
                self.assertIsNot(patch_result[1], bank3)
                self.assertEqual(patch_result[1][1, -1, TARGET_COORD].item(), 0)
                self.assertEqual(bank3[1, -1, TARGET_COORD].item(), 20)
            else:
                self.assertIsNone(patch_result)
        trace.verify()
        self.assertEqual(trace.restored_calls, [7, 8, 9, 10, 11])

    def test_isolated_replay_preserves_clean_and_ablation_rescue(self):
        import torch

        class Reader:
            query = torch.zeros(12, 4, WIDTH)

            def __call__(self, bank2, bank3, index):
                return ((bank2[:, :, 0] + bank3) * 0.5).unsqueeze(-2).expand(-1, -1, 4, -1)

        trace = CalibrationTrace(None, 3, development_roots(None)[0])
        bank3 = torch.ones((len(trace.lanes), 1, WIDTH), dtype=torch.bfloat16)
        bank3[:, -1, TARGET_COORD] = 20
        bank2 = bank3.unsqueeze(-2).clone()
        reader = Reader()
        clean = reader(bank2, bank3, 6)
        recorded = replay_reader(reader, trace, 3, bank2, bank3, 6)
        zero, rescue = (trace.lanes[3, arm] for arm in ("zero_2027", "reader_rescue"))
        np.testing.assert_array_equal(recorded["message"][0], clean[0, -1].float().numpy())
        np.testing.assert_array_equal(recorded["message"][rescue], recorded["message"][zero])
        np.testing.assert_array_equal(recorded["delivered"][rescue], recorded["message"][0])
        self.assertTrue(torch.equal(bank3, bank2.squeeze(-2)))

    def test_clean_capture_keeps_distinct_roots_and_previous_writers(self):
        import torch

        class Reader(torch.nn.Module):
            def forward(self, bank2, bank3, index):
                return bank3.unsqueeze(-2).expand(-1, -1, 4, -1)

        class SyntheticLoop(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(18))
                self.model.message_input = Reader()

            def forward(self, input_ids, **kwargs):
                writer = self.model.layers[17](input_ids.unsqueeze(-1).expand(-1, -1, WIDTH))
                for _ in (2, 3):
                    for visit in range(12):
                        self.model.message_input(writer.unsqueeze(-2), writer, visit)
                    writer = self.model.layers[17](writer + 1)

        model = SyntheticLoop()
        readers, writers = capture(model, {"input_ids": torch.tensor([[1, 4], [3, 7]])})
        self.assertEqual(readers[2]["bank3"][:, 0, 0].tolist(), [4, 7])
        self.assertEqual(readers[3]["bank3"][:, 0, 0].tolist(), [5, 8])
        self.assertEqual(writers[2][:, 0].tolist(), [6, 9])
        self.assertEqual(len(model.model.message_input._forward_hooks), 0)


class RootAnalysisTests(unittest.TestCase):
    def test_root_grid_rejects_duplicates_and_failed_restoration(self):
        rows = [
            {"root_id": "A", "reader_pass": p, "replay": replay, "label": label,
             "restored_calls": list(range(7, 12) if p == 2 else range(19, 24)),
             "reader_scores": {"reconstruction_relative_error": [0] * 4}, "validation_issues": []}
            for p in (2, 3) for replay in (0, 1) for label in labels()
        ]
        self.assertEqual(len(root_grid(rows, ["A"])), 104)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            root_grid(rows + rows[:1], ["A"])
        with patch.dict(rows[0], restored_calls=[7]):
            with self.assertRaisesRegex(ValueError, "instrumentation"):
                root_grid(rows, ["A"])

    def test_placebo_floor_uses_independent_root_count(self):
        beta = {label: 0.5 for label in labels()}
        beta.update(zero_2027=0.4, reference_00=0.2, placebo_00=0.9)
        grid = {("A", 3, replay, label): {"reader_scores": {"beta3_qkvm": [value] * 4}}
                for replay in (0, 1) for label, value in beta.items()}
        [result] = per_endpoint(grid, ["A"], 3, 1)
        self.assertAlmostEqual(result["paired_effect"], 0.08125)
        self.assertIsNone(describe_endpoint([result] * 58)["sham_floor_95_95"])
        self.assertAlmostEqual(describe_endpoint([result] * 59)["sham_floor_95_95"], 0.41875)
        self.assertEqual(describe_endpoint([result])["roots_with_coordinate_dependent_sign"], 1)
