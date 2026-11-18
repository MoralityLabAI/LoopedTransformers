"""Analytic precision fixtures and one untrained native fixture on 61064."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope13c import precision_oracle as precision


class QuadraticBlock(torch.nn.Module):
    def __init__(self, coefficient=0.125):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, state):
        result = state.clone()
        result[..., 0] = state[..., 0] + self.coefficient * state[..., 10:].square().sum(-1)
        return result


class AnalyticModel(torch.nn.Module):
    """Two blocks give a known diagonal Hessian on current-head-null axes."""
    def __init__(self, coefficient=0.125):
        super().__init__()
        self.block = QuadraticBlock(coefficient)
        self.head = torch.nn.Linear(48, 10, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(10, 48))


class PrecisionOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def analytic_fixture(self, coefficient=0.125):
        model32 = AnalyticModel(coefficient).eval().requires_grad_(False)
        model64 = copy.deepcopy(model32).double()
        h32 = torch.zeros((1, 81, 48), dtype=torch.float32)
        selected = torch.zeros(96, dtype=torch.float64)
        selected[48 + 11] = 1.0
        sham = torch.zeros_like(selected)
        sham[48 + 12] = 1.0
        projector = torch.eye(48, dtype=torch.float64)
        projector[:10, :10] = 0.0
        jacobian, _ = precision.native_pair_jacobian(model64, h32.double(), 10)
        kwargs = dict(goal=10, donor_class=0, original_class=1,
                      reference_h64=h32.double(), reference_j64=jacobian,
                      head_projector=projector, target_curvature=4.0 * coefficient)
        return model32, model64, h32, selected, [sham], kwargs

    def test_dense_hessian_preserves_every_coordinate_column(self):
        diagonal = torch.linspace(-2.0, 3.0, 96, dtype=torch.float64)
        v = torch.arange(96, dtype=torch.float64) / 96.0
        matrix = torch.diag(diagonal) + torch.outer(v, v)
        seen = []

        def oracle(direction):
            seen.append(direction.clone())
            return (matrix @ direction).reshape(2, 48)

        symmetric, receipt, raw = precision.dense_margin_hessian(oracle)
        self.assertEqual(receipt["hvp_count"], 96)
        self.assertEqual(len(seen), 96)
        torch.testing.assert_close(torch.stack(seen), torch.eye(96, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(raw, matrix, rtol=0, atol=0)
        torch.testing.assert_close(symmetric, matrix, rtol=0, atol=0)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["skew_frobenius_norm"], 0.0)
        json.dumps(receipt, allow_nan=False)

    def test_dense_skew_failure_keeps_complete_raw_columns(self):
        matrix = torch.eye(96, dtype=torch.float64)
        matrix[0, 1] = 0.1
        symmetric, receipt, raw = precision.dense_margin_hessian(lambda v: matrix @ v)
        self.assertFalse(receipt["passed"])
        self.assertGreater(receipt["skew_frobenius_norm"], receipt["symmetry_limit"])
        self.assertEqual(receipt["hvp_count"], 96)
        torch.testing.assert_close(raw, matrix, rtol=0, atol=0)
        torch.testing.assert_close(symmetric, (matrix + matrix.T) / 2, rtol=0, atol=0)

    def test_dense_rejects_nonfinite_or_non_fp64_columns(self):
        with self.assertRaises(ValueError):
            precision.dense_margin_hessian(lambda v: v.float())
        with self.assertRaises(ValueError):
            precision.dense_margin_hessian(lambda v: torch.full_like(v, float("nan")))
        with self.assertRaises(ValueError):
            precision.dense_margin_hessian(lambda v: v[:95])

    def test_dense_native_analytic_hessian_matches_known_curvature(self):
        model32, model64, h32, selected, shams, kwargs = self.analytic_fixture()
        oracle = precision.native_margin_hvp(model64, h32.double(), 10, 0, 1)
        symmetric, receipt, raw = precision.dense_margin_hessian(oracle)
        expected = torch.zeros((96, 96), dtype=torch.float64)
        expected[58:, 58:] = 0.5 * torch.eye(38, dtype=torch.float64)
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        torch.testing.assert_close(symmetric, expected, rtol=0, atol=0)
        self.assertTrue(receipt["passed"])

    def test_exact_native_geometry_and_curvature_pass_all_gates(self):
        fixture = self.analytic_fixture()
        result = precision.native_precision_check(*fixture[:5], **fixture[5])
        self.assertTrue(result.receipt["native_ready"], result.receipt)
        for key in ("native_physical", "promoted_curvature", "curvature_resolution", "derivative_agreement"):
            self.assertTrue(result.receipt[key])
        self.assertEqual(result.receipt["hvp_counts"], {"native32": 2, "promoted_exact": 2, "promoted_probe": 2})
        self.assertEqual(result.receipt["jacobian_counts"], {"native32": 10, "promoted_same_state": 10})
        torch.testing.assert_close(result.raw["q_native"], torch.full((2,), 0.5, dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(result.raw["jacobian32"].dtype, torch.float32)
        self.assertEqual(result.raw["hvp32"].dtype, torch.float32)
        self.assertEqual(result.raw["hvp64_same_probe"].dtype, torch.float64)
        self.assertEqual(result.raw["hvp64_actual_delta"].dtype, torch.float64)
        json.dumps(result.receipt, allow_nan=False)

    def test_actual_fp32_state_difference_can_exceed_probe_precision(self):
        model32, model64, h32, selected, shams, kwargs = self.analytic_fixture()
        h32[0, 10, 11] = 2.0 ** -30
        kwargs["reference_h64"] = h32.double()
        kwargs["reference_j64"], _ = precision.native_pair_jacobian(model64, h32.double(), 10)
        result = precision.native_precision_check(model32, model64, h32, selected, shams, **kwargs)
        raw = result.raw
        self.assertEqual(float(raw["actual_edits_native_fp64"][0, 59]), 1.0 - 2.0 ** -30)
        self.assertEqual(float(raw["probe32"][0, 59]), 1.0)
        self.assertEqual(float(raw["direction_cast_residual"][0, 59]), -(2.0 ** -30))
        self.assertEqual(result.receipt["arms"][0]["direction_cast_norm"], 2.0 ** -30)
        expected_q = 0.5 * (1.0 - 2.0 ** -30) ** 2
        self.assertAlmostEqual(float(raw["q64_actual"][0]), expected_q, places=16)
        self.assertGreater(result.receipt["arms"][0]["promoted_direction_cast_curvature_drift"], 0.0)
        self.assertFalse(result.receipt["native_physical"])

    def test_probe_resolution_rejects_common_curvature_bias_that_parity_cancels(self):
        fixture = self.analytic_fixture()
        original_factory = precision.native_margin_hvp

        def biased_factory(model, state, *args):
            oracle = original_factory(model, state, *args)
            if state.dtype == torch.float32:
                return lambda direction: oracle(direction) + (2.0 ** -23) * direction
            return oracle

        with patch.object(precision, "native_margin_hvp", side_effect=biased_factory):
            result = precision.native_precision_check(*fixture[:5], **fixture[5])
        self.assertTrue(result.receipt["native_physical"])
        self.assertTrue(result.receipt["promoted_curvature"])
        self.assertTrue(result.receipt["derivative_agreement"])
        self.assertFalse(result.receipt["curvature_resolution"])
        self.assertFalse(result.receipt["native_ready"])
        self.assertEqual(result.receipt["arms"][1]["paired_resolution_error"], 0.0)
        self.assertGreater(result.receipt["arms"][1]["probe_resolution_error"], result.receipt["q_tolerance"])

    def test_paired_resolution_rejects_opposing_errors_inside_each_probe_limit(self):
        fixture = self.analytic_fixture(coefficient=2.0 ** -14)
        original_factory = precision.native_margin_hvp

        def opposing_factory(model, state, *args):
            oracle = original_factory(model, state, *args)
            if state.dtype == torch.float32:
                diagonal = torch.zeros(96, dtype=torch.float32)
                diagonal[59], diagonal[60] = -(2.0 ** -27), 2.0 ** -27
                return lambda direction: oracle(direction) + diagonal * direction
            return oracle

        with patch.object(precision, "native_margin_hvp", side_effect=opposing_factory):
            result = precision.native_precision_check(*fixture[:5], **fixture[5])
        self.assertTrue(result.receipt["promoted_curvature"])
        self.assertTrue(result.receipt["derivative_agreement"])
        self.assertFalse(result.receipt["curvature_resolution"])
        sham = result.receipt["arms"][1]
        self.assertLess(sham["probe_resolution_error"], result.receipt["q_tolerance"])
        self.assertGreater(sham["paired_resolution_error"], result.receipt["q_tolerance"])

    def test_forward_agreement_is_a_separate_derivative_gate(self):
        fixture = self.analytic_fixture()
        original_jacobian = precision.native_pair_jacobian

        def shifted_forward(model, state, goal):
            jacobian, logits = original_jacobian(model, state, goal)
            if state.dtype == torch.float32:
                logits = logits + 1e-4
            return jacobian, logits

        with patch.object(precision, "native_pair_jacobian", side_effect=shifted_forward):
            result = precision.native_precision_check(*fixture[:5], **fixture[5])
        self.assertTrue(result.receipt["jacobian_agreement"]["passed"])
        self.assertFalse(result.receipt["forward_agreement"]["passed"])
        self.assertFalse(result.receipt["derivative_agreement"])
        self.assertFalse(result.receipt["native_ready"])

    def test_reference_trajectory_drift_does_not_change_common_state_oracle(self):
        model32, model64, h32, selected, shams, kwargs = self.analytic_fixture()
        kwargs["reference_h64"] = torch.full((1, 81, 48), 0.1, dtype=torch.float64)
        kwargs["reference_j64"], _ = precision.native_pair_jacobian(model64, kwargs["reference_h64"], 10)
        result = precision.native_precision_check(model32, model64, h32, selected, shams, **kwargs)
        self.assertTrue(result.receipt["native_ready"])
        self.assertEqual(result.receipt["jacobian_agreement"]["error"], 0.0)
        self.assertGreater(result.receipt["state_drift"]["full_l2"], 1.0)
        self.assertGreater(result.receipt["jacobian_state_drift"]["frobenius_norm"], 0.0)
        torch.testing.assert_close(result.raw["native_h32_lifted"], h32.double(), rtol=0, atol=0)

    def test_model_promotion_and_probe_dtype_are_enforced(self):
        model32, model64, h32, selected, shams, kwargs = self.analytic_fixture()
        with torch.no_grad():
            model64.head.weight[0, 0] += 1e-10
        with self.assertRaises(ValueError):
            precision.native_precision_check(model32, model64, h32, selected, shams, **kwargs)
        oracle = precision.native_margin_hvp(model32, h32, 10, 0, 1)
        with self.assertRaises(ValueError):
            oracle(selected)

    def test_native_untrained_fixture_has_explicit_common_state_inputs_and_fd_parity(self):
        torch.manual_seed(61064)
        model32 = TinyRecursiveModel().eval().requires_grad_(False)
        model64 = copy.deepcopy(model32).double()
        inputs, _ = fixed_data(1, 61064)
        with torch.no_grad():
            h32 = model32.reason_states(inputs)[2]
            h64 = model64.reason_states(inputs)[2]
        generator = torch.Generator().manual_seed(61064)
        selected = torch.randn(96, dtype=torch.float64, generator=generator)
        selected = selected / selected.norm() * 0.25
        sham = torch.randn(96, dtype=torch.float64, generator=generator)
        sham = sham / sham.norm() * 0.25
        _, _, vh = torch.linalg.svd(model32.head.weight.detach().double(), full_matrices=True)
        projector = vh[10:].T @ vh[10:]
        reference_j, _ = precision.native_pair_jacobian(model64, h64, 10)
        reference_hvp = precision.native_margin_hvp(model64, h64, 10, 3, 2)
        target = float(selected @ reference_hvp(selected))
        before = h32.clone()
        result = precision.native_precision_check(
            model32, model64, h32, selected, [sham], goal=10, donor_class=3,
            original_class=2, reference_h64=h64, reference_j64=reference_j,
            head_projector=projector, target_curvature=target,
        )
        raw = result.raw
        torch.testing.assert_close(h32, before, rtol=0, atol=0)
        torch.testing.assert_close(raw["native_h32_lifted"], before.double(), rtol=0, atol=0)
        native_after = (raw["native_pair_base_fp32"].numpy()[None]
                        + raw["intended_edits_fp32"].numpy().reshape(-1, 2, 48)).astype(np.float32)
        physical = (native_after.astype(np.float64)
                    - raw["native_pair_base_fp32"].numpy().astype(np.float64)[None]).reshape(-1, 96)
        np.testing.assert_array_equal(raw["actual_edits_native_fp64"].numpy(), physical)
        self.assertGreater(result.receipt["state_drift"]["full_l2"], 0.0)
        self.assertEqual(raw["hvp32"].dtype, torch.float32)
        self.assertTrue(result.receipt["derivative_agreement"], result.receipt)

        # Independent central differences of the pair gradient at the exact
        # promoted native state check the stored fresh directional derivative.
        direction = raw["probe64"][0].reshape(2, 48)

        def gradient(edit):
            edit = edit.clone().requires_grad_(True)
            state = before.double().clone()
            state[0, 9] = state[0, 9] + edit[0]
            state[0, 10] = state[0, 10] + edit[1]
            logits = model64.head(model64.block(model64.block(state)))[0, 10]
            value, = torch.autograd.grad(logits[3] - logits[2], edit)
            return value.detach().reshape(96)

        epsilon = 1e-4
        finite_difference = (gradient(epsilon * direction) - gradient(-epsilon * direction)) / (2 * epsilon)
        self.assertLess(float((finite_difference - raw["hvp64_same_probe"][0]).norm()), 2e-7)
        json.dumps(result.receipt, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
