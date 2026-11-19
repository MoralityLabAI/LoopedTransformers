"""Analytic Richardson witnesses and one exposed-61064 continuation fixture."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope13c.precision_oracle import dense_margin_hessian
from mechinterp.loopscope13d import promoted_precision as precision


class ExactHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(10, 48, dtype=torch.float64), requires_grad=False)

    def forward(self, state):
        return state @ self.weight.T


class PolynomialBlock(torch.nn.Module):
    def __init__(self, degree=2, coefficient=0.125):
        super().__init__()
        self.degree, self.coefficient = degree, coefficient

    def forward(self, state):
        result = state.clone()
        result[..., 0] = state[..., 0] + self.coefficient * state[..., 10:].pow(self.degree).sum(-1)
        return result


class AnalyticPolynomial(torch.nn.Module):
    def __init__(self, degree=2, coefficient=0.125):
        super().__init__()
        self.block = PolynomialBlock(degree, coefficient)
        self.head = ExactHead()


def analytic_fixture(degree=2, coefficient=0.125, radius=1.):
    model = AnalyticPolynomial(degree, coefficient).eval()
    base = torch.full((1, 81, 48), .25, dtype=torch.float64)
    deltas = torch.zeros((9, 96), dtype=torch.float64)
    for index in range(9):
        deltas[index, 58 + index] = radius
    curvature = 2 * coefficient * degree * (degree - 1) * .25 ** (degree - 2) * radius ** 2
    kwargs = dict(goal=10, donor_class=0, original_class=1, target_curvature=curvature)
    return model, base, deltas, kwargs, curvature


class PromotedPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_known_quadratic_has_exact_curvature_and_separate_gates(self):
        model, base, deltas, kwargs, expected = analytic_fixture()
        before_base, before_deltas = base.clone(), deltas.clone()
        result = precision.gradient_richardson_check(model, base, deltas, **kwargs)
        self.assertTrue(result.receipt["oracle_consistency_passed"])
        self.assertTrue(result.receipt["fd_parity_passed"])
        self.assertTrue(result.receipt["passed"])
        self.assertFalse(result.receipt["is_interval_certificate"])
        self.assertEqual(result.receipt["fresh_hvp_count"], 9)
        self.assertEqual(result.receipt["first_gradient_count"], 54)
        for name in ("q_fresh", "q_richardson_coarse", "q_richardson_fine"):
            torch.testing.assert_close(result.raw[name], torch.full((9,), expected, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(base, before_base, rtol=0, atol=0)
        torch.testing.assert_close(deltas, before_deltas, rtol=0, atol=0)
        self.assertTrue(all(not value.requires_grad for value in result.raw.values()))
        json.dumps(result.receipt, allow_nan=False)

    def test_sixth_degree_function_has_correct_richardson_order(self):
        model, base, deltas, kwargs, expected = analytic_fixture(degree=6)
        result = precision.gradient_richardson_check(model, base, deltas, **kwargs)
        self.assertTrue(result.receipt["passed"], result.receipt)
        errors = result.raw["coarse_errors"], result.raw["fine_errors"]
        self.assertGreater(float(errors[0].min()), 1e-12)
        ratios = errors[0] / errors[1]
        self.assertTrue(bool(((ratios > 15.5) & (ratios < 16.5)).all()))
        torch.testing.assert_close(result.raw["q_fresh"], torch.full((9,), expected, dtype=torch.float64), rtol=0, atol=0)

    def test_unit_step_definition_and_quadratic_scaling_are_preserved(self):
        q_values = []
        for radius in (.25, 1., 3.):
            model, base, deltas, kwargs, expected = analytic_fixture(degree=4, radius=radius)
            result = precision.gradient_richardson_check(model, base, deltas.reshape(9, 2, 48), **kwargs)
            self.assertTrue(result.receipt["passed"], result.receipt)
            actual_lengths = torch.linalg.vector_norm(result.raw["probe_edits"], dim=-1)
            expected_lengths = torch.tensor(precision.FD_STEPS, dtype=torch.float64)[None, :, None].expand(9, 3, 2)
            torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
            torch.testing.assert_close(result.raw["q_richardson_fine"],
                                       torch.full((9,), expected, dtype=torch.float64), rtol=0, atol=1e-13)
            q_values.append(float(result.raw["q_richardson_fine"][0]) / radius ** 2)
        self.assertLess(max(q_values) - min(q_values), 1e-13)

    def test_first_gradients_reject_a_biased_hvp_that_matches_its_dense_matrix(self):
        model, base, deltas, kwargs, _ = analytic_fixture()
        factory = precision.native_margin_hvp
        bias = 2.0 ** -22

        def biased_factory(*args):
            real = factory(*args)
            return lambda delta: real(delta) + bias * delta

        biased_oracle = biased_factory(model, base, 10, 0, 1)
        dense, receipt, _ = dense_margin_hessian(biased_oracle)
        self.assertTrue(receipt["passed"])
        torch.testing.assert_close(torch.stack([biased_oracle(delta) for delta in deltas]),
                                   deltas @ dense.T, rtol=0, atol=0)
        with patch.object(precision, "native_margin_hvp", side_effect=biased_factory):
            result = precision.gradient_richardson_check(model, base, deltas, **kwargs)
        self.assertFalse(result.receipt["oracle_consistency_passed"])
        self.assertFalse(result.receipt["passed"])
        self.assertTrue(result.receipt["fd_parity_passed"])
        self.assertGreater(result.receipt["max_errors"]["fine_vs_fresh"], result.receipt["oracle_tolerance"])

    def test_fd_parity_can_fail_with_an_accurate_oracle(self):
        model, base, deltas, kwargs, _ = analytic_fixture()
        deltas[-1] *= 1.01
        result = precision.gradient_richardson_check(model, base, deltas, **kwargs)
        self.assertTrue(result.receipt["oracle_consistency_passed"])
        self.assertFalse(result.receipt["fd_parity_passed"])
        self.assertFalse(result.receipt["passed"])

    def test_all_raw_gradients_reconstruct_the_witness_in_numpy(self):
        model, base, deltas, kwargs, _ = analytic_fixture(degree=6)
        result = precision.gradient_richardson_check(model, base, deltas, **kwargs)
        arrays = {key: value.numpy() for key, value in result.raw.items()}
        gradients, steps = arrays["pair_gradients"], arrays["steps"]
        self.assertEqual(gradients.shape, (9, 3, 2, 96))
        central = (gradients[:, :, 0] - gradients[:, :, 1]) / (2 * steps[None, :, None])
        extrapolated = (4 * central[:, 1:] - central[:, :-1]) / 3
        values = arrays["norms"][:, None] ** 2 * np.einsum("nsi,ni->ns", extrapolated, arrays["unit_directions"])
        np.testing.assert_allclose(values[:, 0], arrays["q_richardson_coarse"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(values[:, 1], arrays["q_richardson_fine"], rtol=0, atol=1e-15)
        pair_base = base[0, [9, 10]].numpy().reshape(96)
        np.testing.assert_array_equal(arrays["probe_actual_deltas"],
                                      (pair_base + arrays["probe_edits"]) - pair_base)
        np.testing.assert_array_equal(arrays["fd_parity"], values[:, 1] - values[0, 1])

    def test_fresh_hvp_is_called_inside_helper_on_each_exact_delta(self):
        model, base, deltas, kwargs, _ = analytic_fixture()
        deltas[0, 59] = 2.0 ** -30
        factory = precision.native_margin_hvp
        seen = []

        def recording_factory(*args):
            self.assertEqual(args[1].dtype, torch.float64)
            real = factory(*args)

            def oracle(delta):
                seen.append(delta.clone())
                return real(delta)
            return oracle

        with patch.object(precision, "native_margin_hvp", side_effect=recording_factory) as called:
            result = precision.gradient_richardson_check(model, base, deltas, **kwargs)
        self.assertEqual(called.call_count, 1)
        self.assertEqual(len(seen), 9)
        torch.testing.assert_close(torch.stack(seen), deltas, rtol=0, atol=0)
        torch.testing.assert_close(result.raw["actual_edits64"], deltas, rtol=0, atol=0)

    def test_invalid_or_incomplete_inputs_fail_before_measurement(self):
        model, base, deltas, kwargs, _ = analytic_fixture()
        bad = deltas.clone(); bad[0] = 0.
        inputs = ((base.float(), deltas), (base, deltas.float()), (base[:, :80], deltas),
                  (base, deltas[:8]), (base, bad), (base * float("nan"), deltas))
        for b, d in inputs:
            with self.subTest(shape=tuple(d.shape), dtype=str(d.dtype)):
                with patch.object(precision, "native_margin_hvp") as called:
                    with self.assertRaises(ValueError):
                        precision.gradient_richardson_check(model, b, d, **kwargs)
                    called.assert_not_called()
        with self.assertRaises(ValueError):
            precision.gradient_richardson_check(model.train(), base, deltas, **kwargs)
        model.eval().float()
        with self.assertRaises(ValueError):
            precision.gradient_richardson_check(model, base, deltas, **kwargs)

    def test_exposed_native_61064_fixture_resolves_fresh_curvature(self):
        torch.manual_seed(61064)
        model32 = TinyRecursiveModel().float().eval().requires_grad_(False)
        model64 = copy.deepcopy(model32).double()
        inputs, _ = fixed_data(1, 61064)
        with torch.no_grad():
            h32 = model32.reason_states(inputs)[2]
        goal = 10
        _, metadata = legal_predecessor_donor(inputs[0], goal)
        generator = torch.Generator().manual_seed(61064)
        intended = torch.randn((9, 96), dtype=torch.float64, generator=generator)
        intended = intended / intended.norm(dim=1, keepdim=True) * 4.
        pair = h32[0, [goal - 1, goal]].reshape(96)
        physical = (pair + intended.float()).double() - pair.double()
        result = precision.gradient_richardson_check(
            model64, h32.double(), physical, goal=goal,
            donor_class=metadata["donor_goal_label"], original_class=metadata["original_goal_label"],
            target_curvature=0.)
        self.assertTrue(result.receipt["oracle_consistency_passed"], result.receipt["max_errors"])
        self.assertFalse(result.receipt["fd_parity_passed"])
        torch.testing.assert_close(result.raw["base_state64"], h32.double(), rtol=0, atol=0)
        self.assertEqual(result.receipt["oracle_tolerance"], 1e-9)


if __name__ == "__main__":
    unittest.main()
