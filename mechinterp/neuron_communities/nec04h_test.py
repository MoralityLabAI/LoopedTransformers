from itertools import product
from unittest import TestCase

import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04f_native import AuxLooper
from mechinterp.neuron_communities.nec04h_frozen import binary_log_odds, compose, verify_frozen, verify_logits
from mechinterp.neuron_communities.nec04h_oracles import fifth_walsh, shuffled_phases, verify_labels, verify_shuffle
from research.pef07.pef07 import model_digest


class RoutingTests(TestCase):
    def test_logit_agreement_rejects_changed_predictions_even_with_small_errors(self):
        original = torch.tensor([[0., 0.00001], [-1., 1.]])
        self.assertEqual(verify_logits(original + 0.00001, original)["changed_predictions"], 0)
        changed = original.clone()
        changed[0, 1] *= -1
        with self.assertRaises(ValueError):
            verify_logits(changed, original)

    def test_exhaustive_boolean_xor_and_mux(self):
        values = torch.tensor(list(product(range(2), repeat=4)))
        raw, phase, selector, native = values.unbind(1)
        hard = [F.one_hot(x, 2).double().log() for x in (raw, phase, selector, native)]
        result, parity = compose(*hard)
        torch.testing.assert_close(parity.argmax(1), raw ^ phase)
        expected = torch.where(selector.bool(), raw ^ phase, native)
        torch.testing.assert_close(result.argmax(1), expected)
        self.assertTrue((expected != torch.where(selector.bool(), native, raw ^ phase)).any())

    def test_extreme_log_odds_have_finite_composed_loss_and_gradient(self):
        phase = torch.tensor([1000., -1000.], requires_grad=True)
        xor = binary_log_odds(torch.tensor([1000., 1000.]))
        selector = binary_log_odds(torch.tensor([1000., -1000.]))
        native = binary_log_odds(torch.tensor([-1000., 1000.]))
        result, _ = compose(xor, binary_log_odds(phase), selector, native)
        self.assertTrue(result.isfinite().all())
        (-result[:, 0].mean()).backward()
        self.assertTrue(phase.grad.isfinite().all())

    def test_scalar_phase_and_wrong_context_index(self):
        focal = [0, 1, 2, 3, 4]
        parity = ["XOR", ["VAR", 6], ["VAR", 7]]
        for i in focal:
            parity = ["XOR", ["VAR", i], parity]
        world = {"context_positions": list(range(5, 12)), "focal_positions": focal,
                 "canonical_program": ["MUX", ["VAR", 5], parity, ["VAR", 8]]}
        for bits in product(range(2), repeat=12):
            expectation = (bits[5], bits[6] ^ bits[7])
            row = {"history_bits12": bits, "xor_prefix": {"5": sum(bits[:5]) % 2},
                   "mux_selector": bits[5], "composed_label": int(oracle(world["canonical_program"], bits))}
            self.assertEqual(verify_labels(world, row), expectation)
        incorrect = {**world, "context_positions": [6, 5, 7, 8, 9, 10, 11]}
        with self.assertRaises(ValueError):
            verify_labels(incorrect, row)

    def test_shuffle_rejects_leaked_and_count_changing_labels(self):
        groups = {i: (i % 2, (i // 2) % 2) for i in range(80)}
        tagged = [(i, *groups[i]) for i in groups for _ in range(32)]
        assignments = shuffled_phases(12, groups)
        self.assertEqual(assignments, shuffled_phases(12, dict(reversed(list(groups.items())))))
        allowed = [assignments[i] for i, _, _ in tagged]
        verify_shuffle(tagged, allowed)
        self.assertTrue(any(assignments[i] != groups[i][1] for i in groups))
        leaked = allowed.copy()
        leaked[0] ^= 1
        with self.assertRaises(ValueError):
            verify_shuffle(tagged, leaked)
        with self.assertRaises(ValueError):
            verify_shuffle(tagged, [0] * len(tagged))

    def test_frozen_guard_detects_gradients_and_altered_weights(self):
        model = AuxLooper().requires_grad_(False)
        expected = model_digest(model)
        head = nn.Linear(32, 1)
        head(model.core.memory(torch.zeros(2, 12, dtype=torch.long))).sum().backward()
        verify_frozen(model, expected)
        with torch.no_grad():
            model.aux_head.bias.add_(1)
        with self.assertRaises(ValueError):
            verify_frozen(model, expected)
        model.aux_head.requires_grad_(True)
        with self.assertRaises(ValueError):
            verify_frozen(model, model_digest(model))

    def test_walsh_alignment_uses_true_phase(self):
        margins = [2. * (2 * (i.bit_count() % 2) - 1) for i in range(32)]
        self.assertEqual(fifth_walsh(margins, 0), {"signed": 2., "normalized": 1.})
        self.assertEqual(fifth_walsh(margins, 1), {"signed": -2., "normalized": -1.})
