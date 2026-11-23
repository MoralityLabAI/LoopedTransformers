from unittest import TestCase

from configs.freeze_ppt_residual_transfer import bpd_shaped, graph_hash
from configs.run_ppt_residual_transfer import compare


class ResidualTransferTests(TestCase):
    def test_horn_fingerprint_preserves_roles_across_aliases_and_rule_order(self):
        left = "∀ (A B C : Prop) (h : A) (d : A → B) (r : B → C), C"
        right = "∀ (Y X Z : Prop) (s : Z → Y) (q : X → Z) (t : X), Y"
        swapped = "∀ (Y X Z : Prop) (s : Z → Y) (q : X → Z) (t : X), Z"
        self.assertEqual(graph_hash(left), graph_hash(right))
        self.assertNotEqual(graph_hash(left), graph_hash(swapped))

    def test_equality_fingerprint_treats_edge_orientation_as_presentation(self):
        left = "∀ (u v w : Nat) (a : u = v) (b : v = w), u = w"
        right = "∀ (z y x : Nat) (q : z = y) (r : y = x), x = z"
        self.assertEqual(graph_hash(left), graph_hash(right))
        original = bpd_shaped(1211, "fresh_bpd_equality")
        self.assertEqual(original, bpd_shaped(1211, "fresh_bpd_equality"))
        self.assertNotEqual(original.root, bpd_shaped(1212, "fresh_bpd_equality").root)

    def test_contrasts_average_budgets_within_the_registered_root(self):
        candidate = [{"module": "a", "case": "a0", "budget": 1, "utility": 0., "proof": False},
                     {"module": "a", "case": "a0", "budget": 2, "utility": .9, "proof": True},
                     {"module": "b", "case": "b0", "budget": 1, "utility": .8, "proof": True}]
        control = [{**row, "utility": 0., "proof": False} for row in candidate]
        result = compare(candidate, control)
        self.assertAlmostEqual(result["mean"], (.45 + .8) / 2)
        self.assertEqual(result["proof_rate_difference"], .75)
        with self.assertRaisesRegex(ValueError, "paired scenarios"):
            compare(candidate, list(reversed(control)))
