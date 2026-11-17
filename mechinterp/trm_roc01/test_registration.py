"""ROC-TRM-01 preregistration checks. Pure stdlib, zero model compute.

Run: python -m unittest discover -s mechinterp/trm_roc01 -p 'test_registration.py' -v
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
PROTOCOL = (ROOT / "PROTOCOL.md").read_text(encoding="utf-8")


def b_block_coord_norm(T: list[list[int]], observed: tuple[int, ...]) -> float:
    """Frobenius norm of U^T(T-I)Q for *coordinate* observers only."""
    d = len(T)
    assert all(len(row) == d for row in T)
    unseen = [j for j in range(d) if j not in observed]
    return math.sqrt(sum(float(T[i][j]) ** 2 for i in observed for j in unseen))


def make_rotations(n: int) -> list[list[int]]:
    """Exact integer matrix T=1 direct-sum n copies of R_90."""
    d = 1 + 2 * n
    T = [[0 for _ in range(d)] for _ in range(d)]
    T[0][0] = 1
    for a in range(n):
        i = 1 + 2 * a
        T[i][i + 1] = -1
        T[i + 1][i] = 1
    return T


class TestRegistration(unittest.TestCase):
    def test_metadata_and_pins(self):
        self.assertEqual(CONFIG["study_id"], "ROC-TRM-01")
        self.assertEqual(CONFIG["registered_utc_date"], "2026-11-17")
        self.assertEqual(CONFIG["status"], "frozen_source_registration_no_new_model_data")
        for key in ("mathcan_roc01", "evalmath_roc01"):
            val = CONFIG["source_pins"][key]
            self.assertEqual(len(val), 40)
            int(val, 16)
        for ck in CONFIG["pretrained_checkpoints"].values():
            self.assertEqual(len(ck["sha256"]), 64)
            int(ck["sha256"], 16)
        for x in ("No claim", "before any new ROC-TRM-01 puzzle generation"):
            self.assertIn(x, PROTOCOL)

    def test_full_state_and_native_readout(self):
        a = CONFIG["source_derived_architecture"]
        self.assertEqual(a["full_state_dimensions"],
                         a["state_buffers"] * (a["board_cells"] + a["prefix_tokens"]) * a["state_width"])
        self.assertEqual(a["full_state_dimensions"], 99328)
        self.assertEqual(a["native_digit_logits_upper_rank"],
                         a["board_cells"] * a["native_digit_rows"])
        self.assertEqual(a["full_vocab_logits_upper_rank"],
                         a["board_cells"] * a["head_vocab_rows"])
        self.assertEqual(a["core_calls_per_outer_step"],
                         a["outer_h_cycles"] * (a["l_updates_per_h_cycle"] + 1))
        self.assertEqual(a["native_digit_rows"], 9)

    def test_closure_clock_and_observer_definitions(self):
        m = CONFIG["measurement"]
        self.assertEqual(m["clock"], "complete_outer_ACT_step_including_both_buffers")
        self.assertEqual(m["transition_horizons"], [1, 2, 4])
        self.assertEqual(m["native_full_state_rank_grid"]["base"],
                         "measured_native_full_board_readout_rank")
        self.assertIn(7, m["native_full_state_rank_grid"]["additional_dimensions"])
        self.assertIn(9, m["local_cell_diagnostic_rank_grid"])
        self.assertTrue(m["local_cell_diagnostic_only"])
        self.assertIn("G_current P = G_current", m["row_space_condition"])

    def test_null_and_physical_arms(self):
        m = CONFIG["measurement"]
        self.assertEqual(m["rank_null_seed_last"] - m["rank_null_seed_first"] + 1, 128)
        self.assertGreaterEqual(m["rank_null_distinct_min"], 100)
        self.assertEqual(len(m["one_sided_tests"]), 6)
        self.assertEqual(len(set(m["one_sided_tests"])), 6)
        self.assertEqual(m["holm_alpha"], 0.05)
        self.assertEqual(m["lambda_reference"], 1.0)
        self.assertIn("observed_to_observed", m["intervention_arms"])
        self.assertNotIn("freeze_U", m["intervention_arms"])
        self.assertIn("ablate_readout", m["intervention_arms"])
        self.assertEqual(m["finite_perturbation_l2_primary"], 0.2)
        self.assertIn("physical_l2_matched_random", m["required_controls"])

    def test_bpd_null_is_distinct_from_roc_null(self):
        b = CONFIG["bpd_gate_prior_to_any_real_TRM"]
        self.assertIn("permute_component_activations", b["null_type"])
        self.assertEqual(b["null_seeds_last"] - b["null_seeds_first"] + 1, 199)
        self.assertEqual(b["planted_active_set"], "2_of_8")
        self.assertEqual(len(b["density_grid"]) * len(b["scale_grid"]), 12)
        self.assertTrue(b["mean_ablation_control_required"])
        self.assertGreaterEqual(b["gate_precision_min"], 0.8)
        self.assertGreaterEqual(b["gate_recall_min"], 0.8)
        self.assertEqual(b["if_fail"], "stop_before_pretrained_TRM")
        self.assertIn("different test", PROTOCOL)

    def test_toy_folds_and_budget(self):
        t = CONFIG["toy"]
        self.assertEqual(t["box_shape"][0] * t["box_shape"][1], t["board_shape"][0])
        self.assertEqual(t["board_shape"][0] * t["board_shape"][1], t["tokens"])
        self.assertEqual(t["model_seeds"], sorted(set(t["model_seeds"])))
        self.assertEqual(len(t["model_seeds"]), 3)
        self.assertLessEqual(t["max_batch_size"], 8)
        self.assertGreater(t["target_heldout_count"], t["minimum_nontrivial_heldout"])
        self.assertNotEqual(t["generator_seed"], t["split_seed"])
        self.assertIn("hash_disjoint", t["board_family_split"])

    def test_real_pilot_hash_selection_and_memory(self):
        r = CONFIG["real_pilot"]
        self.assertEqual(r["per_checkpoint_WB_Prop"], 8)
        self.assertEqual(r["per_checkpoint_WB_Stall"], 8)
        self.assertEqual(r["batch_size"], 1)
        self.assertIn("sha256", r["cohort_selection"])
        self.assertFalse(r["checkpoint_retraining"])
        h = CONFIG["hardware"]
        self.assertLessEqual(h["process_rss_stop_gib"], 6.0)
        self.assertLessEqual(h["process_commit_stop_gib"], 8.0)
        self.assertFalse(h["dense_full_model_jacobian"])

    def test_rotation_exact_rank_parity_witnesses(self):
        # Coordinate witnesses for MathCan's all-subspaces R90 proof.
        t3 = make_rotations(1)
        self.assertEqual(b_block_coord_norm(t3, (0,)), 0.0)
        self.assertEqual(b_block_coord_norm(t3, (0, 1)), 1.0)
        self.assertEqual(b_block_coord_norm(t3, (0, 1, 2)), 0.0)
        t5 = make_rotations(2)
        self.assertEqual(b_block_coord_norm(t5, (0, 1)), 1.0)
        self.assertEqual(b_block_coord_norm(t5, (0, 1, 2)), 0.0)
        self.assertEqual(b_block_coord_norm(t5, (0, 1, 2, 3)), 1.0)
        # These tests verify explicit exact-rational/integer witnesses.
        # The general lower-bound theorem is established in pinned MathCan.

    def test_bpd_selection_ids_and_explicit_abstention(self):
        b = CONFIG["bpd_gate_prior_to_any_real_TRM"]
        ids = [f"{b['toy_name']}|{i}".encode("utf-8")
               for i in range(b["null_seeds_first"], b["null_seeds_last"] + 1)]
        self.assertEqual(len({hashlib.sha256(s).hexdigest() for s in ids}), 199)
        self.assertTrue(CONFIG["evidence"]["null_undefined_means_unclassified"])
        self.assertTrue(CONFIG["evidence"]["local_cell_results_are_not_global_closure"])
        self.assertTrue(CONFIG["evidence"]["partial_spectrum_is_not_certified_global_Lstar"])


if __name__ == "__main__":
    unittest.main()
