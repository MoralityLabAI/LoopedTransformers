"""LS-SAE07 preregistered source-only cohort admission tests."""
from __future__ import annotations
import unittest

from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_hard_task_v1 import admission as ADMISSION
from mechinterp.loopscope02.scripts import sudoku_prop as PROP
from mechinterp.sae_causal_transfer.native_trm_v1 import families as BASE


class SourceAdmissionTests(unittest.TestCase):
    def test_hard_source_validity_and_original_sudoku_propagation_parity(self):
        rows=FAM.roster("pilot_52")
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(row["holes"],52)
            self.assertEqual(row["x"].count(0),52)
            self.assertEqual(BASE.count_solutions(row["x"],2),1)
            self.assertGreater(BASE.rectangles(row["target"]),0)
            q="".join("." if n==0 else str(n) for n in row["x"])
            truth=PROP.trace_propagation(q)
            visible=row["visible_structural_difficulty"]
            self.assertEqual(visible["propagation_solved"],truth.solved)
            self.assertEqual(visible["propagation_rounds"],truth.num_rounds)
            self.assertEqual(visible["deduced_missing_cells"],
                             sum(row["x"][i]==0 and truth.cell_round[i]>=1 for i in range(81)))
            self.assertEqual(visible["first_round_naked_singles"],
                             sum(truth.naked_round1))
    def test_source_frozen_failed_hole_count_remains_failed_and_unreplaced(self):
        receipt=ADMISSION.evaluate()
        self.assertFalse(receipt["full_scientific_cohort_executable"])
        self.assertEqual(receipt["preregistration_commit"],ADMISSION.PROTOCOL)
        self.assertEqual(receipt["per_family"]["test_52"]["admitted_slots"],8)
        self.assertEqual(receipt["per_family"]["pilot_52"]["admitted_slots"],2)
        self.assertEqual(receipt["per_family"]["pilot_54"]["admitted_slots"],2)
        failed=receipt["per_family"]["test_54"]
        self.assertEqual(failed["requested_slots"],8)
        self.assertEqual(failed["admitted_slots"],7)
        self.assertEqual(len(failed["refused_slots"]),1)
        self.assertEqual(failed["refused_slots"][0]["grid_seed"],202627040006)
        self.assertIn("reached=53",failed["refused_slots"][0]["reason"])
    def test_out_of_scope_hole_counts_and_registered_family_rejected(self):
        with self.assertRaises(ValueError):
            FAM.roster("test_55")
        with self.assertRaises(ValueError):
            FAM.dig_hard(BASE.full_solution(202627099999),202628099999,51)


if __name__=="__main__":
    unittest.main()
