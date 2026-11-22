"""NEC04G synthetic signed teacher-phase and independent FWHT tests.

Uses only artificial 32-mask logits and labels. No NEC04F source, trained
model, archived numerical result, GPU, optimizer, SAE or protected cohort.
"""
from __future__ import annotations

from math import isclose
import unittest

from mechinterp.neuron_communities.nec04g_signed_phase import (
    analyze_context,analyze_validation,walsh,summarize,SCHEMA,
    SOURCE,VALIDATION,
)
from mechinterp.neuron_communities.nec04g_independent import (
    hadamard,context as independent_context,
)


def group(phase=0,coefficient=-2.0,aux=0.0):
    rows=[]
    scores=[]
    for mask in range(32):
        truth=(mask.bit_count()%2)^phase
        rows.append({"mux_selector":1,"focal_mask":mask,
                     "composed_label":truth,"xor_prefix":{"5":mask.bit_count()%2}})
        five=-1.0 if mask.bit_count()%2 else 1.0
        one=-1.0 if mask&1 else 1.0
        margin=coefficient*five+aux*one
        scores.append([0.0,margin])
    return rows,scores


class OriginalSignedParityPhaseTests(unittest.TestCase):
    def test_teacher_phase_zero_has_negative_fifth_coefficient(self):
        r,x=group(phase=0,coefficient=-2)
        a=analyze_context(r,x)
        self.assertEqual(a["original_correct_out_of_32"],32)
        self.assertEqual(a["teacher_phase_status"],"teacher_aligned")
        self.assertEqual(a["fifth_order_unsigned_power_share"],1.)
        self.assertAlmostEqual(a["teacher_signed_normalized_fifth_coefficient"],1.)
        self.assertEqual(a["posthoc_teacher_phase_oracle_correct_out_of_32"],32)

    def test_teacher_phase_one_has_positive_fifth_coefficient(self):
        r,x=group(phase=1,coefficient=2)
        a=analyze_context(r,x)
        self.assertEqual(a["context_phase_from_focal_mask_zero"],1)
        self.assertEqual(a["teacher_phase_status"],"teacher_aligned")
        self.assertEqual(a["original_correct_out_of_32"],32)

    def test_wrong_phase_has_full_unsigned_power_and_zero_accuracy(self):
        r,x=group(phase=0,coefficient=2)
        a=analyze_context(r,x)
        self.assertEqual(a["fifth_order_unsigned_power_share"],1.)
        self.assertEqual(a["teacher_phase_status"],"teacher_inverted")
        self.assertEqual(a["original_correct_out_of_32"],0)
        self.assertEqual(a["posthoc_teacher_phase_oracle_correct_out_of_32"],32)
        self.assertEqual(a["teacher_signed_normalized_fifth_coefficient"],-1.)

    def test_orthogonal_lower_order_signal_has_separate_energy(self):
        rows,scores=group(phase=0,coefficient=-2.,aux=1.5)
        actual=analyze_context(rows,scores)
        self.assertAlmostEqual(actual["fifth_order_unsigned_power_share"],4./6.25)
        self.assertAlmostEqual(actual["teacher_signed_normalized_fifth_coefficient"],.8)
        self.assertEqual(actual["original_correct_out_of_32"],32)

    def test_independent_fwht_matches_direct_walsh_all_32_coefficients(self):
        margins=[(mask-7)*.125 + ((mask*13)%17)*.04
                 for mask in range(32)]
        a=walsh(margins)
        b=hadamard(margins)
        self.assertEqual(len(a),32)
        for x,y in zip(a,b):
            self.assertTrue(isclose(x,y,rel_tol=0,abs_tol=1e-13))
        rows,logits=group(phase=1,coefficient=-2.,aux=.5)
        first=analyze_context(rows,logits)
        second=independent_context(rows,logits)
        self.assertTrue(isclose(first["teacher_signed_fifth_order_walsh_coefficient"],
                                second["signed"],abs_tol=1e-13))
        self.assertEqual(first["original_correct_out_of_32"],second["original_correct"])
        self.assertEqual(first["posthoc_teacher_phase_oracle_correct_out_of_32"],
                         second["oracle_correct"])

    def test_original_context_phase_corruption_fails_closed(self):
        rows,logits=group()
        rows[10]["composed_label"]^=1
        with self.assertRaisesRegex(AssertionError,"not raw XOR"):
            analyze_context(rows,logits)
        rows,logits=group()
        rows[1]["focal_mask"]=9
        with self.assertRaisesRegex(AssertionError,"mask ordering"):
            analyze_context(rows,logits)

    def test_zero_signal_does_not_invent_signed_phase(self):
        rows,logits=group(coefficient=0.,aux=0.)
        got=analyze_context(rows,logits)
        self.assertEqual(got["teacher_phase_status"],"near_zero")
        self.assertIsNone(got["fifth_order_unsigned_power_share"])
        self.assertIsNone(got["teacher_signed_normalized_fifth_coefficient"])

    def test_owned_validation_subset_keeps_32_mask_group_denominator(self):
        context_rows,context_logits=group(phase=1,coefficient=2)
        alt=[dict(r,**{"mux_selector":0}) for r in context_rows]
        original_rows=context_rows+alt*23
        original_logits=[[0.,0.] for _ in range(SOURCE)]+context_logits+[[0.,0.] for _ in range(23*32)]
        self.assertEqual(len(original_rows),VALIDATION)
        r=analyze_validation(original_rows,original_logits)
        self.assertEqual(r["original_parity_context_count"],1)
        self.assertEqual(r["original_parity_case_denominator"],32)
        self.assertEqual(r["original_parity_correct"],32)
        self.assertEqual(r["phase_aligned_contexts"],1)
        self.assertEqual(r["phase_inverted_contexts"],0)

    def test_oracle_delta_is_posthoc_and_keeps_full_case_count(self):
        cases=[analyze_context(*group(phase=0,coefficient=2)),
               analyze_context(*group(phase=0,coefficient=-2))]
        r=summarize(cases)
        self.assertEqual(r["original_parity_case_denominator"],64)
        self.assertEqual(r["original_parity_correct"],32)
        self.assertEqual(r["posthoc_teacher_phase_oracle_correct"],64)
        self.assertEqual(r["phase_inverted_contexts"],1)
        self.assertAlmostEqual(r["posthoc_oracle_minus_native_accuracy"],.5)


if __name__=="__main__":
    unittest.main()
