"""Pure-stdlib NEC-03U archival margin verification and regression tests."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from mechinterp.neuron_communities.nec03u_archive_margin import (
    ARMS,BINS,MODES,PANEL,ROOTS,ITT_ZERO,ELIGIBLE,EPS,
    canonical_jsonl,calc_margin,predict,margin_bin,stats,
    verify_dose_receipt,verify_lock,
)


class TestNEC03UArchive(unittest.TestCase):
    def test_source_only_lock_and_fixed_parent(self):
        cfg,digest=verify_lock()
        self.assertEqual(len(digest),64)
        self.assertEqual(cfg["original_nec03t_run"],44913605626)
        self.assertEqual(cfg["original_world_root_indices"],list(range(8)))
        self.assertEqual(cfg["original_source_group_indices"],list(range(24,32)))
        self.assertEqual(cfg["paired_model_arms"],["composed","additive"])
        self.assertEqual(cfg["original_fixed_zero_roots"],[3,5])
        self.assertEqual(cfg["original_accuracy_primary_effect"],1/2048)
        self.assertEqual(cfg["original_source_group_indices"],list(PANEL))
        self.assertFalse(cfg["posthoc_margin_pvalue_computed"])
        self.assertEqual(cfg["margin_bins"],list(BINS))
        self.assertEqual(cfg["correctness_tie_policy"],
                         "argmax first class wins: predict 1 only if logit1>logit0")
        self.assertTrue(all(cfg[k]==0 for k in (
            "native_or_SAE_training_updates","model_forwards","development_eval_scored",
            "reserved_scored","new_recruit_worlds_scored","five_input_embargoed_scored",
            "four_input_confirmation_scored")))

    def test_predictions_ties_and_teacher_signed_margins(self):
        self.assertEqual(calc_margin([3.,5.],1),2.)
        self.assertEqual(calc_margin([3.,5.],0),-2.)
        self.assertEqual(calc_margin([3.,3.],0),0.)
        self.assertEqual(predict([1.,1.]),0)
        self.assertEqual(predict([3.,4.]),1)
        for invalid in ([float("nan"),1.],[1.,float("inf")],[1.],[]):
            with self.assertRaises(AssertionError):
                calc_margin(invalid,1)
        with self.assertRaises(AssertionError):
            calc_margin([0.,1.],2)

    def test_margin_bins_are_fixed_nonoverlap(self):
        values=[-1.,-0.001,0.,1.999,2.,4.999,5.,9.999,10.,11.]
        expected=["negative","negative","zero_to_2","zero_to_2",
                  "2_to_5","2_to_5","5_to_10","5_to_10","10_or_more","10_or_more"]
        self.assertEqual([margin_bin(v) for v in values],expected)

    def test_distribution_uses_a_full_original_denominator(self):
        s=stats([-2.,0.,0.,1.,3.])
        self.assertEqual(s["count"],5)
        self.assertEqual((s["positive"],s["negative"],s["zero"]),(2,1,2))
        self.assertAlmostEqual(s["mean"],.4)
        self.assertEqual(s["median"],0.)
        self.assertEqual(stats([])["count"],0)
        self.assertIsNone(stats([])["median"])

    def test_canonical_case_bytes_have_stable_hash(self):
        rows=[{"a":3,"b":[0,1]},{"c":True,"d":"source"}]
        first=canonical_jsonl(rows)
        second=canonical_jsonl(rows)
        self.assertEqual(first,second)
        self.assertEqual(first.count(b"\n"),2)
        self.assertEqual(len(hashlib.sha256(first).hexdigest()),64)

    def test_reject_physical_dose_amplification(self):
        group={
            "k":1,
            "active_ids":[list(range(16))],
            "source_ids":[[0]],"angular_sham_ids":[[1]],
            "hash_sham_ids":[[2]],"exact_positive16_codes":[True],
            "candidate_count":[15],"angle_gap":[.02],
            "angular_removal_dose":{
                "raw_dose":{"source":[2.],"sham":[4.]},
                "shared_dose":[2.],
                "gains":{"source":[1.],"sham":[.5]},
                "physical_dose_equal_within_1pct":[True],
                "nonzero_shared_dose":[True],
            },
            "hash_removal_dose":{
                "raw_dose":{"source":[2.],"sham":[6.]},
                "shared_dose":[2.],
            },
        }
        # The real source data always contain both original and wide
        # budgets; synthetic test k=1 shares their same fields.
        site={"budget":{"orig":group,"wide":dict(group)}}
        verify_dose_receipt(site,0,valid={"orig":True,"wide":True})
        group["angular_removal_dose"]["gains"]["source"]=[1.4]
        with self.assertRaises(AssertionError):
            verify_dose_receipt(site,0,valid={"orig":True,"wide":True})


if __name__=="__main__":
    unittest.main()
