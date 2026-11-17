"""Pure-stdlib ROC-04A source lock, exact oracle and paired case tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from research_gym.analysis.roc02_delayed_rule_source import rule_table, symbolic_execute
from research_gym.analysis.roc04_permutation_source import (
    DEV_N, LOW_POOL, TRAIN_N, TRAIN_TABLE_N, EXCLUDED_ORIGINAL,
    source_pools, source_schedules, build_train_and_dev, source_manifest,
    write_sources, swapped_row_counterfactual,
    nuisance_distractor_counterfactual,
)


class ROC04SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low, cls.high, cls.dev, cls.reserved = build_train_and_dev()

    def test_preregistration_source_contract(self):
        cfg = json.loads(Path("configs/roc04_permutation_diversity_transfer_v1.json").read_text())
        self.assertEqual(cfg["source"]["per_arm_train_examples"], TRAIN_N)
        self.assertEqual(cfg["source"]["dev_table_pool"], DEV_N)
        self.assertEqual(cfg["source"]["high_train_table_pool"], TRAIN_TABLE_N)
        self.assertEqual(cfg["model"]["steps"], 1920)
        self.assertEqual(cfg["model"]["seeds"],
                         [2026111705, 2026111707, 2026111711,
                          2026111713, 2026111717, 2026111719])
        self.assertEqual(cfg["source"]["original_roc02_sealed_families"], [6, 7])

    def test_source_pools_distinct_and_original_eight_excluded(self):
        tr, dv, reserved = source_pools()
        self.assertEqual((len(tr), len(dv), len(reserved)), (512, 256, 384))
        self.assertFalse(set(tr) & set(dv))
        self.assertFalse(set(tr) & set(reserved))
        self.assertFalse(set(dv) & set(reserved))
        self.assertEqual(len(EXCLUDED_ORIGINAL), 8)
        self.assertFalse(set(tr + dv + reserved) & EXCLUDED_ORIGINAL)

    def test_train_same_skeleton_and_different_table(self):
        self.assertEqual((len(self.low), len(self.high), len(self.dev)),
                         (1024, 1024, 256))
        for a,b in zip(self.low,self.high):
            self.assertEqual(a["source_index"], b["source_index"])
            self.assertEqual(a["events"][1:], b["events"][1:])
            self.assertEqual(a["skeleton_id"], b["skeleton_id"])
            self.assertEqual(a["delay"], b["delay"])
            self.assertEqual(a["events"][0]["src"], "A")
            self.assertEqual(b["events"][0]["src"], "A")
            self.assertEqual(symbolic_execute(a["events"])["label"], a["label"])
            self.assertEqual(symbolic_execute(b["events"])["label"], b["label"])
        self.assertEqual(sorted(Counter(x["rule_table_id"] for x in self.low).values()),
                         [256]*4)
        self.assertEqual(sorted(Counter(x["rule_table_id"] for x in self.high).values()),
                         [2]*512)
        self.assertEqual(len({x["rule_table_id"] for x in self.dev}),256)
        self.assertFalse({x["rule_table_id"] for x in self.high} &
                         {x["rule_table_id"] for x in self.dev})

    def test_dev_legal_pair_and_nuisance_oracle(self):
        eligible = 0
        for original in self.dev:
            altered = swapped_row_counterfactual(original)
            self.assertEqual(len(altered["events"]),len(original["events"]))
            self.assertEqual(altered["events"][1:],original["events"][1:])
            self.assertNotEqual(altered["label"],original["label"])
            self.assertEqual(symbolic_execute(altered["events"])["label"],altered["label"])
            nuisance=nuisance_distractor_counterfactual(original)
            if nuisance is not None:
                eligible+=1
                self.assertEqual(nuisance["label"],original["label"])
                self.assertEqual(nuisance["events"][0],original["events"][0])
                self.assertEqual(nuisance["events"][-1],original["events"][-1])
                self.assertEqual(symbolic_execute(nuisance["events"])["label"],
                                 original["label"])
        self.assertGreater(eligible,0)
        self.assertLess(eligible,len(self.dev))

    def test_deterministic_write_and_reserved_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            m=write_sources(root)
            self.assertFalse(m["reserved_test_examples_generated"])
            self.assertFalse(m["old_roc02_sealed_families_read"])
            self.assertEqual(sorted(x.name for x in root.iterdir()),
                             ["development.jsonl","source_manifest.json",
                              "train_high512.jsonl","train_low4.jsonl"])
            for name,entry in m["source_files"].items():
                raw=(root/name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),entry["sha256"])
                self.assertEqual(len(raw.splitlines()),entry["records"])
            before=m["source_files"]["train_high512.jsonl"]["sha256"]
            second=write_sources(root)
            self.assertEqual(second["source_files"]["train_high512.jsonl"]["sha256"],
                             before)
            self.assertEqual(m,second)


if __name__ == "__main__":
    unittest.main()
