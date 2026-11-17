"""ROC-05A standard-library source lock and independent origination checks."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from research_gym.analysis.roc02_delayed_rule_source import symbolic_execute
from research_gym.analysis.roc04_permutation_source import source_pools as previous_pools
from research_gym.analysis.roc05_permutation_source import (
    EXCLUDED_ORIGINAL, PRIOR_ROC04_TABLES, TRAIN_N, DEV_N,
    source_pools, build_train_and_dev, write_sources,
    swapped_row_counterfactual,nuisance_distractor_counterfactual,
)

class ROC05SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low,cls.high,cls.dev,cls.reserved=build_train_and_dev()

    def test_prospective_lock_and_prior_identity_exclusion(self):
        cfg=json.loads(Path("configs/roc05_fresh_permutation_replication_v1.json").read_text())
        self.assertEqual(cfg["model"]["steps"],1920)
        self.assertEqual(len(cfg["model"]["initialization_seeds"]),6)
        self.assertEqual(cfg["source"]["fresh_pool_counts"],{"high_train":512,"development":256,"reserved_future":384})
        prior={t for group in previous_pools() for t in group}
        self.assertEqual(prior,PRIOR_ROC04_TABLES)
        self.assertEqual(len(prior),1152)
        pools=source_pools()
        self.assertEqual(list(map(len,pools)),[512,256,384])
        self.assertFalse(set(sum((list(x) for x in pools),[])) & (prior|EXCLUDED_ORIGINAL))
        self.assertEqual(len(set(sum((list(x) for x in pools),[]))),1152)

    def test_all_exact_oracles_and_matched_train_events(self):
        self.assertEqual((len(self.low),len(self.high),len(self.dev)),(1024,1024,256))
        for a,b in zip(self.low,self.high):
            self.assertEqual(a["events"][1:],b["events"][1:])
            self.assertEqual(a["skeleton_id"],b["skeleton_id"])
        for r in self.low+self.high+self.dev:
            self.assertEqual(symbolic_execute(r["events"]),{"status":"ok","label":r["label"]})
        self.assertEqual(sorted(Counter(r["rule_table_id"] for r in self.high).values()),[2]*512)
        self.assertEqual(sorted(Counter(r["rule_table_id"] for r in self.low).values()),[256]*4)
        self.assertEqual(len({r["rule_table_id"] for r in self.dev}),256)

    def test_independent_table_id_holdout(self):
        assert len(self.reserved)==384
        sampled={tuple(r["events"][0]["table"]) for r in self.high+self.dev}
        self.assertFalse(sampled & set(self.reserved))
        self.assertFalse(sampled & PRIOR_ROC04_TABLES)
        self.assertFalse(sampled & EXCLUDED_ORIGINAL)

    def test_legal_source_edit_and_nuisance_placebo(self):
        eligible=0
        for r in self.dev:
            swap=swapped_row_counterfactual(r)
            self.assertNotEqual(swap["label"],r["label"])
            self.assertEqual(swap["events"][1:],r["events"][1:])
            self.assertEqual(symbolic_execute(swap["events"])["label"],swap["label"])
            nuisance=nuisance_distractor_counterfactual(r)
            if nuisance is not None:
                eligible+=1
                self.assertEqual(nuisance["label"],r["label"])
                self.assertEqual(nuisance["events"][0],r["events"][0])
                self.assertEqual(symbolic_execute(nuisance["events"])["label"],r["label"])
        self.assertGreater(eligible,0)

    def test_reproducible_locked_files_and_reserved_absence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            a=write_sources(p)
            self.assertEqual(set(q.name for q in p.iterdir()),{"train_low4.jsonl","train_high512.jsonl","development.jsonl","source_manifest.json"})
            self.assertFalse(a["old_roc02_sealed_families_read"])
            self.assertFalse(a["reserved_test_examples_generated"])
            self.assertFalse(a["roc05_reserved_test_examples_generated"])
            self.assertTrue(a["roc04_prior_1152_table_identities_excluded"])
            for name,meta in a["source_files"].items():
                raw=(p/name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),meta["sha256"])
                self.assertEqual(len(raw.splitlines()),meta["records"])
            b=write_sources(p)
            self.assertEqual(a,b)

if __name__=="__main__":
    unittest.main()
