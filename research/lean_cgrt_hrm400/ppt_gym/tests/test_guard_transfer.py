import json
from pathlib import Path
import unittest

from configs.run_ppt_guard_probe_transfer import LOCK, stratified_interval
from configs.run_ppt_mathcan_progress import read_receipt
from ppt_gym.guard_transfer import expressions, public_digest, unseen_pairs


class GuardTransferTests(unittest.TestCase):
    def test_public_plan_is_locked_and_excludes_previous_closed_expressions(self):
        receipts = Path(__file__).resolve().parents[2] / "results" / "ppt_hybrid_rl"
        bank = read_receipt(receipts / "ppt-guard-recovery-bank-1103.json.gz")
        deep = read_receipt(receipts / "ppt-guard-deep-source-1126.json.gz")
        previous = bank["source"] + bank["development"] + deep["source"]
        plan, audit = unseen_pairs(previous)
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        self.assertEqual(public_digest(plan), lock["public_plan_sha256"])
        self.assertEqual(len(plan), 128)
        self.assertEqual(audit["roots"], {"BPD.geom": 32, "MathCan.maskAt": 32})
        self.assertFalse(expressions(plan) & expressions(previous))
        relabeled = [{**row, "false_guard": 99, "truth": []} for row in plan]
        self.assertEqual(public_digest(plan), public_digest(relabeled))

    def test_interval_resamples_paired_roots_within_family(self):
        rows = [{"pair": "a", "family": "BPD.geom"}, {"pair": "b", "family": "MathCan.maskAt"}] * 4
        contrast = {"effects": {"a": .2, "b": 0.}}
        self.assertEqual(stratified_interval(contrast, rows, draws=200), [.1, .1])
