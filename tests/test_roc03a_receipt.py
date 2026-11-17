"""ROC-03A public single-seed feasibility receipt audit (stdlib-only)."""
import json
import math
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/roc03a_source_locked_training_v1.json"
RECEIPT=ROOT/"experiments/roc03a_task_competence/roc03a_R8_S2026111701_metrics.json"

class ROC03AReceipts(unittest.TestCase):
    def test_source_and_locked_budget(self):
        cfg=json.loads(CONFIG.read_text())
        r=json.loads(RECEIPT.read_text())
        self.assertEqual(cfg["pilot"]["updates"],1920)
        self.assertEqual(cfg["pilot"]["batch_size"],32)
        self.assertEqual(cfg["pilot"]["eval_steps"],[240,480,960,1920])
        self.assertEqual(r["eval_steps"],[240,480,960,1920])
        self.assertFalse(r["sealed_test_evaluated"])
        self.assertEqual(cfg["data"]["sealed_test_families"],[6,7])
        self.assertEqual(r["train_sha256"],"843de4a066b456129ff32df8906d076e051cd7352f7fd73a07dbb78beb0f4054")
        self.assertEqual(r["calibration_sha256"],"a9a8190aee2b4c2111afd8ceaf59422b55b483f7e54182795d986cc8739b818c")
        self.assertEqual(len(r["checkpoint_sha256"]),64)

    def test_milestone_accuracy_and_fixed_gate(self):
        cfg=json.loads(CONFIG.read_text())
        r=json.loads(RECEIPT.read_text())
        self.assertEqual(r["gate"],"FAIL")
        self.assertEqual(len(r["checkpoints"]),4)
        self.assertEqual([v["step"] for v in r["checkpoints"]],[240,480,960,1920])
        for v in r["checkpoints"]:
            self.assertEqual(v["train"]["cases"],1024)
            self.assertEqual(v["calibration"]["cases"],256)
            for k in ("train","calibration"):
                self.assertTrue(math.isfinite(v[k]["cross_entropy"]))
                self.assertTrue(0<=v[k]["accuracy"]<=1)
        final=r["checkpoints"][-1]
        self.assertEqual(final["train"]["accuracy"],362/1024)
        self.assertEqual(final["calibration"]["accuracy"],72/256)
        self.assertLess(final["train"]["accuracy"],0.90)
        self.assertLess(final["calibration"]["accuracy"],0.70)
        self.assertTrue(r["checkpoint_replayed_sha_and_metrics"])

if __name__=="__main__":unittest.main()
