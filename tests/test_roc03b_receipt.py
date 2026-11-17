"""ROC-03B fixed-budget negative public-receipt audit; stdlib only."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/roc03b_query_conditioned_table_memory_v1.json"
RPT=ROOT/"experiments/roc03b_query_table_memory/roc03b_R8_S2026111703_metrics.json"

class ROC03BReceipt(unittest.TestCase):
    def test_protocol(self):
        c=json.loads(CFG.read_text());r=json.loads(RPT.read_text())
        self.assertEqual(c["pilot"]["seed"],2026111703)
        self.assertEqual(c["pilot"]["updates"],1920)
        self.assertEqual(c["pilot"]["batch_size"],32)
        self.assertEqual(c["pilot"]["eval_steps"],[240,480,960,1920])
        self.assertEqual(c["source"]["sealed_families"],[6,7])
        self.assertFalse(r["sealed_test_evaluated"])
        self.assertTrue(r["initialization_equal"])
        self.assertEqual(r["parameter_count"],51520)
        self.assertEqual(r["train_sha256"],c["source"]["train_hash"])
        self.assertEqual(r["calibration_sha256"],c["source"]["calibration_hash"])
    def test_eight_milestone_rows(self):
        r=json.loads(RPT.read_text())
        self.assertEqual(len(r["checkpoints"]),8)
        self.assertEqual({v["step"] for v in r["checkpoints"]},{240,480,960,1920})
        self.assertEqual({v["arm"] for v in r["checkpoints"]},{"task_only","table_supervised"})
        self.assertEqual(r["gate"],"FAIL")
        for arm,cal_count in (("task_only",78),("table_supervised",74)):
            row=next(v for v in r["checkpoints"] if v["step"]==1920 and v["arm"]==arm)
            self.assertEqual(row["train"]["accuracy"],1)
            self.assertEqual(row["calibration"]["accuracy"],cal_count/256)
            self.assertEqual(row["calibration"]["whole_table_exact_accuracy"],0)
            self.assertEqual(len(r["saved_models"][arm]["checkpoint_sha256"]),64)
        self.assertEqual(r["posthoc_calibration_table_probe"]["table_supervised_match_any_training_rule"],244)
        self.assertEqual(r["posthoc_calibration_table_probe"]["task_only_match_any_training_rule"],216)
        self.assertTrue(r["posthoc_calibration_table_probe"]["not_registered_primary"])

if __name__=="__main__":unittest.main()
