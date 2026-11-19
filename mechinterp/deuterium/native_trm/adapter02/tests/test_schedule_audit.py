"""Fail-closed checks for the source-only measurement auditor."""
import json
import tempfile
import unittest
from pathlib import Path

from mechinterp.deuterium.native_trm.adapter02.audit_schedule import (
    SCHEMA,UPSTREAM_SHA,TOL,audit,classify,
)


class TestScheduleAudit(unittest.TestCase):
    def setUp(self):
        self.errors = dict.fromkeys(("warmup_parameters","warmup_gradients","main_parameters",
                                    "main_gradients","terminal_gradients","losses"),0.)

    def test_crossing_fixed_tolerance_remains_blocked(self):
        self.assertEqual(classify(self.errors),"SOURCE_ONLY_FULL_SCHEDULE_PASS")
        self.errors["main_gradients"] = TOL*2
        self.assertEqual(classify(self.errors),"BLOCKED_UNMATCHED_OPTIMIZER")

    def test_missing_or_nonfinite_error_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,"family"):
            classify({"main_parameters":0.})
        self.errors["main_parameters"] = float("nan")
        with self.assertRaisesRegex(ValueError,"invalid"):
            classify(self.errors)

    def test_declared_scientific_exposure_rejected_before_trace_read(self):
        record = {"schema":SCHEMA,"source_seed":61077,"source_n":512,
            "trained_teacher_used":False,"scientific_evaluation":False,
            "reserved_teacher_roots_opened":[930101],"reserved_data_splits_opened":[],
            "upstream_commit":UPSTREAM_SHA,"tolerance":TOL}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/"schedule_receipt.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"root exposure"):
                audit(path)


if __name__ == "__main__":
    unittest.main()
