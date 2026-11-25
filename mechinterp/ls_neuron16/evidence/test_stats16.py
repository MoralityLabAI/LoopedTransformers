"""Synthetic receipt checks; this module never generates experimental seeds."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import stats16


def fixture():
    return {"rows": [{"seed": s, "arm": a, "dose": 15,
                      "initial": {"verdict": "POINT_VIOLATION"},
                      "final": {"verdict": "POINT_VIOLATION"}}
                     for s in stats16.SEEDS for a in stats16.ARMS]}


class ReceiptValidation(unittest.TestCase):
    def test_coverage_and_duplicate_rejected(self):
        raw = fixture()
        self.assertEqual(len(stats16.validated_rows(raw)), 48)
        raw["rows"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            stats16.validated_rows(raw)
        raw = fixture()
        raw["rows"].append(copy.deepcopy(raw["rows"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            stats16.validated_rows(raw)

    def test_initial_verdict_disagreement_rejected(self):
        raw = fixture()
        raw["rows"][0]["initial"]["verdict"] = "UNRESOLVED"
        with self.assertRaisesRegex(ValueError, "Initial verdict mismatch"):
            stats16.validated_rows(raw)

    def test_invalid_dose_and_verdict_rejected(self):
        raw = fixture()
        raw["rows"][0]["dose"] = 5
        with self.assertRaises(ValueError):
            stats16.validated_rows(raw)
        raw = fixture()
        raw["rows"][0]["final"]["verdict"] = "APPROXIMATE"
        with self.assertRaises(ValueError):
            stats16.validated_rows(raw)

    def test_unresolved_and_acquisitions_keep_assigned_denominator(self):
        raw = fixture()
        keyed = stats16.validated_rows(raw)
        for a in stats16.ARMS:
            keyed[493, a]["initial"]["verdict"] = "UNRESOLVED"
            keyed[493, a]["final"]["verdict"] = "CERTIFIED"
            keyed[494, a]["initial"]["verdict"] = "CERTIFIED"
        keyed[495, "beam"]["final"]["verdict"] = "CERTIFIED"
        keyed[496, "greedy"]["final"]["verdict"] = "CERTIFIED"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "input.json").write_text(json.dumps(raw))
            result = stats16.run(p / "input.json", p / "out")
            self.assertEqual(result["n_units"], 24)
            self.assertEqual(result["n_final_model_endpoints"], 48)
            for row in result["arm_summaries"]:
                self.assertEqual(row["initial_unresolved"], 1)
                self.assertEqual(row["new_certificates"], 1)
                self.assertEqual(row["certificate_acquisitions"], 2)
                self.assertEqual(row["lost_certificates"], 1)
            report = result["comparisons"][0]
            self.assertEqual(report["candidate_only_seeds"], [495])
            self.assertEqual(report["baseline_only_seeds"], [496])
            self.assertEqual(report["candidate_correct"], 1)
            self.assertEqual(report["baseline_correct"], 1)
            self.assertFalse(report["operational_count_gate"])
            self.assertEqual(result["study_evidence_class"], "PILOT_ONLY")


if __name__ == "__main__":
    unittest.main()
