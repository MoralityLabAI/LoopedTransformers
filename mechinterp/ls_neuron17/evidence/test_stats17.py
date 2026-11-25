import copy
import unittest
from stats17 import validate_rows, repaired, SEEDS, ARMS


def fixture():
    return {"rows": [{"seed": s, "arm": a, "dose": 15, "initial": {"verdict": "POINT_VIOLATION"},
                       "final": {"verdict": "CERTIFIED"}} for s in SEEDS for a in ARMS]}


class EvidenceTests(unittest.TestCase):
    def test_complete_fixed_cohort(self):
        self.assertEqual(len(validate_rows(fixture())), 48)

    def test_missing_and_duplicate_are_rejected(self):
        raw = fixture(); raw["rows"].pop()
        with self.assertRaises(ValueError): validate_rows(raw)
        raw = fixture(); raw["rows"].append(copy.deepcopy(raw["rows"][0]))
        with self.assertRaises(ValueError): validate_rows(raw)

    def test_initial_disagreement_is_rejected(self):
        raw = fixture(); raw["rows"][0]["initial"]["verdict"] = "UNRESOLVED"
        with self.assertRaises(ValueError): validate_rows(raw)

    def test_unresolved_initial_is_no_proven_repair(self):
        raw = fixture()["rows"][0]; raw["initial"]["verdict"] = "UNRESOLVED"
        self.assertEqual(repaired(raw), 0)


if __name__ == "__main__":
    unittest.main()
