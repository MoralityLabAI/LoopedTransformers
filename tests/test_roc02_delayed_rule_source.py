"""Independent exact oracle and split-seal checks for ROC-02 prospective source."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from research_gym.analysis.roc02_delayed_rule_source import (
    DOMAIN, FAMILIES, FEATURE_DIM, SPLITS, build_split, canonical_hash,
    encode_event, invalid_fixture_suite, rule_table, symbolic_execute,
    write_dataset)
except ImportError:
    from roc02_delayed_rule_source import (
        DOMAIN, FAMILIES, FEATURE_DIM, SPLITS, build_split, canonical_hash,
        encode_event, invalid_fixture_suite, rule_table, symbolic_execute,
        write_dataset)


class ROC02ExactGymTests(unittest.TestCase):
    def test_registered_families_are_distinct_permutations(self):
        tables = [tuple(rule_table(i)) for i in range(8)]
        self.assertEqual(len(set(tables)),8)
        self.assertEqual(FEATURE_DIM,79)
        for table in tables:
            self.assertEqual(sorted(table),list(range(DOMAIN)))
        with self.assertRaises(ValueError):
            rule_table(8)

    def test_binding_and_distractor_overwrite(self):
        events = [
            {"op":"bind","src":"A","table":rule_table(0)},
            {"op":"distractor","src":"B","table":rule_table(2)},
            {"op":"distractor","src":"B","table":rule_table(3)},
            {"op":"query","src":"A","value":4},
        ]
        self.assertEqual(symbolic_execute(events),{"status":"ok","label":5})
        events[-1] = {"op":"query","src":"B","value":4}
        self.assertEqual(symbolic_execute(events),{"status":"ok","label":rule_table(3)[4]})

    def test_neural_input_encoding_exact_width(self):
        events=[{"op":"bind","src":"A","table":rule_table(4)},
                {"op":"carry"},
                {"op":"query","src":"A","value":5}]
        for ev in events:
            x=encode_event(ev)
            self.assertEqual(len(x),79)
            self.assertEqual(x[:4].count(1.0),1)
        self.assertEqual(sum(encode_event(events[0])),10.)
        self.assertEqual(sum(encode_event(events[-1])),3.)
        with self.assertRaises(ValueError):
            encode_event({"op":"query","src":"A","value":8})

    def test_frozen_split_counts_exact_symbolic(self):
        seen = set()
        families_by_split=[]
        for name, cfg in SPLITS.items():
            cases=build_split(name)
            self.assertEqual(len(cases),cfg["n"])
            ids={case["id"] for case in cases}
            self.assertEqual(len(ids),len(cases))
            self.assertFalse(ids & seen)
            seen |= ids
            families_by_split.append({c["family"] for c in cases})
            for case in cases:
                self.assertTrue(2<=case["delay"]<=16)
                self.assertEqual(canonical_hash(case),case["id"])
                self.assertEqual(symbolic_execute(case["events"])["label"],case["label"])
                self.assertEqual(case["events"][-1]["op"],"query")
        for i in range(len(families_by_split)):
            for j in range(i):
                self.assertFalse(families_by_split[i]&families_by_split[j])

    def test_invalid_no_rule_and_out_of_domain(self):
        invalid=invalid_fixture_suite()
        self.assertEqual(len(invalid),24)
        for row in invalid:
            self.assertEqual(symbolic_execute(row["events"])["status"],row["expected"])

    def test_sealed_public_inputs_exclude_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_dataset(Path(tmp))
            self.assertEqual(manifest["splits"]["train"]["n"],1024)
            self.assertEqual(manifest["splits"]["calibration"]["n"],256)
            self.assertEqual(manifest["splits"]["sealed_test"]["n"],384)
            for name,meta in manifest["splits"].items():
                data=(Path(tmp)/meta["file_name"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),meta["file_sha256"])
                first=json.loads(data.splitlines()[0])
                self.assertEqual("label" in first,name!="sealed_test")


if __name__ == "__main__":
    unittest.main()
