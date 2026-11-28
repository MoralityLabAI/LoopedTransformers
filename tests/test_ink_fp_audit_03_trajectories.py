import json
import unittest
from pathlib import Path

from research_gym.ink_fp_audit_03 import trajectories as T

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_gym/ink_fp_audit_03/receipts/trajectories_20261128"


class TrajectoryTests(unittest.TestCase):
    def test_provenance_required(self):
        v = {f: "x" for f in T.FIELDS}
        with self.assertRaises(ValueError):
            T.make("c", 0, v, {f: "guess" for f in T.FIELDS}, {}, "negative")
        with self.assertRaises(ValueError):
            T.make("c", 0, {k: v[k] for k in T.FIELDS[:-1]}, {f: "curated" for f in T.FIELDS}, {}, "negative")
        t = T.make("c", 0, v, {f: "curated" for f in T.FIELDS}, {}, "negative")
        self.assertEqual(t["schema"], T.SCHEMA)

    def test_next_action_rows_hide_results(self):
        t = T.audit02_effort(ROOT)[0]
        row = T.next_action_row(t)
        blob = json.dumps(row["input"]) + json.dumps(row["target"])
        self.assertNotIn("rescues_vs_low", blob)
        self.assertNotIn(t["decision"], blob)
        self.assertEqual(set(row["input"]) - {"exercise"}, set(T.PREFIX))

    def test_audit02_numbers_mechanical(self):
        t = T.audit02_effort(ROOT)[0]
        self.assertTrue(t["field_sources"]["actual_result"].startswith("mechanical"))
        self.assertEqual(t["actual_result"]["high"]["correct"], 84)
        self.assertEqual(t["actual_result"]["truncated_failures"], 28)
        self.assertEqual(t["field_sources"]["interpretation"], "curated")

    def test_committed_build(self):
        if not (OUT / "manifest.json").exists():
            self.skipTest("trajectories not built")
        m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["trajectories"], 1502)
        n = 0
        for line in (OUT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            n += 1
            self.assertEqual(set(t["field_sources"]), set(T.FIELDS))
            for f, src in t["field_sources"].items():
                self.assertIn(src.split(":")[0], ("mechanical", "rule", "transcribed", "curated"))
        self.assertEqual(n, 1502)


if __name__ == "__main__":
    unittest.main()
