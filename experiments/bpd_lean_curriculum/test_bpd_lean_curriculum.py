import json
import tempfile
import unittest
from pathlib import Path

from experiments.bpd_lean_curriculum.generate import (
    FAMILIES, DEPTHS, mk_case, roster, export, check_countermodel,
)
from experiments.bpd_lean_curriculum.bridge import plan


class BpdLeanCurriculum(unittest.TestCase):
    def test_roster_balanced_and_proof_root_disjoint(self):
        rows = roster()
        self.assertEqual(len(rows), 384)
        self.assertEqual(sum(r["critical"] for r in rows), 192)
        self.assertEqual(sum(not r["critical"] for r in rows), 192)
        self.assertEqual({r["family"] for r in rows}, set(FAMILIES))
        self.assertEqual({r["structural_depth"] for r in rows}, set(DEPTHS))
        self.assertEqual(len({r["case_id"] for r in rows}), len(rows))
        roots = {s: {r["root_id"] for r in rows if r["split"] == s}
                 for s in ("train", "dev", "test")}
        self.assertTrue(all(roots[a].isdisjoint(roots[b])
                            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))))
        self.assertTrue(all(check_countermodel(row) for row in rows))

    def test_matched_minimal_pairs_and_no_admitted_proofs(self):
        groups = {}
        for row in roster():
            groups.setdefault((row["root_id"], row["nuisance"]), []).append(row)
            self.assertNotIn("sorry", row["lean_theorem"])
            self.assertNotIn("admit", row["lean_theorem"])
            self.assertEqual(row["gold_proof"] is not None, row["critical"])
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual({r["critical"] for r in pair}, {True, False})

    def test_induction_counterexample(self):
        for d in DEPTHS:
            case = mk_case("induction", d, 0, False, True)
            self.assertEqual(case["countermodel"]["actual_at_zero"], d)
            self.assertTrue(check_countermodel(case))

    def test_export_is_reproducible(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            a = export(Path(left))
            b = export(Path(right))
            self.assertEqual(a, b)
            self.assertEqual(a["split_counts"], {"train": 256, "dev": 64, "test": 64})
            self.assertEqual(a["positive_templates"], 192)
            self.assertTrue(a["sha256"])
            for split in ("train", "dev", "test"):
                for r in (json.loads(s) for s in (Path(left)/f"{split}.jsonl").read_text().splitlines()):
                    self.assertEqual(r["split"], split)

    def test_unknown_is_not_false_for_incidence(self):
        data = {"split": "train", "quotas": {"exact_local": 1, "rw_local": 1},
                "states": [{"id": "a", "allowed": ["exact_local"]},
                           {"id": "b", "allowed": []}]}
        got = plan(data, cap=4)
        self.assertEqual(got["known_positive_edges"], 1)
        self.assertEqual(got["unknown_action_family_cells"], 3)
        self.assertEqual(got["verified_negative_cells"], 0)
        self.assertEqual(got["data_split"], "train")
        self.assertGreaterEqual(got["hall_lower_bound_current_certified_edges"]["minimum_unfilled_slots"], 1)
        self.assertTrue(all(p["status"] == "unknown_requires_original_lean_compile"
                            for p in got["candidate_probes"]))
        with self.assertRaises(ValueError):
            plan({**data, "split": "test"})


if __name__ == "__main__":
    unittest.main()
