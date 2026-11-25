"""Corrupt old-seed receipts to test branch isolation and audit rejection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import audit17 as audited

ROOT = Path(__file__).resolve().parent / "development" / "469"


class BranchAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = json.loads((ROOT / "initial" / "469.json").read_text())
        cls.common = json.loads((ROOT / "common" / "469.json").read_text())
        cls.raw = json.loads((ROOT / "results" / "multistart-469.json").read_text())
        cls.common_sha = audited.base.sha(ROOT / "common" / "469.json")

    def replay(self, value):
        return audited.check_multistart(self.initial, self.common, value, self.common_sha)

    def test_valid_receipt_preserves_three_independent_paths(self):
        checked = self.replay(self.raw)
        self.assertEqual(checked["ls_calls"], 223)
        self.assertEqual(checked["lp_calls"], 222)
        self.assertEqual(len(checked["branch_diagnostics"]), 3)
        self.assertEqual(sum(b["worsening_after_bootstrap"] for b in checked["branch_diagnostics"]), 0)

    def test_bootstrap_branch_order_is_fixed(self):
        bad = copy.deepcopy(self.raw)
        bad["initial_branch_ids"][0], bad["initial_branch_ids"][1] = bad["initial_branch_ids"][1], bad["initial_branch_ids"][0]
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_cross_branch_parent_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["rounds"][0]["expansions"][0]["parent_id"] = bad["initial_branch_ids"][1]
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_cross_branch_selected_child_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["rounds"][0]["expansions"][0]["selected_after"] = [bad["initial_branch_ids"][1]]
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_beam_style_branch_reallocation_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        retained = bad["rounds"][0]["retained"]
        retained[0], retained[1] = retained[1], retained[0]
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_wrong_round_best_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        row = bad["rounds"][0]
        row["best"] = next(value for value in row["retained"] if value != row["best"])
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_wrong_final_branch_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["winning_branch"] = (bad["winning_branch"] + 1) % 3
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_common_work_is_excluded_from_new_compute(self):
        bad = copy.deepcopy(self.raw)
        bad["new_compute_budget"]["ls_calls"] = 223
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_candidate_score_corruption_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["rounds"][0]["expansions"][0]["anchors"][0]["fits"][0]["first"]["scores"]["bound"] += .001
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_sparse_dual_corruption_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["rounds"][0]["expansions"][0]["anchors"][0]["fits"][0]["secondary"]["lower_duals"][0] += 1
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_short_round_budget_is_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["rounds"].pop()
        with self.assertRaises(AssertionError):
            self.replay(bad)

    def test_stable_final_tie_uses_earliest_branch(self):
        def node(identity, code):
            return {"id": identity, "codes": [code] * 52, "scores": {"bound": .2}}
        branches = [node("branch0", 0), node("branch1", 1), node("branch2", 2)]
        self.assertEqual(audited.base.keep(branches, 1)[0]["id"], "branch0")
        branches[2]["scores"]["bound"] -= 1e-13
        self.assertEqual(audited.base.keep(branches, 1)[0]["id"], "branch2")

    def test_missing_or_extra_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("initial", "common", "beam", "results"):
                (root / name).mkdir()
            for folder, name in (("initial", "469.json"), ("common", "469.json"), ("beam", "beam-469.json")):
                (root / folder / name).touch()
            with self.assertRaises(AssertionError):
                audited.check_coverage(root, (469,))
            (root / "results" / "multistart-469.json").touch()
            audited.check_coverage(root, (469,))
            (root / "results" / "multistart-470.json").touch()
            with self.assertRaises(AssertionError):
                audited.check_coverage(root, (469,))

    def test_all_parent_receipt_hashes_are_checked(self):
        manifest = audited.check_parent_manifest(ROOT, (469,))
        self.assertEqual(len(manifest["files"]), 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupted = copy.deepcopy(manifest)
            corrupted["files"]["baseline_certificates.json"]["sha256"] = "0" * 64
            for relative in manifest["files"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / relative).read_bytes())
            (root / "parent_manifest.json").write_text(json.dumps(corrupted))
            with self.assertRaises(AssertionError):
                audited.check_parent_manifest(root, (469,))


if __name__ == "__main__":
    unittest.main()
