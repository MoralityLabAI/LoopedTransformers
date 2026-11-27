"""Roster audit against a synthetic spent inventory."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
import olr08_spent_cohorts as S  # noqa: E402


class RosterAuditTests(unittest.TestCase):
    def test_overlap_reserved_and_novelty(self):
        tmp = Path(tempfile.mkdtemp())
        inv = {"spent_files": {"src/Init/Data/Nat/Basic.lean": ["olr07a_roots:train"]},
               "spent_families": {"Nat": ["train"]}, "reserved_family": "SInt",
               "spent_goal_sha256": [hashlib.sha256(b"g0").hexdigest()],
               "spent_decls": [["src/Init/Data/Nat/Gcd.lean", "Nat.d"]]}
        roster = {"train": ["src/Init/Data/Nat/Basic.lean", "src/Init/Data/Nat/Gcd.lean"],
                  "calibration": ["src/Init/Data/SInt/Lemmas.lean"],
                  "evaluation": ["src/Init/Data/BitVec/Lemmas.lean"],
                  "roots": [{"id": "a", "file": "src/Init/Data/Nat/Gcd.lean", "decl": "Nat.d", "goal": "g1"},
                            {"id": "b", "file": "src/Init/Data/BitVec/Lemmas.lean", "decl": "B.x", "goal": "g0"}]}
        (tmp / "inv.json").write_text(json.dumps(inv))
        (tmp / "roster.json").write_text(json.dumps(roster))
        S.audit(types.SimpleNamespace(inventory=tmp / "inv.json", roster=tmp / "roster.json", out=tmp / "out.json"))
        out = json.loads((tmp / "out.json").read_text())
        self.assertEqual(list(out["roles"]["train"]["spent_overlap"]), ["src/Init/Data/Nat/Basic.lean"])
        self.assertEqual(out["roles"]["train"]["new_file_seen_family"], ["src/Init/Data/Nat/Gcd.lean"])
        self.assertEqual(out["roles"]["calibration"]["reserved_access"], ["src/Init/Data/SInt/Lemmas.lean"])
        self.assertEqual(out["roles"]["evaluation"]["new_family"], ["src/Init/Data/BitVec/Lemmas.lean"])
        self.assertEqual(out["root_collisions"], {"roots": 2, "exact_goal": ["b"], "file_decl": ["a"]})

    def test_family(self):
        self.assertEqual(S.family("src/Init/Data/List/Nat/Basic.lean"), "List")
        self.assertEqual(S.family("src/Init/Data/SInt.lean"), "SInt")


if __name__ == "__main__":
    unittest.main()
