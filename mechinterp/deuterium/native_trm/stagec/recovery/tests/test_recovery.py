"""Prevent accidental alteration of original first-science source/auditor.

The corrected post-result audit differs by one target-row indexing typo, plus
an explanatory two-line preamble. Scoring code and prospective source lock remain
unaltered, so a new model inference would be a historical reproduction only.
"""
import hashlib
import unittest
from pathlib import Path
from mechinterp.deuterium.native_trm.stagec.recovery import replay_first_artifacts as recovery


class TestRecoveredFirstScienceOnly(unittest.TestCase):
    def test_original_bug_is_frozen_and_recovery_patch_exact(self):
        root=recovery.REPO
        original=(root/"mechinterp/deuterium/native_trm/stagec/audit_native.py").read_bytes()
        corrected=(root/"mechinterp/deuterium/native_trm/stagec/recovery/audit_recovered.py").read_text()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         "3cd934d079e9ed2af1765efb77f4783a6d9e6c48443c07989734744e0cad5c44")
        old=original.decode()
        orig_line="require(all(len(row)==81 and all(y[i][j]==x[i][(j-1)%81]"
        new_line="require(all(len(y[i])==81 and all(y[i][j]==x[i][(j-1)%81]"
        self.assertEqual(old.count(orig_line),1)
        self.assertEqual(corrected.count(new_line),1)
        self.assertTrue(corrected.endswith(old.replace(orig_line,new_line)))

    def test_exact_first_science_artifact_roster(self):
        self.assertEqual(recovery.SCORED_RUN_ID,44465490737)
        self.assertEqual(recovery.STAGEB_RUN_ID,44463652662)
        self.assertEqual(recovery.TEACHERS,(103,211,307))
        self.assertEqual(len(recovery.FIRST_LOCK_SHA),64)
        self.assertEqual(set(recovery.FIRST_FILES_SHA),{103,211,307})
        for files in recovery.FIRST_FILES_SHA.values():
            self.assertEqual(len(files),5)
            self.assertTrue(all(len(s)==64 for s in files.values()))


if __name__=="__main__":
    unittest.main()
