"""Synthetic matrix tests only; no TEST theorem name or goal is accessed."""
from __future__ import annotations
import copy,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analysis as a

def example():
    conf=a.CONFIG
    manifest=dict(test_source_commit=conf["test_source_commit"],
                  selected_before_goal_state_access=True,
                  cases=[dict(module=m,git_blob_sha1=b,theorems=[f"synthetic_{i}",f"synthetic_{i+5}"])
                         for i,(m,b) in enumerate(conf["source_modules"]["test"].items())])
    roots=a._locked_roots(manifest,conf)
    rows=[dict(phase="primary",module=mod,theorem=thm,policy=p,
               accepted=False,logical_evaluations=12,elapsed_seconds=8.0)
          for mod,thm in sorted(roots) for p in conf["primary"]["policies"]]
    return manifest,rows

def certify(row):
    row.update(accepted=True,first_source_sha256="a"*64,
               second_source_sha256="a"*64,third_source_sha256="a"*64,
               allowed_axioms=[],independent_third_valid=True)

class MatrixTests(unittest.TestCase):
    def test_recurrent_specific_two_seed_witness(self):
        manifest,rows=example()
        target=(rows[0]["module"],rows[0]["theorem"])
        for row in rows:
            if (row["module"],row["theorem"])==target and row["policy"] in (
                "loop64_structural_s17","loop64_structural_s29"):
                certify(row)
        result=a.analyse(rows,manifest)
        self.assertTrue(result["registered_recurrent_specific_criterion_passed"])
        self.assertEqual(result["cases"],160)
        self.assertEqual(result["per_seed_paired_coverage"]["17"]["loop64"]["difference"],1)
        self.assertEqual(result["per_seed_paired_coverage"]["43"]["loop64"]["difference"],0)
    def test_single_seed_is_not_registered_witness(self):
        manifest,rows=example()
        certify(next(r for r in rows if r["policy"]=="loop64_structural_s17"))
        result=a.analyse(rows,manifest)
        self.assertFalse(result["registered_recurrent_specific_criterion_passed"])
    def test_duplicate_case_or_missing_policy_fails(self):
        manifest,rows=example()
        with self.assertRaises(ValueError):a.analyse(rows+[rows[0]],manifest)
        with self.assertRaises(ValueError):a.analyse(rows[:-1],manifest)
    def test_false_certificate_rejected(self):
        manifest,rows=example()
        rows[0]["accepted"]=True
        with self.assertRaises(ValueError):a.analyse(rows,manifest)
        certify(rows[0]);rows[0]["third_source_sha256"]="b"*64
        with self.assertRaises(ValueError):a.analyse(rows,manifest)
    def test_wrong_budget_and_phase_rejected(self):
        manifest,rows=example()
        rows[0]["logical_evaluations"]=49
        with self.assertRaises(ValueError):a.analyse(rows,manifest)
        rows[0]["logical_evaluations"]=48
        rows[0]["phase"]="secondary"
        with self.assertRaises(ValueError):a.analyse(rows,manifest)
    def test_symbolic_control_blocks_claim(self):
        manifest,rows=example()
        target=(rows[0]["module"],rows[0]["theorem"])
        for row in rows:
            if (row["module"],row["theorem"])==target and row["policy"] in (
                "loop64_structural_s17","loop64_structural_s29","native"):
                certify(row)
        self.assertFalse(a.analyse(rows,manifest)["registered_recurrent_specific_criterion_passed"])

if __name__=="__main__":unittest.main()
