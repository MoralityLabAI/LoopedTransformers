"""RAM-16 immutable first-run and corrected-artifact composition tests.

Synthetic original case receipts only. Never fetches private checkpoints
or touches public prospective MathCan theorem sources.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import compose_postscore_audit as merger


def put(folder:Path,theorem:str,policy:str,*,phase:str,grammar_flag:bool,
        invalid=False):
    folder.mkdir(parents=True)
    rec=dict(policy=policy,theorem=theorem,logical_evaluations=0 if invalid else 2,
        original_source_sha256="a"*64,
        elapsed_seconds=.1,
        accepted=not invalid,
        reason="unexpected_error" if invalid else "independently_certified",
        grammar_receipt=dict(ablation_original_20=grammar_flag))
    (folder/"case.json").write_text(json.dumps(rec))
    (folder/"source.log").write_text("original compiler log\n")


class PostscoreCompose(unittest.TestCase):
    def make_data(self,path):
        manifest=json.loads((HERE/"LOCKED_TEST_MANIFEST.json").read_text())
        original=path/"original";secondary=path/"secondary";ablation=path/"ablation"
        for idx in range(10):
            root=manifest["cases"][idx//2]["theorems"][idx%2]
            for phase,policies in merger.PHASES.items():
                for policy in policies:
                    put(original/phase/f"{idx:02d}"/policy,root,policy,
                        phase=phase,grammar_flag=False,invalid=phase=="secondary")
                    if phase=="secondary":
                        put(secondary/phase/f"{idx:02d}"/policy,root,policy,
                            phase=phase,grammar_flag=False)
                    if phase=="ablation":
                        put(ablation/phase/f"{idx:02d}"/policy,root,policy,
                            phase=phase,grammar_flag=True)
        return original,secondary,ablation

    def test_complete_immutable_result_collection(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,c=self.make_data(root);out=root/"composite"
            result=merger.compose(a,b,c,out,ablation_run_id=43807259096)
            self.assertEqual(result["original_policy_case_count"],140)
            self.assertEqual(len(result["original_byte_hashes"]),200)
            self.assertEqual(result["preserved_invalid_original_secondary_cases"],40)
            self.assertEqual(result["preserved_invalid_original_ablation_cases"],20)
            self.assertFalse(result["original_primary_modified"])
            for label in ("primary","secondary","ablation",
                          "invalid_original_secondary","invalid_original_ablation"):
                self.assertTrue((out/label).is_dir())
            self.assertEqual((out/"primary"/"00"/"native"/"case.json").read_bytes(),
                             (a/"primary"/"00"/"native"/"case.json").read_bytes())
            self.assertTrue(json.loads((out/"ablation"/"00"/"native"/"case.json").read_text())
                            ["grammar_receipt"]["ablation_original_20"])
    def test_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,c=self.make_data(root);out=root/"composite";out.mkdir()
            with self.assertRaises(FileExistsError):
                merger.compose(a,b,c,out,ablation_run_id=43807259096)
    def test_rejects_old_expanded_grammar_under_corrected_ablation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,c=self.make_data(root)
            src=c/"ablation"/"00"/"native"/"case.json";item=json.loads(src.read_text())
            item["grammar_receipt"]["ablation_original_20"]=False
            src.write_text(json.dumps(item))
            with self.assertRaises(AssertionError):
                merger.compose(a,b,c,root/"out",ablation_run_id=43807259096)
    def test_rejects_first_secondary_being_treated_as_proof_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,c=self.make_data(root)
            src=a/"secondary"/"00"/"native"/"case.json";item=json.loads(src.read_text())
            item["reason"]="evaluation_budget"
            src.write_text(json.dumps(item))
            with self.assertRaises(AssertionError):
                merger.compose(a,b,c,root/"out",ablation_run_id=43807259096)
    def test_bad_correction_run_id_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,c=self.make_data(root)
            with self.assertRaises(ValueError):
                merger.compose(a,b,c,root/"out",ablation_run_id=merger.OLD_RUN)


if __name__=="__main__":unittest.main()
