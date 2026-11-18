"""Independent auditor tests: exact sources and adversarial receipt tampering."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from experiments.nanbeige.nan_comp04 import cohort
from experiments.nanbeige.nan_comp04.audit_source import audit


def _write(path:Path, rows:list[dict]):
    path.write_text("".join(cohort.canonical(x)+"\n" for x in rows),encoding="utf-8")


def _rehash_receipt(root:Path,file:str):
    receipt_file=root/"source_receipt.json"
    r=json.loads(receipt_file.read_text(encoding="utf-8"))
    r["files_sha256"][file]=sha256((root/file).read_bytes()).hexdigest()
    receipt_file.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n",encoding="utf-8")


class TestIndependentSourceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=cohort.build()

    def _saved(self,root:Path)->Path:
        out=root/"development"
        cohort.save(out,self.data)
        return out

    def test_complete_source_only_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self._saved(Path(tmp))
            r=audit(out)
            self.assertEqual(r["result"],"PASS")
            self.assertEqual((r["roots"],r["public_rows"],r["host_only_rows"]),(20,220,220))
            self.assertEqual(r["tasks_by_tier"]["classification:first"],80)
            self.assertEqual(r["tasks_by_tier"]["coefficient:query_cross"],20)
            self.assertFalse(r["sealed_ARG03_test_opened"])
            self.assertTrue(r["no_model_calls"])

    def test_host_only_label_tamper_is_detected_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self._saved(Path(tmp))
            path=out/"host_only.jsonl"
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            i=next(i for i,x in enumerate(rows)
                   if x["kind"]=="coefficient" and x["oracle_answer"]!=0)
            rows[i]["oracle_answer"]+=1
            _write(path,rows)
            _rehash_receipt(out,"host_only.jsonl")
            with self.assertRaises(AssertionError):
                audit(out)

    def test_early_reciprocal_public_tamper_is_detected_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=self._saved(Path(tmp))
            path=out/"public.jsonl"
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            i=next(i for i,x in enumerate(rows)
                   if x["tier"]=="early" and x["kind"]=="classification")
            y=rows[i]["demos"][0]["output"]
            y[0]+=1
            rows[i]["public_sha256"]=cohort.fingerprint({k:v for k,v in rows[i].items()
                                                         if k not in ("case_id","public_sha256")})
            _write(path,rows)
            _rehash_receipt(out,"public.jsonl")
            with self.assertRaises(AssertionError):
                audit(out)

    def test_protected_test_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            audit(Path("/tmp/heldout/nanbeige"))
        with self.assertRaises(ValueError):
            audit(Path("/tmp/test/nanbeige"))


if __name__=="__main__":
    unittest.main()
