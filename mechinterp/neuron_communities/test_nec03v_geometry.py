"""NEC-03V geometry gate: stdlib tests, no models or protected cohorts."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from mechinterp.neuron_communities.nec03v_geometry import (
    ACTIVE, DIM, original_lock_digest, read_pack, choose_groups,
    preflight, canonical, file_hash,
)
from mechinterp.neuron_communities.nec03v_independent_geometry_audit import audit
from mechinterp.neuron_communities.nec03v_fixture import make_fixture


class GeometryPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.pack = cls.root / "source_geometry"
        make_fixture(cls.pack,world_index=0)
        cls.result = preflight(cls.pack)
        cls.output = cls.root / "preflight.json"
        cls.output.write_text(canonical(cls.result)+"\n",encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_original_source_custody_and_no_outcomes(self):
        report = self.result
        self.assertEqual(report["source_scope"],list(range(24,32)))
        self.assertEqual(report["world_id"],"dev5-000")
        self.assertEqual(report["model_forwards"],0)
        self.assertEqual(report["labels_accessed"],0)
        self.assertEqual(report["logits_accessed"],0)
        self.assertEqual(report["new_recruited_root_reads"],0)
        self.assertEqual(report["scientific_gate"],"REDESIGN_OR_PILOT_ONLY")
        self.assertEqual(report["evidence_kind"],"synthetic_geometry_fixture")
        self.assertFalse(report["registered_primary_scoring_enabled"])
        self.assertIsNone(report["behavioral_effect"])
        self.assertEqual(len(report["case_receipts"]),512)
        self.assertEqual({r["arm"] for r in report["case_receipts"]},
                         {"composed","additive"})

    def test_32d_dose_matching_and_invalid_case_zeroing(self):
        good=0
        invalid=0
        for case in self.result["case_receipts"]:
            self.assertEqual(set(case["visits"]),{"3","7"})
            if case["eligible_all_visits"]:
                good+=1
                for visit in case["visits"].values():
                    self.assertFalse(visit["effective_zero_pair"])
                    self.assertEqual(len(visit["source"]),4)
                    self.assertEqual(len(visit["sham"]),4)
                    self.assertFalse(set(visit["source"])&set(visit["sham"]))
                    self.assertLessEqual(visit["relative_match_error"],0.01)
                    self.assertLessEqual(visit["source_gain"],1)
                    self.assertLessEqual(visit["sham_gain"],1)
                    self.assertGreaterEqual(visit["shared_dose"],1e-4)
                    self.assertEqual(len(visit["effective_source32"]),32)
            else:
                invalid+=1
                for visit in case["visits"].values():
                    self.assertTrue(visit["effective_zero_pair"])
                    self.assertEqual(visit["effective_source32"],[0.0]*DIM)
                    self.assertEqual(visit["effective_sham32"],[0.0]*DIM)
        self.assertGreater(good,0)
        self.assertGreater(invalid,0)
        self.assertEqual(good+invalid,512)

    def test_separately_implemented_geometry_audit(self):
        proof = audit(self.pack,self.output)
        self.assertTrue(proof["geometry_audit_passed"])
        self.assertFalse(proof["external_source_provenance_authenticated"])
        self.assertEqual(proof["case_arm_records"],512)
        self.assertEqual(proof["new_model_forwards"],0)

    def test_tamper_candidate_pair_or_vector_fails_independent_audit(self):
        record=copy.deepcopy(self.result)
        good=next(r for r in record["case_receipts"] if r["eligible_all_visits"])
        first=next(v for v in good["visits"].values() if v["ok"])
        first["scaled_source32"][0]+=0.1
        corrupt=self.root/"corrupt_output.json"
        corrupt.write_text(canonical(record)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(AssertionError,"incorrect physical geometry"):
            audit(self.pack,corrupt)

    def test_changed_source_manifest_and_outcome_columns_are_rejected(self):
        clone=self.root/"modified_pack"
        clone.mkdir()
        for file in self.pack.iterdir():
            (clone/file.name).write_bytes(file.read_bytes())
        path=clone/"composed.json"
        original=json.loads(path.read_text())
        original["cases"][0]["truth_label"]=1
        path.write_text(canonical(original)+"\n",encoding="utf-8")
        manifest=clone/"manifest.json"
        meta=json.loads(manifest.read_text())
        meta["arm_shards"]["composed"]["sha256"]=file_hash(path)
        manifest.write_text(canonical(meta)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"expected fields"):
            read_pack(clone)

    def test_protected_world_or_changed_source_lock_rejected(self):
        clone=self.root/"protected_pack"
        clone.mkdir()
        for file in self.pack.iterdir():
            (clone/file.name).write_bytes(file.read_bytes())
        manifest=clone/"manifest.json"
        data=json.loads(manifest.read_text())
        data["world_index"]=8
        data["world_id"]="dev5-008"
        manifest.write_text(canonical(data)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"historical eight"):
            read_pack(clone)
        data["world_index"]=0
        data["world_id"]="dev5-000"
        data["original_lock_sha256"]="0"*64
        manifest.write_text(canonical(data)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"lock checksum"):
            read_pack(clone)


if __name__=="__main__":
    unittest.main()
