"""Pure stdlib pre-registration tests; no historical native inference."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from mechinterp.neuron_communities.nec03v_activation_extract import (
    EVIDENCE, extract, positive_codes,
)
from mechinterp.neuron_communities.nec03v_fixture import make_fixture
from mechinterp.neuron_communities.nec03v_geometry import (
    canonical, file_hash, read_pack,
)


class SourceActivationRegistrationTests(unittest.TestCase):
    def test_positive_code_support_retains_zero_selected_entries_as_censored(self):
        raw=[0.0]*128
        raw[3]=0.4
        raw[9]=1e-7
        raw[17]=2e-7
        out=positive_codes(raw)
        self.assertEqual(out,[[3,0.4],[17,2e-7]])
        self.assertEqual(positive_codes([0.0]*128),[])

    def test_refuse_nonfinite_or_wider_than_topk(self):
        with self.assertRaisesRegex(ValueError,"128"):
            positive_codes([1.0]*129)
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            positive_codes([float("inf")]+[0.0]*127)
        with self.assertRaisesRegex(ValueError,"capacity"):
            positive_codes([1.0]*17+[0.0]*111)

    def test_decoder_tuple_list_normalization_and_physical_tamper(self):
        from mechinterp.neuron_communities.nec03v_independent_activation_audit import (
            same_projected_columns,
        )
        old=[[0.01*(feature+1)+0.001*i for i in range(32)]
             for feature in range(128)]
        imported=[tuple(row) for row in old]
        self.assertTrue(same_projected_columns(imported,old))
        tampered=[list(row) for row in old]
        tampered[17][4]+=0.001
        self.assertFalse(same_projected_columns(imported,tampered))
        self.assertFalse(same_projected_columns(imported[:127],old))

    def test_only_original_worlds_authorized_before_checkpoint_or_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            with self.assertRaises(PermissionError):
                extract(8,p,p,None,p,p,p/"never_written",
                        allow_registered_forwards=True)
            self.assertFalse((p/"never_written").exists())

    def _registered_simulated_pack(self,p):
        pack=p/"source"
        meta=make_fixture(pack, world_index=0)
        receipt={
            "schema":"NEC-03V/frozen-source-positive-topk-extraction/v1",
            "world_index":0,
            "source_checkpoint_receipts":meta["source_checkpoint_receipts"],
            "source_fit_context_group_indices":list(range(40)),
            "intervention_source_groups":list(range(24,32)),
            "source_case_records_per_arm":256,
            "new_optimizer_updates":0,
            "task_labels_accessed":0,
            "model_logits_stored":False,
            "recruited_or_reserved_queries":0,
            "embargoed_or_confirmation_queries":0,
            "behavioral_necessity_scoring":False,
            "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        }
        path=pack/"extraction_receipt.json"
        path.write_text(canonical(receipt)+"\n",encoding="utf-8")
        m=pack/"manifest.json"
        meta["evidence"]=EVIDENCE
        meta["extraction_receipt_sha256"]=file_hash(path)
        m.write_text(canonical(meta)+"\n",encoding="utf-8")
        return pack,receipt

    def test_extraction_receipt_custody_validator_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            pack,record=self._registered_simulated_pack(p)
            actual,shards,_sha=read_pack(pack)
            self.assertEqual(actual["evidence"],EVIDENCE)
            self.assertEqual(len(shards["composed"]["cases"]),256)
            source=pack/"extraction_receipt.json"
            record["task_labels_accessed"]=1
            source.write_text(canonical(record)+"\n",encoding="utf-8")
            manifest=pack/"manifest.json"
            meta=json.loads(manifest.read_text(encoding="utf-8"))
            meta["extraction_receipt_sha256"]=file_hash(source)
            manifest.write_text(canonical(meta)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"custody"):
                read_pack(pack)


if __name__=="__main__":
    unittest.main()
