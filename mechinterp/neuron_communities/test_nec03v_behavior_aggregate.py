"""Synthetic-only eight-root aggregation regression tests for NEC03V source pilot.

These fabricate correctness totals and independent-audit receipts to test
the protocol arithmetic. They make NO claims about genuine model behavior.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mechinterp.neuron_communities.nec03v_behavior_aggregate import aggregate
from mechinterp.neuron_communities.nec03v_source_attest import canonical,sha256_file


class EightRootSourceAggregateTests(unittest.TestCase):
    def make_roots(self,root:Path):
        lock=json.loads((Path(__file__).with_name(
            "NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json")).read_text("utf-8"))
        folders=[]
        for i in range(8):
            folder=root/f"root{i}"
            folder.mkdir()
            raw=folder/"source_behavior_raw.jsonl"
            raw.write_text("{}\n",encoding="utf-8")
            result={
                "schema":"NEC-03V/source-only-group-removal-behavior/v1",
                "world_index":i,"world_id":f"dev5-{i:03d}",
                "source_cases_per_arm":256,
                "original_source_groups":list(range(24,32)),
                "source_pack_manifest_sha256":lock["roots"][str(i)]["source_pack_manifest_sha256"],
                "frozen_source_group_roster_sha256":lock["roots"][str(i)]["frozen_group_selection_sha256"],
                "raw_behavior_sha256":sha256_file(raw),
                "source_only_native_forward_calls":112,
                "recruited_or_embargoed_world_queries":0,
                "development_evaluation_or_reserved_queries":0,
                "old_confirmation_queries":0,
                "new_optimizer_updates":0,
                "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
                "confirmatory_population_claim":False,
            }
            modes=("full_affine","group_025","sham_025","group_05",
                   "sham_05","group_1","sham_1")
            c={key:200 for key in modes}
            c["sham_1"]=208
            a={key:200 for key in modes}
            result["correctness_totals"]={"composed":c,"additive":a}
            result["raw_root_composed_minus_additive_correctness_DID"]=8/256
            result["registered_ITT_root_effect"]=0. if i in (3,5) else 8/256
            result["original_fixed_ITT_zero"]=i in (3,5)
            r=folder/"source_behavior_report.json"
            r.write_text(canonical(result)+"\n",encoding="utf-8")
            audited={
                "schema":"NEC-03V/independent-original-source-group-removal-behavior/v1",
                "world_index":i,
                "independent_behavior_replay_passed":True,
                "source_behavior_report_sha256":sha256_file(r),
                "source_behavior_raw_sha256":sha256_file(raw),
                "original_source_case_arm_records":512,
                "independent_replayed_native_forwards":112,
                "independent_correctness_totals":result["correctness_totals"],
                "original_primary_ITT_effect":result["registered_ITT_root_effect"],
                "population_claim_permitted":False,
                "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
            }
            (folder/"source_behavior_independent.json").write_text(
                canonical(audited)+"\n",encoding="utf-8")
            folders.append(folder)
        return folders

    def test_full_eight_root_ITT_includes_fixed_3_5_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders=self.make_roots(Path(tmp))
            result=aggregate(folders)
            self.assertEqual(result["source_only_ITT_root_macro_effect"],6/8*8/256)
            self.assertEqual(result["positive_root_count"],6)
            self.assertEqual(result["fixed_ITT_zero_roots"],[3,5])
            self.assertEqual(result["roots"]["3"]["effect"],0)
            self.assertEqual(result["roots"]["5"]["effect"],0)
            self.assertEqual(result["roots"]["3"]["raw_effect"],8/256)
            self.assertTrue(result["above_meaningful_effect_floor"])
            self.assertFalse(result["root_level_population_p_value_computed"])

    def test_missing_root_or_tampered_independent_audit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders=self.make_roots(Path(tmp))
            with self.assertRaisesRegex(ValueError,"complete original eight-root"):
                aggregate(folders[:7])
            with self.assertRaisesRegex(ValueError,"duplicate"):
                aggregate([folders[0]]*8)
            audit=folders[0]/"source_behavior_independent.json"
            row=json.loads(audit.read_text())
            row["independent_behavior_replay_passed"]=False
            audit.write_text(canonical(row)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"audit"):
                aggregate(folders)


if __name__=="__main__":
    unittest.main()
