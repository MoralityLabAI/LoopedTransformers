"""NEC03W label-blind calibration shortlist synthetic-only tests.

No original data, native model forward, labels, loss, selected task outcome
or saved query source data is accessed in these deterministic tests.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechinterp.neuron_communities.nec03w_calibration_shortlist import (
    shortlist,PHYSICAL_SAMPLE,MIN_CALIBRATION_CO_SUPPORT,
)


class CalibrationShortlistTests(unittest.TestCase):
    def toy_cohort(self,root,*,support=True):
        cols=[[0.0]*32 for _ in range(128)]
        for i in range(16):
            cols[i][i//2]=1.
        quartet={"quartet":[0,1,2,3],
                 "pair_partitions":[[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]]}
        roster={
            "roster_sha256":"synthetic-roster-sha",
            "arms":{
                arm:{"visits":{"0":{"candidates":[quartet]}}}
                for arm in ("composed","additive")
            },
        }
        calibration=[
            [[i,1.0] for i in range(16)] if support else [[i,1.0] for i in range(4,20)]
            for _ in range(256)
        ]
        source={
            "arms":{
                arm:{"positive_codes_by_visit":{"0":calibration}}
                for arm in ("composed","additive")
            },
        }
        shards={
            arm:{"visits":{"0":{"decoder32":cols}}}
            for arm in ("composed","additive")
        }
        original={"world_index":root,"world_id":f"dev5-{root:03d}"}
        checked={"independent_code_audit_passed":True}
        return original,shards,source,checked,roster,"synthetic-original-source-pack"

    def test_fixed_sample_spans_only_eight_source_calibration_groups(self):
        self.assertEqual(len(PHYSICAL_SAMPLE),32)
        self.assertEqual(PHYSICAL_SAMPLE[:4],(0,8,16,24))
        self.assertEqual(PHYSICAL_SAMPLE[-4:],(224,232,240,248))
        self.assertEqual(MIN_CALIBRATION_CO_SUPPORT,64)

    def test_synthetic_source_quartet_is_supported_and_replayable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/"codes.json"
            audit=root/"audit.json"
            source.write_text("{}\n")
            audit.write_text("{}\n")
            from mechinterp.neuron_communities import nec03w_calibration_shortlist as unit
            with patch.object(unit,"load_calibration",
                              return_value=self.toy_cohort(0)):
                result=shortlist(root/"not_native",source,audit)
            self.assertEqual(result["schema"],
                             "NEC-03W/source-fit-calibration-label-blind-selection/v1")
            self.assertEqual(result["new_native_model_forwards"],0)
            self.assertEqual(result["new_task_labels_read"],0)
            self.assertFalse(result["new_factorial_or_rescue_behavior_scored"])
            for arm in ("composed","additive"):
                chosen=result["selected_physical_source_candidate_per_arm"][arm]
                self.assertIsNotNone(chosen)
                self.assertEqual(chosen["full_source_calibration_support_count"],256)
                self.assertEqual(chosen["matched_physical_sample_count"],32)
                self.assertEqual(chosen["visit"],0)

    def test_no_support_must_not_select_any_functional_quartet(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/"codes.json"
            audit=root/"audit.json"
            source.write_text("{}\n")
            audit.write_text("{}\n")
            from mechinterp.neuron_communities import nec03w_calibration_shortlist as unit
            with patch.object(unit,"load_calibration",
                              return_value=self.toy_cohort(0,support=False)):
                result=shortlist(root/"not_native",source,audit)
            self.assertEqual(result["selected_physical_source_candidate_per_arm"],
                             {"composed":None,"additive":None})
            self.assertEqual(result["source_sites"]["composed"]["0"]["tested_partitions"],[])


if __name__=="__main__":
    unittest.main()
