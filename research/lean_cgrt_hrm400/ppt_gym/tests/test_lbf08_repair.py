"""LBF-08: synthetic-only unit checks of native repair data contracts.

FakeLean is an engineering fixture and cannot create scientific evidence.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import torch

from ppt_gym.data import canonical
from ppt_gym.lbf08_repair import (
    _oracle_labels, forced_recovery, frame_schedule,
    visited_frame, shortest_finite_script, VISITED_SCHEMA,
)
from ppt_gym.tests.test_live_lean import FakeLean


class NativeRepairContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.oracle = {
            "table": {
                "P1": {"apply r0": ["P0"]},
                "P0": {"apply h0": []},
            },
            "size": {"P1": 2, "P0": 1},
        }

    def _event(self, goal="P0", menu=None, attempt=2, failed=None):
        if menu is None:
            menu = ["apply h0"]
        return {
            "attempt": attempt,
            "tactic": menu[0],
            "selected_index": 0,
            "status": "kernel_closed" if goal == "P0" else "accepted_nonterminal",
            "input": {"state": {"goals": [{
                "target": {"a": goal}, "hypotheses": [],
                "pp": "⊢ " + goal,
            }]},
            "failed_choices": failed or [],
            "checks_remaining": 8 - attempt + 1},
            "candidate_menu": menu,
            "public_packet_sha256": "fake-hex",
        }

    def test_explicit_censor_on_unreachable_and_missing_shortest(self):
        self.assertIsNone(_oracle_labels({"target": {"a": "P9"}},
                                         ["apply h0"], self.oracle))
        self.assertIsNone(_oracle_labels({"target": {"a": "P1"}},
                                         ["apply h0"], self.oracle))
        self.assertEqual(_oracle_labels({"target": {"a": "P1"}},
                                        ["apply r0"], self.oracle), (1,))

    def test_training_label_is_native_set_and_single_target_is_exact_one(self):
        ep={"root": "R", "lineage": "L"}
        event=self._event(failed=["apply r7"])
        frame,status=visited_frame(event,ep,self.oracle)
        self.assertEqual(status,"witnessed")
        self.assertEqual(frame["schema"],VISITED_SCHEMA)
        self.assertEqual(frame["labels"],[1])
        self.assertEqual(frame["single_labels"],[1])
        self.assertEqual(frame["n_optimal"],1)
        self.assertEqual(frame["failed_choices"],["apply r7"])
        self.assertEqual(frame["source_public_packet_sha256"],
                         hashlib.sha256(frame["goal"].encode()).hexdigest())
        first,event_reason=visited_frame(self._event(attempt=1),ep,self.oracle)
        self.assertIsNone(first)
        self.assertEqual(event_reason,"initial_root_not_repair")

    def test_multiple_optimal_targets_credited_only_in_set_arm(self):
        oracle={
            "table": {"P3": {"apply h0": [],"apply h1": [],"apply r0": ["P2"]},
                      "P2":{"apply h2":[]}},
            "size": {"P3":1,"P2":1},
        }
        frame,status=visited_frame(self._event(goal="P3",menu=["apply h0","apply h1","apply r0"]),
                                  {"root":"R","lineage":"L"},oracle)
        self.assertEqual(status,"witnessed")
        self.assertEqual(frame["labels"],[1,1,0])
        self.assertEqual(frame["single_labels"],[1,0,0])
        self.assertEqual(frame["n_optimal"],2)

    def test_censored_no_witness_stays_unavailable_for_supervision(self):
        event=self._event(goal="P1",menu=["apply h0"])
        frame,status=visited_frame(event,{"root":"R","lineage":"L"},self.oracle)
        self.assertIsNone(frame)
        self.assertEqual(status,"no_public_shortest_witness")
        event=self._event(goal="P0",menu=["apply h0"])
        event["input"]["oracle_label"]=1
        with self.assertRaisesRegex(ValueError,"packet"):
            visited_frame(event,{"root":"R","lineage":"L"},self.oracle)

    def test_exact_shortest_ignores_unreachable_alternatives(self):
        graph = {
            "P1": {"apply r0": ["P0"], "apply r9": ["P8"]},
            "P0": {"apply h0": []},
            "P8": {"apply r8": ["P7"]},
        }
        costs = {"P1": 2, "P0": 1, "P8": None, "P7": None}
        self.assertEqual(shortest_finite_script(graph, costs, "P1"),
                         ["apply r0", "apply h0"])
        with self.assertRaisesRegex(ValueError,"finite shortest"):
            shortest_finite_script(graph, costs, "P8")
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            shortest_finite_script({"P1":{"apply r9":["P8"]}}, costs, "P1")

    def test_forced_recovery_from_native_prefix_is_marked_teacher_assisted(self):
        class Case:
            statement="FAKE GENERATED HORN"
        result=forced_recovery(FakeLean(), Case(), self.oracle, ["apply r0"])
        self.assertEqual(result["status"],"native_kernel_recovered")
        self.assertEqual(result["prefix_tactics"],["apply r0"])
        self.assertEqual(result["continuation_tactics"],["apply h0"])
        self.assertTrue(result["native_kernel_verified"])
        self.assertIsNone(result["standalone_certificate"])
        self.assertTrue(result["within_eight_accepted_tactics"])

    def test_action_roster_is_precomputed_and_arm_independent(self):
        src=[{"lineage":f"l{i}","step":0} for i in range(48)]
        vis=[{"lineage":f"l{i}","step":2} for i in range(48)]
        one=frame_schedule(src,vis,107,steps=720)
        two=frame_schedule(src,vis,107,steps=720)
        self.assertEqual(one,two)
        self.assertEqual(len(one),720)
        self.assertEqual(sum(x["use_model_visited"] for x in one),360)
        self.assertEqual(len({x["lineage"] for x in one}),48)

    def test_missing_visits_remain_on_source_schedule(self):
        src=[{"lineage":f"l{i}","step":0} for i in range(48)]
        vis=[{"lineage":"l0","step":1}]
        schedule=frame_schedule(src,vis,107,steps=720)
        self.assertEqual(len(schedule),720)
        self.assertTrue(any(x["visit_index"] is None for x in schedule))
        self.assertTrue(all(x["use_model_visited"] is False for x in schedule
                            if x["visit_index"] is None))


if __name__=="__main__":
    unittest.main()
