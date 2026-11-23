"""LSA-02H negative and positive off-reference source contracts (synthetic only)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.data import canonical, sha256_file
from ppt_gym.live_lean import candidate_menu
from ppt_gym.native_train import _snapshot_trajectory
from ppt_gym.offpath_source import _collect_case, _completion_script


CONTEXT = [
    {"name": f"P{i}", "type": {"sort": "Prop"}} for i in range(3)
] + [
    {"name": "h0", "type": {"a": "P0"}},
    {"name": "r0", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}},
    {"name": "r1", "type": {"imp": [{"a": "P2"}, {"a": "P1"}]}},
    {"name": "r2", "type": {"imp": [{"a": "P0"}, {"a": "P2"}]}},
]


class SmallLean:
    version = "SIMULATED ONLY"
    def __init__(self):
        self.contexts = {"ctx0": CONTEXT}
        self.states = {}
        self.nextid = 0

    def reset(self):
        self.states = {}
        self.nextid = 0
        return {"ok": True}

    def _state(self, target):
        self.nextid += 1
        goals = [] if target is None else [{
            "ctx": "ctx0", "pp": f"⊢ {target}", "target": {"a": target},
        }]
        s = {"sid": self.nextid, "goals": goals}
        self.states[self.nextid] = s
        return s

    def init(self, statement, pp=True):
        return self._state("P1")

    def run(self, sid, tactic):
        target = self.states[sid]["goals"][0]["target"]["a"]
        routes = {
            ("P1", "apply r0"): "P0",
            ("P1", "apply r1"): "P2",
            ("P2", "apply r2"): "P0",
            ("P0", "apply h0"): None,
        }
        if (target, tactic) not in routes:
            return {"ok": False, "error": "not a proof"}
        return {"ok": True, "state": self._state(routes[(target, tactic)])}

    def kernel(self, sid):
        return {"ok": not self.states[sid]["goals"], "axioms": []}


@dataclass
class FakeCase:
    root: str = "dummy_root"
    lineage: str = "dummy_lineage"
    family: str = "alt"
    tid: str = "dummy_theorem"
    statement: str = "synthetic"

    def theorem(self):
        return object()


ORACLE = {
    "proof": {"script": ["apply r0", "apply h0"],
              "repl_kernel": {"ok": True, "axioms": []}},
    "table": {
        "P1": {"apply r0": ["P0"], "apply r1": ["P2"]},
        "P2": {"apply r2": ["P0"]},
        "P0": {"apply h0": []},
    },
    "size": {"P0": 1, "P2": 2, "P1": 2},
}


class OffReferenceSourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_graph_alternative_reaches_novel_certified_correction(self):
        lean = SmallLean()
        case = FakeCase()
        ref = _snapshot_trajectory(lean, case, ORACLE)
        self.assertEqual(len(ref), 2)
        extra, records, census = _collect_case(lean, case, ORACLE, ref)
        self.assertEqual(len(extra), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][2], ["apply r1", "apply r2", "apply h0"])
        self.assertEqual(len(extra[0]["correction_states"]), 1)
        state = extra[0]["correction_states"][0]
        self.assertEqual(state["source_kind"], "off_reference")
        self.assertEqual(state["candidates"], ["apply r2"])
        self.assertEqual(state["labels"], [1])
        self.assertEqual(state["checks_remaining"], 7)
        self.assertEqual(state["remaining_proof_steps"], 2)
        self.assertNotIn(state["public_state_sha256"],
                         {r["public_state_sha256"] for r in ref})
        self.assertEqual(census["candidate_within_eight_checks"], 1)
        self.assertEqual(census["offreference_training_frames"], 1)

    def test_dead_alternative_is_counted_as_ineligible(self):
        lean = SmallLean()
        case = FakeCase()
        ref = _snapshot_trajectory(lean, case, ORACLE)
        alt = json.loads(canonical(ORACLE))
        alt["size"]["P2"] = None
        new, scripts, census = _collect_case(lean, case, alt, ref)
        self.assertEqual(new, [])
        self.assertEqual(scripts, [])
        self.assertEqual(census["unreachable_or_over_budget"], 1)

    def test_native_dead_edge_is_excluded_from_finite_completion(self):
        oracle = json.loads(canonical(ORACLE))
        oracle["table"]["P2"]["apply r3"] = ["P3"]
        oracle["size"]["P3"] = None
        self.assertEqual(_completion_script(
            oracle, [{"target": {"a": "P2"}}],
        ), ["apply r2", "apply h0"])

    def test_no_fake_full_proof_can_be_claimed_for_an_unprovable_goal(self):
        self.assertIsNone(_completion_script(
            ORACLE, [{"target": {"a": "P3"}}],
        ))
        self.assertEqual(_completion_script(
            ORACLE, [{"target": {"a": "P2"}}],
        ), ["apply r2", "apply h0"])


if __name__ == "__main__":
    unittest.main()
