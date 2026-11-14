"""FC-13 strict wall deadline, process accounting and proof-gate regression."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import time
import unittest
import fc13_policy as fc13

GOAL="x : Nat\n⊢ x = x"
ORIGINAL=("import Init\nnamespace FC13Test\n"
          "theorem toy (x : Nat) : x = x := by\n  rfl\n"
          "end FC13Test\n")


def simulated(*args,probe,timeout_seconds,**kwargs):
    if timeout_seconds<=0.:
        raise AssertionError("called compiler after deadline")
    return {
       "accepted":True,"timed_out":False,
       "real_remaining_goal_states":[],
       "real_remaining_goal_count":0,
       "source_sha256":"synthetic",
       "compiler_returncode":0,"lean_wall_ms":0.,
       "compiler_error_excerpt":"",
    }


class FC13HardDeadline(unittest.TestCase):
    def test_two_real_calls_required_for_final_certificate(self):
        attempts=[]
        def recorder(*args,**kwargs):
            attempts.append(kwargs["probe"])
            return simulated(*args,**kwargs)
        outcome=fc13.run_controller(ORIGINAL,"toy",GOAL,None,
                   compile_fn=recorder,deadline_ms=15000.)
        self.assertTrue(outcome["accepted"])
        self.assertEqual(attempts,[True,False])
        self.assertEqual(outcome["lean_subprocess_count"],2)
        self.assertEqual(len(outcome["tactic_trials"]),2)
        self.assertTrue(outcome["certificate_completed_within_hard_deadline"])

    def test_slow_successful_compiler_can_never_receive_proof_credit(self):
        def late(*args,**kwargs):
            time.sleep(.022)
            return simulated(*args,**kwargs)
        outcome=fc13.run_controller(ORIGINAL,"toy",GOAL,None,
                compile_fn=late,deadline_ms=5.)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"],"hard_deadline")
        self.assertEqual(outcome["lean_subprocess_count"],1)
        self.assertGreater(outcome["deadline_overrun_ms"],0)
        self.assertTrue(outcome["tactic_trials"][0]["expired_at_return"])

    def test_genuine_multigoal_state_abstains(self):
        def split(*args,**kwargs):
            row=simulated(*args,**kwargs)
            row["real_remaining_goal_states"]=["case fst\n⊢ True","case snd\n⊢ True"]
            row["real_remaining_goal_count"]=2
            return row
        outcome=fc13.run_controller(ORIGINAL,"toy",GOAL,None,
                compile_fn=split,deadline_ms=1000.)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"],"multigoal_unsupported")
        self.assertEqual(outcome["lean_subprocess_count"],1)

    def test_never_exceeds_twelve_actual_subprocess_calls(self):
        def failures(*args,**kwargs):
            row=simulated(*args,**kwargs)
            row["accepted"]=False
            row["compiler_returncode"]=1
            row["compiler_error_excerpt"]="tactic failed"
            return row
        busy_goal="x : Nat\nh1 : x = x\nh2 : x = x\nxs : List Nat\nw : ∃ n : Nat, n > 0\n⊢ x = x"
        outcome=fc13.run_controller(ORIGINAL,"toy",busy_goal,None,
                compile_fn=failures,deadline_ms=20000.)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["lean_subprocess_count"],12)
        self.assertEqual(outcome["reason"],"lean_subprocess_budget")

    def test_failed_full_lean_gate_cannot_issue_certificate(self):
        def rejected_full(*args,**kwargs):
            row=simulated(*args,**kwargs)
            if not kwargs["probe"]:
                row["accepted"]=False
                row["compiler_returncode"]=1
                row["compiler_error_excerpt"]="missing proof"
            return row
        outcome=fc13.run_controller(ORIGINAL,"toy",GOAL,None,
                compile_fn=rejected_full,deadline_ms=20000.)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"],"final_lean_rejected")

    def test_static_contract(self):
        self.assertEqual(fc13.DEADLINE_MS,20000.)
        self.assertEqual(fc13.MAX_LEAN_CALLS,12)
        self.assertEqual(fc13.MAX_TRANSITIONS,3)
        self.assertEqual(set(fc13.CHECKPOINTS),{17,29})

    def test_native_original_lean_compilation(self):
        from shutil import which
        if not which("lean"):
            self.skipTest("requires Lean4.35 compiler")
        outcome=fc13.run_controller(ORIGINAL,"toy",GOAL,None,
                deadline_ms=20000.)
        self.assertTrue(outcome["accepted"],outcome)
        self.assertLessEqual(outcome["elapsed_ms"],20000.)
        self.assertLessEqual(outcome["lean_subprocess_count"],12)


if __name__=="__main__":
    unittest.main()
