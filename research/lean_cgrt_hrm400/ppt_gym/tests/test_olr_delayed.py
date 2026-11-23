"""OLR-01D protocol and budget tests. FakeLean is never scientific proof evidence."""
from __future__ import annotations

import copy
import unittest

from ppt_gym.data import digest
from ppt_gym.olr_delayed import (
    DelayedWorld, STMT, raw_root_features, run_path, replay_path, ROOT_MENU,
)
from ppt_gym.olr_delayed_campaign import score_groups


class FakeDelayedLean:
    """Small immutable-state simulator for Python controller correctness only."""
    version="FAKE Lean 4.19"
    scientific_eligible=False

    def __init__(self):
        self.reset()

    def reset(self):
        self.states={}
        self.contexts={}
        self.sid=0
        return {"ok":True}

    def _state(self, goals):
        self.sid+=1
        state={"sid":self.sid,"goals":[{"ctx":"context","target":copy.deepcopy(t)}
                                      for t in goals]}
        self.states[self.sid]=state
        return state

    def init(self, statement, pp=True):
        match=STMT.fullmatch(statement)
        if not match:
            raise ValueError("fake only handles registered multislot grammar")
        a,b,c,d,e,f=map(int,match.groups())
        assert (a,b)==(d,e)
        self.a,self.b=a,b
        def rule(target):
            value={"a":"R"}
            for premise in reversed(({"raw":f"{a} + {b} = {target}"},
                                     {"a":"P"},{"a":"Q"})):
                value={"imp":[premise,value]}
            return value
        self.contexts={"context":[
            {"name":"P","type":{"sort":"Prop"}},
            {"name":"Q","type":{"sort":"Prop"}},
            {"name":"R","type":{"sort":"Prop"}},
            {"name":"h0","type":{"a":"P"}},
            {"name":"h1","type":{"a":"Q"}},
            {"name":"r0","type":rule(c)},
            {"name":"r1","type":rule(f)},
        ]}
        self.values=(c,f)
        return self._state([{"a":"R"}])

    def run(self,sid,tactic):
        old=self.states[sid]
        goals=[g["target"] for g in old["goals"]]
        if not goals:
            return {"ok":False,"error":"already closed"}
        head=goals[0]
        target=None
        if tactic in ("apply r0","apply r1") and head=={"a":"R"}:
            target=self.values[int(tactic[-1])]
            after=[{"raw":f"{self.a} + {self.b} = {target}"},
                   {"a":"P"},{"a":"Q"}] + goals[1:]
        elif tactic=="rfl" and head.get("raw"):
            parsed=head["raw"].split(" = ")
            if int(parsed[1])!=self.a+self.b:
                return {"ok":False,"error":"rfl failed"}
            after=goals[1:]
        elif tactic=="apply h0" and head=={"a":"P"}:
            after=goals[1:]
        elif tactic=="apply h1" and head=={"a":"Q"}:
            after=goals[1:]
        else:
            return {"ok":False,"error":"wrong goal/tactic"}
        return {"ok":True,"state":self._state(after)}

    def focus(self,sid,goal):
        goals=[g["target"] for g in self.states[sid]["goals"]]
        if goal>=len(goals):
            return {"ok":False,"error":"out of bounds"}
        return {"ok":True,"state":self._state([goals[goal],*goals[:goal],*goals[goal+1:]])}

    def kernel(self,sid):
        closed=not self.states[sid]["goals"]
        return {"ok":closed,"axioms":[] if closed else ["sorryAx"],
                "term":"FAKE NOT A LEAN CERTIFICATE"}


class DelayedTests(unittest.TestCase):
    def setUp(self):
        self.world0=DelayedWorld("pair0",0,2,3,(5,6))
        self.world1=DelayedWorld("pair0",1,2,3,(6,5))

    def test_public_parsing_hides_private_winner(self):
        lean=FakeDelayedLean()
        state=lean.init(self.world0.statement())
        from ppt_gym.live_lean import public_state, candidate_menu
        visible=public_state(lean,state)
        self.assertEqual(candidate_menu(lean,state),ROOT_MENU)
        self.assertEqual(raw_root_features(visible,ROOT_MENU),
                         ((2/40,3/40,5/80),(2/40,3/40,6/80)))
        bad=copy.deepcopy(visible)
        bad["goals"][0]["hypotheses"][3]["type"]={"a":"Q"}
        with self.assertRaises(ValueError):
            raw_root_features(bad,ROOT_MENU)

    def test_early_rollback_rescues_wrong_branch_under_six_calls(self):
        for world,first in ((self.world0,1),(self.world1,0)):
            lean=FakeDelayedLean()
            ep=run_path(lean,world,first,"probe_early",6)
            self.assertTrue(ep["kernel_closed"],ep)
            self.assertEqual(ep["tactic_calls"],6)
            self.assertEqual(ep["rollbacks"],1)
            self.assertEqual(ep["focus_actions"],0)
            self.assertTrue(replay_path(FakeDelayedLean(),world,ep)["replay_passed"])
            self.assertEqual(ep["script"],[
                "apply r"+str(1-first),"rfl","apply h0","apply h1"])

    def test_late_guard_exhausts_budget_after_shared_goals(self):
        for world,first in ((self.world0,1),(self.world1,0)):
            ep=run_path(FakeDelayedLean(),world,first,"probe_late",6)
            self.assertFalse(ep["kernel_closed"])
            self.assertEqual(ep["tactic_calls"],4)
            self.assertEqual(ep["focus_actions"],2)
            self.assertEqual(ep["rollbacks"],1)
            self.assertTrue(replay_path(FakeDelayedLean(),world,ep)["replay_passed"])

    def test_both_schedules_close_on_correct_root(self):
        for world,first in ((self.world0,0),(self.world1,1)):
            for sched in ("probe_early","probe_late"):
                ep=run_path(FakeDelayedLean(),world,first,sched,4)
                self.assertTrue(ep["kernel_closed"])
                self.assertEqual(ep["tactic_calls"],4)
                self.assertTrue(replay_path(FakeDelayedLean(),world,ep)["replay_passed"])
                if sched=="probe_late":
                    self.assertEqual(ep["script"],[
                        "apply r"+str(first),"@focus(1)","apply h0",
                        "@focus(1)","apply h1","rfl"])

    def test_wrong_root_under_four_checks_stays_open(self):
        for sched in ("probe_early","probe_late"):
            ep=run_path(FakeDelayedLean(),self.world1,0,sched,4)
            self.assertFalse(ep["kernel_closed"])
            self.assertEqual(ep["tactic_calls"],2 if sched=="probe_early" else 4)
            self.assertTrue(replay_path(FakeDelayedLean(),self.world1,ep)["replay_passed"])

    def test_tampered_event_rejected_by_independent_replay(self):
        ep=run_path(FakeDelayedLean(),self.world0,0,"probe_early",4)
        forged=copy.deepcopy(ep)
        forged["events"][0]["action"]="apply r1"
        with self.assertRaises(ValueError):
            replay_path(FakeDelayedLean(),self.world0,forged)
        forged=copy.deepcopy(ep)
        forged["tactic_calls"]=3
        with self.assertRaisesRegex(ValueError,"resource"):
            replay_path(FakeDelayedLean(),self.world0,forged)

    def test_score_groups_are_pair_rooted(self):
        row=lambda world,closed:dict(root=str(world),pair="a",world=world,
            seed=0,policy="test",budget=6,schedule="probe_early",
            kernel_closed=closed,tactic_calls=4,focus_actions=0,rollbacks=0)
        s=score_groups([row(0,True),row(1,False)])
        self.assertEqual(s["6:probe_early:0:test"]["both_worlds_certified"],0)
        with self.assertRaises(ValueError):
            score_groups([row(0,True)])


if __name__=="__main__":
    unittest.main()
