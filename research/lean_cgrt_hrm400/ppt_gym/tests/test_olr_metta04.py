"""Protocol fixtures for OLR-METTA-04 staged dependent proof control.

FakeDependentLean is a deterministic JSON simulator. It is NEVER scientific
Lean proof evidence. Hosted native Lean and standalone compilation are the
separate evidence source.
"""
from __future__ import annotations

import copy
import hashlib
import re
import unittest
from unittest.mock import patch

import torch

from ppt_gym.data import digest
from ppt_gym import metta_obligations as metta
from ppt_gym.olr_dependent import DependentWorld,STMT,generate_pairs
from ppt_gym.olr_dependent_repair import offer,repair,typed_pending
from ppt_gym.olr_metta04_policy import TiedStagePolicy,DirectStagePolicy,features,act,training_trace,fit
from ppt_gym.olr_metta04_episode import run_episode,replay_episode
from ppt_gym.live_lean import public_state


class FakeDependentLean:
    """Lean-shaped independent immutable states, never a certificate."""
    scientific_eligible=False
    version="FAKE Lean 4.19"
    def __init__(self):
        self.reset()

    def reset(self):
        self.contexts={}
        self.states={}
        self.next_id=0
        return {"ok":True}

    def _state(self,goals):
        self.next_id+=1
        state={"sid":self.next_id,
               "goals":[{"ctx":"context","target":copy.deepcopy(x)}
                        for x in goals]}
        self.states[state["sid"]]=state
        return state

    def init(self,statement,pp=True):
        m=STMT.fullmatch(statement)
        if not m:
            raise ValueError("fake only supports registered dependent grammar")
        a,b,r0a,r0b,c0,p0,q0,r1a,r1b,c1,p1,q1=map(int,m.groups())
        if not(a==r0a==r1a and b==r0b==r1b and c0==p0==q0 and c1==p1==q1):
            raise ValueError("fake source theorem changed")
        self.a,self.b=a,b
        self.values=(c0,c1)
        def imp(c):
            result={"a":"R"}
            for premise in reversed(({"raw":f"{a} + {b} = {c}"},
                                     {"raw":f"P {c}"},{"raw":f"Q {c}"})):
                result={"imp":[premise,result]}
            return result
        self.contexts["context"]=[
            {"name":"P","type":{"raw":"Nat → Prop"}},
            {"name":"Q","type":{"raw":"Nat → Prop"}},
            {"name":"R","type":{"sort":"Prop"}},
            {"name":"h0","type":{"raw":f"P ({a} + {b})"}},
            {"name":"h1","type":{"raw":"∀ n : Nat, P n → Q n"}},
            {"name":"r0","type":imp(c0)},
            {"name":"r1","type":imp(c1)},
        ]
        return self._state([{"a":"R"}])

    def run(self,sid,tactic):
        state=self.states[sid]
        g=[item["target"] for item in state["goals"]]
        if not g:
            return {"ok":False,"error":"already closed"}
        head=g[0]
        if tactic in ("apply r0","apply r1") and head=={"a":"R"}:
            c=self.values[int(tactic[-1])]
            after=[{"raw":f"{self.a} + {self.b} = {c}"},
                   {"raw":f"P {c}"},{"raw":f"Q {c}"},*g[1:]]
        elif tactic=="rfl" and head.get("raw","").split(" = ")[0]==f"{self.a} + {self.b}":
            c=int(head["raw"].split(" = ")[-1])
            if c!=self.a+self.b:
                return {"ok":False,"error":"failed equality"}
            after=g[1:]
        elif tactic=="apply h0" and head.get("raw","").startswith("P "):
            if int(head["raw"].split()[-1])!=self.a+self.b:
                return {"ok":False,"error":"indexed witness unavailable"}
            after=g[1:]
        elif tactic=="apply h1" and head.get("raw","").startswith("Q "):
            c=int(head["raw"].split()[-1])
            after=[{"raw":f"P {c}"},*g[1:]]
        elif tactic=="exact h1 _ h0" and head.get("raw","").startswith("Q "):
            if int(head["raw"].split()[-1])!=self.a+self.b:
                return {"ok":False,"error":"dependent witness mismatch"}
            after=g[1:]
        else:
            return {"ok":False,"error":"unsupported typed tactic"}
        return {"ok":True,"state":self._state(after)}

    def focus(self,sid,goal):
        g=[item["target"] for item in self.states[sid]["goals"]]
        if goal<0 or goal>=len(g):
            return {"ok":False}
        return {"ok":True,"state":self._state([g[goal],*g[:goal],*g[goal+1:]])}

    def kernel(self,sid):
        return {"ok":not self.states[sid]["goals"],"axioms":[],
                "term":"FAKE-not-Lean"}


def fake_hyperon(program,*,offer_pattern):
    if "Repair dep Typed WithinBudget UniqueLineage CurrentRevision DependenciesMet ConflictFree 2" in program:
        result=(("dep","2"),)
    else:
        result=()
    return result,hashlib.sha256(program.encode()).hexdigest()


class StagedControllerTests(unittest.TestCase):
    def setUp(self):
        self.world=DependentWorld("fake_root","fake_pair","train",0,17,19,1)
        self.claims=[[0.,0.,0.,1.,0.,1.] for _ in range(6)]

    def controller(self,first,macro):
        chosen=[]
        def policy(phase,remaining,branch):
            if phase=="root":
                return ("root_r"+str(first),0)
            if phase=="retry":
                return ("root_r"+str(1-first),0)
            if phase=="open":
                return ("probe_guard",0)
            if phase=="failed":
                return ("restore" if macro else "stop",0)
            if phase=="guard_ok":
                return ("metta_repair" if macro else "manual",0)
            if phase=="closed":
                return ("stop",0)
            raise AssertionError(phase)
        return policy

    def test_two_step_metta_repair_is_atomic(self):
        lean=FakeDependentLean()
        root=lean.init(self.world.statement())
        state=lean.run(root["sid"],"apply r0")["state"]
        state=lean.run(state["sid"],"rfl")["state"]
        self.assertTrue(typed_pending(public_state(lean,state)))
        before=digest(public_state(lean,state))
        with patch.object(metta,"_native_metta",side_effect=fake_hyperon):
            # Imported function needs patch on its own module binding.
            from ppt_gym import olr_dependent_repair as module
            with patch.object(module,"_native_metta",side_effect=fake_hyperon):
                absent=repair(lean,state,budget=1,eligible=True)
                bad=repair(lean,state,budget=2,eligible=False)
                good=repair(lean,state,budget=2,eligible=True)
        self.assertFalse(absent["committed"])
        self.assertEqual(absent["calls"],0)
        self.assertFalse(bad["committed"])
        self.assertTrue(good["committed"])
        self.assertEqual(good["calls"],2)
        self.assertEqual(digest(public_state(lean,state)),before)

    def test_native_simulated_multi_decision_restore(self):
        from ppt_gym import olr_dependent_repair as module
        with patch.object(module,"_native_metta",side_effect=fake_hyperon):
            correct=run_episode(FakeDependentLean(),self.world,claims=self.claims,
                     macro_allowed=True,policy=self.controller(0,True))
            repaired=run_episode(FakeDependentLean(),self.world,claims=self.claims,
                     macro_allowed=True,policy=self.controller(1,True))
            manual=run_episode(FakeDependentLean(),self.world,claims=self.claims,
                     macro_allowed=False,policy=self.controller(0,False))
            failed=run_episode(FakeDependentLean(),self.world,claims=self.claims,
                     macro_allowed=False,policy=self.controller(1,False))
            for ep,allowed in ((correct,True),(repaired,True),(manual,False),(failed,False)):
                self.assertTrue(replay_episode(FakeDependentLean(),self.world,ep,macro_allowed=allowed))
        self.assertEqual((correct["kernel_closed"],correct["tactic_calls"]),(True,4))
        self.assertEqual((repaired["kernel_closed"],repaired["tactic_calls"]),(True,6))
        self.assertEqual((manual["kernel_closed"],manual["tactic_calls"]),(True,5))
        self.assertFalse(failed["kernel_closed"])
        self.assertEqual(failed["tactic_calls"],2)
        self.assertEqual(repaired["restores"],1)
        self.assertEqual(correct["terminal_stops"],1)

    def test_work_first_produces_true_native_proof(self):
        from ppt_gym import olr_dependent_repair as module
        def policy(phase,remaining,branch):
            if phase=="root":return "root_r0",0
            if phase=="open":return "work_first",0
            return "stop",0
        with patch.object(module,"_native_metta",side_effect=fake_hyperon):
            ep=run_episode(FakeDependentLean(),self.world,
                claims=self.claims,macro_allowed=True,policy=policy)
            self.assertTrue(replay_episode(FakeDependentLean(),self.world,ep,macro_allowed=True))
        self.assertTrue(ep["kernel_closed"])
        self.assertEqual(ep["tactic_calls"],5)
        self.assertEqual(ep["focus_ops"],1)

    def test_second_restore_is_vetoed_without_touching_native_state(self):
        from ppt_gym import olr_dependent_repair as module
        wrong=1
        def always_bad(phase,remaining,branch):
            if phase in ("root","retry"):
                return "root_r"+str(wrong),0
            if phase=="open":
                return "probe_guard",0
            if phase=="failed":
                return "restore",0
            return "stop",0
        with patch.object(module,"_native_metta",side_effect=fake_hyperon):
            episode=run_episode(FakeDependentLean(),self.world,claims=self.claims,
                macro_allowed=True,policy=always_bad)
            self.assertTrue(replay_episode(FakeDependentLean(),self.world,episode,
                                           macro_allowed=True))
        self.assertFalse(episode["kernel_closed"])
        self.assertEqual(episode["restores"],1)
        self.assertEqual(episode["vetoed_restores"],1)
        self.assertEqual(episode["tactic_calls"],4)

    def test_tampering_with_replay_verdict_fails(self):
        from ppt_gym import olr_dependent_repair as module
        with patch.object(module,"_native_metta",side_effect=fake_hyperon):
            ep=run_episode(FakeDependentLean(),self.world,claims=self.claims,
                  macro_allowed=True,policy=self.controller(1,True))
            forged=copy.deepcopy(ep)
            for event in forged["events"]:
                if event["kind"]=="tactic" and event["action"]=="rfl" and not event["ok"]:
                    event["ok"]=True
                    break
            with self.assertRaises(ValueError):
                replay_episode(FakeDependentLean(),self.world,forged,macro_allowed=True)

    def test_stage_model_is_trained_with_carry_and_mask(self):
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled=False
        torch.use_deterministic_algorithms(True)
        traces=[]
        for winner in (0,1):
            traces.append(training_trace(self.claims,winner=winner,
                                         macro_allowed=True,wrong_start=False))
            traces.append(training_trace(self.claims,winner=winner,
                                         macro_allowed=True,wrong_start=True))
        a,ra=fit(traces,seed=17,arm="loop3",steps=3,batch_size=2)
        b,rb=fit(traces,seed=17,arm="loop3",steps=3,batch_size=2)
        self.assertEqual(ra["parameters"],rb["parameters"])
        self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k])
                            for k in a.state_dict()))
        action,carry,logits=act(a,self.claims,phase="root",remaining=6,
            branch=None,macro_allowed=True,carry=None,arm="loop3")
        self.assertIn(action,("root_r0","root_r1"))
        self.assertEqual(tuple(carry.shape),(1,a.width))
        self.assertEqual(len(logits),2)


if __name__=="__main__":
    unittest.main()
