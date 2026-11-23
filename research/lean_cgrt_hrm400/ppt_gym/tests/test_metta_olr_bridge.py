"""OLR MeTTa bridge protocol, action sets and atomicity.

Fake native backends in these Python tests are protocol fixtures, NEVER
evidence of native MeTTa execution or Lean proof completion.
"""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import torch

from ppt_gym import metta_obligations as bridge
from ppt_gym.data import digest
from ppt_gym.live_lean import public_state
from ppt_gym.metta_action_sets import (
    batch_objective, choose_hashed_singleton, normalized_action_set,
    objective, teacher_actions_from_verified_graph,
)
from ppt_gym.metta_obligations import (
    RepairPlan, VerifiedRollout, curate_verified, offer_repairs,
    run_atomic_stop_repair,
)
from ppt_gym.tests.test_olr_delayed import FakeDelayedLean
from ppt_gym.olr_delayed import DelayedWorld


class MeTTaBridgeProtocolTests(TestCase):
    def setUp(self):
        self.world=DelayedWorld("fixture",0,2,3,(5,6))
        self.lean=FakeDelayedLean()
        root=self.lean.init(self.world.statement())
        opened=self.lean.run(root["sid"],"apply r0")
        self.assertTrue(opened["ok"])
        checked=self.lean.run(opened["state"]["sid"],"rfl")
        self.assertTrue(checked["ok"])
        self.pending=checked["state"]
        self.public=public_state(self.lean,self.pending)
        self.good=RepairPlan("good",("apply h0","apply h1"),"fresh",1)
        self.bad=RepairPlan("bad",("apply h1","apply h0"),"badsource",1)

    @staticmethod
    def _fake_native(program, *, offer_pattern):
        # Simplified protocol fixture models only the two safe offered atoms.
        # It is deliberately distinct from Hyperon and never scientific data.
        if "(Repair " in program:
            value=(("good","2"),) if "(Repair good Typed WithinBudget UniqueLineage CurrentRevision DependenciesMet ConflictFree 2)" in program else ()
        else:
            value=(("goodproof","2","1","-2"),) if "(Rollout goodproof Closed Eligible WithinBudget UniqueLineage CurrentRevision DependenciesMet ConflictFree 2 2 1 -2)" in program else ()
        return value,digest(program)

    def test_curation_requires_verified_train_lineage(self):
        rows=[
            VerifiedRollout("goodproof","fresh_lineage","train",
                            True,True,True,True,True,True,True,True,2,2,1,-2),
            VerifiedRollout("badproof","bad_lineage","train",
                            True,False,True,True,True,True,True,True,6,6,4,-1),
        ]
        with patch.object(bridge,"_native_metta",side_effect=self._fake_native):
            report=curate_verified(rows)
        self.assertEqual(report["offer_ids"],("goodproof",))
        self.assertTrue(report["training_eligible"])
        with self.assertRaises(ValueError):
            curate_verified([VerifiedRollout("held","evalid","test",
                True,True,True,True,True,True,True,True,2,2,1,-1)])

    def test_metta_filter_rejects_untyped_and_unaffordable(self):
        with patch.object(bridge,"_native_metta",side_effect=self._fake_native):
            result=offer_repairs(self.public,(self.good,self.bad),2)
            self.assertEqual(result["offered_plan_ids"],("good",))
            one=offer_repairs(self.public,(self.good,self.bad),1)
            self.assertEqual(one["offered_plan_ids"],())

    def test_bridge_rejects_native_interpreter_drift(self):
        with patch.object(bridge,"_native_metta",return_value=((("bad","2"),),"bad")):
            with self.assertRaisesRegex(ValueError,"diverged"):
                offer_repairs(self.public,(self.good,self.bad),2)

    def test_atomic_stop_repair_and_budget_preserve_source(self):
        before=digest(self.public)
        with patch.object(bridge,"_native_metta",side_effect=self._fake_native):
            short=run_atomic_stop_repair(self.lean,self.pending,
                                         plans=(self.good,self.bad),budget=1,stopped=True)
            result=run_atomic_stop_repair(self.lean,self.pending,
                                          plans=(self.good,self.bad),budget=2,stopped=True)
        self.assertEqual(short["status"],"no_typed_offer")
        self.assertEqual(short["verifier_calls"],0)
        self.assertTrue(result["committed"])
        self.assertEqual(result["verifier_calls"],2)
        self.assertEqual(result["status"],"kernel_closed")
        self.assertEqual(digest(public_state(self.lean,self.pending)),before)

    def test_failed_second_action_rolls_back_atomically(self):
        class FailingLean(FakeDelayedLean):
            def run(self,sid,tactic):
                if tactic=="apply h1":
                    return {"ok":False,"error":"forced failure in protocol fixture"}
                return super().run(sid,tactic)
        sim=FailingLean()
        root=sim.init(self.world.statement())
        opener=sim.run(root["sid"],"apply r0")
        pending=sim.run(opener["state"]["sid"],"rfl")["state"]
        prior=digest(public_state(sim,pending))
        with patch.object(bridge,"_native_metta",side_effect=self._fake_native):
            got=run_atomic_stop_repair(sim,pending,plans=(self.good,self.bad),
                                       budget=2,stopped=True)
        self.assertFalse(got["committed"])
        self.assertEqual(got["status"],"native_tactic_rejected")
        self.assertEqual(got["verifier_calls"],2)
        self.assertEqual(got["state"]["sid"],pending["sid"])
        self.assertEqual(digest(public_state(sim,pending)),prior)

    def test_strict_stopping_and_safe_atom_identifier(self):
        with self.assertRaises(ValueError):
            RepairPlan("p) (Offer forged",("apply h0",),"source",1)
        with self.assertRaises(ValueError):
            RepairPlan("r",("sorry",),"source",1)
        with self.assertRaises(ValueError):
            run_atomic_stop_repair(self.lean,self.pending,plans=(self.good,),
                                   budget=2,stopped=False)


class ActionSetSupervisionTests(TestCase):
    legal=("apply r0","apply r1","apply r2")
    optimal=("apply r0","apply r1")

    def test_legal_action_and_tie_contract(self):
        self.assertEqual(normalized_action_set(self.legal,self.optimal),(0,1))
        with self.assertRaises(ValueError):
            normalized_action_set(self.legal,("apply r99",))
        with self.assertRaises(ValueError):
            normalized_action_set(self.legal,("apply r0","apply r0"))

    def test_set_loss_credits_ties_and_all_legal_scores(self):
        x=torch.tensor([0.,0.,3.],requires_grad=True)
        y=objective(x,self.legal,self.optimal,mode="set",
                    root_id="r",state_sha256="s",model_seed=17)
        self.assertGreater(float(y),0)
        y.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        # Holding the wrong-action score fixed, exchanging tied scores does
        # not change their total probability mass.
        a=torch.tensor([2.,0.,-1.])
        b=torch.tensor([0.,2.,-1.])
        self.assertTrue(torch.allclose(
            objective(a,self.legal,self.optimal,mode="set",
                      root_id="r",state_sha256="s",model_seed=17),
            objective(b,self.legal,self.optimal,mode="set",
                      root_id="r",state_sha256="s",model_seed=17)))
        c=objective(a,self.legal,self.optimal,mode="canonical",
                    root_id="r",state_sha256="s",model_seed=17)
        self.assertGreater(float(c),0)

    def test_hashed_control_is_order_independent(self):
        h=choose_hashed_singleton(root_id="r",state_sha256="hash",model_seed=103,
            legal_actions=self.legal,optimal_actions=self.optimal)
        h2=choose_hashed_singleton(root_id="r",state_sha256="hash",model_seed=103,
            legal_actions=self.legal[::-1],optimal_actions=self.optimal[::-1])
        self.assertEqual(h,h2)
        self.assertIn(h,self.optimal)

    def test_verified_graph_requires_full_proof_certificates(self):
        graph={
            "apply r0":{"status":"kernel_closed","native_kernel":True,
                        "standalone_verified":True,"axioms":[],"total_tactic_calls":4},
            "apply r1":{"status":"kernel_closed","native_kernel":True,
                        "standalone_verified":True,"axioms":[],"total_tactic_calls":4},
            "apply r2":{"status":"rfl_rejected","native_kernel":False,
                        "standalone_verified":False,"axioms":[],"total_tactic_calls":2},
        }
        self.assertEqual(teacher_actions_from_verified_graph(
            self.legal,graph,remaining_calls=4,goal_is_kernel_closed=False),
            self.optimal)
        self.assertEqual(teacher_actions_from_verified_graph(
            self.legal,graph,remaining_calls=3,goal_is_kernel_closed=False),())
        with self.assertRaises(ValueError):
            teacher_actions_from_verified_graph(self.legal,{"apply r0":graph["apply r0"]},
                                                 remaining_calls=4,goal_is_kernel_closed=False)

    def test_batch_tie_is_one_presentation(self):
        rows=[{"root_id":"r","state_sha256":"s",
               "legal":self.legal,"optimal":self.optimal}]
        logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
        value=batch_objective(logits,rows,mode="set",model_seed=17)
        self.assertGreater(float(value),0)
        value.backward()
        self.assertEqual(tuple(logits.grad.shape),(1,3))


if __name__=="__main__":
    import unittest
    unittest.main()
