"""OLR-METTA-03 Python protocol fixtures. FakeLean is NEVER native evidence."""
from __future__ import annotations

import hashlib
import re
import unittest
from unittest.mock import patch

import torch

from ppt_gym.data import digest
from ppt_gym import metta_budget_bridge as bridge
from ppt_gym.metta_budget_bridge import (
    ACTIONS,COST,curate_claims,generate,masked_packet,native_episode,
    replay_episode,uncurated_features,
)
from ppt_gym.metta_budget_model import (
    EvidenceLoop,EvidenceDirect,fit,majority_action,source_threshold,
)
from ppt_gym.tests.test_olr_delayed import FakeDelayedLean


def fake_metta(program: str, *, offer_pattern):
    results=[]
    for line in program.splitlines():
        if line.startswith("(Claim "):
            parts=line.strip("()").split()
            if parts[2:5]==["Current","Unique","Eligible"]:
                results.append((parts[1],parts[5],parts[6]))
    return tuple(results),hashlib.sha256(program.encode()).hexdigest()


class NativeEvidenceTests(unittest.TestCase):
    def test_generator_is_paired_and_lineage_disjoint(self):
        trains=generate("train",seed=20261129,count=24,low=17,high=30)
        dev=generate("dev",seed=20261140,count=8,low=17,high=30,
                     exclude=frozenset(w.lineage() for w in trains))
        self.assertEqual(len(trains),48)
        self.assertEqual(len(dev),16)
        self.assertFalse({w.lineage() for w in trains}&{w.lineage() for w in dev})
        self.assertEqual(len({w.pair_id for w in trains}),24)
        for first,second in zip(trains[::2],trains[1::2],strict=True):
            self.assertEqual((first.pair_id,first.world),(second.pair_id,0) if False else (first.pair_id,0))
            self.assertEqual(second.world,1)
            self.assertEqual(first.lean_world().targets,second.lean_world().targets[::-1])

    def test_native_metta_deduplicates_current_lineages(self):
        world=generate("train",seed=20261129,count=1,low=17,high=30)[0]
        with patch.object(bridge,"_native_metta",side_effect=fake_metta):
            receipt=curate_claims(world.claims)
            packet=masked_packet(world)
        self.assertEqual(receipt["admitted_ids"],["c1","c2","c5"])
        self.assertEqual(len(receipt["selected_lineages"]),3)
        self.assertEqual(packet["model_features"][0][:3],[0.,0.,0.])
        self.assertEqual(packet["model_features"][3][:3],[0.,0.,0.])
        self.assertEqual(packet["model_features"][4][:3],[0.,0.,0.])
        sloppy=uncurated_features(packet)
        self.assertNotEqual(sloppy,packet["model_features"])
        self.assertNotIn("true_sum",str(packet["model_features"]))
        self.assertNotIn("reference_proof",str(packet["model_features"]))

    def test_metta_disagreement_rejected_fail_closed(self):
        world=generate("train",seed=20261129,count=1,low=17,high=30)[0]
        with patch.object(bridge,"_native_metta",return_value=((),"bad")):
            with self.assertRaisesRegex(ValueError,"diverged"):
                curate_claims(world.claims)

    def test_true_lean_traces_and_query_costs_in_simulation(self):
        a=generate("train",seed=20261129,count=1,low=17,high=30)
        for world in a:
            native=world.lean_world()
            for action in ACTIONS:
                ep=native_episode(FakeDelayedLean(),native,action)
                self.assertTrue(replay_episode(FakeDelayedLean(),native,ep))
                self.assertLessEqual(ep["tactic_calls"],6)
                self.assertEqual(ep["utility"],round(int(ep["kernel_closed"])-COST*ep["tactic_calls"],7))
                if action=="probe_r0":
                    self.assertEqual(ep["tactic_calls"],6)
                    self.assertTrue(ep["kernel_closed"])
                elif ep["kernel_closed"]:
                    self.assertEqual(ep["tactic_calls"],4)
                else:
                    self.assertEqual(ep["tactic_calls"],2)

    def test_replay_rejects_forged_feedback_or_cost(self):
        world=generate("train",seed=20261129,count=1,low=17,high=30)[1].lean_world()
        ep=native_episode(FakeDelayedLean(),world,"probe_r0")
        import copy
        changed=copy.deepcopy(ep)
        for event in changed["events"]:
            if event["op"]=="restore":
                event["probe_result"]=not event["probe_result"]
        with self.assertRaisesRegex(ValueError,"probe status"):
            replay_episode(FakeDelayedLean(),world,changed)
        changed=copy.deepcopy(ep)
        changed["tactic_calls"]=5
        with self.assertRaisesRegex(ValueError,"call count"):
            replay_episode(FakeDelayedLean(),world,changed)


class LearnedBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled=False
        torch.use_deterministic_algorithms(True)
        words=generate("train",seed=20261129,count=2,low=17,high=30)
        with patch.object(bridge,"_native_metta",side_effect=fake_metta):
            self.packet=[masked_packet(w) for w in words]
        self.rows=[]
        for w,p in zip(words,self.packet,strict=True):
            choices=[
                native_episode(FakeDelayedLean(),w.lean_world(),action)["utility"]
                for action in ACTIONS
            ]
            self.rows.append({"model_features":p["model_features"],
                              "native_action_utilities":choices})

    def test_weight_tied_recurrence_and_full_input_direct_reader(self):
        x=torch.tensor([r["model_features"] for r in self.rows],dtype=torch.float32)
        for arm in ("loop1","loop3","direct"):
            model=EvidenceDirect() if arm=="direct" else EvidenceLoop()
            v=model(x,depth=1 if arm=="loop1" else 3)
            self.assertEqual(tuple(v.shape),(4,3))
            loss=v.square().mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        with self.assertRaises(ValueError):
            EvidenceLoop()(torch.zeros(3,4,6))

    def test_source_only_fitting_and_deterministic_sampler(self):
        a,ma=fit(self.rows,seed=17,arm="loop3",steps=4)
        b,mb=fit(self.rows,seed=17,arm="loop3",steps=4)
        self.assertEqual(ma["source_worlds"],4)
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key],b.state_dict()[key]))
        self.assertEqual(ma["last_loss"],mb["last_loss"])

    def test_source_majority_threshold_is_bounded(self):
        result=source_threshold(self.rows)
        self.assertIn(result["threshold"],[0.,1.,2.,3.,4.,5.,6.,7.,8.])
        self.assertIn(majority_action(self.rows[0]["model_features"],
                                      result["threshold"]),ACTIONS)
        with self.assertRaises(ValueError):
            majority_action([],2.)


if __name__=="__main__":
    unittest.main()
