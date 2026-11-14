"""FC-09 train/test isolation and frozen dual-certificate policy contracts."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc09_trial as trial
import fc06_joint as actorlib


class FC09TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("FC09_TRAIN_ROOT")
        if not root:
            raise RuntimeError("FC09_TRAIN_ROOT must be the exact old test-free checkout")
        cls.root=Path(root)
        cls.train,cls.dev,cls.digest=trial._load_train(cls.root)
        torch.set_num_threads(2)

    def test_source_is_immutable_train_only_and_family_disjoint(self):
        self.assertEqual(trial.PREREG_SHA,
                         "37ceea8c6e5310fb9fb6fb43fd2aab1e9f3210a4")
        self.assertEqual(self.digest,trial.TRAIN_DATA_SHA)
        self.assertEqual((len(self.train),len(self.dev)),(768,192))
        self.assertFalse((self.root/"research/refutation_bench/fc09/test_data.py").exists())
        self.assertFalse((self.root/"research/refutation_bench/fc09/controls.py").exists())
        self.assertFalse({x["family_id"] for x in self.train}&
                         {x["family_id"] for x in self.dev})

    def test_only_graph_inputs_reach_actor(self):
        task=next(x for x in self.train if x["label"]=="established")
        corrupted=deepcopy(task)
        corrupted.update(label="refuted",min_steps=999,reachable=[],
                         shortest_trace=[999],countermodel={"poison":True},
                         split="test_deep",family_id="other")
        for known in (set(task["facts"]),set(task["facts"])|{task["goal"]}):
            a,am=actorlib._source_input(task,known)
            b,bm=actorlib._source_input(corrupted,known)
            for key in a:
                self.assertTrue(torch.equal(a[key],b[key]),key)
            self.assertTrue(torch.equal(am,bm))

    def test_every_training_positive_has_true_conjunctive_reference(self):
        from research.refutation_bench.fc05 import proofs
        for item in self.train:
            if item["label"]!="established":
                continue
            self.assertEqual(item["min_steps"],item["depth"])
            self.assertTrue(all(len(r["premises"])>=2 for r in item["rules"]
                                if r["head"]==item["goal"]))
            self.assertTrue(proofs.verify_trace(
                item,item["shortest_trace"],budget=5)["verified"])

    def test_actor_joint_gradient_and_expected_parameter_count(self):
        model=actorlib.JointActor()
        self.assertEqual(sum(p.numel() for p in model.parameters()),16226)
        item=next(x for x in self.train if x["label"]=="established")
        features,mask=actorlib._source_input(item,set(item["facts"]))
        scores=model(features)
        label=item["shortest_trace"][0]
        self.assertTrue(mask[label])
        loss=torch.nn.functional.cross_entropy(
           scores.masked_fill(~mask[None],-1e4),torch.tensor([label]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.trunk.cell.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.head.parameters()))

    def test_symbolic_result_releases_only_independently_checked_proof(self):
        from research.refutation_bench.fc05 import proofs
        good=next(x for x in self.train if x["label"]=="established")
        gate={"outcome":"established_provisional","reason":"test",
              "passes":6,"model_forward_ms":1.,"checker_ms":0.,
              "checker_calls":0,"certificate":None}
        bad=trial._symbolic_result(good,gate,"andor_cpu_matched",
              {"trace":[],"search_ms":.2,"states_popped":1,
               "allowed_ms":.1,"budget_overrun_ms":.1,"timed_out":True},proofs)
        self.assertEqual(bad["outcome"],"abstain")
        self.assertEqual(bad["symbolic_allowed_ms"],.1)
        self.assertEqual(bad["symbolic_budget_overrun_ms"],.1)
        valid=trial._symbolic_result(good,gate,"andor_cpu_matched",
              {"trace":good["shortest_trace"],"search_ms":.2,"states_popped":2,
               "allowed_ms":.3,"budget_overrun_ms":0.,
               "timed_out":False},proofs)
        self.assertEqual(valid["outcome"],"established_certified")
        self.assertTrue(valid["positive_checked"])

    def test_paired_counts_and_failing_ids(self):
        records=[{"case_id":"a","label":"established"},
                 {"case_id":"b","label":"established"}]
        learned=[{"case_id":"a","outcome":"established_certified"},
                 {"case_id":"b","outcome":"abstain"}]
        other=[{"case_id":"a","outcome":"abstain"},
               {"case_id":"b","outcome":"established_certified"}]
        self.assertEqual(trial._paired(records,learned,other),
             {"learned_only":1,"symbolic_only":1,"both_certified":0,
              "net_neural":0})
        with self.assertRaises(AssertionError):
            trial._paired(records,learned,[{"case_id":"INVALID","outcome":"abstain"},other[1]])


if __name__=="__main__":
    unittest.main()
