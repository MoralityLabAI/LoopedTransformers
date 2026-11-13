"""FC-04 halting contracts, source-only training, exact release gate."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc04_adaptive as fc04
import fc03_compare as fc03


class ConstantAtomLogits(nn.Module):
    """Test fixture, never a trainable or evaluated model output."""
    def __init__(self,values):
        super().__init__()
        self.register_buffer("logits",torch.tensor(
            [8. if x else -8. for x in values]+[-8.]*(fc04.fs.MAX_A-len(values))))
    def forward(self,atom_hidden):
        return self.logits[None,:,None].expand(atom_hidden.shape[0],-1,-1)


class FC04AdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("FC02_ROOT")
        if not root:
            raise RuntimeError("FC02_ROOT must reference pinned FC-02 training checkout")
        cls.train,cls.dev,cls.hash,cls.verify=fc04.load_train_only(Path(root))
        torch.set_num_threads(2)

    def test_preregistered_settings_and_source_isolation(self):
        self.assertEqual(fc04.PREREG_SHA,"ce7190129617d8387a7acf4ceed3e8d0570cfce4")
        self.assertEqual(fc04.THRESHOLDS,(.55,.70,.85,.93,.98))
        self.assertEqual(fc04.STAGES,(4,6,8,10,12))
        self.assertEqual(self.hash,fc03.FC02_DATA_SHA)
        self.assertEqual((len(self.train),len(self.dev)),(1152,192))
        self.assertFalse(set(x["family_id"] for x in self.train) &
                         set(x["family_id"] for x in self.dev))

    def test_public_projection_excludes_gold_and_metadata(self):
        t=deepcopy(self.dev[0])
        bad=deepcopy(t)
        bad["label"]="refuted" if t["label"]=="established" else "established"
        bad["reachable"]=[]
        bad["countermodel"]={"fake": True}
        bad["split"]="test_deep"
        bad["family_id"]="changed"
        a=fc04.fs.encode([fc04._public_graph(t)])
        b=fc04.fs.encode([fc04._public_graph(bad)])
        for key in ("a","r","prem","head","goal_idx"):
            self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_exact_countermodel_release_stops_early(self):
        t=next(x for x in self.dev if x["label"]=="refuted")
        valid_assignment=t["countermodel"]["assignment"]
        logits=[valid_assignment[name] for name in t["atoms"]]
        model=fc03.make_model("tied4_width24")
        model.atom_out=ConstantAtomLogits(logits)
        response=fc04.run_one(model,t,.93,"adaptive",type(self).verify)
        self.assertEqual(response["outcome"],"refuted_certified")
        self.assertEqual(response["passes"],4)
        self.assertEqual(response["checker_calls"],1)
        self.assertTrue(response["countermodel_verified"])

    def test_stable_positive_two_checkpoints(self):
        t=next(x for x in self.dev if x["label"]=="established")
        model=fc03.make_model("tied4_width24")
        model.atom_out=ConstantAtomLogits([True]*len(t["atoms"]))
        positive=fc04.run_one(model,t,.93,"adaptive",type(self).verify)
        self.assertEqual(positive["outcome"],"established_provisional")
        self.assertEqual(positive["passes"],6)
        self.assertEqual(positive["checker_calls"],0)
        negative_only=fc04.run_one(model,t,.93,"cert_only_first",type(self).verify)
        self.assertEqual(negative_only["outcome"],"abstain")
        self.assertEqual(negative_only["passes"],12)

    def test_never_release_unverified_negative(self):
        t=next(x for x in self.dev if x["label"]=="established")
        model=fc03.make_model("tied4_width24")
        model.atom_out=ConstantAtomLogits([False]*len(t["atoms"]))
        answer=fc04.run_one(model,t,.93,"adaptive",type(self).verify)
        self.assertEqual(answer["outcome"],"abstain")
        self.assertEqual(answer["passes"],12)
        self.assertGreater(answer["checker_calls"],0)
        self.assertFalse(answer["countermodel_verified"])

    def test_grid_selection_and_stopping_reason(self):
        frames=[{"passes":i,"goal_probability":p,"countermodel_verified":False}
                for i,p in zip(fc04.STAGES,(.5,.80,.95,.97,.99))]
        self.assertEqual(fc04.decision_from_frames(frames,.93,"adaptive"),
                         {"outcome":"established_provisional","passes":10})
        self.assertEqual(fc04.decision_from_frames(frames,.98,"adaptive"),
                         {"outcome":"abstain","passes":12})
        verified=deepcopy(frames);verified[0]["countermodel_verified"]=True
        self.assertEqual(fc04.decision_from_frames(verified,.55,"adaptive"),
                         {"outcome":"refuted_certified","passes":4})

    def test_scoring_rejects_impossible_certificate(self):
        example=next(x for x in self.dev if x["label"]=="established")
        with self.assertRaises(AssertionError):
            fc04.score([example],[{"case_id":example["case_id"],
              "outcome":"refuted_certified","passes":4,"checker_calls":1,
              "model_forward_ms":1.,"checker_ms":1.}])

if __name__=="__main__":
    unittest.main()
