"""LGA-02 source-independent CPU regression tests.

Check the exact terminal autodiff through the remaining tied block, SAE
physical decoder projection, whole-sequence bootstrap and source-matched
native causal interventions before opening the prospective fresh test run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope05.fit_and_freeze import (
    TopKSAE,fit_sae,train_and_freeze,load_frozen,protocol_sha
)
from mechinterp.loopscope05.terminal_sae_test import (
    terminal_gradients,active_components,bootstrap,geometry_batch,
    fixed_new_test,zero_edit_and_causal,causal_stats,make_model,make_sae,
    run
)


class TerminalGradientAndSAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registered_source_and_new_split_independent(self):
        self.assertEqual(len(protocol_sha()),64)
        x,y,manifest=fixed_new_test(smoke=True)
        self.assertEqual(len(x),16)
        self.assertEqual(manifest["seed"],31044)
        self.assertTrue(manifest["no_exact_sequence_overlap"])
        self.assertTrue(torch.equal(y,x.roll(1,1)))

    def test_future_gradient_includes_cross_token_recurrence(self):
        torch.manual_seed(123)
        model=TinyRecursiveModel().eval().requires_grad_(False)
        x,y=fixed_data(2,31044)
        states,grads=terminal_gradients(model,x,y)
        self.assertEqual(len(states),5)
        self.assertEqual(len(grads),4)
        self.assertTrue(all(t.shape==(2,81,48) for t in grads))
        # Recompute a true directional finite difference of terminal summed CE.
        k=1;token=10
        delta=torch.randn(48)*.2
        h=states[k].clone()
        def loss(add):
            z=h.clone()
            z[0,token]+=add*delta
            for _ in range(4-k):
                z=model.block(z)
            return F.cross_entropy(model.head(z).reshape(-1,10),
                                   y.reshape(-1),reduction="sum")
        numeric=float((loss(1e-3)-loss(-1e-3))/.002)
        predicted=float(grads[k][0,token]@delta)
        self.assertAlmostEqual(numeric,predicted,delta=.025)

    def test_projected_sae_physically_orthogonal_and_norm_preserved(self):
        torch.manual_seed(42)
        sae=TopKSAE()
        mu=torch.randn(48)*.1
        std=torch.rand(48)+.5
        h=torch.randn(7,81,48)
        v=torch.randn(7,81,48)
        grad=torch.randn_like(v)
        proj=active_components(sae,mu,std,h,v,grad)
        p=proj["active"];q=proj["remainder"]
        self.assertEqual(tuple(p.shape),(7,81,48))
        self.assertTrue(torch.allclose(p+q,v,atol=1e-5))
        dot=(p*q).sum(-1)
        self.assertLess(float(dot.abs().mean()),1e-4)
        self.assertTrue(bool((proj["active_energy_fraction"]>=0).all()))
        self.assertTrue(bool((proj["active_energy_fraction"]<1.0001).all()))
        self.assertEqual(tuple(proj["admitted"].shape),(7,81))

    def test_rank_deficiency_flags_ineligible(self):
        sae=TopKSAE()
        with torch.no_grad():
            sae.decoder.weight.zero_()
            sae.encoder.weight.zero_()
            sae.encoder.bias.fill_(1.)
        h=torch.randn(5,48)
        v=torch.randn(5,48)
        projected=active_components(sae,torch.zeros(48),torch.ones(48),h,v)
        self.assertFalse(bool(projected["admitted"].any()))

    def test_train_only_SAE_uses_validation_checkpoint(self):
        generator=torch.Generator().manual_seed(71)
        train=torch.randn(256,48,generator=generator)
        val=torch.randn(128,48,generator=generator)
        sae,mu,std,info=fit_sae(train,val,init_seed=71001,steps=10)
        self.assertEqual(info["sae_init_seed"],71001)
        self.assertEqual(info["train_tokens"],256)
        self.assertEqual(info["val_tokens"],128)
        self.assertTrue(info["best_step"] in (1,10))
        self.assertEqual(len(info["sae_tensor_sha256"]),64)
        self.assertEqual(len(info["normalizer_sha256"]),64)
        self.assertEqual(tuple(sae(torch.zeros(2,48))[0].shape),(2,48))

    def test_whole_sequence_bootstrap_determinism(self):
        vals=[-.2,.3,.8,1.1,.4,.7]
        first=bootstrap(vals,103,draws=200)
        second=bootstrap(vals,103,draws=200)
        self.assertEqual(first,second)
        self.assertLess(first[0],first[1])

    def test_native_causal_uses_full_remaining_shared_block(self):
        model=TinyRecursiveModel().eval().requires_grad_(False)
        sae=TopKSAE().eval().requires_grad_(False)
        sae_info={k:(sae,torch.zeros(48),torch.ones(48)) for k in (1,2)}
        x,y=fixed_data(2,31044)
        rows=zero_edit_and_causal(model,sae_info,x,y,seed_id=103,smoke=True)
        self.assertEqual(set(rows),{"1","2"})
        self.assertEqual(len(rows["1"]),2)
        self.assertEqual(len(rows["1"][0]["goals"]),8)
        for k in ("1","2"):
            for sample in rows[k]:
                for goal in sample["goals"]:
                    if goal["admitted"]:
                        self.assertEqual(set(goal["arms"]),{
                            "full","sae_active","sae_remainder","head_null","random"})
                        for entry in goal["arms"].values():
                            self.assertLess(abs(entry["dose"]-goal["dose"]),1e-5)
                            self.assertEqual(len(entry["native_goal_full_logits"]),10)
                            self.assertAlmostEqual(
                                entry["global_CE_change"],
                                entry["native_global_mean_CE"]-
                                entry["clean_global_mean_CE"],places=5)

    def test_smoke_fits_and_scores_only_new_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            training=train_and_freeze(root/"bundle.pt",root/"train.json",smoke=True)
            self.assertFalse(training["new_test_generated"])
            self.assertTrue(training["teacher_receipts"]["103"]["original_source_reproduction_pass"])
            loaded,bundle=load_frozen(root/"bundle.pt",root/"train.json")
            self.assertEqual(loaded["source_bundle_sha256"],training["source_bundle_sha256"])
            result=run(root/"bundle.pt",root/"train.json",
                       root/"test.json",smoke=True)
            self.assertEqual(result["registered_primary_status"],"PILOT_ONLY")
            self.assertEqual(result["fresh_test"]["n_sequences"],16)
            self.assertEqual(len(result["results"]["103"]["gradient_by_loop"]),4)
            self.assertEqual(set(result["results"]["103"]["native_causal"]),{"1","2"})


if __name__=="__main__":
    unittest.main()
