"""NW-01 pre-model tests: NOR42 original SHA, paired information and causal API.

All tests are synthetic or read only the previously published source/grade
NOR42 first artifact; no new NOR42 experiment identity is generated.
"""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.nor42_recurrent_witness_v0 import data as x
from experiments.nor42_recurrent_witness_v0 import model as m
from experiments.nor42_recurrent_witness_v0 import run


class NW01PublicAndRecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        archive=os.environ.get("NW01_NOR42_ARCHIVE")
        cls.roots=x.validate_archive(Path(archive)) if archive else None

    def test_training_root_public_feature_contract(self):
        roots=x.synthetic_public_roots(96)
        self.assertEqual(len(roots),96)
        self.assertEqual(len({r.identity for r in roots}),96)
        self.assertTrue(all(r.identity.startswith("NW01-synthetic-fit-") for r in roots))
        self.assertTrue(all(len(r.features)==x.PUBLIC_DIM for r in roots))
        self.assertTrue(all(r.hidden_beta in x.AMPLITUDES for r in roots))
        self.assertEqual(x.feature_manifest(roots),
                         x.feature_manifest(x.synthetic_public_roots(96)))
        self.assertTrue(all(r.replicate==-1 for r in roots))

    def test_locked_archived_64_source_pairs_when_available(self):
        if self.roots is None:self.skipTest("set NW01_NOR42_ARCHIVE to verify original first SHA")
        self.assertEqual(len(self.roots),64)
        self.assertEqual({r.replicate for r in self.roots},set(range(100,116)))
        self.assertEqual({r.family for r in self.roots},set(x.FAMILIES))
        for rep in x.REPS:
            self.assertEqual({(r.degree,r.family) for r in self.roots
                              if r.replicate==rep},
                             {(d,f) for d in (512,1024) for f in x.FAMILIES})
        self.assertTrue(all(len(r.features)==x.PUBLIC_DIM for r in self.roots))
        self.assertFalse(set(r.identity for r in self.roots)&
                         set(r.identity for r in x.synthetic_public_roots()))

    def test_public_pair_exact_state_equality_at_eight_passes(self):
        torch.manual_seed(11)
        verifier=m.RecurrentWitnessVerifier(x.PUBLIC_DIM)
        features=torch.randn(7,x.PUBLIC_DIM)
        beta=torch.tensor([-2.,1.,2.,-1.,1.,2.,-2.])
        public=m.root_paired_public(features)
        witness=m.witness_vectors(beta,"public")
        out=verifier(public,witness)
        self.assertEqual(tuple(out.states.shape),(14,8,48))
        self.assertTrue(torch.equal(out.states[0::2],out.states[1::2]))
        self.assertTrue(torch.equal(out.logits[0::2],out.logits[1::2]))
        self.assertTrue(torch.equal(witness,torch.zeros_like(witness)))

    def test_label_independent_sham_witness_still_exact_pair_equality(self):
        roots=x.synthetic_public_roots(8)
        w=run.sham_witness(roots)
        self.assertTrue(torch.equal(w[0::2],w[1::2]))
        self.assertTrue(torch.all(w[:,0]==1))
        torch.manual_seed(23)
        verifier=m.RecurrentWitnessVerifier(x.PUBLIC_DIM)
        pub=m.root_paired_public(torch.tensor([r.features for r in roots]))
        result=verifier(pub,w)
        self.assertTrue(torch.equal(result.states[0::2],result.states[1::2]))

    def test_opposite_world_donor_swaps_exact_final_readouts(self):
        torch.manual_seed(47)
        verifier=m.RecurrentWitnessVerifier(x.PUBLIC_DIM)
        roots=x.synthetic_public_roots(7)
        public=m.root_paired_public(torch.tensor([r.features for r in roots]))
        beta=torch.tensor([float(r.hidden_beta) for r in roots])
        auth=m.witness_vectors(beta,"authentic")
        no_patch=verifier(public,auth)
        for depth in (1,2,4,6):
            t=depth-1
            state=no_patch.states[:,t].detach()
            donor=m.matched_world_donor(state)
            swapped=verifier(public,auth,patch_after=t,donor_state=donor)
            expected=no_patch.logits[:,-1].reshape(-1,2).flip(1).reshape(-1)
            self.assertLess(float((swapped.logits[:,-1]-expected).abs().max()),
                            1e-6)
            same=verifier(public,auth,patch_after=t,donor_state=state)
            self.assertLess(float((same.logits[:,-1]-no_patch.logits[:,-1]).abs().max()),
                            1e-6)
        with self.assertRaises(ValueError):
            verifier(public,auth,patch_after=3)

    def test_paired_world_accuracy_and_brier_floor(self):
        for p in (.1,.3,.49,.5,.8,.99):
            probs=torch.full((128,),p)
            y=m.root_paired_targets(64)
            got=run.paired_metric(probs,y)
            self.assertEqual(got["accuracy"],.5)
            self.assertGreaterEqual(got["brier"],.25-1e-6)

    def test_source_paired_balanced_training_batch(self):
        roots=x.synthetic_public_roots(128)
        features=torch.tensor([r.features for r in roots])
        beta=torch.tensor([r.hidden_beta for r in roots],dtype=torch.float32)
        x0,w,y=run.training_batch(features,beta,
                                 torch.Generator().manual_seed(712))
        self.assertEqual(tuple(x0.shape),(128,x.PUBLIC_DIM))
        self.assertEqual(tuple(w.shape),(128,3))
        self.assertEqual(tuple(y.shape),(128,))
        self.assertTrue(torch.equal(x0[0::4],x0[1::4]))
        self.assertTrue(torch.equal(x0[1::4],x0[2::4]))
        self.assertTrue(torch.equal(x0[2::4],x0[3::4]))
        self.assertEqual(y.mean().item(),.5)
        self.assertTrue(torch.all(w[0::4]==0))
        self.assertTrue(torch.all(w[1::4]==0))
        self.assertTrue(torch.all(w[2::4,0]==1))
        self.assertTrue(torch.all(w[2::4,1:]==0))
        self.assertTrue(torch.all(w[3::4,0]==1))
        self.assertTrue(torch.all(w[3::4,2]>0))

    def test_exact_train_seed_replay_two_optimizer_updates(self):
        roots=x.synthetic_public_roots(32)
        features=torch.tensor([r.features for r in roots])
        beta=torch.tensor([r.hidden_beta for r in roots],dtype=torch.float32)
        def twice():
            torch.manual_seed(11)
            net=m.RecurrentWitnessVerifier(x.PUBLIC_DIM)
            optim=torch.optim.AdamW(net.parameters(),lr=.002)
            rng=torch.Generator().manual_seed(11+390002)
            for _ in range(2):
                x0,w,y=run.training_batch(features,beta,rng)
                optim.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(
                    net(x0,w).logits[:,-1],y)
                loss.backward();optim.step()
            return net.state_dict()
        first=twice()
        second=twice()
        self.assertTrue(all(torch.equal(first[k],second[k]) for k in first))


if __name__=="__main__":
    unittest.main()
