"""Controls for the native readout-silent intervention test, without training."""
import unittest
import numpy as np
import torch
from research_gym.scripts.loopscope_toy import LoopedChase,make_batch
from research_gym.scripts.loopscope_readout_silent import (ARMS,embedding,step,trace,
                  make_patches,select_pairs,parity,summarize_arrays)

class SilentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(45)
        self.model=LoopedChase(4,16,1,2).eval()
        self.tokens,self.hops=make_batch(4,8,531,8)
    @torch.inference_mode()
    def test_native_replay(self):
        e,h=trace(self.model,self.tokens,8)
        self.assertTrue(torch.equal(h[1:,:,-1],self.model(self.tokens,8)))
        self.assertTrue(torch.equal(step(self.model,h[4],e),h[5]))
    @torch.inference_mode()
    def test_all_silent_arms_preserve_entire_head(self):
        e,h=trace(self.model,self.tokens,8)
        patches,_=make_patches(h[6],h[2],4,7)
        baseline=self.model.readout(h[6,:,-1])
        got=self.model.readout(patches[:,:,-1])
        self.assertEqual(float((got[:-1]-baseline).abs().max()),0.)
        self.assertFalse(torch.equal(patches[3,:,:4],h[6,:,:4]))
    @torch.inference_mode()
    def test_noise_is_matched_per_item(self):
        _,h=trace(self.model,self.tokens,8)
        patches,norm=make_patches(h[6],h[2],4,7)
        norms=torch.linalg.vector_norm((patches-h[6]).flatten(2),dim=-1)
        for i in (1,2):self.assertTrue(torch.allclose(norms[i],.25*norm,atol=2e-6))
        for i in (3,4,5):self.assertTrue(torch.allclose(norms[i],norm,atol=2e-6))
    @torch.inference_mode()
    def test_layernorm_null_and_sham(self):
        e,h=trace(self.model,self.tokens,8)
        patches,_=make_patches(h[6],h[2],4,7)
        sham=step(self.model,patches[0],e)
        null=step(self.model,patches[5],e)
        self.assertTrue(torch.equal(sham,h[7]))
        self.assertTrue(torch.allclose(null,sham,atol=1e-5,rtol=1e-5))
    @torch.inference_mode()
    def test_full_donor_positive_replay(self):
        e,h=trace(self.model,self.tokens,8)
        patches,_=make_patches(h[6],h[2],4,7)
        self.assertTrue(torch.equal(step(self.model,patches[-1],e),h[3]))
    def test_selection_ignores_unobserved_futures(self):
        oracle=np.zeros((12,12),dtype=int);pred=oracle.copy()
        a=select_pairs(pred,oracle,6,8,43)
        pred[6:]=1;oracle[6:]=2
        self.assertEqual(a,select_pairs(pred,oracle,6,8,43))
        rows=a[0];self.assertEqual(len(set(x[0] for x in rows)),len(rows))
    def test_wrong_endpoints_excluded(self):
        oracle=np.zeros((12,4),dtype=int);pred=np.ones_like(oracle)
        self.assertEqual(select_pairs(pred,oracle,6,8,43),([],0,0))
    def test_input_validation(self):
        with self.assertRaises(ValueError):make_patches(torch.zeros(3,4,8),torch.ones(3,4,8),4,1)
        with self.assertRaises(RuntimeError):parity(torch.ones(4,2),torch.zeros(4,2),'mutation')
    @torch.inference_mode()
    def test_directions_reproducible(self):
        _,h=trace(self.model,self.tokens,8)
        a,_=make_patches(h[6],h[2],4,7);b,_=make_patches(h[6],h[2],4,7)
        self.assertTrue(torch.equal(a,b))
if __name__=='__main__':unittest.main()
