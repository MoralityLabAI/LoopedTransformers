"""NEC-03W native CPU intervention and next-visit rescue hook smoke.

A synthetic 12-pass recurrent model is used. All tests execute CPU only.
These validate instrumentation and known causal effects, not real SAE evidence.
"""
from __future__ import annotations

import unittest

from mechinterp.neuron_communities.nec03w_native import (
    DelayedRescue,MODES,native_forward,source_donor_modes,
)
from mechinterp.neuron_communities.nec03w_factorial import matched_pair_geometry


class NativeRecurrentHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch

    def toy_model(self):
        torch=self.torch

        class ToyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_down=torch.nn.Linear(64,32,bias=False)
                with torch.no_grad():
                    self.mlp_down.weight.zero_()
                    for j in range(32):
                        self.mlp_down.weight[j,j]=.3
                        self.mlp_down.weight[j,j+32]=.3

            def forward(self,x):
                x=x+self.mlp_down(torch.cat((x,x),dim=-1))
                return x

        class TinyLooper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block=ToyBlock()
                self.head=torch.nn.Linear(32,2,bias=False)
                with torch.no_grad():
                    self.head.weight.zero_()
                    self.head.weight[0,0]=1.
                    self.head.weight[1,1]=1.

            def logits(self,bits):
                mem=self.head.weight.new_zeros((len(bits),32))
                for v in range(12):
                    token=self.head.weight.new_zeros((len(bits),2,32))
                    token[:,0,0]=bits[:,v].float()
                    token[:,1,:]=mem
                    mem=self.block(token)[:,1,:]
                return self.head(mem)

        return TinyLooper().eval()

    def source_case(self,model):
        torch=self.torch
        sae=torch.nn.Module()
        sae.decoder=torch.nn.Linear(128,64,bias=True)
        with torch.no_grad():
            sae.decoder.weight.zero_()
            sae.decoder.bias.zero_()
            for i in range(16):
                sae.decoder.weight[i//2,i]=1.
        projected=[]
        with torch.no_grad():
            scale=torch.ones(64)
            down=model.block.mlp_down.weight
            for f in range(128):
                projected.append((down@(scale*sae.decoder.weight[:,f])).tolist())
        return sae,projected

    def test_single_visit_twelve_pass_all_hooks_and_delayed_restoration(self):
        torch=self.torch
        model=self.toy_model()
        bits=torch.ones((1,12),dtype=torch.long)
        control=native_forward(model,bits)
        donor=torch.ones((1,64))
        mask=torch.zeros(64)
        changed=native_forward(model,bits,donors={0:donor},gates={0:mask})
        self.assertEqual(tuple(changed.shape),(1,2))
        self.assertFalse(torch.allclose(changed,control))
        corrective=DelayedRescue(visit=1,direction32=torch.ones((1,32)))
        late=native_forward(model,bits,donors={0:donor},gates={0:mask},
                            rescue=corrective)
        self.assertFalse(torch.allclose(changed,late))
        self.assertEqual(len(model.block.mlp_down._forward_pre_hooks),0)
        self.assertEqual(len(model.block._forward_hooks),0)

    def test_exact_addback_and_distinct_late_rescue_are_testable(self):
        torch=self.torch
        model=self.toy_model()
        sae,cols=self.source_case(model)
        registered=matched_pair_geometry(
            [[i,1.] for i in range(16)],cols,(0,1,2,3),[[0,1],[2,3]],
            root=0,arm="composed",group=24,mask=0,visit=0,trials=512)
        self.assertTrue(registered["admissible"],registered)
        original={
            "model":model,"sae":sae,
            "mean":torch.zeros(64),"scale":torch.ones(64),
            "source_sites":{0:set([0])},
            "source_hubs":{},
        }
        z=torch.zeros((1,128))
        z[0,:16]=1.
        plans=source_donor_modes(
            original,{0:z},target_visit=0,matched_geometry=[registered])
        self.assertEqual(set(plans),set(MODES))
        x=torch.ones((1,12),dtype=torch.long)
        baseline=native_forward(model,x,**plans["baseline"])
        identity=native_forward(model,x,**plans["same_visit_exact_addback"])
        self.assertTrue(torch.allclose(baseline,identity,atol=1e-5))
        ab=native_forward(model,x,**plans["remove_AB"])
        self.assertFalse(torch.allclose(ab,baseline))
        late=native_forward(model,x,**plans["late_target_rescue"])
        sham=native_forward(model,x,**plans["late_matched_sham_rescue"])
        self.assertFalse(torch.allclose(late,sham))
        self.assertEqual(plans["late_target_rescue"]["rescue"].visit,1)
        self.assertEqual(len(model.block._forward_hooks),0)

    def test_invalid_donor_or_wrong_late_site_fails_closed(self):
        torch=self.torch
        model=self.toy_model()
        bits=torch.ones((1,12),dtype=torch.long)
        with self.assertRaises(ValueError):
            native_forward(model,bits,donors={12:torch.zeros((1,64))},
                           gates={12:torch.zeros(64)})
        with self.assertRaises(ValueError):
            native_forward(model,bits,rescue=DelayedRescue(
                visit=12,direction32=torch.zeros((1,32))))
        with self.assertRaises(ValueError):
            native_forward(model,bits,rescue=DelayedRescue(
                visit=1,direction32=torch.zeros((1,33))))
        self.assertEqual(len(model.block.mlp_down._forward_pre_hooks),0)

    def test_source_ineligible_no_forced_feature_geometry(self):
        torch=self.torch
        model=self.toy_model()
        sae,_=self.source_case(model)
        prior={"model":model,"sae":sae,"mean":torch.zeros(64),
               "scale":torch.ones(64),"source_sites":{},
               "source_hubs":{}}
        with self.assertRaisesRegex(ValueError,"only one originally registered"):
            source_donor_modes(prior,{},target_visit=0,matched_geometry=[])


if __name__=="__main__":
    unittest.main()
