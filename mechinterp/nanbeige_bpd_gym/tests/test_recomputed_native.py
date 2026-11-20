from types import SimpleNamespace
import unittest
import torch
from torch import nn
from mechinterp.nanbeige_bpd_gym.recomputed_native import ResidentFrozenHead,ResidentSavedTensors,native_pairs


class RecomputedGradientTests(unittest.TestCase):
    def test_resident_head_derivative_and_frozen_version_guard(self):
        torch.manual_seed(9);h=torch.randn(1,1,7,requires_grad=True);weight=torch.randn(13,7);gradient=torch.randn(1,1,13)
        out=ResidentFrozenHead.apply(h,weight,4)
        torch.testing.assert_close(out,torch.nn.functional.linear(h,weight),rtol=1e-6,atol=1e-6)
        out.backward(gradient);torch.testing.assert_close(h.grad,gradient@weight,rtol=1e-6,atol=1e-6)
        h=torch.randn(1,1,7,requires_grad=True);out=ResidentFrozenHead.apply(h,weight,4)
        weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,'changed before backward'):out.sum().backward()

    def test_checkpoint_excludes_stateful_selected_hook_and_preserves_gradient(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(7,7,bias=False);self.calls=0
            def forward(self,x,**kwargs):
                self.calls+=1;return (x+torch.tanh(self.linear(x)),)
        torch.manual_seed(9);layers=nn.ModuleList([Layer() for _ in range(3)]);model=nn.ModuleDict({'layers':layers});model.requires_grad_(False)
        factor=nn.Parameter(torch.tensor(.2));hook=layers[1].register_forward_hook(lambda m,args,out:(out[0]+factor*args[0],))
        loop=SimpleNamespace(model=model,layers=layers,n=3,n_gpu_layers=3,stage_layer=None,phase=0,active_layer=-1,base=SimpleNamespace(norm=nn.Identity()))
        context=dict(mask=None,pos=torch.arange(5)[None,:],cache_position=torch.arange(5))
        x=torch.randn(1,5,7,requires_grad=True)
        ordinary=native_pairs(loop,[(x,context)],None,{1},recompute=False)[0];ordinary.square().mean().backward()
        gx=x.grad.clone();gf=factor.grad.clone();expected=ordinary.detach().clone();x.grad=None;factor.grad=None
        for layer in layers:layer.calls=0
        owner=ResidentSavedTensors(loop,1024**2)
        changed=native_pairs(loop,[(x,context)],owner,{1},recompute=True)[0];changed.square().mean().backward()
        torch.testing.assert_close(changed,expected,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,gx,rtol=1e-6,atol=1e-6);torch.testing.assert_close(factor.grad,gf,rtol=1e-6,atol=1e-6)
        self.assertEqual(layers[1].calls,1);self.assertEqual(layers[0].calls,2);self.assertEqual(layers[2].calls,2)
        self.assertEqual(owner.checkpointed_blocks,2);self.assertEqual(owner.recomputed_blocks,2);hook.remove()


if __name__=='__main__':unittest.main()
