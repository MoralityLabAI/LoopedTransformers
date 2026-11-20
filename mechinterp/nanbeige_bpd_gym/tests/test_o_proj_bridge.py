from types import SimpleNamespace
import unittest
import torch
from torch import nn
from mechinterp.nanbeige_bpd_gym.o_proj_bridge import bridge_o_proj


class RMS(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor([.8,1.1,.9],dtype=torch.float64),requires_grad=False)
    def forward(self,x):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+.01)*self.weight


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj=nn.Linear(3,4,bias=False,dtype=torch.float64)
        self.up_proj=nn.Linear(3,4,bias=False,dtype=torch.float64)
        self.down_proj=nn.Linear(4,3,bias=False,dtype=torch.float64)
        self.requires_grad_(False)
    def forward(self,x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))


class OProjectionBridgeTests(unittest.TestCase):
    def layer(self):
        torch.manual_seed(13)
        return SimpleNamespace(post_attention_layernorm=RMS(),mlp=MLP(),enable_hyper_connection=False)

    def test_matches_independent_analytic_RMS_SwiGLU_jacobian_and_finite_writes(self):
        layer=self.layer();residual=torch.tensor([[[.1,-.1,.2]]],dtype=torch.float64);attention=torch.tensor([[[.6,-.3,.7]]],dtype=torch.float64);x=residual+attention
        rows=torch.tensor([[1.,.2,-.3],[-.2,.8,.4]],dtype=torch.float64)
        before=[p.clone() for m in (layer.post_attention_layernorm,layer.mlp) for p in m.parameters()]
        r=bridge_o_proj(layer,residual,attention,rows,actual_pre_mlp_state=x,relative_doses=(.0001,))
        v=x.flatten();inv=torch.rsqrt(v.square().mean()+.01);D=3
        norm_jac=torch.diag(layer.post_attention_layernorm.weight)@(inv*torch.eye(D,dtype=v.dtype)-inv**3*torch.outer(v,v)/D)
        norm=v*inv*layer.post_attention_layernorm.weight
        gate=layer.mlp.gate_proj.weight@norm;up=layer.mlp.up_proj.weight@norm;sig=torch.sigmoid(gate)
        silu=gate*sig;derivative=sig+gate*sig*(1-sig)
        mlp_jac=layer.mlp.down_proj.weight@(torch.diag(derivative*up)@layer.mlp.gate_proj.weight+torch.diag(silu)@layer.mlp.up_proj.weight)
        jac=torch.eye(D,dtype=v.dtype)+mlp_jac@norm_jac
        torch.testing.assert_close(r['physical_JVP_rows'],rows@jac.T,rtol=1e-12,atol=1e-12)
        self.assertGreater(float((r['physical_JVP_rows']-rows).norm()),.001)
        self.assertLess(max(z['central_vs_autograd_relative_l2'] for z in r['finite_writes']),1e-7)
        after=[p for m in (layer.post_attention_layernorm,layer.mlp) for p in m.parameters()]
        for p,q in zip(before,after):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_rejects_ambiguous_coordinates_live_hooks_and_training(self):
        layer=self.layer();x=torch.ones(1,1,3,dtype=torch.float64);directions=torch.eye(3,dtype=torch.float64)
        with self.assertRaises(ValueError):bridge_o_proj(layer,x,x,torch.eye(4,dtype=torch.float64),actual_pre_mlp_state=2*x)
        with self.assertRaises(ValueError):bridge_o_proj(layer,x,x,directions,actual_pre_mlp_state=x)
        hook=layer.mlp.register_forward_hook(lambda *args:None)
        with self.assertRaises(ValueError):bridge_o_proj(layer,x,x,directions,actual_pre_mlp_state=2*x)
        hook.remove();layer.mlp.gate_proj.weight.requires_grad_(True)
        with self.assertRaises(ValueError):bridge_o_proj(layer,x,x,directions,actual_pre_mlp_state=2*x)
        with torch.inference_mode():
            with self.assertRaises(RuntimeError):bridge_o_proj(self.layer(),x,x,directions,actual_pre_mlp_state=2*x)

    def test_preserves_full_prefix_shape_with_last_token_only_tangent(self):
        layer=self.layer();residual=torch.ones(1,5,3,dtype=torch.float64)*.1;output=torch.randn(1,5,3,dtype=torch.float64)
        directions=torch.eye(3,dtype=torch.float64)
        r=bridge_o_proj(layer,residual,output,directions,actual_pre_mlp_state=residual+output,relative_doses=(.01,))
        self.assertEqual(r['native_remainder'].shape,(1,5,3))
        self.assertEqual(r['physical_JVP_rows'].shape,(3,3))
        self.assertEqual(len(r['finite_writes']),3)


if __name__=='__main__':unittest.main()
