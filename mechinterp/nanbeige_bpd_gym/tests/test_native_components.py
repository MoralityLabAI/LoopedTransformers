"""Engineering parity only: no pretrained weights or task performance claims."""
import os
import copy
from types import SimpleNamespace
from pathlib import Path
import unittest
import torch
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd,component_class,OffloadedHead,attach_native_components


class NativeComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(os.environ.get('NAN_BPD_SOURCE',str(Path(__file__).resolve().parents[4]/'BPD')))
        if not root.exists():raise RuntimeError('Pinned BPD checkout required; set NAN_BPD_SOURCE')
        cls.bpd_root=root;cls.framework=load_bpd(root);cls.Kernel=component_class(root)

    def setUp(self):
        torch.manual_seed(2026112007)
        self.linear=torch.nn.Linear(7,5,bias=True)
        cfg=self.framework['config'].AblationConfig(mode='zero',sphere_rescale=False)
        self.reference=self.framework['components'].TiedComponentLinear(self.linear,3,init_svd=False,ablation=cfg)
        self.native=self.Kernel(self.linear,3,init_svd=False,ablation=cfg)
        self.native.load_state_dict(self.reference.state_dict())

    def schedules(self,mask,delta):
        return self.framework['components'].PassMaskSchedule(mask,delta)

    def test_native_kernel_clean_closure_even_for_unfaithful_factors(self):
        x=torch.randn(1,4,7);actual=torch.sin(self.linear(x))
        with torch.no_grad():self.native.V.mul_(100);self.native.U.mul_(100)
        self.native.mode='component';self.native.schedule=self.schedules(torch.ones(1,1,4,3),torch.ones(1,1,4))
        got=self.native.forward_kernel(x,actual)
        self.assertTrue(torch.equal(got,actual))

    def test_framework_mask_algebra_and_factor_gradients(self):
        x=torch.randn(1,4,7);mask=torch.rand(1,1,4,3);delta=torch.rand(1,1,4)
        reference_x=x.clone().requires_grad_();native_x=x.clone().requires_grad_()
        for model in (self.reference,self.native):model.mode='component';model.schedule=self.schedules(mask,delta)
        a=self.reference(reference_x);b=self.native(native_x)
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
        a.square().sum().backward();b.square().sum().backward()
        for expected,actual in ((reference_x.grad,native_x.grad),(self.reference.V.grad,self.native.V.grad),(self.reference.U.grad,self.native.U.grad)):
            torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-5)
        la=self.framework['losses'].faithfulness_loss({'site':self.reference})
        lb=self.framework['losses'].faithfulness_loss({'site':self.native})
        torch.testing.assert_close(la,lb)

    def test_pass_mean_sphere_matches_framework(self):
        config=self.framework['config'].AblationConfig(mode='pass_mean',sphere_rescale=True)
        a=self.framework['components'].TiedComponentLinear(self.linear,3,init_svd=False,ablation=config)
        b=self.Kernel(self.linear,3,init_svd=False,ablation=config);b.load_state_dict(a.state_dict())
        calibration=torch.randn(1,4,7)
        for model in (a,b):
            model.update_running_means=True
            with torch.no_grad():model(calibration)
            model.update_running_means=False;model.reset_pass_state();model.mode='component'
            model.schedule=self.schedules(torch.full((1,1,4,3),.3),torch.full((1,1,4),.7))
        torch.testing.assert_close(a(calibration+1),b(calibration+1),rtol=1e-5,atol=1e-5)

    def test_physical_stage_hook_uses_selected_layer_only(self):
        def block():
            b=torch.nn.Module();b.mlp=torch.nn.Module();b.mlp.down_proj=torch.nn.Linear(7,7,bias=False);return b
        target=torch.nn.Module();target.model=torch.nn.Module();target.model.layers=torch.nn.ModuleList([block(),block()])
        target.lm_head=torch.nn.Linear(7,23,bias=False)
        loop=SimpleNamespace(model=target,device=torch.device('cpu'),n_gpu_layers=1,
            stage_layer=copy.deepcopy(target.model.layers[1]),active_layer=0)
        key='model.layers.1.mlp.down_proj'
        instrument=attach_native_components(loop,{key:3},bpd_root=self.bpd_root,ablation=self.reference.ablation)
        self.addCleanup(instrument.close);instrument.to('cpu')
        x=torch.randn(1,4,7);expected=loop.stage_layer.mlp.down_proj(x).detach()
        core=instrument.wrappers[key]
        self.assertEqual(core.calls,0)
        schedule=self.schedules(torch.ones(1,2,4,3),torch.ones(1,2,4))
        with instrument.masked_components({key:schedule}):
            loop.active_layer=1
            a=loop.stage_layer.mlp.down_proj(x);b=loop.stage_layer.mlp.down_proj(x)
            self.assertTrue(torch.equal(a,expected));self.assertTrue(torch.equal(b,expected));self.assertEqual(core.calls,2)
        self.assertEqual(target.lm_head.weight.device.type,'cpu')
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))
        self.assertTrue(all(not p.requires_grad for p in loop.stage_layer.parameters()))

    def test_online_controller_observes_actual_changed_second_pass(self):
        first=torch.randn(1,4,7);second=first+3
        self.native.mode='component';self.native.schedule=self.schedules(torch.zeros(1,2,4,3),torch.ones(1,2,4))
        self.native.live_controller=lambda reached,tau:torch.sigmoid(reached[...,:3])
        self.native(first);self.native(second)
        log=self.native.controller_observations
        self.assertEqual([r['pass'] for r in log],[0,1])
        self.assertTrue(torch.equal(log[1]['input'],second));self.assertFalse(torch.equal(log[0]['mask'],log[1]['mask']))
        self.assertEqual(len(list(self.native.parameters())),2)  # one shared V/U pair, not two pass copies

    def test_equivalent_diagonal_and_permutation_gauges(self):
        x=torch.randn(1,4,7);mask=torch.rand(1,1,4,3);delta=torch.rand(1,1,4)
        self.native.mode='component';self.native.schedule=self.schedules(mask,delta)
        expected=self.native(x)
        self.native.reset_pass_state();scale=torch.tensor([.2,3.,15.]);perm=torch.tensor([2,0,1])
        with torch.no_grad():
            self.native.V.copy_((self.native.V*scale)[:,perm]);self.native.U.copy_((self.native.U/scale[:,None])[perm])
        self.native.schedule=self.schedules(mask[...,perm],delta)
        torch.testing.assert_close(self.native(x),expected,rtol=1e-5,atol=1e-5)

    def test_full_vocabulary_head_forward_and_input_derivative(self):
        weight=torch.randn(23,7);a=torch.randn(2,3,7,requires_grad=True);b=a.detach().clone().requires_grad_()
        actual=OffloadedHead.apply(a,weight,5);expected=torch.nn.functional.linear(b,weight)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        target=torch.softmax(torch.randn_like(actual),dim=-1)
        loss_a=-(target*torch.log_softmax(actual,dim=-1)).sum();loss_b=-(target*torch.log_softmax(expected,dim=-1)).sum()
        loss_a.backward();loss_b.backward();torch.testing.assert_close(a.grad,b.grad,rtol=2e-6,atol=2e-6)


if __name__=='__main__':unittest.main()
