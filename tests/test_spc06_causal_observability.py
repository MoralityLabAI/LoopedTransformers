"""Local, bounded SPC-06 instruments: checkpoint-independent CPU controls."""
import unittest

import numpy as np
import torch

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import new_model
from experiments.spectral_state_cost_v0.spc06_phase_necessity import (
    run_ablation_batch, suppress_answer_increment,
)
from experiments.spectral_state_cost_v0.spc06_readout_observability import (
    exact_readout_fiber_rotate, random_equal_chord_fiber,
    logit_margin_jacobian, split_readout_directions, evaluate_readout_null,
)
from experiments.spectral_state_cost_v0.spc06_future_observability import native_observability
from experiments.spectral_state_cost_v0.spc05_phase_subspace import fit_phase_svd


class SPC06Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_logits_and_equal_chord_random_control(self):
        m=new_model(41522).eval().double()
        rng=torch.Generator().manual_seed(19034)
        x=torch.randn(6,48,dtype=torch.float64,generator=rng)
        delta=torch.randn(6,48,dtype=torch.float64,generator=rng)
        with torch.no_grad():
            for scale in (0.5,1.0,2.0,4.0):
                z=exact_readout_fiber_rotate(m,x,delta,scale,preserve='logits')
                r=random_equal_chord_fiber(m,x,z,177+int(scale*10))
                self.assertLess(float((m.readout(z)-m.readout(x)).abs().max()),1e-10)
                self.assertLess(float((m.readout(r)-m.readout(x)).abs().max()),1e-10)
                self.assertLess(float((torch.linalg.vector_norm(z-x,dim=1)-
                                      torch.linalg.vector_norm(r-x,dim=1)).abs().max()),1e-10)
                self.assertGreater(float(torch.linalg.vector_norm(z-x)),1e-4)
            with self.assertRaises(ValueError):
                exact_readout_fiber_rotate(m,x,delta,1.0,preserve='unknown')

    def test_margin_jacobian_matches_autograd_and_orthogonal_projection(self):
        model=new_model(41523).eval().double()
        torch.manual_seed(4115)
        x=torch.randn(4,48,dtype=torch.float64,requires_grad=True)
        dx=torch.randn_like(x)
        with torch.no_grad():J=logit_margin_jacobian(model,x.detach())
        def fun(t):
            logits=model.readout(t)
            return logits[:,:-1]-logits[:,-1:]
        _,right=torch.autograd.functional.jvp(fun,x,(dx,))
        self.assertLess(float(torch.linalg.vector_norm((J@dx[:,:,None]).squeeze(-1)-right)),1e-9)
        with torch.no_grad():
            row,null,diag=split_readout_directions(model,x.detach(),dx)
            self.assertEqual(diag['min_readout_margin_rank'],7)
            self.assertLess(diag['max_null_first_order_margin_change'],1e-9)
            self.assertLess(float(torch.linalg.vector_norm(row+null-dx)),1e-11)
            self.assertLess(float(torch.max(torch.abs((row*null).sum(-1)))),1e-11)

    def test_repeated_answer_freeze_keeps_exact_readout(self):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.spectral_state_cost_v0.spc05_phase_interventions import prepare_step
        from research_gym.scripts.loopscope_toy import make_batch
        model=new_model(41522).eval().double()
        tokens,_=make_batch(8,6,905501,12)
        with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]):
            F,x=prepare_step(model,tokens)
            for _ in range(3):x=F(x)
            frozen=model.readout(x[:,-1,:])
            for _ in range(9):
                after=F(x)
                x=suppress_answer_increment(x,after,torch.eye(48,dtype=torch.float64))
                self.assertTrue(torch.equal(model.readout(x[:,-1,:]),frozen))

    def test_rank_growth_on_untrained_native_model(self):
        m=new_model(41522).eval().double()
        r=native_observability(m,905501,depth=12,horizon=3,eps=2e-4)
        self.assertLess(r['instant_analytic_vs_fd_relative_error'],1e-7)
        self.assertEqual(r['steps']['0']['rank_rel_1e-3'],7)
        self.assertEqual(r['steps']['1']['rank_rel_1e-3'],14)
        self.assertLessEqual(r['steps']['2']['rank_rel_1e-3'],21)
        self.assertGreaterEqual(r['steps']['3']['rank_rel_1e-6'],14)

    def test_frozen_subspace_evaluation_smoke(self):
        m=new_model(41522).eval().double()
        Vh,_=fit_phase_svd(m,n_prompts=24,depths=(3,5))
        r=run_ablation_batch(m,Vh,905501,batch=7,random_repeats=1,
                            horizons=(6,12),ranks=(7,8))
        for h in ('6','12'):
            self.assertEqual(r['conditions']['freeze_answer'][h]['matches_frozen_loop3_readout'],7)
        self.assertIn('pca_rank7',r['conditions'])
        self.assertIn('random8_equal_removed_norm_trial0',r['conditions'])

    def test_readout_fiber_heldout_schema(self):
        m=new_model(41522).eval().double()
        r=evaluate_readout_null(m,905501,batch=6,depth=14,follow=3)
        self.assertEqual(r['readout_jacobian']['min_readout_margin_rank'],7)
        for k,v in r['condition'].items():
            if k.startswith(('exact_logits_','random_fiber_')):
                self.assertLess(v['max_abs_immediate_logit_change'],1e-9)
                self.assertEqual(v['immediate_readout_label_unchanged_all'],6)
        self.assertEqual(r['condition']['full_answer']['n_registered'],r['registered_cycles'])


if __name__=='__main__':unittest.main()
