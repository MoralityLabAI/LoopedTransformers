"""Source-only original-stock/pass-indexed parity contract.

Only a new, untrained TinyRecursiveModel(48,4) and the excluded smoke generator
seed 61077 are used. Original 12 planned new teacher roots and every registered
source cohort 940xxx--943xxx remain unopened by this suite.
"""
from __future__ import annotations

import copy
import unittest

import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.deuterium.native_trm.fit_and_select import config
from mechinterp.deuterium.native_trm.adapter02.pass_indexed import (
    PassIndexedComponentModel, pass_indexed_mask_infos, TARGET
)
from mechinterp.deuterium.native_trm.adapter02.pgd_shared import (
    pgd_pass_indexed_recon_loss, shared_adversarial_source_shape,
    interpolate_shared_noise
)
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import make_mask_infos
from param_decomp.models.batch_and_loss_fns import (
    run_batch_first_element, recon_loss_kl
)
from param_decomp.metrics.pgd_masked_recon_loss import pgd_recon_loss
from param_decomp.metrics.importance_minimality_loss import importance_minimality_loss
from param_decomp.utils.module_utils import expand_module_patterns


SMOKE_SEED=61077
TOL=2e-5


def fit_pair(n=2):
    torch.manual_seed(SMOKE_SEED)
    original=TinyRecursiveModel(48,4).eval().requires_grad_(False)
    twin=TinyRecursiveModel(48,4).eval().requires_grad_(False)
    twin.load_state_dict(original.state_dict(),strict=True)
    cfg=config(SMOKE_SEED,smoke=True)
    info=expand_module_patterns(original,cfg.all_module_info)
    twin_info=expand_module_patterns(twin,cfg.all_module_info)
    stock=ComponentModel(
        target_model=original,run_batch=run_batch_first_element,
        module_path_info=info,ci_config=cfg.ci_config,sigmoid_type=cfg.sigmoid_type
    )
    indexed=PassIndexedComponentModel(
        target_model=twin,run_batch=run_batch_first_element,
        module_path_info=twin_info,ci_config=cfg.ci_config,sigmoid_type=cfg.sigmoid_type
    )
    indexed.load_state_dict(stock.state_dict(),strict=True)
    x,_=fixed_data(n,SMOKE_SEED)
    stock.eval()
    indexed.eval()
    return stock,indexed,cfg,(x,)


def pgd_cfg(cfg):
    return next(one for one in cfg.loss_metric_configs if one.classname=="PGDReconLoss")


def err(a,b):
    return float((a.detach()-b.detach()).abs().max())


class TestPassIndexedOriginalParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_four_clean_inputs_and_original_last_cache(self):
        stock,indexed,cfg,batch=fit_pair()
        original=stock(batch,cache_type="input")
        reached=indexed(batch,cache_type="input")
        self.assertEqual(set(reached.cache),{TARGET})
        self.assertEqual(tuple(reached.cache[TARGET].shape),(4,2,81,128))
        self.assertTrue(torch.equal(reached.cache[TARGET][-1],original.cache[TARGET]))
        self.assertTrue(torch.equal(reached.output,original.output))
        self.assertGreater(float((reached.cache[TARGET][0]-reached.cache[TARGET][-1]).abs().max()),1e-6)
        self.assertEqual(len(indexed.target_module_paths),1)

    def test_stock_mask_identical_to_four_forced_equal_masks_and_delta_replay(self):
        stock,indexed,cfg,batch=fit_pair()
        clean=stock(batch,cache_type="input")
        ci=stock.calc_causal_importances(clean.cache,sampling=cfg.sampling).lower_leaky[TARGET]
        delta=stock.calc_weight_deltas()[TARGET]
        noise=torch.full((2,81,1),0.7)
        for residual in (False,True):
            mask=ci
            if residual:
                si=make_mask_infos({TARGET:mask},weight_deltas_and_masks={
                    TARGET:(delta,noise.squeeze(-1))})
                pi=pass_indexed_mask_infos(mask.unsqueeze(0).expand(4,-1,-1,-1),
                   weight_delta=delta,residual_mask=noise.unsqueeze(0).expand(4,-1,-1,-1))
            else:
                si=make_mask_infos({TARGET:mask})
                pi=pass_indexed_mask_infos(mask.unsqueeze(0).expand(4,-1,-1,-1))
            original=stock(batch,mask_infos=si)
            indexed_logits=indexed(batch,mask_infos=pi)
            self.assertLessEqual(err(original,indexed_logits),TOL)

    def test_pass_indexed_ci_is_same_original_network_over_four_depths(self):
        stock,indexed,cfg,batch=fit_pair()
        s=stock(batch,cache_type="input")
        p=indexed(batch,cache_type="input")
        c_s=stock.calc_causal_importances(s.cache,sampling=cfg.sampling)
        c_p=indexed.calc_causal_importances(p.cache,sampling=cfg.sampling)
        self.assertEqual(tuple(c_p.lower_leaky[TARGET].shape),(4,2,81,16))
        self.assertLessEqual(err(c_p.lower_leaky[TARGET][-1],c_s.lower_leaky[TARGET]),TOL)
        delta=float((c_p.lower_leaky[TARGET][0]-c_p.lower_leaky[TARGET][-1]).abs().mean())
        self.assertGreater(delta,1e-6)
        fresh=indexed(batch,mask_infos=pass_indexed_mask_infos(c_p.lower_leaky[TARGET]))
        frozen=indexed(batch,mask_infos=pass_indexed_mask_infos(
            c_p.lower_leaky[TARGET][-1:].expand(4,-1,-1,-1)))
        self.assertGreater(err(fresh,frozen),1e-7)

    def test_zero_and_malformed_mask_guards(self):
        stock,indexed,cfg,batch=fit_pair()
        good=torch.ones(4,2,81,16)
        wrong=torch.ones(5,2,81,16)
        with self.assertRaisesRegex(ValueError,"four passes"):
            pass_indexed_mask_infos(wrong)
        with self.assertRaisesRegex(ValueError,"weight residual"):
            pass_indexed_mask_infos(good,weight_delta=torch.zeros(48,128))
        with self.assertRaisesRegex(ValueError,"residual mask"):
            pass_indexed_mask_infos(good,weight_delta=torch.zeros(48,128),
                                    residual_mask=torch.ones(2,81,1))
        with self.assertRaisesRegex(ValueError,"expected"):
            indexed(batch,mask_infos=make_mask_infos({TARGET:torch.ones(2,16)}))
        self.assertEqual(indexed.target_module_paths,[TARGET])

    def test_equal_mask_native_component_and_ci_gradients(self):
        stock,indexed,cfg,batch=fit_pair()
        sc=stock.calc_causal_importances(stock(batch,cache_type="input").cache,
                                          sampling=cfg.sampling)
        pc=indexed.calc_causal_importances(indexed(batch,cache_type="input").cache,
                                            sampling=cfg.sampling)
        sm=sc.lower_leaky[TARGET]
        pm=pc.lower_leaky[TARGET][-1:].expand(4,-1,-1,-1)
        out_s=stock(batch,mask_infos=make_mask_infos({TARGET:sm}))
        out_p=indexed(batch,mask_infos=pass_indexed_mask_infos(pm))
        self.assertLessEqual(err(out_s,out_p),TOL)
        params_s=list(stock._components.parameters())+list(stock.ci_fn.parameters())
        params_p=list(indexed._components.parameters())+list(indexed.ci_fn.parameters())
        g_s=torch.autograd.grad(out_s.square().mean(),params_s)
        g_p=torch.autograd.grad(out_p.square().mean(),params_p)
        self.assertEqual(len(g_s),len(g_p))
        self.assertLessEqual(max(err(a,b) for a,b in zip(g_s,g_p)),TOL)

    def test_stock_vs_pass_indexed_PGD_same_adversary_shape_and_gradient(self):
        stock,indexed,cfg,batch=fit_pair()
        sc=stock(batch,cache_type="input")
        pc=indexed(batch,cache_type="input")
        ci_s=stock.calc_causal_importances(sc.cache,sampling=cfg.sampling).lower_leaky[TARGET]
        ci_p=ci_s.unsqueeze(0).expand(4,-1,-1,-1)
        self.assertEqual(shared_adversarial_source_shape(ci_p,stock.calc_weight_deltas()[TARGET]),(2,81,17))
        pgd=pgd_cfg(cfg)
        torch.manual_seed(1971)
        s_loss=pgd_recon_loss(
            model=stock,batch=batch,target_out=sc.output,
            ci={TARGET:ci_s},weight_deltas=stock.calc_weight_deltas(),
            pgd_config=pgd,reconstruction_loss=recon_loss_kl)
        torch.manual_seed(1971)
        p_loss=pgd_pass_indexed_recon_loss(
            model=indexed,batch=batch,ci=ci_p,target_out=pc.output,
            weight_deltas=indexed.calc_weight_deltas(),pgd_config=pgd,
            reconstruction_loss=recon_loss_kl)
        self.assertLessEqual(err(s_loss,p_loss),TOL)
        grad_s=torch.autograd.grad(s_loss,list(stock._components.parameters()))
        grad_p=torch.autograd.grad(p_loss,list(indexed._components.parameters()))
        self.assertLessEqual(max(err(a,b) for a,b in zip(grad_s,grad_p)),TOL)

    def test_importance_penalty_mean_normalizes_four_depths(self):
        stock,indexed,cfg,batch=fit_pair()
        s=stock.calc_causal_importances(stock(batch,cache_type="input").cache,
                                          sampling=cfg.sampling)
        c=s.upper_leaky[TARGET]
        repeated=c.unsqueeze(0).expand(4,-1,-1,-1)
        co=next(one for one in cfg.loss_metric_configs
                if one.classname=="ImportanceMinimalityLoss")
        kw={"current_frac_of_training":1.0,"eps":co.eps,"pnorm":co.pnorm,
            "beta":co.beta,"p_anneal_start_frac":co.p_anneal_start_frac,
            "p_anneal_final_p":co.p_anneal_final_p,
            "p_anneal_end_frac":co.p_anneal_end_frac}
        s_loss=importance_minimality_loss(ci_upper_leaky={TARGET:c},**kw)
        p_loss=importance_minimality_loss(ci_upper_leaky={TARGET:repeated},**kw)
        self.assertLessEqual(err(s_loss,p_loss),TOL)


if __name__=="__main__":
    unittest.main()
