"""SPC-05 controls: deterministic state replay, role-restricted interventions."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import new_model, weight_sha256
from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    model_from_receipt, causal_phase_batch, pointer_periods,
)
from experiments.spectral_state_cost_v0.spc05_phase_subspace import (
    fit_phase_svd, evaluate_rank_batch, random_basis, project_increment,
)
from research_gym.scripts.loopscope_toy import make_batch
from experiments.spectral_state_cost_v0.spc05_ablation import (
    suppress_answer_increment, run_ablation_batch,
)


class PhaseInterventionUnitTests(unittest.TestCase):
    def test_checkpoint_sha_and_seed_guard(self):
        model=new_model(41522)
        record={'seed':41522, 'model':model.state_dict(),
                'weight_sha256':weight_sha256(model),
                'hyperparameters':{'n_nodes':8,'d_model':48}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt'
            torch.save(record,path)
            recovered,digest=model_from_receipt(path,41522)
            self.assertEqual(digest,record['weight_sha256'])
            self.assertEqual(recovered.tok.weight.dtype,torch.float64)
            with self.assertRaises(ValueError):model_from_receipt(path,41523)
            record['weight_sha256']='wrong'
            torch.save(record,path)
            with self.assertRaises(ValueError):model_from_receipt(path,41522)

    def test_exact_whole_state_transplant_control(self):
        model=new_model(41522).eval().double()
        r=causal_phase_batch(model,905501,batch=8,phase_depth=12,follow=4,noise_levels=(0.05,))
        self.assertEqual(r['native_forward_gap'],0.0)
        self.assertLess(r['models']['full_phase_plus1']['full_shift_exact_replay_gap'],1e-11)
        self.assertEqual(r['models']['unperturbed']['n_nearest_reference_zero'],
                         r['models']['unperturbed']['n_phase_evaluable'])
        self.assertGreaterEqual(r['n_registered_nonfixed_task_cycles'],0)

    def test_all_interventions_are_counted_and_bounded(self):
        r=causal_phase_batch(new_model(41522).eval().double(),905502,
                             batch=10,phase_depth=16,follow=3,noise_levels=(0.25,))
        self.assertIn('noise_answer_0.25',r['models'])
        self.assertIn('noise_context_0.25',r['models'])
        self.assertEqual(r['models']['answer_phase_plus1']['n_phase_evaluable'],
                         r['n_distinguishable_phase_reference'])
        for row in r['models'].values():
            self.assertLessEqual(row['n_registered_nearest_zero'],row['n_registered_task_cycles'])
            self.assertLessEqual(row['n_registered_nearest_plus_one'],row['n_registered_task_cycles'])

    def test_orthonormal_projection_and_equal_norm_control(self):
        torch.manual_seed(1501)
        X=torch.randn((12,48),dtype=torch.float64)
        Q=random_basis(48,8,2903)
        P=X@Q.T@Q
        self.assertTrue(torch.allclose(project_increment(P,Q,'raw'),P,atol=1e-12))
        self.assertTrue(torch.allclose(project_increment(X,Q,'raw'),P,atol=1e-12))
        Z=project_increment(X,Q,'norm_matched')
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(Z,dim=1),
                                        torch.linalg.vector_norm(X,dim=1),atol=1e-10))
        with self.assertRaises(ValueError):project_increment(X,Q,'unknown')

    def test_svd_has_heldout_contract(self):
        model=new_model(41522).eval().double()
        V,basis=fit_phase_svd(model,seed=904001,n_prompts=32,depths=(2,4))
        self.assertEqual(V.shape,(48,48))
        self.assertEqual(basis['rows'],64)
        self.assertLessEqual(basis['energy_capture_by_rank']['8'],1.0000001)
        result=evaluate_rank_batch(model,V,905501,batch=6,ranks=(2,4),
                                   random_repeats=2,depth=12,follow=4)
        self.assertIn('4',result['rank_rows'])
        self.assertEqual(len(result['rank_rows']['2']['random_norm_matched']),2)
        row=result['rank_rows']['4']['pca_raw']
        self.assertEqual(row['registered_evaluable'],result['registered_cycles'])

    def test_ablation_projected_increment_is_exactly_suppressed(self):
        torch.manual_seed(41522)
        current=torch.randn(3,9,48,dtype=torch.float64)
        following=current+torch.randn_like(current)
        Q=random_basis(48,7,918)
        removed=suppress_answer_increment(current,following,Q)
        self.assertTrue(torch.equal(removed[:,:-1],following[:,:-1]))
        expected=project_increment(following[:,-1]-current[:,-1],Q,'raw')
        self.assertTrue(torch.allclose(following[:,-1]-removed[:,-1],expected,atol=1e-12))
        residual=removed[:,-1]-current[:,-1]
        self.assertLess(float((residual@Q.T).abs().max()),1e-12)
        full=suppress_answer_increment(current,following,torch.eye(48,dtype=current.dtype))
        self.assertTrue(torch.allclose(full[:,-1],current[:,-1],atol=1e-12))
        self.assertTrue(torch.equal(full[:,:-1],following[:,:-1]))

    def test_ablation_frozen_answer_smoke_on_native_recurrence(self):
        model=new_model(41522).eval().double()
        Vh,_=fit_phase_svd(model,seed=904001,n_prompts=32,depths=(2,4))
        receipt=run_ablation_batch(model,Vh,905501,batch=6,
                                   random_repeats=2,horizons=(6,8),ranks=(4,7,8))
        self.assertEqual(receipt['items'],6)
        self.assertEqual(receipt['horizons'],[6,8])
        for horizon in ('6','8'):
            self.assertEqual(receipt['conditions']['freeze_answer'][horizon]
                             ['matches_frozen_loop3_readout'],6)
            for rank in (4,7,8):
                self.assertEqual(receipt['conditions'][f'pca_rank{rank}'][horizon]
                                 ['n_items'],6)
            self.assertIn('random8_raw_trial1',receipt['conditions'])
            self.assertIn('random8_equal_removed_norm_trial1',receipt['conditions'])

    def test_pointer_period_is_graph_grounded(self):
        tokens,_=make_batch(8,8,905501,20)
        periods=pointer_periods(tokens)
        self.assertEqual(len(periods),8)
        self.assertTrue(all(1<=x<=8 for x in periods))


if __name__=='__main__':unittest.main()
