"""SPC-07 finite gauge, observability, and frozen-checkpoint controls.

The CI-only core tests work from source. Tests using the original trained
checkpoints require SPC04_CHECKPOINT_DIR and verify the internal SHA256 gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import numpy as np

from experiments.spectral_state_cost_v0.spc07_transfer import (
    D_MODEL, TRAIN_PROMPT_SEED, EVAL_PROMPT_SEEDS, fit_temporal_basis,
    finite_temporal_jacobian, gramian_capture, orthogonal_procrustes,
    paired_cluster_bootstrap, projector_alignment, token_hashes,
    train_alignment,
)
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES, new_model
from experiments.spectral_state_cost_v0.spc05_phase_interventions import model_from_receipt
from experiments.spectral_state_cost_v0.spc06_readout_observability import logit_margin_jacobian
from research_gym.scripts.loopscope_toy import make_batch


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        torch.set_num_threads(min(2, torch.get_num_threads()))

    def test_exact_orthogonal_calibration_on_synthetic_coordinate_gauge(self):
        torch=self.torch
        g=torch.Generator().manual_seed(44071)
        X=torch.randn((144,12),generator=g,dtype=torch.float64)
        Q=torch.linalg.qr(torch.randn((12,12),generator=g,dtype=torch.float64)).Q
        T=X@Q
        estimate, residual=orthogonal_procrustes(X,T)
        self.assertLess(residual,1e-12)
        self.assertLess(torch.linalg.vector_norm(estimate-Q).item(),1e-11)
        self.assertLess(torch.linalg.vector_norm(estimate.T@estimate-torch.eye(12)).item(),1e-11)

    def test_missing_pair_alignment_rejected(self):
        torch=self.torch
        with self.assertRaises(ValueError):
            orthogonal_procrustes(torch.zeros(7,8), torch.zeros(7,8))
        with self.assertRaises(ValueError):
            orthogonal_procrustes(torch.zeros(12,5), torch.zeros(12,7))

    def test_source_target_subspace_alignment(self):
        torch=self.torch
        eye=torch.eye(8,dtype=torch.float64)
        same=projector_alignment(eye,eye,3)
        self.assertAlmostEqual(same['overlap_trace_fraction'],1,places=12)
        self.assertLess(same['max_principal_angle_deg'],1e-6)
        other=eye[[3,4,5,0,1,2,6,7]]
        disjoint=projector_alignment(eye,other,3)
        self.assertAlmostEqual(disjoint['overlap_trace_fraction'],0,places=12)
        G=torch.diag(torch.tensor([4.,3.,2.,1.],dtype=torch.float64))
        self.assertAlmostEqual(gramian_capture(torch.eye(4,dtype=torch.float64),G,2),0.7,places=12)

    def test_analytical_readout_derivative_matches_native_finite_difference(self):
        torch=self.torch
        model=new_model(41522).eval().double()
        tokens,_=make_batch(N_NODES,1,917710,40)
        O,x=finite_temporal_jacobian(model,tokens,depth=20,horizon=3)
        D=logit_margin_jacobian(model,x[:,-1,:])[0]
        self.assertEqual(tuple(O.shape),(4*(N_NODES-1),D_MODEL))
        error=float((torch.linalg.vector_norm(O[:N_NODES-1]-D)/torch.linalg.vector_norm(D)).detach())
        self.assertLess(error,1e-6)
        self.assertTrue(torch.isfinite(O).all())

    def test_fitted_gramian_psd_and_prompt_hash_partition(self):
        torch=self.torch
        model=new_model(41523).eval().double()
        V,G,hashes,receipt=fit_temporal_basis(model,n_prompts=3,horizon=3)
        self.assertEqual(tuple(G.shape),(D_MODEL,D_MODEL))
        self.assertGreaterEqual(torch.linalg.eigvalsh(G).min().item(),-1e-12)
        self.assertTrue(torch.allclose(V@V.T,torch.eye(D_MODEL,dtype=torch.float64),atol=1e-10))
        self.assertLess(receipt['fd_checks'][0]['two_step_fd_relative_difference'],1e-5)
        target_tokens,_=make_batch(N_NODES,64,EVAL_PROMPT_SEEDS[0],40)
        self.assertFalse(hashes & token_hashes(target_tokens))
        self.assertNotIn(TRAIN_PROMPT_SEED,EVAL_PROMPT_SEEDS)

    def test_cluster_bootstrap_preserves_registered_denominator(self):
        cases=[]
        for idx in range(2):
            a=[1,1,0,1,1]
            b=[1,0,0,0,1]
            cases.append({'registered':5,'conditions':{
                'A':{'shift_bits':a},'B':{'shift_bits':b}}})
        result=paired_cluster_bootstrap({'heldout':{'p':cases}},comparisons=(('A','B'),),repeats=80)
        d=result['A__vs__B']
        self.assertEqual(d['registered'],10)
        self.assertEqual(d['discordant_gain'],4)
        self.assertEqual(d['discordant_loss'],0)
        self.assertAlmostEqual(d['difference_rate'],0.4)
        self.assertEqual(d['bootstrap_95pct_cluster_ci'],[0.4,0.4])

    def test_pinned_public_summary_sanity(self):
        file=Path(__file__).resolve().parents[1]/'experiments/spectral_state_cost_v0/receipts/spc07_summary.json'
        if not file.exists():
            self.skipTest('frozen numerical receipt absent')
        summary=json.loads(file.read_text())
        self.assertEqual(len(summary['pairs']),2)
        for direction,data in summary['pairs'].items():
            self.assertGreater(data['registered'],0)
            self.assertEqual(data['conditions']['full_phase_step']['n'],data['registered'])
            self.assertEqual(data['conditions']['no_step']['shift'],0)
            for condition,metrics in data['conditions'].items():
                self.assertLessEqual(metrics['shift'],data['registered'])
                self.assertLessEqual(metrics['shift_and_correct'],metrics['shift'])
        self.assertTrue(summary['paired_bootstrap'])

    def test_saved_trained_checkpoints_and_true_pair_alignment(self):
        root=os.getenv('SPC04_CHECKPOINT_DIR')
        if not root:
            self.skipTest('SPC04_CHECKPOINT_DIR not configured in hosted core checks')
        folder=Path(root)
        if not all((folder/f'spc04_seed{s}.pt').is_file() for s in (41522,41523)):
            self.skipTest('pinned SPC-04 checkpoint files unavailable')
        src,sha1=model_from_receipt(folder/'spc04_seed41522.pt',41522)
        tgt,sha2=model_from_receipt(folder/'spc04_seed41523.pt',41523)
        self.assertEqual(sha1,'67c61b2c43a6005925c23a9ac90e71fde5f6248480740eb088541a9fa381baa4')
        self.assertEqual(sha2,'0a8dedc860aea4d80f70cfc1b95f73aa44191f465dc4554e1578dc97bcb58631')
        Q,Qshuffle,hashes,data=train_alignment(src,tgt,n_prompts=48)
        self.assertEqual(data['paired_rows'],48*3)
        self.assertLess(data['paired_procrustes_relative_error'],
                        data['shuffled_procrustes_error_evaluated_on_true_pairs'])
        self.assertTrue(self.torch.allclose(Q@Q.T,self.torch.eye(D_MODEL,dtype=self.torch.float64),atol=1e-10))
        self.assertEqual(len(hashes),48)

if __name__=='__main__':unittest.main()
