"""Bounded native replay and algebraic invariants; no trained-result fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from experiments.spectral_state_cost_v0 import spc09_common as c
from experiments.spectral_state_cost_v0.spc09_causal_compression import (
    retention_map, compression_maps, compression_displacement, magnitude_control, evaluate, summarize)
from experiments.spectral_state_cost_v0.spc09_observability_steering import local_decomposition, compare_prediction, steer_prompt
from experiments.spectral_state_cost_v0.spc09_cross_checkpoint import procrustes, transfer_bases, matched_update
from experiments.spectral_state_cost_v0.spc09_training_regularizer import auxiliary_losses, require_validation, compressed_snapshot
from experiments.spectral_state_cost_v0.spc09_reproduce import bounded_original, compare_numbers, initial_control
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import new_model
from experiments.spectral_state_cost_v0.spc05_phase_interventions import prepare_step
from research_gym.scripts.loopscope_toy import make_batch


class SPC09Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = new_model(41522).eval().double()
        for p in cls.model.parameters(): p.requires_grad_(False)
        cls.config = c.load_config()

    def test_canonical_graph_ignores_edge_order_and_query(self):
        tokens, _ = make_batch(8,4,10,3)
        x = tokens[0].clone(); y = x.clone()
        y[:16] = x[:16].reshape(8,2).flip(0).reshape(-1); y[-1] = (y[-1]+1)%8
        self.assertEqual(c.graph_id(x), c.graph_id(y))

    def test_train_validation_test_graph_disjoint(self):
        cohorts = [c.cohort(s,16,930901,3)[0] for s in ('train','validation','test')]
        sets = [set(c.fingerprints(t)['graph_ids']) for t in cohorts]
        self.assertTrue(all(len(s)==16 for s in sets))
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_cohort_deterministic(self):
        a = c.cohort('test',4,21,3); b = c.cohort('test',4,21,3)
        self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]))

    def test_config_rejects_hardware_violation(self):
        bad = dict(self.config, batch_size=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'; c.save_json(path,bad)
            with self.assertRaises(ValueError): c.load_config(path)

    def test_checkpoint_preload_hash_rejects_self_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spc04_seed41522.pt'; path.write_bytes(b'untrusted fixture')
            ready, rows = c.verify_checkpoints(tmp,[41522])
            self.assertFalse(ready); self.assertEqual(rows[0]['status'],'hash_mismatch')
            with mock.patch('torch.load',side_effect=AssertionError('must not deserialize')):
                with self.assertRaises(ValueError): c.load_model(tmp,41522)

    def test_missing_checkpoint_is_not_synthetic_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready, rows = c.verify_checkpoints(tmp,[41522,41523])
            self.assertFalse(ready); self.assertTrue(all(r['status']=='missing' for r in rows))

    def test_native_state_reconstruction(self):
        t,_ = make_batch(8,4,21,4)
        _,state = c.state_at(self.model,t,3)
        with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]): native = self.model(t,3)[-1]
        torch.testing.assert_close(state[:,-1],native,atol=1e-12,rtol=1e-12)

    def test_bounded_original_preserves_fd_and_restores_modules(self):
        from experiments.spectral_state_cost_v0 import spc08_phase_readout as orig
        tokens,_ = make_batch(8,1,21,6)
        prior = orig.finite_temporal_jacobian
        reference,_ = prior(self.model,tokens,depth=3,horizon=1)
        with bounded_original(4):
            bounded,_ = orig.finite_temporal_jacobian(self.model,tokens,depth=3,horizon=1)
        self.assertIs(orig.finite_temporal_jacobian,prior)
        torch.testing.assert_close(bounded,reference,atol=2e-10,rtol=2e-7)

    def test_bounded_adapter_restores_on_exception(self):
        from experiments.spectral_state_cost_v0 import spc08_phase_readout as orig
        prior=orig.prepare_step
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with bounded_original(4): raise RuntimeError('fixture')
        self.assertIs(orig.prepare_step,prior)

    def test_sequential_jvp_matches_finite_native_replay(self):
        tokens,_=make_batch(8,1,27,6)
        O,step,state=c.temporal_jacobian(self.model,tokens,3,2)
        f=c.observation_function(self.model,step,state,2)
        v=torch.linspace(-1,1,48,dtype=torch.float64); v/=v.norm()
        x=state[0,-1]; eps=1e-5
        with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]): fd=(f(x+eps*v)-f(x-eps*v))/(2*eps)
        torch.testing.assert_close(O@v,fd,atol=2e-8,rtol=2e-6)
        self.assertEqual(tuple(O.shape),(21,48)); self.assertFalse(O.requires_grad)

    def test_local_deferred_space_and_invisible_space(self):
        O=torch.zeros(21,48,dtype=torch.float64)
        O[:7,:7]=torch.eye(7,dtype=torch.float64)
        O[7:14,7:14]=2*torch.eye(7,dtype=torch.float64)
        split=local_decomposition(O)
        self.assertEqual(len(split['immediate']),7)
        self.assertEqual(len(split['future_in_current_null']),7)
        self.assertEqual(len(split['horizon_approximately_invisible']),34)
        self.assertLess(float((O[:7]@split['future_in_current_null'].T).norm()),1e-12)
        self.assertLess(float((O@split['horizon_approximately_invisible'].T).norm()),1e-12)

    def test_zero_gramian_does_not_invent_visible_rank(self):
        O=torch.zeros(14,48,dtype=torch.float64); s=local_decomposition(O)
        self.assertEqual(len(s['immediate']),0); self.assertEqual(len(s['future_in_current_null']),0)
        self.assertEqual(s['numerical_observation_rank'],0)

    def test_oblique_map_fixes_P_and_annihilates_D(self):
        I=torch.eye(48,dtype=torch.float64); P=I[:4]; D=(I[:4]+I[4:8])/2**.5
        Q=retention_map(P,D)
        torch.testing.assert_close(Q@P.T,P.T,atol=1e-12,rtol=1e-12)
        self.assertLess(float((Q@D.T).norm()),1e-12)
        self.assertGreater(float(torch.linalg.matrix_norm(Q,2)),1)

    def test_oblique_map_rejects_overlapping_bases(self):
        I=torch.eye(48,dtype=torch.float64)
        with self.assertRaises(ValueError): retention_map(I[:4],I[:4])

    def test_compression_union_uses_total_rank_budget(self):
        I=torch.eye(48,dtype=torch.float64)
        bases={'P':I,'D':I.roll(24,0),'PCA':I,'mean':torch.zeros(48,dtype=torch.float64)}
        for rank in (2,4,6,8,12,16):
            maps,invalid=compression_maps(bases,rank,1,I[:7])
            self.assertFalse(invalid)
            self.assertEqual(int(torch.linalg.matrix_rank(maps['retain_union'])),rank)
            self.assertEqual(int(torch.linalg.matrix_rank(maps['suppress_immediate_invisible'])),7)

    def test_centered_projection_preserves_mean(self):
        I=torch.eye(48,dtype=torch.float64); mean=torch.arange(48,dtype=torch.float64)
        self.assertEqual(float(compression_displacement(mean[None],mean,I[:4].T@I[:4]).norm()),0)

    def test_magnitude_control_exact_energy(self):
        x=torch.randn(4,48,dtype=torch.float64); y=magnitude_control(x,19)
        torch.testing.assert_close(x.norm(dim=-1),y.norm(dim=-1),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(y,magnitude_control(x,19))

    def test_zero_magnitude_control_remains_zero(self):
        self.assertEqual(float(magnitude_control(torch.zeros(4,48,dtype=torch.float64),19).norm()),0)

    def test_native_compression_baseline_and_controls(self):
        I=torch.eye(48,dtype=torch.float64)
        bases={'P':I,'D':I.roll(24,0),'PCA':I,'mean':torch.zeros(48,dtype=torch.float64)}
        config=dict(self.config,future_horizon=2)
        t,y=c.cohort('test',4,99,16)
        row=evaluate(self.model,bases,t,y,config,2,3)
        baseline=row['conditions']['baseline']
        self.assertEqual(max(baseline['immediate_logit_distortion']),0)
        self.assertEqual(max(baseline['future_logit_distortion']),0)
        self.assertEqual(sum(baseline['registered']),0)
        for label,z in row['conditions'].items():
            if label.endswith('_magnitude_control'):
                np.testing.assert_allclose(z['intervention_norm'],row['conditions'][label.removesuffix('_magnitude_control')]['intervention_norm'],atol=1e-10)

    def test_prediction_metrics_handle_true_zero(self):
        z=compare_prediction([0,0],[0,1]); self.assertIsNone(z['relative_error']); self.assertIsNone(z['sign_agreement'])
        z=compare_prediction([1,-2],[1,-2]); self.assertEqual(z['relative_error'],0); self.assertEqual(z['sign_agreement'],1)

    def test_procrustes_row_convention_and_reverse(self):
        X=torch.randn(96,48,dtype=torch.float64); Q=c.random_rows(48,19).T
        fit=procrustes(X,X@Q)
        torch.testing.assert_close(fit,Q,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(procrustes(X@Q,X),Q.T,atol=1e-10,rtol=1e-10)

    def test_transfer_matched_energy_and_rank(self):
        I=torch.eye(48,dtype=torch.float64); bases={'P':I,'D':I.roll(24,0)}
        Q=c.random_rows(48,3).T; arms=transfer_bases(bases,bases,Q,Q.T,4,8)
        natural=torch.randn(4,48,dtype=torch.float64)
        for group in arms.values():
            for basis in group.values():
                self.assertEqual(int(torch.linalg.matrix_rank(basis)),4)
                delta=matched_update(natural,basis)
                torch.testing.assert_close(delta.norm(dim=-1),natural.norm(dim=-1),atol=1e-12,rtol=1e-12)

    def test_zero_transfer_projection_is_explicit_failure(self):
        I=torch.eye(48,dtype=torch.float64)
        with self.assertRaises(ValueError): matched_update(I[4:8],I[:4])

    def test_transfer_preserves_zero_and_tiny_nonzero_energy(self):
        I=torch.eye(48,dtype=torch.float64)
        self.assertEqual(float(matched_update(torch.zeros(4,48,dtype=torch.float64),I[:4]).norm()),0)
        tiny=I[:4]*1e-14
        torch.testing.assert_close(matched_update(tiny,I[:4]).norm(dim=-1),tiny.norm(dim=-1),atol=1e-28,rtol=1e-12)

    def test_bootstrap_reproducible_and_empty_phase_safe(self):
        a=c.paired_interval([(2,4),(-1,4)],100,21)
        self.assertEqual(a,c.paired_interval([(2,4),(-1,4)],100,21)); self.assertEqual(a['difference'],.125)
        self.assertIsNone(c.paired_interval([(0,0)],100,21)['interval95'])

    def test_reproduction_counts_exact_and_float_tolerance(self):
        self.assertFalse(compare_numbers({'count':2,'rate':.5},{'count':2,'rate':.50000001},1e-5))
        self.assertEqual(len(compare_numbers({'count':2},{'count':3},1)),1)
        self.assertEqual(compare_numbers({'count':2},{},1)[0]['status'],'missing')

    def test_initialization_mismatch_is_not_substituted_for_original_control(self):
        from experiments.spectral_state_cost_v0.spc04_trained_dynamics import weight_sha256
        actual=weight_sha256(new_model(41522))
        with mock.patch.object(c,'pinned_manifest',return_value={41522:{'initial_weight_sha256':'0'*64}}):
            model,record=initial_control(41522)
            self.assertIsNone(model)
            self.assertFalse(record['same_initialization'])
            self.assertEqual(record['actual_initial_weight_sha256'],actual)
        with mock.patch.object(c,'pinned_manifest',return_value={41522:{'initial_weight_sha256':actual}}):
            model,record=initial_control(41522)
            self.assertIsNotNone(model)
            self.assertTrue(record['same_initialization'])

    def test_auxiliary_losses_zero_at_teacher_and_finite_gradients(self):
        I=torch.eye(48,dtype=torch.float64); teacher=torch.randn(3,4,48,dtype=torch.float64)
        p,d=auxiliary_losses(teacher,teacher,I[:8],I[8:16]); self.assertEqual(float(p+d),0)
        student=(teacher+torch.randn_like(teacher)*.01).requires_grad_(True)
        p,d=auxiliary_losses(student,teacher,I[:8],I[8:16]); (p+d).backward()
        self.assertTrue(torch.isfinite(student.grad).all()); self.assertGreater(float(student.grad.norm()),0)

    def test_training_gate_rejects_missing_or_stale_validation(self):
        with self.assertRaises(ValueError): require_validation(None,'sha',{},[])
        with tempfile.TemporaryDirectory() as tmp:
            c.save_json(Path(tmp)/'spc09_A.json',{'stage':'A','status':'incomplete','config_sha256':'sha'})
            with self.assertRaises(ValueError): require_validation(tmp,'sha',{},[])

    def test_training_gate_binds_sources_checkpoints_and_all_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stage in 'ABCD':
                c.save_json(Path(tmp)/f'spc09_{stage}.json',{'stage':stage,'status':'complete','config_sha256':'sha',
                    'source_sha256':{'code':'one'},'checkpoints':[1],
                    'result':{'reproduction_pass':True,'frozen_interventions_validated':True}})
            require_validation(tmp,'sha',{'code':'one'},[1])
            with self.assertRaises(ValueError): require_validation(tmp,'sha',{'code':'two'},[1])

    def test_compressed_snapshot_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.npz'; row=compressed_snapshot(self.model,path,{'update':50})
            self.assertEqual(row['file_sha256'],c.sha256(path))
            with np.load(path,allow_pickle=False) as loaded:
                for key,value in self.model.state_dict().items(): np.testing.assert_array_equal(loaded[key],value.numpy())
            timestamp=path.stat().st_mtime_ns
            self.assertEqual(row,compressed_snapshot(self.model,path,{'update':50}))
            self.assertEqual(timestamp,path.stat().st_mtime_ns)
            altered=copy.deepcopy(self.model)
            with torch.no_grad(): altered.head.bias.add_(.001)
            with self.assertRaises(FileExistsError): compressed_snapshot(altered,path,{'update':50})
            self.assertEqual(row['file_sha256'],c.sha256(path))

    def test_memory_measurement_and_receipt_json_are_finite(self):
        budget=c.Budget(self.config); budget.tick(); metrics=budget.finish()
        self.assertGreater(metrics['sampled_peak_rss_bytes'],0)
        json.dumps(metrics,allow_nan=False)

    def test_small_training_only_basis_fit_and_native_steering_smoke(self):
        config=dict(self.config,phase_prompts=16,phase_depths=[3,6,12],
                    fit_prompts=2,fit_depth=3,fit_horizon=1,future_horizon=1,radius_grid=[.0001])
        bases=c.fit_bases(self.model,config)
        self.assertTrue(all(c.partition(key)=='train' for key in bases['fit_graphs']))
        self.assertGreaterEqual(len(bases['P']),2)
        I=torch.eye(48,dtype=torch.float64)
        torch.testing.assert_close(bases['D']@bases['D'].T,I,atol=1e-10,rtol=1e-10)
        token,_=c.cohort('test',1,129,6)
        self.assertNotIn(c.graph_id(token[0]),bases['fit_graphs'])
        row=steer_prompt(self.model,token,3,config)
        self.assertEqual(len(row['rows']),6)
        self.assertTrue(all(r['status']=='measured' for r in row['rows']))

    def test_continuation_auxiliary_loss_updates_student_only(self):
        from experiments.spectral_state_cost_v0.spc04_trained_dynamics import weight_sha256
        teacher=copy.deepcopy(self.model); student=copy.deepcopy(self.model)
        frozen_hash=weight_sha256(teacher)
        for parameter in student.parameters(): parameter.requires_grad_(True)
        with torch.no_grad(): student.head.weight.add_(.001)
        tokens,hops=c.cohort('train',4,313,3)
        optim=torch.optim.AdamW(student.parameters(),lr=1e-4)
        before=weight_sha256(student)
        with torch.no_grad(): reference=teacher(tokens,3)
        predicted=student(tokens,3)
        I=torch.eye(48,dtype=torch.float64)
        phase,deferred=auxiliary_losses(predicted,reference,I[:8],I[8:16])
        loss=torch.nn.functional.cross_entropy(student.readout(predicted).reshape(-1,8),hops.T.reshape(-1))+.1*(phase+deferred)
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        self.assertNotEqual(weight_sha256(student),before)
        self.assertEqual(weight_sha256(teacher),frozen_hash)


if __name__=='__main__': unittest.main()
