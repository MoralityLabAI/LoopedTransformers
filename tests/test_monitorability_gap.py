from __future__ import annotations
import unittest
import numpy as np
from research_gym.analysis.monitorability_gap import (
    strongest_null_direction, null_project, taylor_bounds, discrete_cmi,
    log_loss, balanced_bayes_accuracy, finite_effects, torch_jacobians,
)
from research_gym.scripts.monitorability_gap_controls import controls


class MonitorabilityGapTests(unittest.TestCase):
    def test_axis_gap(self):
        r=strongest_null_direction([[1,0]],[[0,3]])
        self.assertAlmostEqual(r.sensitivity,3)
        self.assertAlmostEqual(r.observation_derivative_norm,0)
    def test_full_observation(self):
        r=strongest_null_direction(np.eye(3),[[1,2,3]])
        self.assertEqual(r.sensitivity,0)
        self.assertEqual(r.nullity,0)
    def test_future_factors_through_observation(self):
        r=strongest_null_direction([[1,2,3]],[[2,4,6],[-1,-2,-3]])
        self.assertEqual(r.sensitivity,0)
    def test_empty_observation(self):
        r=strongest_null_direction(np.empty((0,2)),[[3,4]])
        self.assertAlmostEqual(r.sensitivity,5)
    def test_zero_observation(self):
        r=strongest_null_direction(np.zeros((2,3)),[[0,0,2]])
        self.assertAlmostEqual(r.sensitivity,2)
    def test_zero_future(self):
        r=strongest_null_direction([[1,0]],[[0,0]])
        self.assertEqual(r.sensitivity,0)
    def test_projection_idempotent(self):
        rng=np.random.default_rng(1)
        c=rng.normal(size=(2,7)); x=rng.normal(size=(3,7))
        y,_=null_project(x,c); yy,_=null_project(y,c)
        np.testing.assert_allclose(y,yy,atol=1e-12)
        np.testing.assert_allclose(c@y.T,0,atol=1e-12)
    def test_orthogonal_coordinate_invariance(self):
        rng=np.random.default_rng(2)
        c=rng.normal(size=(2,7)); b=rng.normal(size=(3,7))
        q,_=np.linalg.qr(rng.normal(size=(7,7)))
        a=strongest_null_direction(c,b); z=strongest_null_direction(c@q,b@q)
        self.assertAlmostEqual(a.sensitivity,z.sensitivity,places=11)
    def test_duplicate_observations(self):
        r=strongest_null_direction([[1,0],[2,0]],[[0,1]])
        self.assertEqual(r.observation_rank,1)
        self.assertAlmostEqual(r.sensitivity,1)
    def test_numerical_rank_threshold_is_explicit(self):
        r=strongest_null_direction(np.diag([1,1e-13]),[[0,1]],atol=1e-12)
        self.assertEqual(r.observation_rank,1)
        self.assertGreater(r.observation_derivative_norm,0)
    def test_matrix_mismatch_rejected(self):
        with self.assertRaises(ValueError): strongest_null_direction([[1,0]],[[1,0,0]])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): strongest_null_direction([[np.nan,0]],[[0,1]])
    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError): null_project([[1]],[[1]],rtol=-1)
    def test_taylor_exact_linear(self):
        r=taylor_bounds(.1,3,0,0)
        self.assertEqual(r['observation_drift_upper'],0)
        self.assertAlmostEqual(r['future_drift_lower'],.3)
    def test_taylor_curvature(self):
        r=taylor_bounds(.1,1,2,2)
        self.assertAlmostEqual(r['observation_drift_upper'],.01)
        self.assertAlmostEqual(r['future_drift_lower'],.09)
    def test_taylor_residual(self):
        r=taylor_bounds(.1,1,2,0,.03)
        self.assertAlmostEqual(r['observation_drift_upper'],.013)
    def test_taylor_lower_clipped(self):
        self.assertEqual(taylor_bounds(2,1,0,10)['future_drift_lower'],0)
    def test_taylor_negative_rejected(self):
        with self.assertRaises(ValueError): taylor_bounds(-1,1,0,0)
    def test_deterministic_cmi(self):
        p=np.zeros((2,2,4))
        for s in range(4): p[s%2,s%2,s]=1
        self.assertAlmostEqual(discrete_cmi(p),0)
    def test_cmi_one_bit(self):
        p=np.zeros((2,2,1)); p[0,0,0]=p[1,1,0]=.5
        self.assertAlmostEqual(discrete_cmi(p),np.log(2))
    def test_cmi_bad_table(self):
        with self.assertRaises(ValueError): discrete_cmi(np.zeros((2,2,2)))
    def test_loss_uniform(self):
        self.assertAlmostEqual(log_loss([0,1],[[.5,.5],[.5,.5]]),np.log(2))
    def test_loss_zero_probability(self):
        self.assertTrue(np.isinf(log_loss([0],[[0,1]])))
    def test_loss_bad_probabilities(self):
        with self.assertRaises(ValueError): log_loss([0],[[.5,.7]])
    def test_loss_bad_labels(self):
        with self.assertRaises(ValueError): log_loss([.2],[[.5,.5]])
    def test_bayes_equal(self):
        self.assertEqual(balanced_bayes_accuracy([.4,.6],[.4,.6]),.5)
    def test_bayes_disjoint(self):
        self.assertEqual(balanced_bayes_accuracy([1,0],[0,1]),1)
    def test_bayes_partial(self):
        self.assertEqual(balanced_bayes_accuracy([.75,.25],[.25,.75]),.75)
    def test_bayes_invalid(self):
        with self.assertRaises(ValueError): balanced_bayes_accuracy([1,1],[0,1])
    def test_finite_effects(self):
        r=finite_effects([1,2],[1,2],[0,0],[3,4])
        self.assertEqual(r['observation_l2'],0)
        self.assertEqual(r['future_l2'],5)
    def test_finite_invalid(self):
        with self.assertRaises(ValueError): finite_effects([1],[1,2],[0],[1])
    def test_all_controls_pass(self):
        r=controls()
        self.assertEqual(r['passed'],r['total'])
        self.assertFalse(r['trained_model_evidence'])


class TorchTests(unittest.TestCase):
    def test_linear_jacobian(self):
        import torch
        h=torch.tensor([1.,2.],dtype=torch.float64)
        c,b=torch_jacobians(h,lambda h:h[:1],lambda h:3*h[1:])
        np.testing.assert_allclose(c,[[1,0]])
        np.testing.assert_allclose(b,[[0,3]])
    def test_nonlinear_finite_scaling(self):
        import torch
        h=torch.zeros(2,dtype=torch.float64)
        obs=lambda h:(h[0]+h[1]**2).reshape(1)
        future=lambda h:h[1:].clone()
        c,b=torch_jacobians(h,obs,future)
        r=strongest_null_direction(c,b)
        for radius in [.001,.01,.1]:
            hh=h+torch.tensor(r.direction)*radius
            self.assertAlmostEqual(float(obs(hh)),radius**2)
            self.assertAlmostEqual(float(future(hh)),radius)
    def test_state_shape_preserved(self):
        import torch
        h=torch.ones((1,2,3),dtype=torch.float64)
        c,b=torch_jacobians(h,lambda h:h[:,-1],lambda h:h.sum().reshape(1))
        self.assertEqual(c.shape,(3,6)); self.assertEqual(b.shape,(1,6))
    def test_nonfinite_torch_rejected(self):
        import torch
        with self.assertRaises(ValueError): torch_jacobians(torch.tensor([float('nan')]),lambda x:x,lambda x:x)


if __name__=='__main__': unittest.main()
