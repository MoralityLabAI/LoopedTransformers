"""ROC-02B geometry, seeded nulls, source-only state and rank-p tests."""
import hashlib
import math
import sys
import unittest
from pathlib import Path

import torch
sys.path.insert(0,str(Path(__file__).parent))
try:
    from research_gym.analysis.roc02b_calibration_card import (
        TiedRuleModel,stage_model,suffix,tangent,jacobian_metrics,random_U,
        fingerprint,null_calibrate,ridge_predict,holm,RANKS,NULL_SEEDS)
except ModuleNotFoundError:
    from roc02b_calibration_card import (
    TiedRuleModel,stage_model,suffix,tangent,jacobian_metrics,random_U,
    fingerprint,null_calibrate,ridge_predict,holm,RANKS,NULL_SEEDS)

class CalibrationGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_original_model_state_is_identical(self):
        torch.manual_seed(13)
        model=TiedRuleModel(8).eval()
        x=torch.randn(3,7,79); m=torch.ones(3,7);m[1,4:]=0
        with torch.no_grad():
            hmid,hfin=stage_model(model,x,m)
            self.assertTrue(torch.allclose(model(x,m),model.readout(hfin),atol=0,rtol=0))
            self.assertTrue(torch.allclose(suffix(model,hmid),hfin,atol=0,rtol=0))

    def test_suffix_jacobian_matches_autograd(self):
        torch.manual_seed(3)
        model=TiedRuleModel(16).eval()
        h=torch.randn(128)*.3
        target=tangent(model,h).float()
        h=h.requires_grad_(True)
        actual=torch.autograd.functional.jacobian(lambda v:suffix(model,v[None,:])[0],h,vectorize=True)
        self.assertLess(float(torch.max(torch.abs(actual-target))),2e-5)

    def test_block_energy_and_gauge(self):
        torch.manual_seed(7)
        J=torch.randn(128,128,dtype=torch.float64)*.05
        U=torch.linalg.qr(torch.randn(128,7,dtype=torch.float64))[0]
        R=torch.randn(8,128,dtype=torch.float64)
        m=jacobian_metrics(J,U,R)
        Q=torch.eye(128,dtype=torch.float64)-U@U.T
        self.assertAlmostEqual(m['B']**2,float(torch.linalg.norm(U.T@J@Q)**2),places=10)
        self.assertAlmostEqual(m['C']**2,float(torch.linalg.norm(Q@J@U)**2),places=10)
        G=torch.linalg.qr(torch.randn(7,7,dtype=torch.float64))[0]
        z=jacobian_metrics(J,U@G,R)
        for key in ('A','B','C','s'):
            self.assertAlmostEqual(m[key],z[key],places=10)

    def test_keyed_nulls_distinct_gauge_invariant(self):
        cksha='ab'*32
        for k in (1,7,8,16,32):
            U=random_U(cksha,0,k,941001)
            self.assertTrue(torch.equal(U,random_U(cksha,0,k,941001)))
            self.assertEqual(fingerprint(U),fingerprint(-U))
            self.assertNotEqual(fingerprint(U),fingerprint(random_U(cksha,1,k,941001)))
            self.assertLess(float(torch.linalg.norm(U.T@U-torch.eye(k,dtype=torch.float64))),1e-12)

    def test_empirical_holm_ties_and_tags(self):
        p=[1/129,2/129,1.,1.,1.,1.]
        q=holm(p)
        self.assertAlmostEqual(q[0],6/129)
        self.assertAlmostEqual(q[1],10/129)
        null=[{'a':1.,'b':1.,'c':1.,'s':.2} for i in range(128)]
        obs={'a':0.,'b':0.,'c':0.,'s':.9}
        x=null_calibrate(obs,null)
        # Null ties are counted extreme: empirical p and Holm are conservative.
        self.assertIn('closed_quotient',x['tags'])
        self.assertIn('invariant_subspace',x['tags'])
        self.assertIn('latched_memory',x['tags'])

if __name__=='__main__':unittest.main()