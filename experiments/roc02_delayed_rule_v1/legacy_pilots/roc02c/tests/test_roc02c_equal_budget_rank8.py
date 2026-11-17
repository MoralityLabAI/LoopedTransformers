"""ROC-02C numerical regression tests for the equal-budget rank-eight pilot."""
import math
import unittest
from pathlib import Path
import sys

import torch
sys.path.insert(0,str(Path(__file__).parent))
try:
    from research_gym.analysis.roc02c_equal_budget_rank8 import (
        fit_one, b_relative_squared, evaluate, ridge_for_basis, RANK, UPDATES, GAMMAS)
except ModuleNotFoundError:
    from roc02c_equal_budget_rank8 import (
        fit_one, b_relative_squared, evaluate, ridge_for_basis, RANK, UPDATES, GAMMAS)

class ROC02CGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_penalty_identity_vs_explicit_projector(self):
        g=torch.Generator().manual_seed(17)
        U=torch.linalg.qr(torch.randn(128,RANK,dtype=torch.float64,generator=g))[0]
        D=torch.randn(2,128,128,dtype=torch.float64,generator=g)*.2
        norms=torch.linalg.norm(D,dim=(1,2))
        meds=torch.tensor([.12,.2],dtype=torch.float64)
        penalty,r=b_relative_squared(U,D,norms,meds)
        Q=torch.eye(128,dtype=torch.float64)-U@U.T
        exact=torch.stack([(torch.linalg.norm(U.T@d@Q)/n/m) for d,n,m in zip(D,norms,meds)])
        self.assertLess(float(torch.max(torch.abs(r-exact))),1e-12)
        self.assertAlmostEqual(float(penalty),float((exact*exact).mean()),places=12)

    def test_prior_restored_in_classification(self):
        g=torch.Generator().manual_seed(13)
        X=torch.randn(32,128,dtype=torch.float64,generator=g)
        Z=torch.randn(13,128,dtype=torch.float64,generator=g)
        U=torch.linalg.qr(torch.randn(128,RANK,dtype=torch.float64,generator=g))[0]
        ys=torch.nn.functional.one_hot(torch.randint(0,8,(32,),generator=g),8).double()
        yd=torch.nn.functional.one_hot(torch.randint(0,8,(13,),generator=g),8).double()
        prior=ys.mean(0)
        H,Et,Ev,acc=ridge_for_basis(X,ys-prior,Z,yd-prior,U,label_prior=prior)
        target=prior+(Z@U@H)
        self.assertEqual(float(acc),float((target.argmax(-1)==yd.argmax(-1)).double().mean()))
        self.assertAlmostEqual(float(Ev),float((target-yd).square().mean()),places=12)

    def test_fixed_eighty_update_arm_and_orthogonality(self):
        g=torch.Generator().manual_seed(15)
        U0=torch.linalg.qr(torch.randn(128,8,dtype=torch.float64,generator=g))[0]
        X=torch.randn(28,128,dtype=torch.float64,generator=g)
        Z=torch.randn(12,128,dtype=torch.float64,generator=g)
        Y=torch.randn(28,8,dtype=torch.float64,generator=g)
        V=torch.randn(12,8,dtype=torch.float64,generator=g)
        D=torch.randn(2,128,128,dtype=torch.float64,generator=g)*.05
        N=torch.linalg.norm(D,dim=(1,2))
        m=torch.tensor([.16,.17],dtype=torch.float64)
        val=fit_one(U0,X,Y,Z,V,D,N,m,0.2)
        self.assertEqual(len(val['trace']),3)
        self.assertEqual(val['trace'][-1]['step'],UPDATES)
        self.assertLess(val['orthogonality_error'],1e-10)
        self.assertTrue(math.isfinite(val['calibration_J_lambda1']))
        self.assertEqual(tuple(val['U'].shape),(128,RANK))
        with self.assertRaises(ValueError):fit_one(U0,X,Y,Z,V,D,N,m,0.2,updates=1)

if __name__=='__main__':unittest.main()
