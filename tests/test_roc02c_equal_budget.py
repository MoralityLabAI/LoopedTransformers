"""ROC-02C independent numeric/provenance tests, CPU and source only."""
import json
import math
import os
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from roc02c_equal_budget import (RANKS,GAMMAS,SEEDS,DEPTHS,UPDATES,
    data_matrices,score_components,random_U,null_medians,b_fraction_squared,optimize)
from roc02c_summarize import audit

class ROC02CChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_grid_80(self):
        self.assertEqual(RANKS,(7,8));self.assertEqual(GAMMAS,(0,.05,.2,1.))
        self.assertEqual(UPDATES,80);self.assertEqual(len(SEEDS)*len(DEPTHS),12)

    def test_exact_train_mse_quadratic_identity(self):
        torch.manual_seed(307)
        H=torch.randn(30,128,dtype=torch.float64)
        y=torch.randint(0,8,(30,))
        mat=data_matrices(H,y)
        U=torch.linalg.qr(torch.randn(128,7,dtype=torch.float64))[0]
        mse,W=score_components(U,*mat[:3],len(H))
        x=H-H.mean(0); target=torch.nn.functional.one_hot(y,8).double(); target-=target.mean(0)
        ref=(target-x@U@W).square().mean()
        self.assertLess(abs(float(mse-ref)),1e-12)

    def test_B_fraction_direct_projector(self):
        torch.manual_seed(18)
        D=torch.randn(128,128,dtype=torch.float64)*.03
        U=torch.linalg.qr(torch.randn(128,7,dtype=torch.float64))[0]
        med=.14
        val=b_fraction_squared(U,D,med)
        Q=torch.eye(128,dtype=torch.float64)-U@U.T
        ref=(U.T@D@Q).square().sum()/D.square().sum()/med**2
        self.assertLess(float(abs(val-ref)),1e-12)

    def test_null_fingerprint(self):
        torch.manual_seed(12)
        D=torch.randn(128,128,dtype=torch.float64)*.04
        T=torch.eye(128,dtype=torch.float64)+D
        one=null_medians([T],'ab'*32,7,(0,))[0]
        two=null_medians([T],'ab'*32,7,(0,))[0]
        self.assertEqual(one,two)
        self.assertGreater(one['median_b'],0)
        self.assertTrue(one['null_128_distinct'])

    def test_optimizer_budget_and_frozen_U(self):
        torch.manual_seed(13)
        H=torch.randn(48,128,dtype=torch.float64)
        y=torch.randint(0,8,(48,))
        mats=data_matrices(H,y)
        U=torch.linalg.qr(torch.randn(128,7,dtype=torch.float64))[0]
        J=torch.randn(128,128,dtype=torch.float64)*.01+torch.eye(128,dtype=torch.float64)
        m=[{'state_index':i,'median_b':.04} for i in (0,1)]
        original=U.clone()
        out=optimize(U,*mats[:3],len(H),[J-torch.eye(128,dtype=torch.float64)]*2,m,0.05)
        self.assertTrue(torch.equal(U,original))
        self.assertLess(float(torch.linalg.norm(out.T@out-torch.eye(7,dtype=torch.float64))),1e-12)

    def test_receipts_calibration_only(self):
        root=Path(os.environ.get('ROC02C_RECEIPTS_DIR','/mnt/data/roc02c/receipts'))
        if not root.is_dir():self.skipTest('source receipts unavailable')
        rows,nulls,summary=audit(root)
        self.assertEqual(len(rows),96)
        self.assertEqual(len(nulls),144)
        self.assertFalse(summary['sealed_test_evaluated'])
        self.assertEqual(sum(v['joint_J_better_count'] for v in summary['by_depth_rank']),24)

if __name__=='__main__':unittest.main()
