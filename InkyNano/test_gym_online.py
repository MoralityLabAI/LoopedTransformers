import unittest
import torch
from gym_online import same_prefix

class OnlineTraceTests(unittest.TestCase):
    def test_unsorted_native_topk_has_same_identity(self):
        a={6:[(torch.tensor([[0,1,2,3,4,5,6,7]]),torch.tensor([[.1,.2,.3,.4,.5,.6,.7,.8]]))]}
        b={6:[(torch.tensor([[7,6,5,4,3,2,1,0]]),torch.tensor([[.8,.7,.6,.5,.4,.3,.2,.1]]))]}
        result=same_prefix(a,b,1)
        self.assertEqual(result['id_mismatch_rows'],0)
        self.assertAlmostEqual(result['max_abs_weight_diff'],0.)
    def test_mismatch_is_not_accepted(self):
        a={0:[(torch.tensor([[0,1,2,3,4,5,6,7]]),torch.ones(1,8))]}
        b={0:[(torch.tensor([[0,1,2,3,4,5,6,8]]),torch.ones(1,8))]}
        self.assertEqual(same_prefix(a,b,1)['id_mismatch_rows'],1)

if __name__=='__main__':unittest.main()
