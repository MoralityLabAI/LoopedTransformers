import unittest
import torch
from experiments.nan_bpd01.edits import dose_edit,phase_direction,routed_mask

class EditsTest(unittest.TestCase):
    def test_physical_norm_after_bfloat_rounding(self):
        torch.manual_seed(37);h=torch.randn(16,32,64).bfloat16()
        a,ra=dose_edit(h,phase_direction(h,7,context_width=8))
        b,rb=dose_edit(h,phase_direction(h,11,context_width=8,pattern=-h[:,-11:-3,7].float()))
        self.assertTrue(ra['eligible'] and rb['eligible']);self.assertTrue(torch.equal(a[:,-3:],h[:,-3:]))
        self.assertLess(abs(ra['actual_l2']/rb['actual_l2']-1),.01)
        z,rz=dose_edit(h,torch.zeros_like(h));self.assertTrue(rz['zero_dose']);self.assertFalse(rz['eligible'])
    def test_reached_state_mask_differs_from_clean_oracle(self):
        clean=torch.tensor([[4.,1.,0.]]);reached=torch.tensor([[0.,1.,5.]])
        self.assertFalse(torch.equal(routed_mask(clean,torch.eye(3),1),routed_mask(reached,torch.eye(3),1)))

if __name__=='__main__':unittest.main()
