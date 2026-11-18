import unittest
import torch
from experiments.nan_bpd01.components import canonical_factors,spectral_groups

class DegenerateGroupTest(unittest.TestCase):
    def test_grouped_write_survives_rotation_of_equal_singular_components(self):
        v=torch.diag(torch.tensor([3.,3.,1.],dtype=torch.float64).sqrt());u=v.clone()
        g=torch.tensor([[.6,-.8,0.],[.8,.6,0.],[0.,0.,1.]],dtype=torch.float64)
        rotated_v=v@g;rotated_u=g.T@u
        self.assertTrue(torch.allclose(v@u,rotated_v@rotated_u,atol=1e-14,rtol=0))
        self.assertGreater(float((v[:,:1]@u[:1]-rotated_v[:,:1]@rotated_u[:1]).norm()),1)
        cv,cu,s=canonical_factors(v,u);rv,ru,rs=canonical_factors(rotated_v,rotated_u)
        groups=spectral_groups(s);self.assertEqual(groups,[[0,1],[2]])
        # A group selection retains BOTH ambiguous identities, not one label.
        self.assertTrue(torch.allclose(cv[:,groups[0]]@cu[groups[0]],rv[:,groups[0]]@ru[groups[0]],atol=1e-13,rtol=0))

if __name__=='__main__':unittest.main()
