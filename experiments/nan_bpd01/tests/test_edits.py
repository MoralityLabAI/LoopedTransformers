import unittest
import torch
from experiments.nan_bpd01.edits import dose_edit,phase_direction,routed_mask,iter_phase_directions

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
    def test_lazy_directions_are_byte_identical_to_registered_eager_plan(self):
        torch.manual_seed(41);h=torch.randn(16,19,64).bfloat16();root=18211
        cfg={'phase_channel':7,'context_width':8,'neighbor_channels':[6,8],'random_channels':[17,31]}
        eager={'phase_global_suppress':phase_direction(h,7),'phase_context_suppress':phase_direction(h,7,context_width=8),
               'phase_context_replace':phase_direction(h,7,context_width=8,replacement=-2.)}
        for c in [6,8,17,31]:eager[f'channel{c}_global']=phase_direction(h,c,pattern=-h[:,:,7].float())
        for c in [17,31]:eager[f'random_dark_{c}']=phase_direction(h,c,context_width=8,pattern=-h[:,-11:-3,7].float())
        for seed in [741,947]:
            g=torch.Generator(device=h.device).manual_seed(root+seed);d=torch.zeros_like(h,dtype=torch.float32)
            d[:,-11:-3]=torch.randn((16,8,64),generator=g);eager[f'random_dense_dark_{seed}']=d
        lazy=dict(iter_phase_directions(h,cfg,root,-2.));self.assertEqual(list(eager),list(lazy))
        for name in eager:self.assertTrue(torch.equal(eager[name],lazy[name]),name)

if __name__=='__main__':unittest.main()
