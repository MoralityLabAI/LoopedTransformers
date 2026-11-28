import unittest
import torch
from experiments.nan_bpd01.edits import shared_random_dark,prefix_hidden_error,dose_edit,iter_phase_directions
from experiments.nan_bpd01.head import ChunkedHead,prefix_logprob_spread

class PrefixContractTest(unittest.TestCase):
    def test_default_plan_uses_shared_noise_and_legacy_requires_explicit_opt_in(self):
        h=torch.randn(1,19,32).repeat(16,1,1).bfloat16();cfg={'phase_channel':7,'context_width':8,'neighbor_channels':[6,8],'random_channels':[17,31]}
        plan=dict(iter_phase_directions(h,cfg,18211,-2.))
        self.assertNotIn('random_dense_dark_741',plan)
        self.assertTrue(torch.equal(plan['random_shared_dark_741'],plan['random_shared_dark_741'][0:1].expand_as(h)))
    def test_random_control_preserves_shared_prefix_before_and_after_causal_toy(self):
        torch.manual_seed(67);h=torch.randn(1,19,32).repeat(16,1,1).bfloat16()
        h[:,-2:,0]=torch.arange(16).view(16,1).bfloat16()
        # Second scored position must agree within each first-op group.
        h[:,-2,0]=torch.arange(4).repeat_interleave(4).bfloat16()
        direction=shared_random_dark(h,width=8,seed=18952)
        self.assertTrue(torch.equal(direction,direction[0:1].expand_as(direction)))
        edited,dose=dose_edit(h,direction);self.assertTrue(dose['eligible'])
        def causal(x):return x.float().cumsum(1).to(x.dtype)
        self.assertEqual(prefix_hidden_error(causal(edited)),0)
        head=ChunkedHead(torch.randn(101,32).bfloat16(),dict(zip('INSR',[1,2,3,4])),device='cpu',chunk_rows=17)
        self.assertEqual(prefix_logprob_spread(head.score([causal(edited)])[0]),0)
        bad=direction.clone();bad[1,-5,0]+=2
        bad_state=(h.float()+bad).bfloat16();self.assertGreater(prefix_hidden_error(causal(bad_state)),0)

if __name__=='__main__':unittest.main()
