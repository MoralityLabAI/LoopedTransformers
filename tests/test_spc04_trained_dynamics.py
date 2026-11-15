"""SPC-04: exact native pointer labels and controlled training protocol."""
import math
import unittest

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import (
    new_model, train_model, evaluate_capability, diagnostic_trajectory,
    prompt_and_recurrence, weight_sha256,
)
from experiments.spectral_state_cost_v0.spc04_floquet_cycle import (
    pointer_cycle_length, phase_alignment_report, floquet_probe,
)


class TrainedDynamicsTests(unittest.TestCase):
    def test_deterministic_training_and_matched_weights(self):
        model,init,receipt=train_model(41522,steps=12,batch=8)
        self.assertEqual(receipt['initial_weights_sha256'],weight_sha256(init))
        self.assertNotEqual(receipt['trained_weights_sha256'],weight_sha256(init))
        model2,init2,receipt2=train_model(41522,steps=12,batch=8)
        self.assertEqual(receipt['trained_weights_sha256'],receipt2['trained_weights_sha256'])
        self.assertEqual(receipt['initial_weights_sha256'],receipt2['initial_weights_sha256'])

    def test_labels_are_per_loop_not_frozen_third_hop(self):
        from research_gym.scripts.loopscope_toy import make_batch
        t3,h3=make_batch(8,4,881001,3)
        t42,h42=make_batch(8,4,881001,42)
        self.assertTrue((t3==t42).all().item())
        self.assertTrue((h3==h42[:,:3]).all().item())
        self.assertEqual(h42.shape[1],42)

    def test_capability_metric_shapes(self):
        m=new_model(41522)
        r=evaluate_capability(m,eval_items=10,eval_seeds=(881001,))
        self.assertEqual(len(r['mean_aligned_hop_accuracies']),3)
        self.assertEqual(set(r['mean_current_hop_accuracy_by_depth']),{'1','2','3','6','12','24','42'})
        self.assertTrue(all(0<=v<=1 for v in r['mean_current_hop_accuracy_by_depth'].values()))

    def test_native_full_state_replay(self):
        m=new_model(41522).eval().double()
        step,seq,tokens,hops,gap=prompt_and_recurrence(m,982001,loops=48)
        self.assertEqual(gap,0)
        self.assertEqual(seq[0].shape,(1,9,48))
        self.assertEqual(len(seq),49)

    def test_graph_cycle_length(self):
        import torch
        # Three-cycle (0,2,1); all other nodes fixed.
        edges=[(0,2),(1,0),(2,1),(3,3),(4,4),(5,5),(6,6),(7,7)]
        data=[v for e in edges for v in e]
        self.assertEqual(pointer_cycle_length(torch.tensor([data+[0]])),3)
        self.assertEqual(pointer_cycle_length(torch.tensor([data+[3]])),1)

    def test_untrained_control_has_fixed_point(self):
        r=phase_alignment_report(new_model(41522),n_prompts=12,loops=90,comparison_depth=70)
        self.assertLess(r['median_one_lag_residual'],1e-7)
        self.assertEqual(r['n_nontrivial_stable_matching_period'],0)

    def test_full_cycle_probe_without_arnoldi(self):
        r=floquet_probe(new_model(41522),seed=982001,
                        comparison_depth=20,loops=45,include_eigs=False)
        self.assertEqual(r['full_state_dim'],432)
        self.assertLess(r['max_native_forward_gap'],1e-10)
        self.assertTrue(math.isfinite(r['one_step_relative_residual']))
        self.assertNotIn('floquet_spectral_radius_estimate',r)

if __name__=='__main__':unittest.main()
