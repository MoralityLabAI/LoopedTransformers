from copy import deepcopy
import unittest
from mechinterp.nanbeige_bpd_gym.audit_fit import registered_sources,check_steps,diagnostics_audit


class FitAuditTests(unittest.TestCase):
    def setUp(self):
        self.seal,self.cfg,self.public,self.gold=registered_sources()

    def test_rejects_DEV_training_even_with_valid_numeric_losses(self):
        cal=[r for r in self.public if r['split']=='calibration' and not r['copy_control']]
        steps=[dict(step=i,source_id=r['id'],mode='closed_loop',faith_loss=1e-4,min_loss=.2,stoch_loss=.4,adv_loss=.8,total_loss=10.8) for i,r in enumerate(cal)]
        self.assertTrue(check_steps(steps,self.public,self.cfg)['four_term_loss_accounting_checked'])
        bad=deepcopy(steps);bad[0]['source_id']=next(r['id'] for r in self.public if r['split']=='development')
        with self.assertRaises(AssertionError):check_steps(bad,self.public,self.cfg)
        bad=deepcopy(steps);bad[0]['total_loss']+=.1
        with self.assertRaises(AssertionError):check_steps(bad,self.public,self.cfg)

    def test_requires_every_root_once_and_preserves_all_one_closure(self):
        phase=dict(pass_index_zero_based=0,free_token_id=1,free_token_text='A',full_log_normalizer=4.,free_token_logprob=-1.,
            full_vocabulary_KL=0.,actual_last_token_module_write_relative_l2=0.,local_intervention_l2_at_actual_reached_input=0.,
            local_intervention_relative_l2_at_actual_reached_input=0.,reached_input_relative_l2=0.,reached_vs_clean_mask_max_abs=None)
        records=[dict(id=r['id'],root=r['root'],split=r['split'],mode='all_one',phases=[dict(phase)]) for r in self.public if not r['copy_control']]
        data=dict(records=records,static_CAL_mask=[[.5]*8])
        self.assertEqual(sum(r['contexts'] for r in diagnostics_audit(data,self.public,self.gold,1,['all_one'])),16)
        bad=deepcopy(data);bad['records'].pop()
        with self.assertRaises(AssertionError):diagnostics_audit(bad,self.public,self.gold,1,['all_one'])
        bad=deepcopy(data);bad['records'][0]['phases'][0]['actual_last_token_module_write_relative_l2']=.01
        with self.assertRaises(AssertionError):diagnostics_audit(bad,self.public,self.gold,1,['all_one'])


if __name__=='__main__':unittest.main()
