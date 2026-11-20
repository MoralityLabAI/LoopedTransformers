import unittest
import numpy as np
from mechinterp.nanbeige_bpd_gym.task_rows import load_rows,score_reply
from mechinterp.nanbeige_bpd_gym.crossref import probe_and_crossref,REVISION
from mechinterp.nanbeige_bpd_gym.suite_rows import load_suite_rows,score_suite_reply

class TaskAdapterTests(unittest.TestCase):
    def test_fresh_private_answers_are_not_injected_into_unassisted_prompt(self):
        rows=load_rows(component='free');self.assertEqual(len(rows),32);self.assertEqual(len({r['root_id'] for r in rows}),8)
        self.assertTrue(all(not r['gold_assistance'] for r in rows));self.assertTrue(all('verified inverse coefficient' not in r['prompt'] for r in rows))
        self.assertTrue(all(score_reply(f"[{r['answer']}]",r['kind'],r['answer'])==1 for r in rows))
    def test_erasure_is_scored_as_two_world_macro_not_both_correct(self):
        self.assertEqual(score_reply('**R**','operator',['R','N']),.5)
        self.assertEqual(score_reply('R or N','operator',['R','N']),0)
        with self.assertRaises(ValueError):load_rows(split='test')

class CoordinateTests(unittest.TestCase):
    def data(self):
        E=np.zeros((4,3072));E[:,:4]=np.eye(4);D=E.T.copy();mean=np.zeros(3072);scale=np.ones(3072);h=np.zeros(3072);h[:4]=[1,2,3,4]
        sae={'encoder':E,'bias':np.zeros(4),'decoder':D,'mean':mean,'scale':scale}
        meta={'checkpoint_revision':REVISION,'backend':'bf16','layer_zero_based':21,'loop':2,'site':'residual_after_layer','width':3072,'artifact_sha256':'a'*64}
        return h,sae,E.copy(),meta
    def test_same_site_reconstructs_and_crossrefs_without_edit(self):
        h,sae,V,m=self.data();r=probe_and_crossref(h,sae,V,state_meta=m,sae_meta=m|{'fit_split':'calibration','fit_manifest_sha256':'b'*64},vpd_meta=m|{'factor_role':'physical_output_directions'})
        self.assertEqual(r['reconstruction_l2'],0);self.assertFalse(r['model_edit_performed']);self.assertFalse(r['BPD_transfer_attempted'])
        np.testing.assert_allclose(r['encoder_vpd_cosine'],np.eye(4))
    def test_equal_width_different_loop_or_site_is_refused(self):
        h,sae,V,m=self.data()
        for bad in (m|{'loop':1},m|{'site':'post_final_norm'},m|{'backend':'nf4'},m|{'width':8192}):
            with self.assertRaises(ValueError):probe_and_crossref(h,sae,V,state_meta=m,sae_meta=m|{'fit_split':'calibration','fit_manifest_sha256':'b'*64},vpd_meta=bad|{'factor_role':'physical_output_directions'})
    def test_dev_fit_and_vpd_input_factors_are_refused(self):
        h,sae,V,m=self.data()
        with self.assertRaises(ValueError):probe_and_crossref(h,sae,V,state_meta=m,sae_meta=m|{'fit_split':'development'},vpd_meta=m|{'factor_role':'physical_output_directions'})
        with self.assertRaises(ValueError):probe_and_crossref(h,sae,V,state_meta=m,sae_meta=m|{'fit_split':'calibration','fit_manifest_sha256':'b'*64},vpd_meta=m|{'factor_role':'input_factors'})

class WiderSuiteTests(unittest.TestCase):
    def test_public_prime_rows_preserve_native_task_scope_and_exclude_local_routes(self):
        rows=load_suite_rows();self.assertEqual(len(rows),89);self.assertEqual(len({r['root_id'] for r in rows}),41)
        self.assertEqual({r['suite'] for r in rows},{'pointer','arc1','arc2','sudoku','storyworld'})
        self.assertTrue(all(not r['private_label']['copy_control'] for r in rows))
        with self.assertRaises(ValueError):load_suite_rows(split='test')
        with self.assertRaises(ValueError):load_suite_rows(scope='unregistered')
    def test_prime_complete_reply_endpoint_refuses_trailing_prose(self):
        r=next(r for r in load_suite_rows() if r['suite']=='arc1')
        import json
        answer=json.dumps(r['private_label']['answer'])
        self.assertEqual(score_suite_reply(answer,r['schema'],r['private_label'])['correct'],1)
        self.assertEqual(score_suite_reply(answer+' Explanation follows.',r['schema'],r['private_label'])['correct'],0)

if __name__=='__main__':unittest.main()
