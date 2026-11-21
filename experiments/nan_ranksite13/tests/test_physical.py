import ast
from pathlib import Path
import unittest
import numpy as np
import torch
from experiments.nan_ranksite13.physical_audit import bf16
from experiments.nan_ranksite13.diagnostic_audit import diagnostics_audit
from experiments.nan_ranksite13.common import condition,key


class PhysicalAuditContracts(unittest.TestCase):
    def test_cpu_BF16_rounding_matches_native_cast_including_signed_ties(self):
        rng=np.random.default_rng(13);values=rng.normal(size=20000).astype(np.float32)*100
        ties=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x80000000],dtype=np.uint32).view(np.float32)
        values=np.concatenate([values,ties])
        actual=torch.from_numpy(values).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(bf16(values),actual)

    def test_rank16_diagnostic_retains_all_root_loop_checks_and_rejects_wrong_static_shape(self):
        public=[];gold={};records=[]
        for split in ('calibration','development'):
            for i in range(8):
                name=f'{split}{i}';public.append(dict(id=name,root=name,split=split,copy_control=False,schema=dict(type='label',allowed=['A','B'])))
                gold[name]='A';phases=[]
                for t in (0,1):
                    phases.append(dict(pass_index_zero_based=t,free_token_id=1,free_token_text='A',full_log_normalizer=1.,free_token_logprob=-1.,
                        full_vocabulary_KL=0.,actual_last_token_module_write_relative_l2=0.,local_intervention_l2_at_actual_reached_input=0.,
                        local_intervention_relative_l2_at_actual_reached_input=0.,reached_input_relative_l2=0.,reached_vs_clean_mask_max_abs=None))
                records.append(dict(id=name,root=name,split=split,mode='component_only',phases=phases))
        data=dict(records=records,static_CAL_mask=np.ones((2,16)).tolist())
        result=diagnostics_audit(data,public,gold,2,['component_only'],16)
        self.assertEqual(len(result),4);self.assertTrue(all(r['correct']==8 for r in result))
        data['static_CAL_mask']=np.ones((2,8)).tolist()
        with self.assertRaises(AssertionError):diagnostics_audit(data,public,gold,2,['component_only'],16)

    def test_worker_condition_key_and_oracle_isolation_do_not_depend_on_dependency_loop_names(self):
        self.assertEqual(len({key(condition(i)) for i in range(6)}),6)
        p=Path('experiments/nan_ranksite13/worker.py');t=p.read_text();tree=ast.parse(t)
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.endswith(('.report','.gate','.probe','.physical_audit','.cell_audit')) for n in imports))
        for n in ast.walk(tree):
            if isinstance(n,ast.For):self.assertNotEqual(getattr(n.target,'id',None),'name')
        self.assertNotIn('host_labels.jsonl',t);self.assertIn("public_file=parent/'fixtures/public.jsonl'",t)


if __name__=='__main__':unittest.main()
