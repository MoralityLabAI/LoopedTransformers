import copy
import unittest
from pathlib import Path
from experiments.nan_story23.common import ROOT,read,rows,qualified
from experiments.nan_story23.parser import parse
from experiments.nan_story23.oracle import grade,trace_packet
from experiments.nan_story23.source_audit import validate
from experiments.nan_story23.audit import coverage
from experiments.nan_prime20.wire import OrderedSlots


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public=rows(ROOT/'fixtures/public.jsonl');cls.host=rows(ROOT/'fixtures/host_labels.jsonl')

    def test_all_sources_independent_replay(self):
        self.assertEqual(validate(self.public,self.host)['registered_trajectories'],144)

    def test_parser_rejects_rescue_and_accepts_whitespace(self):
        r=self.public[0];g=self.host[0];text=g['canonical_trace']
        for wrong in ('Answer: '+text,text+' extra',text.replace(';END',''),'x'*513, text.replace('PATH=','PATH=X>')):
            self.assertIsNone(parse(wrong,r['schema']))
            self.assertIsNone(trace_packet(wrong,r['schema']))
        for value in (text,' \n'+text+'\n',text.replace('=', ' = ').replace('>', ' > ')):
            self.assertEqual(parse(value,r['schema']),trace_packet(value,r['schema']))
            self.assertEqual(grade(value,r['schema'],g)['trace_exact'],1)

    def test_component_grades_remain_separate(self):
        r=self.public[0];g=self.host[0]
        text=g['canonical_trace'].replace('FINAL='+g['final_room'],'FINAL=R'+str((int(g['final_room'][1])+1)%8))
        result=grade(text,r['schema'],g)
        self.assertEqual(result['format_valid'],1);self.assertEqual(result['path_correct'],1)
        self.assertEqual(result['intermediate_correct'],[1,1,1]);self.assertEqual(result['final_correct'],0)

    def test_ordered_slots_never_retry(self):
        cfg=read(ROOT/'config.json');slots=OrderedSlots(self.public)
        def body(r):return dict(model='nanbeige-451ed48c-nf4',messages=[dict(role='user',content=r['prompt'])],max_tokens=64)
        i,k,r=slots.claim(body(self.public[0]),'nf4',cfg)
        with self.assertRaises(ValueError):slots.claim(body(self.public[0]),'nf4',cfg)
        slots.commit(r['id'])
        with self.assertRaises(ValueError):slots.claim(body(self.public[0]),'nf4',cfg)
        i,k,r=slots.claim(body(self.public[1]),'nf4',cfg);self.assertEqual(i,1)

    def test_missing_is_bounded_not_zero_scored(self):
        groups=coverage(self.public,{},set())
        self.assertEqual(sum(r['registered_trajectories'] for r in groups),144)
        self.assertTrue(all(r['final_correct']['whole_roster_upper_bound']==1 for r in groups))
        self.assertTrue(all(r['graded_trajectories']==0 for r in groups))

    def test_two_step_fit_never_qualifies(self):
        audit=dict(source_manifest_sha256=read(ROOT/'config.json')['FIT16_IO18_seal'],status='passed',
            cells=[dict(passes=2,status='passed',training=dict(optimizer_steps=2))])
        self.assertFalse(qualified(audit,2));audit['cells'][0]['training']['optimizer_steps']=16
        self.assertTrue(qualified(audit,2))

    def test_cache_and_model_imports_do_not_import_oracle(self):
        import ast
        for name in ('engine.py','server.py','instrument.py','adapters.py'):
            tree=ast.parse((ROOT/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse(any(part in (node.module or '').split('.') for part in ('oracle','taskset','worlds','sources','source_audit')))
        self.assertLessEqual(read(ROOT/'fixtures/tokenizer_receipt.json')['maximum_canonical_trace_tokens'],64)

    def test_SAE_report_uses_equal_root_weights(self):
        from experiments.nan_story23.report import root_probe_groups
        data=[]
        for root,values in [('a',[0,0]),('b',[1])]:
            for value in values:
                data.append(dict(root=root,split='development',query_type='actual',event_count=32,arm='main',mode='native',
                    phase_metrics=[dict(layer_zero_based=21,loop=2,normalized_MSE=value,random_initial_MSE=value,
                        physical_reconstruction_L2=value,feature_active_fraction=[value]*128)]))
        result=root_probe_groups(data)[0]
        self.assertEqual(result['independent_roots'],2)
        self.assertEqual(result['normalized_MSE_root_mean'],.5)
        self.assertEqual(result['feature_active_fraction_root_mean'],[.5]*128)


if __name__=='__main__':unittest.main()
