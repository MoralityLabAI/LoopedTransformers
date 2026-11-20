import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.nan_diag04.parser import parse
from experiments.nan_diag04.prompt import make_prompts
from experiments.nan_diag04.grade import root_metrics,diagnose
from experiments.nan_diag04.telemetry import Telemetry
from experiments.nan_diag04.audit import final_value
from experiments.nan_diag04.decode import generate
from experiments.nan_bpd01.verify_sources import unary
import torch

class ParserTests(unittest.TestCase):
    def test_independent_regex_acceptance_agrees(self):
        for s in ('[17]','**17**','`-3`','**[-17]**','17\n','-0.','1 7','[1,7]','[1000]'):
            p=parse(s,'integer');self.assertEqual(final_value(s,'integer',False),p['value'])
    def test_multidigit_not_accepted_without_delimiter(self):
        self.assertEqual(parse('1','integer')['status'],'pending')
        self.assertEqual(parse('17','integer')['status'],'pending')
        self.assertEqual(parse('17','integer',eos=True)['value'],17)
        self.assertEqual(parse('17','integer',limit=True)['status'],'rejected')
        for s in ('[17]','**17**','**[17]**','`17`','17\n','17.','−17\n','[ -17 ]'):
            self.assertEqual(abs(parse(s,'integer')['value']),17)
    def test_prose_multiple_numbers_and_unclosed_wrappers_rejected(self):
        for s in ('1 7','[1,7]','It is 17','17.0','[1.','[1000]','**17** extra'):
            self.assertEqual(parse(s,'integer')['status'],'rejected',s)
        self.assertEqual(parse('**17','integer',limit=True)['status'],'rejected')
    def test_operator_and_code_alphabets_are_separate(self):
        for s,v,k in [('**R**','R','operator'),('`D`','D','code')]:self.assertEqual(parse(s,k)['value'],v)
        self.assertEqual(parse('R','code')['status'],'rejected')
        self.assertEqual(parse('D','operator')['status'],'rejected')
        self.assertEqual(parse('R or N','operator')['status'],'rejected')

class SourceTests(unittest.TestCase):
    def setup_source(self):
        xs=[[1,1,-2,1,2],[1,-1,0,2,0]];refs={op:[unary(x,op) for x in xs] for op in 'INSR'}
        return xs,refs,make_prompts(xs,refs,['R','N'],dict(zip('INSR','ABCD')),list('INSR'))
    def test_assistance_is_local_and_explicit(self):
        xs,refs,p=self.setup_source();self.assertEqual(len(p),24)
        self.assertNotIn('R(f)',p['free/b4']['text']);self.assertNotIn('Entry',p['select/0']['text'])
        self.assertIn('Entry R:',p['table/0']['text']);self.assertNotIn('Entry R:',p['coded/0']['text'])
        self.assertIn('verified operator label is R',p['labelcopy/0']['text'])
        self.assertEqual(p['erased'],p['erased_repeat'])
    def test_full_vector_requires_every_coefficient(self):
        xs,refs,p=self.setup_source();c={'refs':refs,'worlds':['R','N'],'codes':dict(zip('INSR','ABCD'))}
        pred={}
        for arm in ('free','scaffold','copy'):
            for n in range(1,5):pred[f'{arm}/b{n}']=refs['R'][0][n]
        for i,op in enumerate(c['worlds']):
            for arm in ('select','table','coded','labelcopy','codecopy'):pred[f'{arm}/{i}']=c['codes'][op] if arm in ('coded','codecopy') else op
        pred['erased']=pred['erased_repeat']='R';m=root_metrics(c,pred)
        self.assertEqual(m['free_full'],1);self.assertEqual(m['controls'],1);self.assertEqual(m['erased_macro'],.5);self.assertEqual(m['erased_both'],0)
        pred['free/b4']+=1;self.assertEqual(root_metrics(c,pred)['free_full'],0)
    def test_cannot_diagnose_arithmetic_from_failed_copy_controls(self):
        s={'controls':7,'free_full':0,'scaffold_full':0,'table_R':8,'table_alt':8,'coded_R':8,'coded_alt':8,'select_R':0}
        self.assertEqual(diagnose(s),'readout_control_failure');s['controls']=8
        self.assertEqual(diagnose(s),'arithmetic_bottleneck_compatible');s['scaffold_full']=8
        self.assertEqual(diagnose(s),'scaffold_sensitive_computation')

class TelemetryTests(unittest.TestCase):
    def test_live_log_failure_is_isolated_from_inference(self):
        t=Telemetry(Path('unused.json'));t.start=0
        replies=[SimpleNamespace(stdout='0, 4096',returncode=0),SimpleNamespace(stdout='[]',returncode=0)]
        with patch('experiments.nan_diag04.telemetry.subprocess.run',side_effect=replies),patch.object(Path,'open',side_effect=PermissionError('test')):
            t.sample()
        self.assertEqual(len(t.samples),1);self.assertEqual(len(t.errors),1)

class Loop:
    def prepare(self,ids):return ids.float().unsqueeze(-1),None
    def run_pairs(self,items):return [h for h,c in items]
class Head:
    def score(self,h):return {'free_top_token_id':{99:1,1:2,2:3,98:10,10:11,11:12}[int(h[0,-1,0])]}
class Tokenizer:
    eos_token_id=9999
    def decode(self,ids,**kw):return ''.join({1:'1',2:'7',3:'\n',10:'**',11:'R',12:'**'}[i] for i in ids)
class DecodeTests(unittest.TestCase):
    def test_numeric_boundary_and_operator_stop_are_independent(self):
        out=generate(Loop(),Head(),Tokenizer(),{'n':[99],'o':[98]},{'n':'integer','o':'operator'},{'maximum_integer_tokens':12,'maximum_operator_tokens':8})
        self.assertEqual(out['n']['parser']['value'],17);self.assertEqual(out['n']['generated_ids'],[1,2,3])
        self.assertEqual(out['o']['parser']['value'],'R')
    def test_budget_truncation_cannot_pass_as_a_complete_number(self):
        out=generate(Loop(),Head(),Tokenizer(),{'n':[99]},{'n':'integer'},{'maximum_integer_tokens':2,'maximum_operator_tokens':8})
        self.assertEqual(out['n']['parser']['status'],'rejected');self.assertIsNone(out['n']['parser']['value'])

if __name__=='__main__':unittest.main()
