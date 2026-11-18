import json
from pathlib import Path
import unittest
import torch
from experiments.nan_comp03.parser import parse
from experiments.nan_comp03.decode import generate
from experiments.nan_comp03.prompt import body,RECIPROCAL_EXAMPLES
from experiments.nan_comp03.grade import gate
from experiments.nan_comp03.audit import answer
from experiments.nan_bpd01.verify_sources import unary

class ParserContracts(unittest.TestCase):
    def test_independent_regex_agrees_on_declared_language_and_prefixes(self):
        for op in 'INSR':
            for left,right in (('',''),('**','**'),('*','*'),('`','`')):
                for end in ('','.'):
                    text=' '+left+op+right+end+'\n'
                    expected=parse(text)
                    self.assertEqual(answer(text,False),(expected['status'],expected['operator']))
        for text in ('**R','R or N','\n','**N** prose','`S','R'*65):
            for final in (False,True):
                expected=parse(text,final=final)
                self.assertEqual(answer(text,final),(expected['status'],expected['operator']))
    def test_wrappers_and_whitespace(self):
        for s in ('R',' R ','**R**',' * R * ','`R`','\n** R **\n','R.'):
            self.assertEqual(parse(s)['operator'],'R')
    def test_no_prose_scanning_or_ambiguous_answer(self):
        for s in ('R or N','The answer is R','**R** extra','RR','Operator: R','N/R','<think>R','Reciprocal'):
            self.assertEqual(parse(s)['status'],'rejected',s)
    def test_incomplete_and_character_limits(self):
        for s in ('','**','**R','`R','   '):
            self.assertEqual(parse(s)['status'],'pending')
            self.assertEqual(parse(s,final=True)['status'],'rejected')
        self.assertEqual(parse(' '*65)['reason'],'character_limit')
    def test_reciprocal_examples_and_source_only_erasure(self):
        for x,y in RECIPROCAL_EXAMPLES:self.assertEqual(unary(x,'R'),y)
        row={'case_id':'unused','root':1,'split':'development','family':'unused','inputs':[[1,1,1,1,1],[1,-1,1,1,1]],'outputs':[[1,-1,0,0,0],[1,1,0,0,0]],'query':[1,0,1,0,1],'public_sha256':'unused'}
        other=row|{'outputs':[[1,0,1,0,1],[1,0,-1,0,1]]}
        self.assertEqual(body(row,'erased'),body(other,'erased'))
        self.assertNotEqual(body(row,'curriculum'),body(other,'curriculum'))
        self.assertNotIn('unused',body(row,'curriculum'))
        with self.assertRaises(ValueError):body(row|{'operator':'R'},'curriculum')

class Loop:
    def prepare(self,ids):return ids.float().unsqueeze(-1),None
    def run_pairs(self,items):return [h for h,ctx in items]
class Head:
    def score(self,h):return {'free_top_token_id':{99:1,1:2,2:3,3:4}[int(h[0,-1,0])]}
class Tokenizer:
    eos_token_id=999
    def decode(self,ids,**kwargs):return ''.join({1:' **',2:'R',3:'**',4:' explanation'}[i] for i in ids)

class GreedyContracts(unittest.TestCase):
    def test_unforced_greedy_stops_on_balanced_wrapper(self):
        result=generate(Loop(),Head(),Tokenizer(),{'a':[99],'b':[99]},max_tokens=8)
        self.assertEqual(result['a']['generated_ids'],[1,2,3])
        self.assertEqual(result['a']['parser']['operator'],'R')
        self.assertEqual(result['a'],result['b'])
    def test_token_budget_does_not_accept_unclosed_markup(self):
        result=generate(Loop(),Head(),Tokenizer(),{'a':[99]},max_tokens=2)['a']
        self.assertIsNone(result['parser']['operator'])
        self.assertEqual(result['stop_reason'],'token_limit')

class GateContracts(unittest.TestCase):
    def setup_data(self,wrong_R=False):
        cfg=json.loads(Path('experiments/nan_comp03/config.json').read_text());labels={};pairs=[];g={}
        for i,ops in enumerate(cfg['development_pairs']):
            p={'root':i,'split':'development','recipient':f'{i}a','donor':f'{i}b'};pairs.append(p)
            for cid,op in zip((p['recipient'],p['donor']),ops):
                labels[cid]={'operator':op};g[cid]={'curriculum':{'future_correct':int(not wrong_R or op!='R')},'erased':{'future_correct':0}}
        return g,pairs,labels,cfg
    def test_perfect_sources_pass(self):self.assertTrue(gate(*self.setup_data())['passed'])
    def test_aggregate_cannot_hide_reciprocal_failure(self):
        result=gate(*self.setup_data(wrong_R=True))
        self.assertEqual(result['future_correct'],12)
        self.assertFalse(result['requirements']['reciprocal'])
        self.assertFalse(result['passed'])
    def test_source_blind_predictor_cannot_pass(self):
        g,pairs,labels,cfg=self.setup_data()
        for cid in labels:
            g[cid]['curriculum']['future_correct']=g[cid]['erased']['future_correct']=int(labels[cid]['operator']=='R')
        result=gate(g,pairs,labels,cfg);self.assertEqual(result['both_correct_pairs'],0);self.assertFalse(result['passed'])

if __name__=='__main__':unittest.main()
