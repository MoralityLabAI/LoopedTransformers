import copy
import math
import unittest
import torch
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp02.prompt import body,TUTORIAL,aliases
from experiments.nan_bpd01.verify_sources import unary

class Contracts(unittest.TestCase):
    def test_aliases_include_bare_piece_when_encoder_prepends_spaces(self):
        class T:
            def encode(self,text,**kw):return [2*'INSR'.index(text.strip())+2]
            def convert_tokens_to_ids(self,text):return 2*'INSR'.index(text)+1
            def decode(self,values,**kw):
                i=values[0];return (' ' if i%2==0 else '')+'INSR'[(i-1)//2]
        class Wrapper:tok=T()
        self.assertEqual(aliases(Wrapper()),{'I':[1,2],'N':[3,4],'S':[5,6],'R':[7,8]})
    def test_full_vocabulary_and_alias_marginals(self):
        torch.manual_seed(31);w=torch.randn(113,17,dtype=torch.float64);h=torch.randn(1,7,17,dtype=torch.float64)
        alias={'I':[1,4],'N':[9,10],'S':[61,62],'R':[111,112]};r=OperatorHead(w,alias,device='cpu',chunk_rows=19).score(h)
        raw=torch.nn.functional.linear(h[:,-1],w).float().flatten();z=float(torch.logsumexp(raw.double(),0))
        self.assertLess(abs(r['full_log_normalizer']-z),1e-12);self.assertEqual(r['free_top_token_id'],int(raw.argmax()))
        for op,ids in alias.items():self.assertLess(abs(r['class_logprobs'][op]-(float(torch.logsumexp(raw[ids].double(),0))-z)),1e-12)
    def test_unrestricted_output_is_not_forced_into_four_choices(self):
        w=torch.zeros(25,4);w[24]=20;h=torch.ones(1,3,4);alias=dict(zip('INSR',[[1,2],[3,4],[5,6],[7,8]]))
        r=OperatorHead(w,alias,device='cpu',chunk_rows=7).score(h)
        self.assertIsNone(r['free_operator']);self.assertEqual(r['free_top_token_id'],24);self.assertLess(r['grammar_mass'],1e-20)
        with self.assertRaises(ValueError):OperatorHead(w,{'I':[1],'N':[1],'S':[2],'R':[3]},device='cpu')
    def test_alias_mass_and_argmax_ties_are_explicit(self):
        w=torch.zeros(19,1);w[1]=3;w[3]=2.9;w[4]=2.9;h=torch.ones(1,3,1)
        alias={'I':[1,2],'N':[3,4],'S':[5,6],'R':[7,8]};r=OperatorHead(w,alias,device='cpu',chunk_rows=3).score(h)
        self.assertEqual(r['free_operator'],'I');self.assertEqual(r['constrained_operator'],'N')
        w[17]=4;w[18]=4;r=OperatorHead(w,alias,device='cpu',chunk_rows=3).score(h);self.assertEqual(r['free_top_token_id'],17)
    def test_erasure_removes_pair_target_information_and_rejects_labels(self):
        r={'case_id':'a','root':1,'split':'development','family':'dense_p4','inputs':[[1,1,1,0,0],[1,-1,0,1,0]],'outputs':[[1,-1,1,0,0],[1,1,0,-1,0]],'query':[1,1,0,0,0],'public_sha256':'x'}
        d=copy.deepcopy(r);d['case_id']='b';d['outputs']=[[1,0,1,0,1],[1,0,-1,0,0]]
        self.assertEqual(body(r,'erased'),body(d,'erased'));self.assertNotEqual(body(r,'worked'),body(d,'worked'))
        with self.assertRaises(ValueError):body(r|{'operator':'N'},'worked')
        expected={'I':[1,1,0,0,0],'N':[1,-1,0,0,0],'S':[1,0,1,0,0],'R':[1,-1,1,-1,1]}
        for op,y in expected.items():self.assertEqual(unary([1,1,0,0,0],op),y);self.assertIn(str(y),TUTORIAL)

if __name__=='__main__':unittest.main()
