import unittest
import torch
from experiments.nan_bpd01.head import ChunkedHead,program_scores,scores_at_temperatures
from mechinterp.dark_fiber_gyms.core import arg02_program_scores

class HeadContractTests(unittest.TestCase):
    def test_full_normalization_and_prefixes(self):
        torch.manual_seed(31)
        w=torch.randn(67,13,dtype=torch.float64)
        h=torch.randn(16,7,13,dtype=torch.float64)
        ids=dict(zip('INSR',(2,11,40,66)))
        rec=ChunkedHead(w,ids,device='cpu',chunk_rows=17).score([h,h.clone()])
        logits=torch.nn.functional.linear(h[:,-3:],w).float()
        for temp in (.5,1.,2.):
            lp=torch.log_softmax(logits.double()/temp,dim=-1).float()
            expected=arg02_program_scores(lp,ids)
            got=scores_at_temperatures(rec[0],(temp,)*3)
            self.assertLess(max(abs(a-b) for a,b in zip(expected,got)),2e-5)
        self.assertEqual(rec[1]['full_vocab_max_logprob_error_vs_reference'],0.)
        self.assertTrue(torch.equal(torch.tensor(rec[0]['operator_logits']),logits[:,:,[2,11,40,66]]))

    def test_stable_extreme_logits_and_all_rows(self):
        torch.manual_seed(7)
        w=torch.randn(71,5)*100
        h=torch.randn(16,5,5)*100
        out=ChunkedHead(w,dict(zip('INSR',(0,10,35,70))),device='cpu',chunk_rows=13).score([h])[0]
        self.assertTrue(torch.isfinite(torch.tensor(out['scores'])).all())
        # A four-token denominator must fail this adversarial comparison.
        raw=torch.nn.functional.linear(h[:,-3:],w).float()
        restricted=torch.log_softmax(raw[:,:,[0,10,35,70]],-1)
        self.assertGreater(max(abs(a-b) for a,b in zip(out['scores'],program_scores(restricted))),100)

    def test_full_vocab_difference_moments(self):
        torch.manual_seed(9)
        w=torch.randn(59,11)
        a=torch.randn(16,4,11);b=a+0.02*torch.randn_like(a)
        ids=dict(zip('INSR',(3,9,20,58)))
        result=ChunkedHead(w,ids,device='cpu',chunk_rows=7).score([a,b])
        x=torch.log_softmax(torch.nn.functional.linear(a[:,-3:],w).double(),-1)
        y=torch.log_softmax(torch.nn.functional.linear(b[:,-3:],w).double(),-1)
        self.assertAlmostEqual(result[1]['full_vocab_logprob_l2_vs_reference'],float((y-x).norm()),places=5)
        self.assertAlmostEqual(result[1]['full_vocab_max_logprob_error_vs_reference'],float((y-x).abs().max()),places=5)

    def test_bias_and_invalid_operator_ids(self):
        torch.manual_seed(2)
        w=torch.randn(13,5);h=torch.randn(16,3,5);bias=torch.randn(13)
        ids=dict(zip('INSR',(0,3,8,12)))
        r=ChunkedHead(w,ids,device='cpu',chunk_rows=8,bias=bias).score([h])[0]
        lp=torch.log_softmax(torch.nn.functional.linear(h,w,bias).double(),-1).float()
        self.assertLess(max(abs(a-b) for a,b in zip(r['scores'],arg02_program_scores(lp,ids))),1e-5)
        with self.assertRaises(ValueError):ChunkedHead(w,dict(zip('INSR',(0,0,3,4))),device='cpu')

if __name__=='__main__':unittest.main()
