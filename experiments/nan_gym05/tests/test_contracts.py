import unittest
import torch
import numpy as np
from experiments.nan_gym05.parser import parse
from experiments.nan_gym05.sources import materialize,pointer_root
from experiments.nan_gym05.common import read
from experiments.nan_gym05.grade import score_value
from experiments.nan_gym05.sae import SmallSAE
from experiments.nan_gym05.linear import decompose

class GrammarTests(unittest.TestCase):
    def test_whole_grid_and_closed_fence(self):
        s={'type':'grid'};tick=chr(96)*3
        self.assertEqual(parse('[[1,2],[3,4]]',s)['value'],[[1,2],[3,4]])
        self.assertEqual(parse(tick+'json\n[[1,2],[3,4]]',s)['status'],'pending')
        self.assertEqual(parse(tick+'json\n[[1,2],[3,4]]\n'+tick,s)['status'],'accepted')
        self.assertEqual(parse('The grid is [[1]]',s)['status'],'rejected')
        self.assertEqual(parse('[[1],[2,3]]',s)['status'],'rejected')
    def test_no_scanning_or_incomplete_plan_fallback(self):
        s={'type':'plan','allowed':['wait','rush']}
        self.assertEqual(parse('["wait","rush"]',s)['value'],['wait','rush'])
        self.assertEqual(parse('["rush"',s,final=True)['status'],'rejected')
        self.assertEqual(parse('["invented"]',s)['status'],'rejected')
        self.assertEqual(parse('A or B',{'type':'label','allowed':['A','B']})['status'],'rejected')
    def test_bare_and_wrapped_environment_names(self):
        s={'type':'label','allowed':['arc_easy','gsm8k']}
        self.assertEqual(parse('**arc_easy**',s)['value'],'arc_easy')
        self.assertEqual(parse('arc_',s)['status'],'pending')
        self.assertEqual(parse('arc_',s,final=True)['status'],'rejected')

class SourceTests(unittest.TestCase):
    def test_native_suite_sources_and_calibration_separation(self):
        public,host=materialize(read('experiments/nan_gym05/config.json'))
        self.assertEqual(len(public),290);self.assertEqual(len(host),290)
        self.assertEqual(len({r['root'] for r in public if r['split']=='calibration'}),32)
        self.assertEqual({r['suite'] for r in public},{'pointer','arc1','arc2','sudoku','storyworld'})
        self.assertTrue(all('answer' not in r for r in public))
        labels={r['id']:r for r in host}
        for r in public:
            if r['id'].endswith(':copy'):self.assertEqual(score_value(labels[r['id']]['answer'],labels[r['id']])['correct'],1)
    def test_pointer_source_is_deterministic_and_has_no_shared_current_answers(self):
        self.assertEqual(pointer_root(2026112005,[1,2,3,4]),pointer_root(2026112005,[1,2,3,4]))
        self.assertNotEqual(pointer_root(2026112005,[1]),pointer_root(2026212005,[1]))
    def test_valid_sudoku_solution_kept_distinct_from_upstream_exact_target(self):
        _,host=materialize(read('experiments/nan_gym05/config.json'))
        label=next(r for r in host if r['suite']=='sudoku' and not r['copy_control'])
        score=score_value(label['answer'],label);self.assertEqual(score['correct'],1);self.assertEqual(score['valid_givens_preserving_solution'],1)
        self.assertEqual(score_value(None,label)['valid_givens_preserving_solution'],0)
    def test_storyworld_verifies_rollout_instead_of_literal_witness(self):
        _,host=materialize(read('experiments/nan_gym05/config.json'))
        for label in host:
            if label['suite']=='storyworld' and not label['copy_control']:
                self.assertEqual(score_value(label['answer'],label)['correct'],1)
                self.assertEqual(score_value(None,label)['correct'],0)

class DictionaryTests(unittest.TestCase):
    def test_sparse_encoder_learns_and_has_bounded_active_features(self):
        torch.manual_seed(19);model=SmallSAE(16,8,4);x=torch.randn(32,16);opt=torch.optim.Adam(model.parameters(),lr=.01)
        before=float((model(x)-x).square().mean().detach())
        for _ in range(60):
            loss=(model(x)-x).square().mean();opt.zero_grad();loss.backward();opt.step()
        self.assertLess(float((model(x)-x).square().mean().detach()),before)
        self.assertTrue(bool(((model.encode(x)>0).sum(-1)<=4).all()))
    def test_factor_plus_residual_closure_keeps_evidence_class(self):
        arrays,receipt=decompose(torch.randn(32,48),16,9)
        self.assertEqual(arrays['V16'].shape,(48,16));self.assertFalse(receipt['trained_VPD']);self.assertFalse(receipt['trained_BPD'])
        self.assertLess(receipt['ranks']['16']['closure_relative_l2'],1e-5)

if __name__=='__main__':unittest.main()
