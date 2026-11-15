from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import search as s

BUDGET=dict(seconds=30,evaluations=48,transitions=8,request_seconds=3,max_goals=8)
class Guard:
    def check(self):pass
class FakeSession:
    def __init__(self,*args):self.initial=dict(goals=['⊢ P ∧ Q'],definitions=[])
    def query(self,actions,timeout):
        goals={1:['⊢ P','⊢ Q'],2:['⊢ Q'],3:[]}[len(actions)]
        return dict(accepted=True,goals=goals,definitions=[])
    def close(self):pass

class SearchTests(unittest.TestCase):
    def test_complete_branch_tree_requires_independent_compile_and_replay(self):
        def certificate(lean,original,theorem,actions,prefix,guard,timeout):
            p=prefix.with_suffix('.lean');p.write_text('test only')
            return dict(accepted=True,source_file=str(p),source_sha256='same')
        def replay(*args):return dict(accepted=True,source_sha256='same')
        with tempfile.TemporaryDirectory() as td,patch('ram03_core.candidates',return_value=[('constructor','constructor')]):
            result=s.search('','', 'toy',None,Path(td),Guard(),BUDGET,session_type=FakeSession,certificate=certificate,replayer=replay)
        self.assertTrue(result['accepted']);self.assertEqual(result['logical_evaluations'],3)
        self.assertEqual(result['actual_lean_processes'],3);self.assertEqual(result['max_path_transitions'],3)
    def test_partial_branch_never_certifies(self):
        with tempfile.TemporaryDirectory() as td,patch('ram03_core.candidates',return_value=[('constructor','constructor')]):
            result=s.search('','', 'toy',None,Path(td),Guard(),dict(BUDGET,transitions=2),session_type=FakeSession)
        self.assertFalse(result['accepted']);self.assertIsNone(result['certificate'])
    def test_fresh_replay_failure_rejects_final_compile(self):
        def certificate(*args):
            p=args[4].with_suffix('.lean');p.write_text('test only')
            return dict(accepted=True,source_file=str(p),source_sha256='same')
        with tempfile.TemporaryDirectory() as td,patch('ram03_core.candidates',return_value=[('constructor','constructor')]):
            result=s.search('','', 'toy',None,Path(td),Guard(),BUDGET,session_type=FakeSession,
              certificate=certificate,replayer=lambda *args:dict(accepted=False,source_sha256='same'))
        self.assertFalse(result['accepted'])
    def test_ranker_cannot_change_candidates(self):
        class Bad:
            def rank(self,*args):return [('omega','sorry')]
        with tempfile.TemporaryDirectory() as td,patch('ram03_core.candidates',return_value=[('constructor','constructor')]),self.assertRaises(AssertionError):
            s.search('','', 'toy',Bad(),Path(td),Guard(),BUDGET,session_type=FakeSession)
    def test_actual_evaluation_budget(self):
        with tempfile.TemporaryDirectory() as td,patch('ram03_core.candidates',return_value=[('constructor','constructor')]):
            result=s.search('','', 'toy',None,Path(td),Guard(),dict(BUDGET,evaluations=1),session_type=FakeSession)
        self.assertFalse(result['accepted']);self.assertEqual(result['logical_evaluations'],1)
if __name__=='__main__':unittest.main()
