from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bounded_multigoal_search as s

class SearchTests(unittest.TestCase):
    def test_signature_order_multiplicity_and_case_normalization(self):
        self.assertEqual(s.signature(['case left\nh : True\n⊢ True']),s.signature(['case right\nh : True\n⊢ True']))
        self.assertNotEqual(s.signature(['⊢ P','⊢ Q'])[0],s.signature(['⊢ Q','⊢ P'])[0])
        self.assertNotEqual(s.signature(['⊢ P'])[0],s.signature(['⊢ P','⊢ P'])[0])

    def test_all_branches_close_before_certificate(self):
        def fake(original,theorem,actions,probe,*_):
            goals={1:['⊢ P','⊢ Q'],2:['⊢ Q'],3:[]}[len(actions)]
            return dict(accepted=True,timed_out=False,remaining_goals=goals,goal_count=len(goals),probe=probe)
        with tempfile.TemporaryDirectory() as td:
            r=s.solve('','toy',['⊢ P ∧ Q'],None,Path(td),fake,generator=lambda g:{'constructor':['constructor']})
        self.assertTrue(r['accepted'])
        self.assertEqual(r['lean_subprocesses'],4)
        self.assertEqual(r['max_path_transitions'],3)
        self.assertTrue(r['calls'][0]['creates_branches'])
        self.assertFalse(r['calls'][-1]['probe'])

    def test_partial_branch_is_never_a_proof(self):
        def fake(*_): return dict(accepted=True,timed_out=False,remaining_goals=['⊢ P','⊢ Q'],goal_count=2)
        with tempfile.TemporaryDirectory() as td:
            r=s.solve('','toy',['⊢ P ∧ Q'],None,Path(td),fake,generator=lambda g:{'constructor':['constructor']})
        self.assertFalse(r['accepted'])
        self.assertEqual(r['diagnostics']['deduplicated_states'],1)

    def test_actual_calls_include_final_and_deadline(self):
        ticks=[0.]
        def fake(*_):
            ticks[0]+=.1
            return dict(accepted=True,timed_out=False,remaining_goals=[],goal_count=0)
        with tempfile.TemporaryDirectory() as td:
            a=s.solve('','toy',['⊢ P'],None,Path(td),fake,generator=lambda g:{'rfl':['rfl']},max_calls=1)
            self.assertFalse(a['accepted']);self.assertEqual(a['lean_subprocesses'],1)
            b=s.solve('','toy',['⊢ P'],None,Path(td),fake,generator=lambda g:{'rfl':['rfl']},clock=lambda:ticks[0],seconds=.15)
            self.assertFalse(b['accepted']);self.assertEqual(b['reason'],'hard_deadline')

    def test_axiom_gate_and_goal_parser(self):
        wrap=lambda x:'RAM02_AXIOMS_BEGIN\n'+x+'\nRAM02_AXIOMS_END'
        self.assertTrue(s.axiom_audit(wrap("'toy' depends on axioms: [propext, Classical.choice, Quot.sound]"))[0])
        for value in ['sorryAx','poisoned','Lean.trustCompiler']:
            self.assertFalse(s.axiom_audit(wrap("'toy' depends on axioms: ["+value+"]"))[0])
        self.assertFalse(s.axiom_audit('success')[0])
        log='RAM02_GOAL_COUNT=2\nRAM02_GOAL_BEGIN\n⊢ P\nRAM02_GOAL_END\nRAM02_GOAL_BEGIN\n⊢ Q\nRAM02_GOAL_END'
        self.assertEqual(s.extract_goals(log),['⊢ P','⊢ Q'])
        with self.assertRaises(ValueError): s.extract_goals(log.replace('COUNT=2','COUNT=1'))

    def test_candidate_injection_rejected(self):
        class Bad:
            def rank(self,g,c): return [('rfl','sorry')]
        with tempfile.TemporaryDirectory() as td,self.assertRaises(AssertionError):
            s.solve('','toy',['⊢ P'],Bad(),Path(td),None,generator=lambda g:{'rfl':['rfl']})

if __name__=='__main__': unittest.main()
