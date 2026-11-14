from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ram04_core as c
import model

def local(name,typ,prop=True,head='',function=False,domain='',rewritable=False):
    return dict(name=name,type=typ,prop=prop,head=head,function=function,domain=domain,rewritable=rewritable)

class FamilyGrammarTests(unittest.TestCase):
    def test_typed_locals_enable_application_symmetry_and_elimination(self):
        state=dict(goals=['h : P → Q\np : P\ne : a = b\nu : P ∨ Q\n⊢ Q'],definitions=[],locals=[
          local('h','P → Q',function=True,domain='P'),local('p','P'),local('e','a = b',head='Eq',rewritable=True),local('u','P ∨ Q',head='Or')])
        pool=c.candidates(state)
        self.assertIn(('exact_local','exact h p'),pool)
        self.assertIn(('exact_local','exact e.symm'),pool)
        self.assertIn(('rw_local','rw [e] at *'),pool)
        self.assertIn(('rcases_local','rcases u with r4a0 | r4b0'),pool)
        self.assertTrue(all(c.valid_script(s) for _,s in pool))
    def test_multiline_metadata_does_not_depend_on_printed_line_parser(self):
        state=dict(goals=['h :\n  P ∧ Q\n⊢ P'],definitions=[],locals=[local('h','P ∧ Q',head='And')])
        self.assertIn(('exact_local','exact h.1'),c.candidates(state))
        self.assertIn(('rcases_local','rcases h with ⟨r4a0, r4b0⟩'),c.candidates(state))
    def test_local_operand_type_enters_features_without_labels(self):
        state=dict(goals=['h : P\n⊢ P'],definitions=[],locals=[local('h','P')])
        row=dict(before=state,family='exact_local',script='exact h')
        x=model.encode([row]);changed=dict(row,before=dict(state,locals=[local('h','Q')]))
        y=model.encode([changed]);self.assertFalse(x['operand'].equal(y['operand']))
        z=model.encode([dict(row,label=3,origin='authored',template='secret',after={'goals':[]})])
        self.assertTrue(all(x[k].equal(z[k]) for k in x))
    def test_family_caps_and_script_injection(self):
        for script in ('exact h; sorry','rw [h] at *\naxiom bad : False','rcases h with a | b; sorry'):
            self.assertFalse(c.valid_script(script))
        state=dict(goals=['⊢ P'],definitions=[],locals=[local('h'+str(i),'P') for i in range(30)])
        pool=c.candidates(state);self.assertLessEqual(sum(f=='exact_local' for f,_ in pool),12)

if __name__=='__main__':unittest.main()
