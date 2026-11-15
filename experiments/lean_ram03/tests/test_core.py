from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ram03_core as c

class CoreTests(unittest.TestCase):
    def test_signatures_preserve_all_siblings(self):
        self.assertNotEqual(c.signature(['⊢ P']),c.signature(['⊢ P','⊢ P']))
        self.assertNotEqual(c.signature(['⊢ P','⊢ Q']),c.signature(['⊢ Q','⊢ P']))
        self.assertEqual(c.signature(['case left\nh : P\n⊢ P']),c.signature(['case right\nh : P\n⊢ P']))
    def test_goal_reduction_and_branch_growth_are_distinct(self):
        before={'goals':['⊢ P','⊢ Q']}
        self.assertEqual(c.label(before,dict(accepted=True,goals=['⊢ Q'])),1)
        self.assertEqual(c.label(before,dict(accepted=True,goals=['⊢ A','⊢ B','⊢ Q'])),2)
        self.assertEqual(c.label(before,dict(accepted=True,goals=[])),3)
        self.assertEqual(c.label(before,dict(accepted=False)),0)
        self.assertEqual(c.label(before,dict(accepted=True,goals=before['goals'])),0)
    def test_registered_grammar_rejects_injection(self):
        for bad in ('sorry','exact h\naxiom bad : False','simp_all [target_theorem]','run_tac pure ()'):
            self.assertFalse(c.valid_script(bad))
        for good in ('induction xs generalizing s','all_goals simp_all [MathCan.HookState.endpoint]',c.ARITH):
            self.assertTrue(c.valid_script(good))
    def test_future_reference_commands_are_removed(self):
        source='import Init\nnamespace T\ntheorem toy (P : Prop) (h : P) : P := by\n  exact hidden_reference_command\nend T\n'
        generated=c.make_source(source,'toy',['exact h'])
        self.assertNotIn('hidden_reference_command',generated)
        self.assertNotIn('sorry',generated)
        self.assertIn('exact h',generated)
    def test_heartbeat_option_preserves_documentation_attachment(self):
        source='import Init\nnamespace T\n/-- Documented theorem. -/\ntheorem toy (P : Prop) (h : P) : P := by\n  exact h\nend T\n'
        generated=c.make_source(source,'toy',['exact h'])
        self.assertLess(generated.index('set_option'),generated.index('/-- Documented'))
        self.assertIn('/-- Documented theorem. -/\ntheorem toy',generated)
    def test_axiom_audit_requires_explicit_allowed_dependencies(self):
        wrap=lambda s:'RAM03_AXIOMS_BEGIN\n'+s+'\nRAM03_AXIOMS_END'
        self.assertTrue(c.axiom_audit(wrap("'toy' does not depend on any axioms"))[0])
        self.assertFalse(c.axiom_audit(wrap("'toy' depends on axioms: [sorryAx]"))[0])
        self.assertFalse(c.axiom_audit(wrap("'toy' depends on axioms: [custom]"))[0])
        self.assertFalse(c.axiom_audit('success')[0])
if __name__=='__main__':unittest.main()
