"""RAM-15 pre-search contract tests; synthetic goals only."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ram15_grammar as gram

SRC="""import Init
namespace Sample
def iterateState (f : Nat → Nat) : Nat → Nat := fun x => f x
structure StateGauge where
  toFun : Nat → Nat
  invFun : Nat → Nat
  left_inv : ∀ x, invFun (toFun x) = x
def transport (g : StateGauge) : Nat → Nat := fun x => g.toFun x
theorem earlier_transport (n : Nat) (x : Nat) : n = n := by
  rfl
theorem target (g : StateGauge) (n : Nat) (x : Nat) :
    iterateState (transport g) n = iterateState (transport g) n := by
  rfl
end Sample
"""

class Registered(unittest.TestCase):
    def test_manifest_and_source_split(self):
        m=json.loads((HERE/'LOCKED_MANIFEST.json').read_text())
        self.assertEqual(len(m['cases']),7)
        self.assertEqual(sum(map(lambda x:len(x['root_theorems']),m['cases'])),14)
        self.assertEqual(len({x['source_file'] for x in m['cases']}),7)
        self.assertEqual(m['registration_commit'],'b5b7836755a457734fa612e335c48c1c64549f40')
        self.assertEqual(len(gram.FAMILIES),20)
        self.assertEqual(m['primary']['policies'],['native','frequency','shape_frequency','loop32','loop64','flat64'])
    def test_old_grammar_ablation_byte_determinism(self):
        state={"goals":["n : Nat\nx : Nat\n⊢ n = n"],"definitions":[]}
        gram.set_context(SRC,'target',ablation=True)
        self.assertEqual(gram.candidates(state),gram.ORIGINAL.candidates(state))
        self.assertEqual(gram.audit_summary()['statically_generated_extra'],0)
    def test_frozen_proxy_candidate_interleaving(self):
        state={"goals":["n : Nat\nx : Nat\n⊢ n = n"],"definitions":[]}
        original=gram.ORIGINAL.candidates(state)
        gram.set_context(SRC,'target')
        changed=gram.candidates(state)
        self.assertGreaterEqual(len(changed),len(original))
        self.assertEqual(set(original)-set(changed),set())
        self.assertTrue(all(f in gram.FAMILIES for f,_ in changed))
        self.assertTrue(all(gram.valid_script(s) for _,s in changed))
        self.assertLessEqual(len(changed)-len(original),32)
        self.assertEqual(len(changed),len(set(changed)))
        receipt=gram.audit_summary()
        self.assertEqual(receipt['theorem'],'target')
        self.assertFalse(receipt['ablation_original_twenty_families'])
    def test_no_reference_proof_body_leakage(self):
        gram.set_context(SRC,'target')
        first=gram.audit_summary()['generated_actions']
        changed=SRC.replace('theorem target (g', 'theorem target (g')
        changed=changed.replace('  rfl\nend Sample','  exact (by trivial)\nend Sample')
        gram.set_context(changed,'target')
        self.assertEqual(first,gram.audit_summary()['generated_actions'])
    def test_invalid_macro_rejected(self):
        gram.set_context(SRC,'target')
        for script in ('sorry','induction n generalizing x <;> sorry',
                       'rw [h]; exact False.elim sorry','run_tac unsafe'):
            self.assertFalse(gram.valid_script(script),script)
    def test_extra_generated_source_safety(self):
        gram.set_context(SRC,'target')
        for item in gram.audit_summary()['generated_actions']:
            self.assertTrue(gram.PROXY.safe_extended_script(item['family'],item['script']))
        self.assertFalse(any('earlier_transport' not in item['script']
                             for item in gram.audit_summary()['generated_actions']
                             if item['family']=='rewrite_prior'))

if __name__=='__main__':unittest.main()
