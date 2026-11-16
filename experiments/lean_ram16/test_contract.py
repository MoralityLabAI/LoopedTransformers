"""RAM-16 pre-TRAIN macro generator, source split, censoring contracts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import grammar

SYNTHETIC="""import Init
namespace Synthetic
def iterateState (f : Nat → Nat) (x : Nat) : Nat := f x
structure StateGauge where
  toFun : Nat → Nat
  invFun : Nat → Nat
  left_inv : ∀ x, invFun (toFun x) = x
theorem previous_result (f : Nat → Nat) (n : Nat) (x : Nat) :
    iterateState f n = iterateState f n := by
  rfl
theorem target_result (f : Nat → Nat) (n : Nat) (x : Nat) :
    iterateState f n = iterateState f n := by
  exact previous_result f n x
theorem future_secret : True := by
  trivial
end Synthetic
"""

class Schema(unittest.TestCase):
    def test_machine_registration_matches_protocol(self):
        conf=json.loads((HERE/"config.json").read_text())
        self.assertEqual(conf["registration_commit"],"4208074aa528c3e2b6b515b5b14b37642a9a340a")
        self.assertEqual(conf["amendment_commit"],"75389e026650653f62eaed96fab741cbc6b09261")
        self.assertEqual(len(conf["source_modules"]["train"]),10)
        self.assertEqual(len(conf["source_modules"]["dev"]),3)
        self.assertEqual(len(conf["source_modules"]["test"]),5)
        self.assertEqual(len(set(conf["source_modules"]["train"]) &
                             set(conf["source_modules"]["test"])),0)
        self.assertEqual(len(set(conf["source_modules"]["dev"]) &
                             set(conf["source_modules"]["test"])),0)
        self.assertEqual(conf["test_source_commit"],"fdbe1581e95427016699169fd19c9e915a40bcf9")
        self.assertEqual(conf["gate"]["min_new_macro_positives"],6)
        self.assertEqual(conf["learning"]["updates_per_seed"],400)
    def test_candidate_range_and_typed_binder(self):
        candidates=grammar.generate(SYNTHETIC,"target_result")
        self.assertTrue(any(x.family=="nat_induction" and
                            x.script.startswith("induction n") for x in candidates))
        self.assertTrue(all(grammar._safe(x.family,x.script) for x in candidates))
        self.assertLessEqual(len(candidates),48)
        self.assertTrue(all(sum(c.family==fam for c in candidates)<=16
                            for fam in grammar.FAMILIES))
        self.assertEqual(len(candidates),len({c.script for c in candidates}))
    def test_prior_header_use_is_stable_under_gold_proof_edits(self):
        first=[x.as_dict() for x in grammar.generate(SYNTHETIC,"target_result")]
        changed=SYNTHETIC.replace("  exact previous_result f n x",
                                   "  omega\n  -- completely different reference target proof")
        second=[x.as_dict() for x in grammar.generate(changed,"target_result")]
        self.assertEqual(first,second)
    def test_future_declaration_is_absent(self):
        actions=grammar.generate(SYNTHETIC,"target_result")
        self.assertFalse(any("future_secret" in x.script for x in actions))
    def test_previous_lemma_header_intersection(self):
        scripts=[x.script for x in grammar.generate(SYNTHETIC,"target_result")]
        self.assertTrue(any("previous_result" in script for script in scripts))
        self.assertFalse(any("theorem target_result" in script for script in scripts))
    def test_unregistered_unsafe_scripts_rejected(self):
        for s in ("sorry","admit","induction n <;> sorry",
                  "rw [previous_result]; exact h", "run_tac foo",
                  "simp [foo]\nunsafe axiom y"):
            for f in grammar.FAMILIES:
                self.assertFalse(grammar._safe(f,s))
    def test_ram14_historical_prefix_is_retained(self):
        prior=grammar.old.generate(SYNTHETIC,"target_result")
        current=grammar.generate(SYNTHETIC,"target_result")
        self.assertEqual(current[:len(prior)],prior)
    def test_source_provenance_is_stable(self):
        a=grammar.describe(SYNTHETIC,"target_result")
        b=grammar.describe(SYNTHETIC,"target_result")
        self.assertEqual(a,b)
        self.assertEqual(len(a["source_sha256"]),64)
        self.assertEqual(set(a["counts"]),set(grammar.FAMILIES))
    def test_source_name_masking(self):
        source=SYNTHETIC.replace("theorem future_secret",
            "/- theorem hidden_inside_comment : False := by sorry -/\n" +
            "theorem future_secret")
        mask=grammar.mask_comments(source)
        names=re.findall(r"^(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)",mask,re.M)
        self.assertNotIn("hidden_inside_comment",names)


if __name__=="__main__":unittest.main()
