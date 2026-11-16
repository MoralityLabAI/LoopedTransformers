"""RAM-18 predata quantified Nat grammar, source isolation and censoring contracts."""
from __future__ import annotations
import json,sys,unittest,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_ram16"))
sys.path.insert(0,str(HERE.parent/"lean_ram17"))
sys.path.insert(0,str(HERE))
import grammar

CONFIG=json.loads((HERE/"config.json").read_text())
SYNTHETIC="""import Init
namespace Sample
def tally (n : Nat) : Nat := n
theorem prior (n : Nat) : tally n = n := by
  rfl
theorem quantified : ∀ n : Nat, tally n = n := by
  intro n
  rfl
theorem quantified_parenthesized : ∀ (n : Nat), tally n = n := by
  intro n
  rfl
theorem explicitly_bound (n : Nat) : tally n = n := by
  rfl
theorem after_secret : False := by
  sorry
end Sample
"""

class RAM18Contracts(unittest.TestCase):
    def test_sealed_module_split_and_gate(self):
        mods=CONFIG["source_modules"]
        self.assertEqual((len(mods["train"]),len(mods["dev"]),len(mods["test"])),(22,4,5))
        self.assertFalse(set(mods["train"]) & (set(mods["dev"]) | set(mods["test"])))
        self.assertFalse(set(mods["dev"]) & set(mods["test"]))
        self.assertEqual(CONFIG["selection"]["train_first"],8)
        self.assertEqual(CONFIG["selection"]["dev_first"],6)
        self.assertEqual(CONFIG["selection"]["test_first"],2)
        self.assertEqual(CONFIG["prospective_TEST_scored"],0)
        self.assertTrue(CONFIG["gate"]["require_all_families"])
        self.assertEqual(CONFIG["gate"]["min_distinct_nat_modules"],2)
        self.assertEqual(CONFIG["gate"]["min_distinct_prior_modules"],2)
        self.assertEqual(CONFIG["test_source_commit"],"3749c84cc6d47297faaf94404af6d1fd36916812")
        self.assertTrue(all(len(v)==40 for role in mods.values() for v in role.values()))

    def test_quantified_nat_binders_generate_valid_composites(self):
        for name in ("quantified","quantified_parenthesized"):
            actions=grammar.generate(SYNTHETIC,name)
            scripts=[x.script for x in actions]
            self.assertTrue(any(x=="intro n; induction n <;> simp_all" for x in scripts), name)
            self.assertTrue(any(x=="intro n; induction n <;> omega" for x in scripts), name)
            self.assertTrue(all(grammar._safe(x.family,x.script) for x in actions))
            self.assertLessEqual(len(actions),48)
            self.assertTrue(all(sum(x.family==f for x in actions)<=16 for f in grammar.FAMILIES))
            self.assertFalse(any("after_secret" in x.script for x in actions))

    def test_exact_historical_ram14_prefix_retained(self):
        for name in ("quantified","explicitly_bound"):
            old=grammar.old.generate(SYNTHETIC,name)
            new=grammar.generate(SYNTHETIC,name)
            self.assertEqual(old,new[:len(old)])

    def test_reference_proof_body_independent(self):
        first=[x.as_dict() for x in grammar.generate(SYNTHETIC,"quantified")]
        edited=SYNTHETIC.replace("  intro n\n  rfl","  omega\n  -- reference gold changed")
        second=[x.as_dict() for x in grammar.generate(edited,"quantified")]
        self.assertEqual(first,second)

    def test_induction_action_syntax_whitelist(self):
        self.assertTrue(grammar._safe("nat_induction","intro n; induction n <;> simp_all"))
        self.assertTrue(grammar._safe("nat_induction","intro n; induction n <;> omega"))
        self.assertFalse(grammar._safe("nat_induction","intro n; induction m <;> omega"))
        for s in ("intro n; induction n; sorry","intro n; run_tac", "intro n; induction n\nunsafe"):
            self.assertFalse(grammar._safe("nat_induction",s))

    def test_collector_audit_genuinely_source_only(self):
        import collect,third_audit
        self.assertIs(third_audit.protected_training_checkout,collect.protected_training_checkout)
        self.assertTrue(third_audit.__file__.endswith("lean_ram18/third_audit.py"))
        self.assertTrue(collect.__file__.endswith("lean_ram18/collect.py"))
        self.assertIn('scratch=out/"native_source_compiler_artifacts"',(HERE/"collect.py").read_text())
        self.assertIn('scratch=out/"third_compiler_sources_and_logs"',(HERE/"third_audit.py").read_text())
        self.assertIn('(out/"two_compile_positive_candidates.jsonl").write_bytes(b"")',(HERE/"collect.py").read_text())

if __name__=="__main__":unittest.main()
