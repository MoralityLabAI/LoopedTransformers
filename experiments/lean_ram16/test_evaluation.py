"""RAM16 first prospective scorer and source-family isolation software contracts.

Synthetic Lean contexts only. Model weights and prospective theorem source
contents are absent; the module-level metadata was locked after fit hashes.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import eval_grammar as g
import harness

FAKE="""import Init
namespace Demo
def iterateState (f : Nat → Nat) (n : Nat) : Nat := f n
theorem previous_iterate (f : Nat → Nat) (n : Nat) (x : Nat) :
    iterateState f n = iterateState f n := by rfl
theorem target (f : Nat → Nat) (n : Nat) (x : Nat) :
    iterateState f n = iterateState f n := by rfl
end Demo
"""
GOAL=dict(accepted=True,goals=["f : Nat → Nat\nn x : Nat\n⊢ iterateState f n = iterateState f n"],
          definitions=[])

class Contract(unittest.TestCase):
    def test_exact_source_family_manifest_is_frozen(self):
        m=harness.LOCKED_SOURCE
        self.assertEqual(len(m["cases"]),5)
        self.assertEqual(m["case_count"],10)
        self.assertTrue(m["selected_before_goal_state_access"])
        self.assertEqual(m["source_commit"],harness.CONFIG["test_source_commit"])
        self.assertEqual(m["selected_model_checkpoint_lock_commit"],
                         "8ff3df3cdf10ef7f55dc927f70206c57f2a2b8c0")
        self.assertEqual(sum(len(x["theorems"]) for x in m["cases"]),10)
    def test_23_family_meta_and_proxy_mapping(self):
        self.assertEqual(len(g.FAMILIES),23)
        self.assertEqual(g.FAMILIES[:20],g.OLD.FAMILIES)
        self.assertEqual(g.family_proxy("nat_induction","induction n"),"induction_list")
        self.assertEqual(g.family_proxy("rewrite_prior","rw [prior]"),"rw_local")
        self.assertEqual(g.family_proxy("simp_prior","simp [defn]"),"simp_all_only_visible")
        self.assertEqual(g.family_proxy("rewrite_prior","simpa [prior]"),"simpa_local")
    def test_original_20_family_ablation_is_strict(self):
        g.set_context(FAKE,"target",ablation=True)
        self.assertEqual(g.candidates(GOAL),g.OLD.candidates(GOAL))
        self.assertTrue(g.audit_summary()["ablation_original_20"])
    def test_shared_candidate_pool_and_provenance(self):
        g.set_context(FAKE,"target")
        original=g.OLD.candidates(GOAL)
        extended=g.candidates(GOAL)
        self.assertTrue(set(original).issubset(set(extended)))
        self.assertEqual(len(extended),len(set(extended)))
        self.assertTrue(all(f in g.FAMILIES and g.valid_script(s) for f,s in extended))
        self.assertLessEqual(len(extended)-len(original),32)
        self.assertTrue(g.audit_summary()["original_ram13_grammar_untouched"])
    def test_censoring_of_unsafe_and_unobserved_actions(self):
        g.set_context(FAKE,"target")
        for script in ("sorry","admit","exact?","unsafe run_tac"):
            self.assertFalse(g.valid_script(script))
        self.assertFalse(any(s=="rw [future_lemma]" for fam,s in g.candidates(GOAL)))
    def test_original_certified_engine_budgets(self):
        self.assertEqual(harness.PRIMARY["seconds"],30)
        self.assertEqual(harness.PRIMARY["evaluations"],48)
        self.assertEqual(harness.SECONDARY["seconds"],60)
        self.assertEqual(harness.SECONDARY["evaluations"],96)
        self.assertEqual(len(harness.POLICIES),8)
        self.assertEqual(len(harness.SECOND),4)
        self.assertEqual(len(harness.ABLATED),2)
        self.assertTrue(callable(harness.base.search))
    def test_original_all_model_hashes_locked(self):
        locked=harness.FROZEN["selected"]
        self.assertEqual(len(locked),4)
        for row in locked.values():
            self.assertEqual(len(row["best_checkpoint_sha256"]),64)
            self.assertLessEqual(row["best_step"],400)
        self.assertEqual(harness.FROZEN["new_test_cases_scored"],0)

if __name__=="__main__":unittest.main()
