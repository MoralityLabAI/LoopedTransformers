"""RAM-16 prospective evaluator source/family-isolation preflight.

Synthetic Lean goals only. This test never checks out the protected TEST
module sources and never reads any original proof body or private weights.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import eval_grammar as evaluated

SYNTHETIC="""import Init
namespace Synthetic
def iterateState (f : Nat → Nat) (x : Nat) := f x
theorem prior_lemma (n : Nat) (x : Nat) :
    iterateState id n = iterateState id n := by
  rfl
theorem fresh_test (n : Nat) (x : Nat) :
    iterateState id n = iterateState id n := by
  rfl
theorem later_lemma : False := by
  sorry
end Synthetic
"""
STATE=dict(accepted=True,goals=["n : Nat\nx : Nat\n⊢ n = n"],definitions=[])

class EvaluationFreeze(unittest.TestCase):
    def test_exact_ten_registered_test_theorems(self):
        reg=json.loads((HERE/"LOCKED_TEST_CASES.json").read_text())
        self.assertEqual(reg["case_count"],10)
        self.assertEqual(len(reg["modules"]),5)
        self.assertEqual(sum(len(r["root_theorems"]) for r in reg["modules"]),10)
        self.assertEqual(reg["source_commit"],"fdbe1581e95427016699169fd19c9e915a40bcf9")
        self.assertEqual(len({r["name"] for r in reg["modules"]}),5)
        self.assertEqual(set(reg["primary_policies"]),{
            "native","frequency","shape_frequency","loop64_task_only",
            "loop64_structural","flat71_task_only","flat71_structural",
            "frozen_loop64_proxy"})
        self.assertFalse(reg["source_test_first_scoring"])
    def test_model_selection_frozen_before_source_read(self):
        check=json.loads((HERE/"FROZEN_SELECTED_MODELS.json").read_text())
        test=json.loads((HERE/"LOCKED_TEST_CASES.json").read_text())
        self.assertEqual(check["first_prospective_TEST_cases_scored"],0)
        self.assertEqual(check["source_TEST_opened_for_training"],False)
        self.assertEqual(len(check["selected_by_historical_original_train_root_disjoint_dev_only"]),4)
        self.assertEqual(check["training_run"],43804250337)
        for name,meta in check["selected_by_historical_original_train_root_disjoint_dev_only"].items():
            test_key=name.replace("_task_plus_structural","_structural")
            self.assertEqual(meta["sha256"],test["selected_model_sha256"][test_key])
            self.assertEqual(len(meta["sha256"]),64)
    def test_all_new_families_use_23_ids(self):
        self.assertEqual(len(evaluated.FAMILIES),23)
        self.assertEqual(evaluated.FAMILIES[:20],evaluated.OLD.FAMILIES)
        self.assertEqual(set(evaluated.FAMILIES[20:]),{
            "nat_induction","simp_prior","rewrite_prior"})
        for family in evaluated.FAMILIES[20:]:
            self.assertIn(evaluated.family_proxy(family,
                "rw [prior_lemma]" if family=="rewrite_prior" else
                "induction n <;> simp_all"),evaluated.OLD.FAMILIES)
    def test_shared_candidate_multiset_and_native_order(self):
        evaluated.set_context(SYNTHETIC,"fresh_test")
        old=evaluated.OLD.candidates(STATE)
        new=evaluated.candidates(STATE)
        self.assertEqual(len(new),len(set(new)))
        self.assertTrue(set(old).issubset(set(new)))
        self.assertTrue(all(f in evaluated.FAMILIES for f,s in new))
        self.assertTrue(all(evaluated.valid_script(s) for _,s in new))
        self.assertLessEqual(len(new)-len(old),32)
        self.assertEqual(evaluated.audit_summary()["theorem"],"fresh_test")
    def test_ablation_is_exact_historical_twenty_family_pool(self):
        evaluated.set_context(SYNTHETIC,"fresh_test",ablation=True)
        self.assertEqual(evaluated.candidates(STATE),
                         evaluated.OLD.candidates(STATE))
        self.assertEqual(evaluated.audit_summary()["structural_candidates_source_generated"],0)
    def test_no_reference_tactic_or_future_theorem_leakage(self):
        evaluated.set_context(SYNTHETIC,"fresh_test")
        first=[x["script"] for x in evaluated.audit_summary()["source_visible_structural_candidates"]]
        changed=SYNTHETIC.replace(
            "theorem fresh_test (n : Nat) (x : Nat) :\n"
            "    iterateState id n = iterateState id n := by\n  rfl",
            "theorem fresh_test (n : Nat) (x : Nat) :\n"
            "    iterateState id n = iterateState id n := by\n"
            "  exact (by simp_all)\n  -- gold reference was altered")
        evaluated.set_context(changed,"fresh_test")
        second=[x["script"] for x in evaluated.audit_summary()["source_visible_structural_candidates"]]
        self.assertEqual(first,second)
        self.assertFalse(any("later_lemma" in s for s in second))
        self.assertFalse(any("gold reference" in s for s in second))
    def test_original_twenty_family_ablation_reaches_base_scorer(self):
        # The RAM15 search resets the source-grammar context; RAM16 must
        # explicitly forward the ablation flag to the base scorer.
        import ast
        source=(HERE/"harness.py").read_text()
        tree=ast.parse(source)
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)
                and x.name=="search")
        calls=[x for x in ast.walk(fn) if isinstance(x,ast.Call)
               and isinstance(x.func,ast.Attribute)
               and isinstance(x.func.value,ast.Name)
               and x.func.value.id=="base" and x.func.attr=="search"]
        self.assertEqual(len(calls),1)
        options={x.arg:x.value for x in calls[0].keywords}
        self.assertIn("ablation",options)
        self.assertIsInstance(options["ablation"],ast.Name)
        self.assertEqual(options["ablation"].id,"ablation")

    def test_prohibited_compiler_proof_script_denied(self):
        evaluated.set_context(SYNTHETIC,"fresh_test")
        for script in ("sorry","admit","run_tac unsafe","induction n; sorry",
                       "exact False.elim (by sorry)"):
            self.assertFalse(evaluated.valid_script(script),script)


if __name__=="__main__":
    unittest.main()
