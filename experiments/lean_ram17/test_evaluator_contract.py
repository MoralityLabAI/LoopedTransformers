"""RAM17 source-only evaluator contracts on synthetic examples, with no TEST access."""
from __future__ import annotations
import ast,json,sys,unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_ram16"))
sys.path.insert(0,str(HERE))
import eval_grammar

SRC="""import Init
namespace Synthetic
def iterate (n : Nat) := n
theorem earlier (n : Nat) : iterate n = n := by
  rfl
theorem target (n : Nat) : iterate n = n := by
  exact earlier n
theorem secret : True := by
  trivial
end Synthetic
"""
STATE=dict(accepted=True,goals=["n : Nat\n⊢ iterate n = n"],definitions=[])

class EvaluatorContracts(unittest.TestCase):
    def test_same_source_pool_for_every_policy_and_original_ablation(self):
        config=json.loads((HERE/"config.json").read_text())
        self.assertEqual(len(config["primary"]["policies"]),16)
        self.assertEqual(len(eval_grammar.FAMILIES),23)
        self.assertEqual(eval_grammar.FAMILIES[:20],eval_grammar.OLD.FAMILIES)
        eval_grammar.set_context(SRC,"target")
        expanded=eval_grammar.candidates(STATE)
        original=eval_grammar.OLD.candidates(STATE)
        self.assertTrue(set(original).issubset(set(expanded)))
        self.assertEqual(len(set(expanded)),len(expanded))
        self.assertLessEqual(len(expanded)-len(original),32)
        eval_grammar.set_context(SRC,"target",ablation=True)
        self.assertEqual(eval_grammar.candidates(STATE),original)

    def test_reference_proof_change_does_not_change_registered_candidates(self):
        eval_grammar.set_context(SRC,"target")
        before=eval_grammar.audit_summary()["source_visible_structural_candidates"]
        changed=SRC.replace("  exact earlier n","  omega\n  -- gold reference edited")
        eval_grammar.set_context(changed,"target")
        after=eval_grammar.audit_summary()["source_visible_structural_candidates"]
        self.assertEqual(before,after)
        self.assertTrue(all("secret" not in x["script"] for x in after))

    def test_ablation_flag_reaches_frozen_base_searcher(self):
        tree=ast.parse((HERE/"harness.py").read_text())
        fun=next(x for x in tree.body
                 if isinstance(x,ast.FunctionDef) and x.name=="search")
        calls=[x for x in ast.walk(fun) if isinstance(x,ast.Call)
               and isinstance(x.func,ast.Attribute)
               and isinstance(x.func.value,ast.Name)
               and x.func.value.id=="base" and x.func.attr=="search"]
        self.assertEqual(len(calls),1)
        flag=[k.value for k in calls[0].keywords if k.arg=="ablation"]
        self.assertEqual(len(flag),1)
        self.assertIsInstance(flag[0],ast.Name)
        self.assertEqual(flag[0].id,"ablation")

    def test_twelve_paired_checkpoint_names_are_used(self):
        src=(HERE/"harness.py").read_text()
        self.assertIn('len(ALIASES)==12',src)
        self.assertIn('self.trained["loop64_task_only_s17"].parameters()',src)
        self.assertIn('self.trained["flat71_task_only_s17"].parameters()',src)
        self.assertIn('FROZEN["all_twelve_models_sha256"][key]',src)
        self.assertIn('report["source_split"]["prospective_test_modules_excluded"]',src)
        self.assertIn('base.PROBE=HERE.parent/"lean_ram16"',src)
        self.assertNotIn('self.trained["loop64_task_only"].parameters()',src)
        self.assertNotIn('self.trained["flat71_task_only"].parameters()',src)

    def test_run_case_defaults_to_primary(self):
        tree=ast.parse((HERE/"run_case.py").read_text())
        func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="run")
        self.assertEqual(ast.literal_eval(func.args.defaults[0]),("primary",))
        source=(HERE/"independent_audit.py").read_text()
        self.assertIn('len(rows)!=160',source)
        self.assertIn('analysis.analyse(rows,harness.LOCKED_SOURCE,harness.CONFIG)',source)

if __name__=="__main__":unittest.main()
