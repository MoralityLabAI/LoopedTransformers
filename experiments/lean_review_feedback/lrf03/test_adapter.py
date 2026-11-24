import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import adapter as A

SOURCE="""namespace MathCan.ConstructiveObservationAudit
theorem demo_monitor_unknown_denied (a : Bool) :
    demoMonitor 2 a = false := by
  cases a <;> rfl
theorem later : True := True.intro
end MathCan.ConstructiveObservationAudit
"""
ROOT="demo_monitor_unknown_denied"

class Tests(unittest.TestCase):
    def test_only_exact_header_and_prefix(self):
        first,second,h,line=A.render(SOURCE,ROOT,["enumerate"])
        self.assertNotIn("theorem "+ROOT,first)
        self.assertNotIn("theorem later",first)
        self.assertIn("theorem lrf03Submitted (a : Bool) :",first)
        self.assertIn("  cases a <;> rfl",first)
        self.assertTrue(second.splitlines()[line-1].startswith("theorem lrf03Contract"))
        self.assertEqual(h,A.sha(A.prefix_header(SOURCE,ROOT)[1]))
    def test_invalid_actions_censored_before_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("adapter.native",side_effect=AssertionError("must not execute")):
                for i,actions in enumerate([["exact arbitrary Lean"],["enumerate\n#eval"],["skip"]*9,[],[False]]):
                    result=A.evaluate_actions(Path(tmp),SOURCE,ROOT,actions,Path(tmp)/str(i))
                    self.assertEqual(result["classification"],"malformed_censored")
                    self.assertEqual(result["compiler_calls"],0)
    def test_unknown_second_failure_censored(self):
        with tempfile.TemporaryDirectory() as tmp:
            one={"status":"compiled","reports":{A.SUB:[]}}
            two={"status":"rejected","error_excerpt":"compiler memory error"}
            with patch("adapter.native",side_effect=[one,two]):
                result=A.evaluate_actions(Path(tmp),SOURCE,ROOT,["enumerate"],Path(tmp)/"out")
            self.assertIsNone(result["reward"])
            self.assertEqual(result["classification"],"contract_unknown_censored")
    def test_independent_inventory_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            one={"status":"compiled","reports":{A.SUB:[]}}
            two={"status":"compiled","reports":{A.SUB:[],A.CON:["propext"]}}
            with patch("adapter.native",side_effect=[one,two]):
                result=A.evaluate_actions(Path(tmp),SOURCE,ROOT,["enumerate"],Path(tmp)/"out")
            self.assertIsNone(result["reward"])
            self.assertEqual(result["classification"],"independent_replay_disagreement_censored")

if __name__=="__main__":unittest.main()
