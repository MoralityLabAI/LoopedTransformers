"""RAM-04 frozen prospective source-family and matched-CPU comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate as e
import ram03_core as core
from model import Policy,Config

class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((HERE/"config.json").read_text())

    def test_12_names_and_strictly_new_modules(self):
        e.source_contract(self.config)
        self.assertEqual(len(e.EXACT_CASES),4)
        self.assertEqual(sum(map(len,e.EXACT_CASES.values())),12)
        self.assertTrue(set(e.EXACT_CASES).isdisjoint(e.SOURCE_TRAIN|e.OLD_TEST))

    def test_immutable_training_and_budget(self):
        self.assertEqual(self.config["development"],[
            "MathCan/HookState.lean:endpoint_append",
            "MathCan/HookState.lean:pathSum_gauge",
            "MathCan/TOEGraph.lean:integral_ext",
            "MathCan/Intervention.lean:obsEqOn_symm",
            "MathCan/TOECosmology.lean:anchored_present"])
        self.assertEqual(self.config["collector"]["states_per_theorem"],10)
        self.assertEqual(self.config["collector"]["max_goals"],8)
        self.assertEqual(self.config["training"]["updates"],1600)
        self.assertEqual(self.config["training"]["branch_sample_fraction"],0.5)

    def test_first_three_names_verified_from_original_only(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)
            (source/"lean-toolchain").write_text("leanprover/lean4:v4.35.0\n")
            for filename,names in e.EXACT_CASES.items():
                dest=source/filename
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(
                    "import Init\n" +
                    "\n".join(f"theorem {n} : True := by trivial" for n in names) +
                    "\ntheorem fourth : True := by trivial\n"
                )
            e.source_contract(self.config,source)
            victim=source/next(iter(e.EXACT_CASES))
            victim.write_text(victim.read_text().replace("theorem ","theorem tampered_",1))
            with self.assertRaises(AssertionError):
                e.source_contract(self.config,source)

    def test_architecture_and_recurrent_shared_cell(self):
        for name,width,loops in (("loop32",32,4),("loop64",64,4),("flat64",64,0)):
            policy=Policy(Config(name,width,loops))
            self.assertEqual(policy.config.loops,loops)
            self.assertEqual(policy.config.width,width)
            self.assertLess(sum(p.numel() for p in policy.parameters()),250000)

    def test_current_original_tactic_grammar(self):
        self.assertIn("induction_list",core.FAMILIES)
        self.assertIn("rcases_local",core.FAMILIES)
        self.assertIn("rw_local",core.FAMILIES)
        for action in ("omega","all_goals simp_all","induction xs","rw [h]"):
            self.assertTrue(core.valid_script(action))

    def test_fatal_lean_compiler_exit_is_recorded_as_unknown_restart_event(self):
        with tempfile.TemporaryDirectory() as dirname:
            instance=object.__new__(e.CrashAwareSession)
            instance.out=Path(dirname)
            with mock.patch.object(e.Session,"query",
                    side_effect=RuntimeError("Lean exited: maximum recursion depth has been reached")):
                with self.assertRaises(e.ProbeTimeout):
                    instance.query(["simp_all"],timeout=.25)
            recorded=Path(dirname)/"fatal_exits.jsonl"
            self.assertTrue(recorded.is_file())
            row=json.loads(recorded.read_text().strip())
            self.assertEqual(row["classification"],"compiler_process_exit_unknown")
            self.assertEqual(row["actions"],["simp_all"])
            self.assertIn("maximum recursion depth",row["error_excerpt"])

    def test_nonfatal_protocol_runtime_error_stays_a_real_error(self):
        with tempfile.TemporaryDirectory() as dirname:
            instance=object.__new__(e.CrashAwareSession)
            instance.out=Path(dirname)
            with mock.patch.object(e.Session,"query",
                    side_effect=RuntimeError("Lean protocol desynchronized")):
                with self.assertRaisesRegex(RuntimeError,"protocol desynchronized"):
                    instance.query(["rfl"],timeout=.25)
            self.assertFalse((Path(dirname)/"fatal_exits.jsonl").exists())

    def test_summary_marks_model_only_proofs_by_exact_theorem(self):
        def mk(yes,name):
            return {
                "accepted":yes,"source_file":"MathCan/OpcodeRefinement.lean",
                "theorem":name,"elapsed_seconds":1.0,"logical_evaluations":2,
                "actual_lean_processes":1,"diagnostics":{},"reason":"x",
            }
        rows={"primary__native":[mk(True,"two_step_gross_budget"),
                                  mk(False,"two_step_output_aggregation")],
              "primary__loop32":[mk(True,"two_step_gross_budget"),
                                 mk(True,"two_step_output_aggregation")]}
        summaries=e.summarize(rows)
        model=next(x for x in summaries if x["model"]=="loop32")
        self.assertEqual(model["certified"],2)
        self.assertEqual(model["model_only_vs_native"],[
             "MathCan/OpcodeRefinement.lean:two_step_output_aggregation"])

if __name__=="__main__":
    unittest.main()
