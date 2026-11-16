"""Regression: a verified root goal may outlive a slow original proof body."""
from __future__ import annotations
import json,signal,tempfile,unittest
from pathlib import Path
from unittest import mock
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_ram16"))
sys.path.insert(0,str(HERE))
import goal_probe

def fake_session(exitcode,stop=True,valid=True):
    class Fake:
        def __init__(self,lean,original,theorem,folder,guard,timeout):
            self.out=Path(folder);self.out.mkdir(parents=True)
            self.number=0
            self.initial={"accepted":valid,"goals":["x : Nat\n⊢ x = x"] if valid else [],"definitions":[]}
            self.proc=type("Process",(),{"returncode":None})()
        def close(self):
            self.proc.returncode=exitcode
            if stop:
                (self.out/"request-1.json").write_text(json.dumps({"id":1,"actions":[],"stop":True}))
    return Fake

class ShutdownTests(unittest.TestCase):
    def read(self,code,stop=True,valid=True):
        with tempfile.TemporaryDirectory() as where:
            with mock.patch.object(goal_probe,"Session",fake_session(code,stop,valid)):
                result=goal_probe.observe_original_goal("lean","source","target",Path(where)/"goal",None)
            receipt=json.loads((Path(where)/"goal"/"goal_probe_receipt.json").read_text())
            return result,receipt
    def test_clean_shutdown(self):
        result,receipt=self.read(0)
        self.assertTrue(result["accepted"])
        self.assertFalse(receipt["deliberately_forced_shutdown"])
        self.assertFalse(receipt["complete_proof_source_certified"])
    def test_intentional_kill_after_genuine_goal(self):
        result,receipt=self.read(-signal.SIGKILL)
        self.assertTrue(result["accepted"])
        self.assertTrue(receipt["deliberately_forced_shutdown"])
        self.assertEqual(receipt["exitcode"],-9)
        self.assertFalse(receipt["complete_proof_source_certified"])
    def test_intentional_term_after_genuine_goal(self):
        result,receipt=self.read(-signal.SIGTERM)
        self.assertTrue(receipt["deliberately_forced_shutdown"])
    def test_unexpected_nonzero_exit_rejected(self):
        with self.assertRaises(goal_probe.ProbeTimeout):
            self.read(1)
    def test_forced_exit_without_stop_receipt_rejected(self):
        with self.assertRaises(goal_probe.ProbeTimeout):
            self.read(-signal.SIGKILL,stop=False)
    def test_invalid_original_goal_rejected(self):
        with self.assertRaises(AssertionError):
            self.read(-signal.SIGKILL,valid=False)
    def test_collector_and_third_audit_reuse_same_observer(self):
        from pathlib import Path
        collector=(HERE/"collect.py").read_text()
        third=(HERE/"third_audit.py").read_text()
        self.assertIn("return observe_original_goal(lean,source,theorem,path,guard,timeout=15.)",collector)
        self.assertIn("native=observe_original_goal(lean,original,theorem,folder,guard,timeout=15.)",third)
        self.assertIn('scratch=out/"native_source_compiler_artifacts"',collector)
        self.assertIn('scratch=out/"third_compiler_sources_and_logs"',third)
if __name__=="__main__":
    unittest.main()
