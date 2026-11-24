import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import reward as R

TASK={"n":3,"facts":[0],"rules":[[[0],1],[[1],2]],"goal":2}

class Feedback(unittest.TestCase):
    def test_native_render(self):
        text=R.render(TASK,{"trace":[0,1]})
        self.assertIn("def intended : Prop :=",text)
        self.assertIn("def advertised : Prop :=",text)
        self.assertIn("exact s1",text)
        self.assertNotIn("sorry",text)
    def test_censored_before_execution(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("reward.compile_source",side_effect=AssertionError("unexpected native execution")):
                for i,c in enumerate(({"trace":[0]},{"trace":[1]},{"trace":[True]},{"trace":["#eval"]})):
                    result=R.evaluate_candidate(TASK,c,Path(d)/str(i))
                    self.assertIsNone(result["reward"])
                    self.assertEqual(result["compiler_calls"],0)
    def test_infrastructure_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("reward.compile_source",return_value={"status":"timeout_unknown"}):
                result=R.evaluate_candidate(TASK,{"trace":[0,1]},Path(d)/"one")
            self.assertIsNone(result["reward"])
    def test_replay_mismatch_censored(self):
        a={"status":"compiled","reports":{R.ROOT:[]}}
        b={"status":"compiled","reports":{R.ROOT:[],R.CONTRACT:[],R.APPLIED:["Classical.choice"]}}
        with tempfile.TemporaryDirectory() as d:
            with patch("reward.compile_source",side_effect=[a,b]):
                result=R.evaluate_candidate(TASK,{"trace":[0,1]},Path(d)/"one")
            self.assertIsNone(result["reward"])
            self.assertEqual(result["classification"],"replay_disagreement_censored")
    def test_unapproved_dependency(self):
        ax=["LRFSubmission.forbiddenPremise"]
        a={"status":"compiled","reports":{R.ROOT:ax}}
        b={"status":"compiled","reports":{R.ROOT:ax,R.CONTRACT:[],R.APPLIED:ax}}
        with tempfile.TemporaryDirectory() as d:
            with patch("reward.compile_source",side_effect=[a,b]):
                result=R.evaluate_candidate(TASK,{"mode":"fixture_axiom"},Path(d)/"one")
            self.assertEqual(result["reward"],0)
            self.assertTrue(result["kernel"])
            self.assertFalse(result["foundation"])
            self.assertTrue(result["contract"])

if __name__=="__main__":unittest.main()
