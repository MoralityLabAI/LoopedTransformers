import unittest
from bench import score,local_url,run
from make_roster import make
from policy import evaluate

class GymTests(unittest.TestCase):
    def test_score(self):
        self.assertTrue(score("Reasoning\nFINAL: 42","42"))
        self.assertFalse(score("FINAL: 420","42"))
        self.assertFalse(score("FINAL: 42\nFINAL: 1","42"))

    def test_local(self):
        self.assertEqual(local_url("http://127.0.0.1:8080"),"http://127.0.0.1:8080/v1/chat/completions")
        with self.assertRaises(ValueError):local_url("https://example.com")

    def test_family_disjoint(self):
        rows=make()
        self.assertEqual(len(rows),16)
        self.assertFalse({r["family"] for r in rows if r["split"]=="train"}&{r["family"] for r in rows if r["split"]=="dev"})

    def test_mock_request(self):
        r=make()[0]
        def fake(url,payload,timeout):
            return {"choices":[{"message":{"content":"FINAL: "+r["answer"]},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":10,"completion_tokens":4}}
        got=run(r,"http://localhost",32,17,1,fake)
        self.assertTrue(got["correct"])
        self.assertEqual(got["completion_tokens"],4)

    def test_train_only(self):
        arms={}
        for k in (1,2,4):
            arms[k]={r["root"]:{"root":r,"correct":k==1 if r["split"]=="train" else k==4,
                      "prompt_tokens":10,"completion_tokens":5,"wall_seconds":1,"finish_reason":"stop"} for r in make()}
        report=evaluate(arms)
        self.assertEqual(report["train_only_lookup"]["mean_depth"],1)
        self.assertEqual(report["train_only_lookup"]["family_macro_accuracy"],0)
        self.assertEqual(report["hindsight_min_correct_depth"]["family_macro_accuracy"],1)

if __name__=="__main__":unittest.main()
