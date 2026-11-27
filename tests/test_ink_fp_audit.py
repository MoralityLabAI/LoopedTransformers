import unittest
from research_gym.ink_fp_audit_01.core import acceptance, grade, replay, roots


def gate(a,b,c,comp=80,rep=0):
    return {"R3": a*100/64, "R4": b*100/64, "RK2": c*100/64,
            "total_correct":a+b+c, "mean_comp":comp, "max_rep":rep}


class AuditTests(unittest.TestCase):
    def test_accepted(self):
        self.assertTrue(acceptance(gate(35,37,39), gate(36,38,38))["accepted_by_rule"])

    def test_mode_drop(self):
        self.assertFalse(acceptance(gate(36,38,38), gate(40,40,34))["accepted_by_rule"])

    def test_total_drop(self):
        self.assertFalse(acceptance(gate(37,41,40), gate(39,40,35))["accepted_by_rule"])

    def test_repetition(self):
        self.assertFalse(acceptance(gate(35,37,39), gate(36,38,39,rep=5))["accepted_by_rule"])

    def test_completion_tie(self):
        self.assertTrue(acceptance(gate(35,37,39,comp=60), gate(35,37,39,comp=64))["accepted_by_rule"])

    def test_invalid_counts(self):
        bad = gate(35,37,39)
        bad["total_correct"] += 1
        with self.assertRaises(ValueError):
            acceptance(bad, gate(36,38,38))

    def test_reported_predecessor(self):
        initial = gate(35,37,39)
        first = {**gate(40,40,34), "scale": 1}
        second = {**gate(41,40,34), "scale": 1}
        row = lambda i,c: {"gen":i,"accepted":True,"winner_scale":1,"candidates":[c]}
        r = replay({"gate_size_per_mode":64,"total_gate_evals":192,"init_gate":initial,
                    "generations":[row(0,first),row(1,second)]})
        self.assertEqual(r["disagreements"],1)
        self.assertEqual(r["rows"][1]["previous"],114)

    def test_roots(self):
        a=roots()
        self.assertEqual(a,roots())
        self.assertEqual(len({r["root_id"] for r in a}),8)
        self.assertTrue(all(0<=r["answer"]<=10 for r in a))
        self.assertNotEqual(a,roots(seed=729002))

    def test_grader(self):
        self.assertTrue(grade("reason\nANSWER: 5",5)["correct"])
        self.assertFalse(grade("ANSWER: 5\nANSWER: 5",5)["correct"])
        self.assertFalse(grade("ANSWER: 5\nextra",5)["correct"])
        self.assertFalse(grade("5",5)["correct"])
        self.assertFalse(grade(("a b c d e f g h "*6)+"\nANSWER: 5",5)["correct"])


if __name__ == "__main__":
    unittest.main()
