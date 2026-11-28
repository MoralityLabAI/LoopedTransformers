"""LSA-02K structured equality and dependent transport engineering tests."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ppt_gym.rewrite_gym import (
    Case, Move, apply_move, certify, freeze, generate, lineage, load_split,
    neighbors, public_packet, render_proof, replay, search,
)


def fixture(family="equality"):
    return Case("fixture","synthetic","train",family,"aabc","dbbc",
                (("aa","db"),("ab","bc"))).validate()


class RewriteTests(unittest.TestCase):
    def test_nested_rewrite_occurrence_and_inverse(self):
        case=fixture()
        menu=neighbors(case,case.source)
        first=next(m for m in menu if m.successor==case.target)
        self.assertEqual(apply_move(case,case.source,first),case.target)
        self.assertEqual(replay(case,[first]),[case.source,case.target])
        inverse=Move(first.rule,1-first.direction,first.position,case.source)
        self.assertEqual(apply_move(case,case.target,inverse),case.source)
        with self.assertRaisesRegex(ValueError,"successor"):
            apply_move(case,case.source,replace(first,successor="aaaa"))
        with self.assertRaisesRegex(ValueError,"does not close"):
            replay(case,[])

    def test_reversible_inner_position_and_transport_renderer(self):
        case=Case("nested","test","train","dependent_transport","aabc","abcc",
                  (("ab","bc"),)).validate()
        move=Move(0,0,1,"abcc")
        self.assertEqual(replay(case,[move])[-1],case.target)
        text=render_proof(case,[move],"nested_test")
        self.assertIn("congrArg (fun z : A => (f_a z))",text)
        self.assertIn("Eq.mp (congrArg P eqproof) witness",text)
        self.assertIn("(P : A → Type)",text)
        inverse=replace(case,source=case.target,target=case.source)
        proof=render_proof(inverse,[Move(0,1,1,case.source)],"inverse_test")
        self.assertIn("Eq.symm",proof)

    def test_same_cost_accounting_and_model_free_search(self):
        case=fixture()
        for policy in ("astar","bidir","uniform"):
            row=search(case,policy,budget=64,depth=16,seed=17)
            self.assertTrue(row["closed"])
            self.assertLessEqual(row["work"],64)
            self.assertEqual(row["model_calls"],0)
            self.assertEqual(replay(case,row["path"])[-1],case.target)
            empty=search(case,policy,budget=0)
            self.assertFalse(empty["closed"])
            self.assertEqual(empty["work"],0)

    def test_rule_lineage_is_permutation_direction_and_order_invariant(self):
        rules=(("ab","bc"),("aa","de"))
        perm=str.maketrans("abcdef","fedcba")
        transformed=tuple((b.translate(perm),a.translate(perm)) for a,b in reversed(rules))
        self.assertEqual(lineage(rules),lineage(transformed))

    def test_deterministic_suite_and_locked_outcomes(self):
        counts={"train":2,"calibration":2,"ordinary":2,"ood":2}
        cases,witnesses=generate(12,counts)
        self.assertEqual(len({c.lineage for split in cases.values() for c in split}),8)
        for split in cases:
            for c,w in zip(cases[split],witnesses[split]):
                self.assertEqual(replay(c,w["path"])[-1],c.target)
                self.assertLessEqual(len(public_packet(c,c.source,16).encode()),384)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"suite"
            freeze(out,12,counts)
            train,m,w=load_split(out,"train",witnesses=True)
            self.assertEqual(len(train),2)
            self.assertFalse((out/"ordinary_witnesses.jsonl").exists())
            with self.assertRaisesRegex(ValueError,"sealed"):
                load_split(out,"ood",witnesses=True)
            p=out/"train.jsonl"
            p.write_text(p.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"hash"):
                load_split(out,"train")

    def test_learned_search_does_not_receive_reference(self):
        case=fixture()
        calls=[]
        def ranker(c,w,m,r):
            calls.append((w,r))
            return [-float(i) for i in range(len(m))]
        result=search(case,"learned",budget=8,ranker=ranker)
        self.assertEqual(result["model_calls"],len(calls))
        self.assertLessEqual(result["work"],8)


if __name__=="__main__":
    unittest.main()
