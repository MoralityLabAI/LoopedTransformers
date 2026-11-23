"""Engineering regressions; simulated graph checks are not Lean certificates."""
from collections import deque
from dataclasses import replace
from itertools import permutations
import unittest

from ppt_gym.data import digest
from ppt_gym.equality_suite import EqCase, system_lineage, term, neighbors
from ppt_gym.equality_search import plan, path_moves
from ppt_gym.equality_preflight import _check_final_script


RULES=(("ab","cd"),("ba","dc"),("aa","bb"),("ac","bd"),("ca","db"),("ad","bc"))


def fixture():
    source="ababaaa"
    target="cdcdbaa"
    lineage=system_lineage(RULES)
    root=digest([lineage,"train",RULES,source,target])[:24]
    return EqCase(root,lineage,"train","indexed_unary_equations",RULES,source,target).validate()


def distance(case):
    todo=deque([(case.target,0)])
    seen={case.target}
    while todo:
        w,d=todo.popleft()
        if w==case.source:
            return d
        for m in neighbors(w,case.equations):
            if m.next_word not in seen:
                seen.add(m.next_word)
                todo.append((m.next_word,d+1))
    return None


class EqualityTests(unittest.TestCase):
    def test_public_binder_and_instantiated_tactic(self):
        c=fixture()
        self.assertEqual(term("abc"),"a (b (c x))")
        self.assertIn("∀ z : Nat, a (b z) = c (d z)",c.statement)
        self.assertIn("(w : P (",c.statement)
        m=next(m for m in neighbors("abcd",RULES) if m.rule==0 and m.position==0)
        self.assertEqual(m.tactic("abcd"),"rw [e0 (c (d x))]")
        self.assertEqual(m.next_word,"cdcd")

    def test_lineage_disjointness_up_to_alpha_and_orientation(self):
        c=fixture()
        for p in permutations("abcd"):
            trans=str.maketrans("abcd","".join(p))
            transformed=tuple((b.translate(trans),a.translate(trans)) for a,b in reversed(RULES))
            self.assertEqual(system_lineage(transformed),c.lineage)
        with self.assertRaises(ValueError):
            replace(c,lineage="fake").validate()

    def test_exact_planning_matches_unidirectional_bfs(self):
        c=fixture()
        r=plan(c)
        self.assertTrue(r["found"])
        self.assertEqual(r["rewrite_distance"],distance(c))
        self.assertEqual(r["words"][0],c.target)
        self.assertEqual(r["words"][-1],c.source)
        for cap in (1,2,4,16,64):
            x=plan(c,cap)
            self.assertEqual(x["found"],r["expansions"]<=cap)
            if x["found"]:
                self.assertEqual(x["words"],r["words"])
            self.assertLessEqual(x["expansions"],cap)

    def test_final_script_is_reconstructed_from_frozen_public_grammar(self):
        c=fixture()
        path=plan(c)["words"]
        script=[m.tactic(w) for m,w in zip(path_moves(c,path),path)]+["exact w"]
        _check_final_script(c,script)
        with self.assertRaises(ValueError):
            _check_final_script(c,["sorry","exact w"])
        with self.assertRaises(ValueError):
            _check_final_script(c,["exact w"])
        with self.assertRaises(ValueError):
            path_moves(c,[c.target,"ddddddd",c.source])


if __name__=="__main__":
    unittest.main()
