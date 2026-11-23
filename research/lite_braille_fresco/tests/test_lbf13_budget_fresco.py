"""Exact work labels and surgical sibling/frontier treatment tests."""
import os
from pathlib import Path
import sys
import unittest
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!="torch":raise
    raise unittest.SkipTest("optional CPU Torch is absent")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lbf13_budget_fresco as b


class BudgetLabels(unittest.TestCase):
    def test_buckets_and_censoring(self):
        self.assertEqual([b.budget_bucket(x) for x in (1,3,4,15,64,127,128)],[1,1,4,8,64,64,128])
        self.assertEqual(b.labels([1,4,9,None],4),[1,1,0,0])
        self.assertEqual(b.labels([1,4,9,None],128),[1,1,1,0])
        with self.assertRaises(ValueError):b.budget_bucket(0)

    def test_monotone_exact_curve(self):
        costs=[None,1,4,8,127,128]
        curve=[b.labels(costs,x) for x in b.BUCKETS]
        for a,z in zip(curve,curve[1:]):
            self.assertTrue(all(i<=j for i,j in zip(a,z)))


@unittest.skipUnless(os.environ.get("LBF13_UPSTREAM"),"requires pinned rewrite source in native CPU job")
class PinnedSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rg=b.base.load_upstream(Path(os.environ["LBF13_UPSTREAM"]))

    def fixture(self,target):
        return self.rg.Case("fixture","fixture","train","equality","aab",target,(("aa","ba"),("ab","cc"))).validate()

    def test_frontier_vs_sibling_order(self):
        c=self.fixture("bab")
        ranker=lambda case,word,menu,d,w:[-100. if m.successor==case.target else 100. for m in menu]
        f=b.search(self.rg,c,"frontier",ranker,budget=1)
        s=b.search(self.rg,c,"sibling",ranker,budget=1)
        self.assertTrue(f["closed"])
        self.assertFalse(s["closed"])
        self.assertEqual(f["work"],s["work"])
        self.assertEqual(f["model_calls"],1)

    def test_exact_forced_prefix(self):
        c=self.fixture("bcc")
        for move in self.rg.neighbors(c,c.source):
            full,path,charged=b.force(self.rg,c,c.source,move,4)
            if full is not None:
                self.assertEqual(full,charged)
                self.rg.replay(c,path)
            for budget in b.BUCKETS:
                small,_,w=b.force(self.rg,c,c.source,move,4,budget)
                self.assertEqual(small,full if full is not None and full<=budget else None)
                self.assertLessEqual(w,budget)
        direct=self.fixture("bab")
        move=next(m for m in self.rg.neighbors(direct,direct.source) if m.successor==direct.target)
        self.assertEqual(b.force(self.rg,direct,direct.source,move,1,1)[0],1)

    def test_symbolic_equivalence(self):
        c=self.fixture("bcc")
        for work in (1,4,8,128):
            self.assertEqual(b.search(self.rg,c,budget=work,depth=4),
                             self.rg.search(c,"astar",budget=work,depth=4))


if __name__=="__main__":unittest.main()
