"""OLR-06F abstract proof bounds, monotonic rekey and total-cap accounting."""
from collections import deque
from itertools import product
import math
import unittest

from ppt_gym.equality_suite import neighbors
from ppt_gym.olr06d_search import PatternDB
from ppt_gym.olr06e_multigoal import Root
from ppt_gym.olr06f_anytime import AnytimePDB,PartialTable,anytime,ARMS

RULES=(("aa","ab"),("ab","bb"),("bc","ca"),("cc","dd"),("ad","dc"),("ba","cb"))
ROOT=Root("dev",RULES,"aaaaaaa",("bbbbbaa","abbaaab","aaaaabb")).validate()


class AnytimeTests(unittest.TestCase):
    def test_partial_lower_bound_monotonicity_and_full_pdb_equality(self):
        goal="abaa"
        p=AnytimePDB(RULES,goal)
        full=PatternDB(RULES,goal)
        dist={goal:0};q=deque([goal])
        while q:
            w=q.popleft()
            for m in neighbors(w,RULES):
                if m.next_word not in dist:
                    dist[m.next_word]=dist[w]+1;q.append(m.next_word)
        words=list(dist)
        old={w:p.bound(w) for w in words}
        while True:
            key=p.choose("cyclic")
            if key is None:break
            p.step(key)
            for w in words:
                h=p.bound(w)
                self.assertLessEqual(old[w],h)
                self.assertLessEqual(h,dist[w])
                old[w]=h
        for w in map("".join,product("abcd",repeat=4)):
            self.assertEqual(p.bound(w),full.bound(w))
        self.assertEqual(p.expansions,full.expansions)
        self.assertEqual(p.edges,full.edges)

    def test_unsettled_frontier_and_unreachable(self):
        p=PartialTable((("aa","ab"),),"a",False,True)
        self.assertEqual(p.bound("c"),0)
        p.step()
        self.assertTrue(p.complete())
        self.assertEqual(p.bound("a"),0)
        self.assertTrue(math.isinf(p.bound("c")))

    def test_total_budget_and_prefix(self):
        for arm in ARMS:
            large=anytime(ROOT,arm,128)
            for cap in (1,8,32,64):
                small=anytime(ROOT,arm,cap)
                self.assertLessEqual(small["total_work"],cap)
                self.assertEqual(small["total_work"],small["abstract_expansions"]+small["expansions"])
                self.assertEqual(small["found"],
                    {t:v for t,v in large["found"].items() if v["found_at"]<=cap})

    def test_eager_setup_charged_before_search(self):
        r=anytime(ROOT,"eager3",1)
        self.assertEqual(r["abstract_expansions"],1)
        self.assertEqual(r["expansions"],0)
        self.assertFalse(r["found"])

    def test_lazy_work_and_cache_counters(self):
        r=anytime(ROOT,"demand_quarter",128)
        self.assertGreater(r["abstract_expansions"],0)
        self.assertGreater(r["expansions"],0)
        self.assertEqual(r["logical"],r["expansions"]+r["hits"])
        self.assertEqual(r["total_work"],r["abstract_expansions"]+r["expansions"])


if __name__=="__main__":unittest.main()
