"""Protocol-only tests for shared frontier search."""
from collections import deque
import unittest
from ppt_gym.equality_suite import neighbors
from ppt_gym.olr06e_multigoal import Root,ARMS,bounded,exact,ProposalLedger,Meter,count_at

RULES=(("aa","ab"),("ab","bb"),("bc","ca"),("cc","dd"),("ad","dc"),("ba","cb"))
ROOT=Root("dev",RULES,"aaaaaaa",("bbbbbaa","abbaaab","aaaaabb")).validate()


class SharedFrontierTests(unittest.TestCase):
    def test_all_bounded_paths_and_cap_prefixes(self):
        for arm in ARMS:
            a=bounded(ROOT,arm,32);b=bounded(ROOT,arm,128)
            self.assertLessEqual(a["expansions"],32)
            self.assertLessEqual(b["expansions"],128)
            self.assertEqual(a["logical"],a["expansions"]+a["hits"])
            self.assertEqual(a["found"],{t:q for t,q in b["found"].items() if q["found_at"]<=32})
        self.assertEqual(len(exact(ROOT)["found"]),3)

    def test_ledger_merge_and_exact_paths(self):
        ledger=ProposalLedger(ROOT);meter=Meter(RULES,128,False)
        ledger.expand(ROOT.source,meter)
        for m in neighbors(ROOT.source,RULES):
            self.assertEqual(ledger.find(m.next_word),ledger.find(ROOT.source))
        result=bounded(ROOT,"demand",128)
        self.assertTrue(result["found"])

    def test_cached_round_robin_meter(self):
        a=bounded(ROOT,"cached_independent",512)
        b=bounded(ROOT,"independent",512)
        self.assertEqual(set(a["found"]),set(b["found"]))
        self.assertLessEqual(a["expansions"],b["expansions"])
        self.assertEqual(a["logical"],b["logical"])
        self.assertEqual(a["edges"]<=b["edges"],True)

    def test_combined_cap_accounts_for_setup(self):
        r={"found":{"a":{"found_at":10},"b":{"found_at":20}},"abstract_expansions":100}
        self.assertEqual(count_at(r,20),2)
        self.assertEqual(count_at(r,20,True),0)
        self.assertEqual(count_at(r,110,True),1)


if __name__=="__main__":unittest.main()
