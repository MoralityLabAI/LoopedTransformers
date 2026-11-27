"""Protocol-only tests for the frontier block intervention."""
from copy import deepcopy
import unittest

from ppt_gym.olr06e_multigoal import Root,bounded
from ppt_gym.olr06h_frontier import (
    State, prefix, observe, controls, suffix, verify_identity,
    PREFIX,BLOCK,CAP,SLOTS,CONTROLS,screen,add_fixed,
)

RULES=(("aa","ab"),("ab","bb"),("bc","ca"),("cc","dd"),("ad","dc"),("ba","cb"))
ROOT=Root("dev",RULES,"aaaaaaa",("bbbbbaa","abbaaab","aaaaabb")).validate()


class FrontierTests(unittest.TestCase):
    def test_unmodified_demand_identity(self):
        s=State(ROOT);s.advance(PREFIX)
        verify_identity(s,PREFIX)
        p=observe(s)
        t,r=suffix(s,0,0.,p["legal"])
        verify_identity(t,CAP)
        self.assertEqual(r["total_work"],r["prefix_work"]+r["continuation_work"])
        self.assertEqual(r["block_work"],0)

    def test_live_component_packet_and_clone(self):
        # Earlier prefix for a reliable multi-component test fixture.
        s=State(ROOT);s.advance(1);p=observe(s)
        self.assertGreater(sum(p["legal"]),1)
        self.assertEqual(len(p["components"]),SLOTS)
        self.assertEqual(len(p["contacts"]),15)
        self.assertEqual(p["pair_comparisons"],sum(c["comparisons"] for c in p["contacts"]))
        before=deepcopy(s.found);nodes=set(s.graph.adj);count=s.meter.expansions
        for a,allowed in enumerate(p["legal"]):
            if not allowed:continue
            t,r=suffix(s,a,0.,p["legal"])
            self.assertLessEqual(r["block_work"],BLOCK)
            self.assertLessEqual(r["total_work"],CAP)
            self.assertTrue(nodes<=set(t.graph.adj))
            self.assertTrue(set(before)<=set(t.found))
            self.assertEqual(r["total_work"],r["prefix_work"]+r["block_work"]+r["continuation_work"])
        self.assertEqual(s.found,before)
        self.assertEqual(s.meter.expansions,count)
        self.assertEqual(set(s.graph.adj),nodes)
        for a in controls(p).values():self.assertTrue(p["legal"][a])

    def test_terminal_prefix_has_only_continue(self):
        s=State(ROOT);s.advance(CAP);p=observe(s)
        self.assertEqual(p["legal"],[True]+[False]*SLOTS)
        self.assertTrue(all(a==0 for a in controls(p).values()))
        self.assertRaises(ValueError,suffix,s,1,0.,p["legal"])


if __name__=="__main__":unittest.main()
