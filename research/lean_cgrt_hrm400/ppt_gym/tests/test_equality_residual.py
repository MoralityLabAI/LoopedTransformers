"""Engineering tests; native certificates are checked in the registered CI run."""
import unittest
import torch

from ppt_gym.data import digest
from ppt_gym.equality_suite import EqCase, neighbors, system_lineage
from ppt_gym.equality_search import plan, path_moves
from ppt_gym.equality_ordering import best_first, hamming
from ppt_gym.equality_residual import (
    Features, Residual, reverse_teacher, teacher_path, groups, retarget, residual_search,
)
from ppt_gym.tests.test_equality_preflight import fixture


class ResidualTests(unittest.TestCase):
    def test_reverse_distances_and_offpath_pairs(self):
        c = fixture()
        d, parent, expansions, edges = reverse_teacher(c, cap=1024, depth=10)
        self.assertLessEqual(expansions,1024)
        self.assertGreater(edges,0)
        self.assertEqual(d[c.target],plan(c)["rewrite_distance"])
        for g in groups(c,d,set(plan(c)["words"])):
            w, good, alt = g["word"], g["good"], g["alternate"]
            self.assertEqual(d[good],d[w]-1)
            self.assertGreaterEqual(d[alt],d[w])
            self.assertIn(d[alt]-d[good],(1,2))
            for v in (w,good,alt):
                path = teacher_path(v,parent)
                self.assertEqual(len(path)-1,d[v])
                path_moves(retarget(c,v),path)
            self.assertGreaterEqual(g["escape_depth"],1)

    def test_invariant_local_features(self):
        c = fixture()
        tr = str.maketrans("abcd","dcab")
        rules = tuple((b.translate(tr),a.translate(tr)) for a,b in reversed(c.equations))
        source,target = c.source.translate(tr),c.target.translate(tr)
        root = digest([c.lineage,c.split,rules,source,target])[:24]
        other = EqCase(root,system_lineage(rules),c.split,c.family,rules,source,target).validate()
        a,b = Features(c),Features(other)
        for w in [c.source,c.target,*(m.next_word for m in neighbors(c.target,c.equations))]:
            self.assertEqual(a.encode(w),b.encode(w.translate(tr)))
            hs = [hamming(m.next_word,c.source) for m in neighbors(w,c.equations)]
            self.assertEqual(a.public_score(w,"local_greedy"),min([hamming(w,c.source)]+hs))

    def test_zero_residual_identical_greedy_search(self):
        c = fixture()
        a,b = Residual(4),Residual(1)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        f = Features(c)
        x = torch.tensor([f.encode(c.target)])
        mask = torch.tensor([f.mask],dtype=torch.bool)
        self.assertEqual(a(x,mask).item(),0)
        for cap in (1,16,64,1024):
            ref = best_first(c,cap,"hamming_greedy")
            for actual in (residual_search(c,cap,"hamming_greedy"),residual_search(c,cap,"residual",a)):
                for key in ("found","words","expansions","candidate_edges_enumerated"):
                    self.assertEqual(ref[key],actual[key])
        a(x,mask).sum().backward()
        self.assertGreater(float(a.readout.weight.grad.abs().sum()),0)


if __name__ == "__main__":
    unittest.main()
