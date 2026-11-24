"""Exact abstraction and source/reader contracts for OLR-06D."""
from collections import deque
from itertools import product
import random
import unittest

import torch

from ppt_gym.equality_suite import neighbors,ALPHABET
from ppt_gym.olr06d_search import PatternDB,make_case,source_frames,search,partitions
from ppt_gym.olr06d_model import PacketMaker,DistanceReader,weight_hash


RULES=(("aa","ab"),("ab","bb"),("bc","ca"),("cc","dd"),("ad","dc"),("ba","cb"))


class OpaqueRewriteTests(unittest.TestCase):
    def test_partition_cost_and_admissibility_exhaustive(self):
        rng=random.Random(6620)
        pairs=["".join(x) for x in product(ALPHABET,repeat=2)]
        for _ in range(12):
            rules=tuple(tuple(rng.sample(pairs,2)) for _ in range(6))
            goal="".join(rng.choice(ALPHABET) for _ in range(4))
            p=PatternDB(rules,goal)
            dist={goal:0};todo=deque([goal])
            while todo:
                w=todo.popleft()
                self.assertLessEqual(p.bound(w),dist[w])
                for m in neighbors(w,rules):
                    n=m.next_word
                    self.assertLessEqual(p.bound(w),1+p.bound(n))
                    if n not in dist:dist[n]=dist[w]+1;todo.append(n)
        for n in range(2,11):
            for part in partitions(n):
                self.assertEqual([i for a,b in part for i in range(a,b)],list(range(n)))
                self.assertTrue(all(1<=b-a<=3 for a,b in part))

    def test_certifiable_source_and_bounded_search(self):
        c=make_case(RULES,"aaaaaaa","bbbbbaa","train")
        frames,_,_=source_frames(c);self.assertEqual(len(frames),12)
        for w,d,words in frames:
            self.assertEqual(words[0],w);self.assertEqual(words[-1],c.source)
            self.assertEqual(len(words),d+1)
        p=PatternDB(c.equations,c.source)
        got=search(c,p.bound,cap=128)
        self.assertTrue(got["found"])
        self.assertLessEqual(got["rewrite_distance"],15)
        self.assertLessEqual(got["expansions"],128)
        for a,b in zip(got["words"],got["words"][1:]):
            self.assertIn(b,[m.next_word for m in neighbors(a,RULES)])

    def test_matched_reader_and_public_packet(self):
        torch.set_num_threads(1)
        c=make_case(RULES,"aaaaaaa","bbbbbaa","train")
        p=PatternDB(c.equations,c.source);maker=PacketMaker(c,p)
        x,i=maker.batch([c.source,c.target])
        torch.manual_seed(17);a=DistanceReader("bi")
        torch.manual_seed(17);b=DistanceReader("one_way")
        self.assertEqual(weight_hash(a),weight_hash(b))
        self.assertEqual(tuple(x.shape),(2,19,132))
        self.assertEqual(tuple(i.shape),(2,19,19))
        for arm in ("bi","one_way","local","dense","full"):
            m=DistanceReader(arm);pred=m(x,i)
            self.assertEqual(tuple(pred.shape),(2,))
            self.assertTrue(torch.isfinite(pred).all())
            pred.sum().backward()


if __name__=="__main__":unittest.main()
