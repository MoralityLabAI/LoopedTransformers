"""Independent protocol and posterior tests. Native CI checks real receipts."""
from itertools import product
from fractions import Fraction
import unittest
import torch

from ppt_gym.olr06c_native import (
    Root,Offer,bank,from_row,parse,flatten,truth_table,constant,outcomes,closure,transport,UNKNOWN,STOP
)
from ppt_gym.olr06c_planner import Planner,execute
from ppt_gym.olr06c_model import SemanticPrior,packet,weight_hash,query_probs


def fixture():
    return Root((("or","x","not","x"),("and","y","not","y")),(
        Offer("anchor",0,0,1),Offer("anchor",0,0,-1),
        Offer("relation",0,1,-1),Offer("relation",0,1,1)),2,"fixture")


def brute(root,ws,a=0,s=0):
    """Independent explicit outcome recursion over uniformly weighted assignments."""
    count=sum(v!=UNKNOWN for v in closure(root,s).values())
    best=Fraction(int(count==root.n))+Fraction(count,4*root.n)-Fraction(a.bit_count(),100)
    if a.bit_count()==root.budget:return best
    for j in range(len(root.offers)):
        if a>>j&1:continue
        yes=[w for w in ws if outcomes(root,w)[j]]
        no=[w for w in ws if not outcomes(root,w)[j]]
        val=0
        if yes:val+=Fraction(len(yes),len(ws))*brute(root,yes,a|1<<j,s|1<<j)
        if no:val+=Fraction(len(no),len(ws))*brute(root,no,a|1<<j,s)
        best=max(best,val)
    return best


class NativeSemanticTests(unittest.TestCase):
    def test_public_generator_disjoint_and_constant(self):
        b=bank();seen=set()
        self.assertEqual(b,bank())
        for rows in b.values():
            for r in rows:
                root=from_row(r["root"])
                self.assertFalse(seen&set(root.exprs))
                for e in root.exprs:
                    self.assertEqual(flatten(parse(e)),e)
                    self.assertEqual(len(set(truth_table(e))),1)
                seen.update(root.exprs)

    def test_exact_posterior_matches_brute_and_positive_only_transport(self):
        root=fixture()
        p=Planner(root,[.5,.5])
        self.assertAlmostEqual(p.value(0,0),float(brute(root,list(product((-1,1),repeat=2)))))
        p=Planner(root,[1.,0.])
        self.assertAlmostEqual(p.value(0,0),1.23)
        self.assertEqual(closure(root,1<<2),{0:UNKNOWN,1:UNKNOWN})
        nodes,term=transport(root,(1<<0)|(1<<2),"r0")
        self.assertEqual(nodes,(0,1));self.assertIn("congrArg Bool.not",term)
        self.assertNotIn("r0_q1",term);self.assertNotIn("r0_q3",term)
        self.assertEqual(transport(root,1<<2,"r0"),((),None))

    def test_observed_rejection_updates_belief_not_proof(self):
        root=fixture();p=Planner(root,[.5,.5])
        self.assertEqual(p.coverage(0),0)
        mask=p.possible(1<<1,0)
        self.assertEqual(mask.bit_count(),2)
        self.assertEqual(p.coverage(0),0)
        receipt=[{"accepted":v} for v in outcomes(root,(1,-1))]
        result=execute(root,receipt,[1.,0.],None)
        self.assertTrue(result["full"]);self.assertEqual(result["queries"],2)

    def test_semantic_packet_and_equal_direction_initialization(self):
        root=fixture();x,g,inc,end,legal=packet(root)
        torch.set_num_threads(1)
        torch.manual_seed(7);a=SemanticPrior("bi")
        torch.manual_seed(7);b=SemanticPrior("one_way")
        self.assertEqual(weight_hash(a),weight_hash(b))
        z=a(x[None],g[None],inc[None])
        p=query_probs(z,g[None],end[None])
        self.assertEqual(tuple(p.shape),(1,len(legal)))
        self.assertAlmostEqual(float(p[0,0]+p[0,1]),1.,places=5)
        self.assertAlmostEqual(float(p[0,2]+p[0,3]),1.,places=5)


if __name__=="__main__":unittest.main()
