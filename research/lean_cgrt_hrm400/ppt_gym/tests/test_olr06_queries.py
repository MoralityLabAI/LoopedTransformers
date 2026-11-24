"""Independent small-instance checks of OLR-06B planning semantics."""
import itertools
import random
import unittest
from fractions import Fraction

import torch

from ppt_gym.olr06_queries import Offer,Root,STOP,Planner,delayed_example,execute,bank,from_row
from ppt_gym.olr06_exact_ledger import solve,UNKNOWN
from ppt_gym.olr06_query_model import packet,make,hash_weights,macs


def brute(root,attempted=(),accepted=()):
    """Slow independent recursion using actual signed constraints and fractions."""
    edges=[];anchors=[(i,1) for i in root.anchored]
    for j in accepted:
        o=root.offers[j]
        if o.kind=="anchor":anchors.append((o.u,1))
        else:edges.append((o.u,o.v,1))
    _,labels=solve(range(len(root.weights)),edges,anchors)
    cover=sum(root.weights[i] for i,v in labels.items() if v!=UNKNOWN)
    cost=sum(root.offers[j].cost for j in attempted)
    best=Fraction(cover,root.total)-Fraction(cost,100)
    for j,o in enumerate(root.offers):
        if j in attempted or cost+o.cost>root.budget:continue
        a=attempted+(j,)
        yes=brute(root,a,accepted+(j,))
        no=brute(root,a,accepted) if o.p4<4 else yes
        best=max(best,Fraction(o.p4,4)*yes+Fraction(4-o.p4,4)*no)
    return best


class QueryTests(unittest.TestCase):
    def test_delayed_relation_gain(self):
        root=delayed_example();p=Planner(root)
        self.assertEqual(p.closure(1)[-1],0)
        self.assertIn(p.choose(0,0,1),(4,5))
        self.assertIn(p.choose(0,0,2),(4,5))
        q=p.values(0,0,2)
        self.assertGreater(q[4],max(q[j] for j in (0,1,2,3,STOP)))
        greedy=p.expectation(lambda a,s:p.choose(a,s,1))
        two=p.expectation(lambda a,s:p.choose(a,s,2))
        opt=p.expectation(lambda a,s:p.choose(a,s))
        self.assertEqual(greedy["resolved"],6)
        self.assertEqual(two["resolved"],6)
        self.assertEqual(opt["resolved"],8)
        self.assertAlmostEqual(opt["utility"],p.value())

    def test_reference_against_independent_fraction_recursion(self):
        rng=random.Random(6601)
        for _ in range(50):
            offers=(Offer("relation",0,1,1,rng.randint(1,4)),
                    Offer("relation",1,2,1,rng.randint(1,4)),
                    Offer("anchor",0,0,rng.randint(1,2),rng.randint(1,4)),
                    Offer("anchor",2,2,rng.randint(1,2),rng.randint(1,4)))
            root=Root(tuple(rng.randint(1,4) for _ in range(3)),(),offers,
                      rng.randint(1,4),"test")
            p=Planner(root)
            self.assertAlmostEqual(p.value(),float(brute(root)),places=10)
            self.assertAlmostEqual(p.expectation(lambda a,s:p.choose(a,s))["utility"],
                                   p.value(),places=10)

    def test_all_outcomes_audited_and_censoring(self):
        root=Root((1,2),(),(Offer("anchor",0,0,1,1),
                            Offer("relation",0,1,1,1)),2,"test")
        p=Planner(root)
        def query(a,s):
            return 0 if a==0 else 1 if a==1 else STOP
        seen=set()
        for seed in range(100):
            result=execute(root,query,seed)
            self.assertEqual(result["spent"],2)
            self.assertEqual(result["queries"],2)
            for e in result["events"]:
                seen.add(e["status"])
                self.assertEqual(e["receipt"] is None,e["status"]=="censored")
        self.assertEqual(seen,{"accepted","censored"})
        self.assertAlmostEqual(p.expectation(query)["resolved"],.25+.25*.25*2)

    def test_relabeling_signature_and_values(self):
        a=delayed_example()
        order=(2,0,3,1);inv={u:j for j,u in enumerate(order)}
        b=Root(tuple(a.weights[u] for u in order),(),
               tuple(Offer(o.kind,inv[o.u],inv[o.v],o.cost,o.p4)
                     for o in reversed(a.offers)),a.budget,"renamed")
        self.assertEqual(a.signature(),b.signature())
        self.assertAlmostEqual(Planner(a).value(),Planner(b).value())

    def test_split_signature_uniqueness(self):
        data=bank();sigs=[r["signature"] for rows in data.values() for r in rows]
        self.assertEqual(len(sigs),240)
        self.assertEqual(len(set(sigs)),240)
        self.assertEqual(data,bank())

    def test_matched_direction_initialization_and_legal_packet(self):
        p=Planner(delayed_example());x,i,m=packet(p,0,0)
        torch.set_num_threads(1)
        checks=[]
        for arm in ("bi","component_to_query","query_to_component"):
            torch.manual_seed(17);model=make(arm)
            checks.append((hash_weights(model),macs(arm),
                           sum(v.numel() for v in model.parameters())))
            self.assertEqual(model(x[None],i[None]).shape,(1,len(m)))
        self.assertEqual(len(set(checks)),1)
        self.assertEqual(set(torch.nonzero(m).flatten().tolist()),set(p.legal(0)))
        self.assertTrue(m[STOP])


if __name__=="__main__":unittest.main()
