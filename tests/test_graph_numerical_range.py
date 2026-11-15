"""NRF-02 exact graph and guarded-policy regression tests."""
import unittest
from fractions import Fraction as Q
from itertools import product
from research_gym.core.numerical_range_certificate import Phase, sqdist
from research_gym.core.graph_numerical_range import (
    Edge,Graph,Guard,UncertifiedGuard,certify_path,choose,
    example_graph,example_policy,guarded_trace,potential,replay_perturbed,
)

def constant_phase(rate:Q):
    if rate>=1:
        return Phase(Q(0),Q(1),(rate/2,),Q(1),"identity")
    if rate==0:
        return Phase(Q(0),Q(1),(Q(0),),Q(1),"identity")
    return Phase(Q(0),Q(1),(Q(0),),1-rate,"identity")

class GraphNumericalRangeTests(unittest.TestCase):
    def setUp(self):
        self.graph=example_graph()
        self.policies=example_policy()

    def test_expanding_edge_and_weighted_witness(self):
        g=self.graph
        self.assertEqual(g.edge("sample").rate,Q(3,2))
        self.assertEqual(g.edge("approve").rate,Q(1,4))
        w=potential(g,Q(3,4))
        self.assertEqual(w,{"idle":Q(1),"probe":Q(2),"audit":Q(4,3)})
        for e in g.edges:
            self.assertLessEqual(e.rate*w[e.source],Q(3,4)*w[e.target])

    def test_path_certificates_exhaustive_to_depth_seven(self):
        g=self.graph; w=potential(g,Q(3,4));alpha=Q(3,4)
        front=[(v,()) for v in g.vertices]; count=0
        for _ in range(8):
            next_front=[]
            for start,path in front:
                target,exact,bound=certify_path(g,w,alpha,start,path)
                self.assertLessEqual(exact,bound); count+=1
                if len(path)<7:
                    next_front.extend((start,path+(e.name,))
                                      for e in g.edges if e.source==target)
            front=next_front
        self.assertGreater(count,150)

    def test_bad_cycle_rejected_even_if_many_good_edges(self):
        g=Graph(("a","b"),(
            Edge("up","a","b",constant_phase(Q(3,2))),
            Edge("down","b","a",constant_phase(Q(3,4))),
            Edge("stay","a","a",constant_phase(Q(1,4))),
        ))
        with self.assertRaisesRegex(ValueError,"excessive directed cycle"):
            potential(g,Q(3,4))

    def test_positive_cycle_criterion_exhaustive_two_nodes(self):
        alpha=Q(3,4); trials=0
        for a,b,c,d in product((Q(0),Q(1,4),Q(3,4),Q(3,2)),repeat=4):
            g=Graph(("u","v"),(
                Edge("uu","u","u",constant_phase(a)),
                Edge("uv","u","v",constant_phase(b)),
                Edge("vu","v","u",constant_phase(c)),
                Edge("vv","v","v",constant_phase(d)),
            ))
            allowed=a<=alpha and d<=alpha and b*c<=alpha*alpha
            try:
                weights=potential(g,alpha)
                self.assertTrue(allowed)
                self.assertTrue(all(e.rate*weights[e.source]<=alpha*weights[e.target]
                                    for e in g.edges))
            except ValueError:
                self.assertFalse(allowed)
            trials+=1
        self.assertEqual(trials,256)

    def test_disconnected_acyclic_expansion_is_admissible(self):
        g=Graph(("x","y"),(Edge("jump","x","y",constant_phase(Q(4))),))
        w=potential(g,Q(1,2))
        self.assertEqual(w,{"x":Q(1),"y":Q(8)})
        self.assertEqual(certify_path(g,w,Q(1,2),"x",("jump",))[1],Q(4))

    def test_branch_guard_separation_with_exact_error_replay(self):
        start=(Q(1,2),Q(1,2))
        perturbed_start=(start[0]+Q(1,1000),start[1])
        delta=Q(1,20000)
        receipts=guarded_trace(self.graph,self.policies,"idle",start,Q(1,1000),
                               (delta,)*12)
        endpoint,perturbed,edges=replay_perturbed(
            self.graph,self.policies,"idle",perturbed_start,(delta,Q(0)),12)
        self.assertEqual(edges,tuple(r.edge for r in receipts))
        self.assertEqual(endpoint,receipts[-1].next_vertex)
        self.assertLessEqual(sqdist(perturbed,receipts[-1].nominal_after),
                             receipts[-1].error_after**2)
        self.assertTrue(all(r.guard_score is None or r.guard_score!=0 for r in receipts))

    def test_reject_tie_and_straddling_guard(self):
        with self.assertRaises(UncertifiedGuard):
            guarded_trace(self.graph,self.policies,"idle",(Q(0),Q(1,2)),
                          Q(0),(Q(0),))
        with self.assertRaises(UncertifiedGuard):
            guarded_trace(self.graph,self.policies,"idle",(Q(1,1000),Q(1,2)),
                          Q(1,500),(Q(0),))
        with self.assertRaises(ValueError):
            Guard((Q(0),Q(0)),Q(0),"sample","hold")

    def test_state_dependent_branch_jump_counterexample(self):
        x=(Q(1,1000),Q(1,2)); y=(Q(-1,1000),Q(1,2))
        a=choose(self.graph,self.policies,"idle",x)
        b=choose(self.graph,self.policies,"idle",y)
        self.assertNotEqual(a.name,b.name)
        self.assertGreater(sqdist(a.phase.step(x),b.phase.step(y)),
                           100**2*sqdist(x,y))
        self.assertEqual(potential(self.graph,Q(3,4))["probe"],Q(2))

    def test_invalid_potential_rejected(self):
        with self.assertRaises(ValueError):
            certify_path(self.graph,{"idle":Q(1),"probe":Q(1),"audit":Q(1)},
                         Q(3,4),"idle",("sample",))
        with self.assertRaises(ValueError):
            potential(self.graph,Q(1))
        with self.assertRaises(ValueError):
            certify_path(self.graph,potential(self.graph,Q(3,4)),Q(3,4),
                         "probe",("sample",))

if __name__=="__main__":
    unittest.main()
