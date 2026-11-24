"""Check projection relaxation and compiled-potential engineering identities."""
import unittest
from ppt_gym.equality_search import path_moves
from ppt_gym.equality_residual import reverse_teacher, Features, residual_search
from ppt_gym.equality_compiled import Projection, Scorer, search, basis, avg, Q
from ppt_gym.tests.test_equality_preflight import fixture


class CompiledTests(unittest.TestCase):
    def test_projection_admissibility(self):
        c=fixture()
        d,_,_,_=reverse_teacher(c,cap=1024,depth=10)
        for k in (2,3):
            p=Projection(c,k)
            self.assertEqual(p(c.source),0)
            self.assertGreater(p.states,0)
            for w,exact in d.items():
                self.assertLessEqual(p(w),exact)
        p=Projection(c,2)
        f=Features(c)
        for w in d:
            self.assertEqual(p.values(w),[e[1] for e in f.entries(w)])

    def test_zero_compiled_residual_preserves_prior(self):
        c=fixture()
        for cap in (1,16,64,256):
            ref=residual_search(c,cap,"pair_greedy")
            for arm,coeff in (("pair_quarter",None),("quadratic",[0.0]*77),("linear",[0.0]*11)):
                row=search(c,cap,arm,coeff)
                for k in ("found","words","expansions","candidate_edges_enumerated"):
                    self.assertEqual(ref[k],row[k])
                if row["found"]:
                    path_moves(c,row["words"])

    def test_compilation_error_bound(self):
        c=fixture()
        coeff=[((i%5)-2)/10 for i in range(77)]
        score=Scorer(c,"quadratic",coeff)
        base=Scorer(c,"pair_quarter")
        f=Features(c)
        goal=avg([e[0] for e in f.entries(c.source)],True)
        d,_,_,_=reverse_teacher(c,cap=256,depth=6)
        for w in d:
            x=avg([e[0] for e in f.entries(w)],True)
            direct=max(-2,min(2,sum(a*(b-g) for a,b,g in zip(coeff,x,goal))))
            self.assertLessEqual(abs((score(w)-base(w))/Q-direct),1.5/Q+1e-9)
        self.assertEqual(score(c.source),0)


if __name__=="__main__":
    unittest.main()
