"""Finite-dimensional ranking guard proof by exhaustive representative cases."""
import math
import unittest

def guarded(base, corrections, alpha):
    return [s + alpha*math.tanh(v) for s,v in zip(base,corrections)]

class RankGuardTest(unittest.TestCase):
    def test_every_score_correction_is_bounded(self):
        for alpha in (0.,.05,.35,1.0):
            base=[-3.,0.,4.,16.]
            perturbed=guarded(base,[-1000.,-1.,1.,1000.],alpha)
            for original,new in zip(base,perturbed):
                self.assertLessEqual(abs(new-original),alpha+1e-12)
    def test_margin_over_twice_alpha_preserves_best_action(self):
        alpha=.35
        base=[0.,.7+1e-4,.1,-3.]
        for left in (-100.,0.,100.):
            for right in (-100.,0.,100.):
                values=guarded(base,[right,left,right,left],alpha)
                self.assertEqual(max(range(len(values)),key=values.__getitem__),1)
    def test_weak_margin_can_change_winner(self):
        alpha=.35
        base=[0.,.1]
        altered=guarded(base,[100.,-100.],alpha)
        self.assertEqual(max(range(len(base)),key=base.__getitem__),1)
        self.assertEqual(max(range(len(altered)),key=altered.__getitem__),0)

if __name__=='__main__': unittest.main()
