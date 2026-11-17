from __future__ import annotations
import itertools
import unittest
import numpy as np
try:
    from .label_alignment import (MODES,rot,rev,distance,validate_codes,nearest_label_costs,
        score_permutation,conditional_shuffle_null,score_fit_heldout,projected_words,
        separation_invariant_under_label_permutation)
except ImportError:
    from label_alignment import (MODES,rot,rev,distance,validate_codes,nearest_label_costs,
        score_permutation,conditional_shuffle_null,score_fit_heldout,projected_words,
        separation_invariant_under_label_permutation)

class LabelAlignmentTests(unittest.TestCase):
    def test_cyclic_and_dihedral_invariance(self):
        for T in (4,8):
            for x in range(1<<T):
                for t in range(T):
                    self.assertEqual(distance(x,rot(x,t,T),'cyclic',T),0)
                    self.assertEqual(distance(x,rot(rev(x,T),t,T),'dihedral',T),0)
                self.assertEqual(distance(x,rev(x,T),'dihedral',T),0)

    def test_exact_nested_distances(self):
        for a,b in itertools.product(range(16),repeat=2):
            self.assertLessEqual(distance(a,b,'dihedral',4),distance(a,b,'cyclic',4))
            self.assertLessEqual(distance(a,b,'cyclic',4),distance(a,b,'ordered',4))
            self.assertLessEqual(distance(a,b,'count',4),distance(a,b,'cyclic',4))
            self.assertGreaterEqual(distance(a,b,'final',4),0)

    def test_separation_invariant_but_alignment_can_fail(self):
        words=[0,1,2]
        self.assertEqual(sum(words[i]!=words[j] for i in range(3) for j in range(i+1,3)),3)
        for p in itertools.permutations(range(3)):
            self.assertTrue(separation_invariant_under_label_permutation(words,list(p)))
        fit=np.asarray([[0,1,2]],dtype=np.int64)
        # Heldout word 0 belongs to label 1, word 1 label 2, word 2 label 0.
        test=np.asarray([[2,0,1]],dtype=np.int64)
        costs=nearest_label_costs(fit,test,'ordered',2)
        correct,decoded=score_permutation(costs,(0,1,2))
        self.assertEqual(correct,0)
        self.assertEqual(decoded,[2,0,1])

    def test_perfect_prototype_alignment(self):
        codes=np.asarray([[1,2,4,8],[1,2,4,8]],dtype=np.int64)
        for mode in ('ordered','cyclic','dihedral'):
            costs=nearest_label_costs(codes,codes,mode,4)
            if mode == 'ordered':
                self.assertEqual(score_permutation(costs,tuple(range(4)))[0],8)
            self.assertEqual(len(costs),8)

    def test_null_determinism_and_no_test_label_leak(self):
        fit=np.asarray([[1,2,4,8,3],[1,2,4,8,3]],dtype=np.int64)
        test=fit.copy()
        costs=nearest_label_costs(fit,test,'ordered',4)
        a=conditional_shuffle_null(costs,permutations=99,seed=7)
        b=conditional_shuffle_null(costs,permutations=99,seed=7)
        self.assertEqual(a,b)
        self.assertEqual(a['total'],10)
        self.assertEqual(a['observed_correct'],10)
        self.assertLessEqual(a['null_mean'],10)

    def test_projected_words_numpy_frozen(self):
        state=np.zeros((2,3,4,2),dtype=float)
        state[:,:,:,0]=np.asarray([0,1,0,1])[None,None,:]
        code=projected_words(state,np.asarray([1.,0.]),.5)
        self.assertEqual(code.tolist(),[[10,10,10],[10,10,10]])
        with self.assertRaises(ValueError):
            projected_words(state,np.array([1.,2.,3.]),0.)

    def test_input_validation(self):
        for x in (np.array([1,2]),np.array([[-1,0]],dtype=int),np.array([[1.5,2.5]])):
            with self.assertRaises(ValueError):validate_codes(x,np.array([[1,2]]))
        with self.assertRaises(ValueError):score_permutation([[0,1],[1,0]],(0,0))
        with self.assertRaises(ValueError):distance(17,1,'ordered',4)

    def test_group_diagnostics_and_threshold(self):
        fit=np.asarray([[0,1,2],[0,1,2]],dtype=np.int64)
        tst=np.asarray([[0,1,2],[0,1,2]],dtype=np.int64)
        r=score_fit_heldout(fit,tst,width=2,repetitions=15,seed=4)
        self.assertEqual(r['modes']['ordered']['correct_per_group'],[3,3])
        self.assertEqual(r['modes']['count']['total'],6)

if __name__=='__main__':unittest.main()
