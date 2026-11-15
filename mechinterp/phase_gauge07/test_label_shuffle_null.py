"""COE-08 independent exact rational null checks and pinned compact fixtures."""
import itertools
import unittest
from fractions import Fraction

try:
    from .label_shuffle_null import (
        eligible_count, analyze, exact_refinement_null_variance, expected_distinctions_under_shuffle,
        expected_refinement_gain, observed_refinement_gain, permutation_null,
    )
except ImportError:
    from label_shuffle_null import (
        eligible_count, analyze, exact_refinement_null_variance, expected_distinctions_under_shuffle,
        expected_refinement_gain, observed_refinement_gain, permutation_null,
    )

# Pinned S4 original LoopedTransformers 64 rows, from benchmark_bundle.json
# Git blob 01ec63b4b2a4ae5399f783c6036a7fc018825a87 and
# trm_results.json Git blob 3c35daeb71da32f5c9ebbf03da539b8ee718b0ee.
LABELS = [1,8,6,2,7,5,9,6,1,9,9,4,7,5,1,1,9,2,8,8,9,9,1,2,7,1,6,5,5,6,3,4,9,2,8,6,8,1,2,2,1,7,3,8,5,4,8,5,6,1,8,6,1,5,8,8,4,9,4,2,7,7,2,5]
W8 = [255,254,0,0,255,248,248,192,253,255,255,255,239,255,0,246,255,11,255,255,248,255,255,0,252,79,255,0,0,0,255,248,225,255,191,254,255,128,129,255,16,36,255,255,254,0,255,254,255,255,0,0,255,255,255,0,0,255,91,255,223,115,199,255]
W16 = [65535,65534,0,0,65535,65528,65528,65472,65533,65535,65535,65535,27375,65535,0,49142,65535,5643,65535,65535,65528,65535,65535,0,65532,11343,65535,0,0,65024,65535,65528,65505,65535,65471,65534,65535,65408,65409,65535,64528,292,65535,65535,65534,0,65535,65534,65535,65535,0,0,65535,65535,65535,0,0,65535,91,57343,49119,64627,65479,65535]

class TestExactNull(unittest.TestCase):
    def test_pinned_label_hist_and_eligible(self):
        self.assertEqual(len(LABELS),64)
        self.assertEqual(eligible_count(LABELS),1795)
        self.assertEqual(sum(LABELS.count(x) for x in range(1,10)),64)

    def test_heldout_s4_eight_and_sixteen_steps(self):
        self.assertEqual(observed_refinement_gain(W8,LABELS,8),20)
        self.assertEqual(observed_refinement_gain(W16,LABELS,16),6)
        a=expected_refinement_gain(W8,LABELS,8)
        b=expected_refinement_gain(W16,LABELS,16)
        self.assertAlmostEqual(float(a),18.697916666666668)
        self.assertAlmostEqual(float(b),7.123015873015873)
        self.assertEqual(a, Fraction(1795*21,2016))
        self.assertEqual(b, Fraction(1795*8,2016))

    def test_quotient_refinement_and_zero_gain(self):
        self.assertEqual(expected_refinement_gain([0,1,3,7],[2,2,1,1],3,
                                                 'ordered','ordered'),0)
        self.assertEqual(observed_refinement_gain([0,7,0,7],[0,1,0,1],3),0)

    def test_hand_checkable_three_point_null(self):
        words=[1,2,3]
        labels=[0,1,1]
        # Ordered distinguishes every trace. 2/3 of pairs differ in labels.
        self.assertEqual(eligible_count(labels),2)
        self.assertEqual(expected_distinctions_under_shuffle(words,labels,3,'ordered'),2)

    def test_independent_permutation_enumeration_small(self):
        # An exhaustive independent permutation oracle for every three-state
        # binary labelled instance of width 3.
        for words in itertools.product(range(8),repeat=3):
            for labels in itertools.product(range(2),repeat=3):
                exact=expected_refinement_gain(list(words),list(labels),3)
                allcases=sum(observed_refinement_gain([words[i] for i in perm],
                            list(labels),3) for perm in itertools.permutations(range(3)))
                self.assertEqual(exact,Fraction(allcases,6))

    def test_shuffle_is_reproducible_and_within_exact_bound(self):
        a=permutation_null(W8,LABELS,8,repetitions=1500,seed=71)
        b=permutation_null(W8,LABELS,8,repetitions=1500,seed=71)
        self.assertEqual(a,b)
        self.assertEqual(a['observed_extra'],20)
        self.assertLess(abs(a['shuffle_mean']-a['exact_null_float']),0.5)
        self.assertTrue(0<a['upper_tail_p_plus_one']<=1)

    def test_exact_variance_at_pinned_s4_cells(self):
        d8=exact_refinement_null_variance(W8,LABELS,8)
        self.assertEqual(d8['edges'],21)
        self.assertEqual(d8['adjacent_edge_pairs'],79)
        self.assertEqual(d8['disjoint_edge_pairs'],131)
        self.assertEqual(d8['mean'],Fraction(1795,96))
        self.assertEqual(d8['variance'],Fraction(11285015,5809152))
        d16=exact_refinement_null_variance(W16,LABELS,16)
        self.assertEqual(d16['edges'],8)
        self.assertEqual(d16['variance'],Fraction(46069057,60043032))

    def test_exact_second_moment_against_all_permutations(self):
        # Independent enumeration: every assignment of the fixed label multiset
        # to a chosen Hamming-code graph, with no probabilistic approximation.
        from random import Random
        g=Random(840)
        for _ in range(50):
            w=[g.randrange(16) for _ in range(4)]
            labels=[g.randrange(3) for _ in range(4)]
            actual=[observed_refinement_gain(w,list(p),4)
                    for p in set(itertools.permutations(labels))]
            mean=Fraction(sum(actual),len(actual))
            variance=Fraction(sum(x*x for x in actual),len(actual))-mean*mean
            formula=exact_refinement_null_variance(w,labels,4)
            self.assertEqual(mean,formula['mean'])
            self.assertEqual(variance,formula['variance'])

    def test_analyze_on_compact_pinned_shaped_receipt(self):
        # Whole pipeline from a self-contained receipt with exactly the same
        # native S4 words, while the full original JSON is checked by hosted CI.
        import hashlib
        try:
            from .score import SOURCE_BUNDLE_SHA256, MODEL_SHA256
        except ImportError:
            from score import SOURCE_BUNDLE_SHA256, MODEL_SHA256
        bundle={'bundle_sha256':SOURCE_BUNDLE_SHA256,
                'items':[{'stratum':'S4_stalled','target_digit':y} for y in LABELS]}
        trace={'checkpoint_sha256':MODEL_SHA256, 'bundle_sha256':SOURCE_BUNDLE_SHA256,
               'steps':16,'items_target_correct_by_step':
               [[(w>>t)&1 for w in W16] for t in range(16)]}
        receipt=analyze(bundle,trace,repetitions=50,seed=12)
        self.assertEqual(receipt['schema'],'COE-08-null-v1')
        self.assertEqual([r['observed_extra'] for r in receipt['windows']], [0,20,6])
        self.assertEqual(receipt['windows'][1]['exact_null_expectation'],'1795/96')

    def test_tamper_detection(self):
        with self.assertRaises(ValueError):
            expected_refinement_gain([1,2],[0],3)
        with self.assertRaises(ValueError):
            expected_refinement_gain([1,2],[0,1],3,'count','ordered')

if __name__=='__main__':unittest.main()
