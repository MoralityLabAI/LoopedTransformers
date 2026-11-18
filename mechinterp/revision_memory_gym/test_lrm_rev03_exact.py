"""Independent finite and exhaustive tests for LRM-REV-03."""
import random
import unittest
from fractions import Fraction as F
from itertools import combinations

from lrm_rev03_exact import (
    N, J, MATCH, PRIORS, HALF, binom, brute_posterior, count_fiber_bound,
    count_posterior, count_posterior_from_public, exact_population_regret,
    full_posterior, initial_count_posterior, named_posterior,
    partial_named_posterior,
)


def brute_partial_named(old, known, replacements, n=12, j=6):
    """Enumerate every possible remaining correction schedule directly."""
    available=[i for i in range(n) if i not in set(known)]
    suffix=[i for i in range(n) if i > (known[-1] if known else -1)]
    h=j-len(known)
    weights=[]
    for prior,p in zip(PRIORS,MATCH):
        total=F(0)
        for rest in combinations(suffix,h):
            unknown=set(rest)
            probability=F(1)
            for i in available:
                probability*=HALF if i in unknown else (p if old[i] else (1-p))
            total+=probability
        y=sum(replacements)
        weights.append(prior*total*p**y*(1-p)**(len(known)-y))
    denominator=sum(weights,F(0))
    return tuple(w/denominator for w in weights)


class FiniteContinuationTests(unittest.TestCase):
    def test_full_six_edit_continuation_distinguishes_all_4096_inputs(self):
        result=count_fiber_bound()
        self.assertEqual(result['distinct_continuations'],4096)
        self.assertEqual(result['schedules'],924)
        self.assertEqual(result['largest_count_fiber'],924)
        self.assertEqual(result['extra_fixed_width_bits'],10)
        self.assertEqual(result['unconstrained_fixed_width_bits'],12)
        self.assertEqual(result['fiber_sizes'],[1,12,66,220,495,792,924,792,495,220,66,12,1])

    def test_small_finite_cases(self):
        for n in range(2,7):
            for j in range(1,n):
                info=count_fiber_bound(n,j)
                self.assertEqual(info['distinct_continuations'],1<<n)
                self.assertEqual(info['largest_count_fiber'],max(__import__('math').comb(n,k) for k in range(n+1)))

    def test_invalid_bounds_are_rejected(self):
        for n,j in [(0,0),(12,0),(12,12),(13,6)]:
            with self.assertRaises(ValueError): count_fiber_bound(n,j)


class BayesianOracleTests(unittest.TestCase):
    def test_exact_conditional_on_arbitrary_disclosed_old_positions(self):
        rng=random.Random(203012)
        for _ in range(32):
            old=tuple(rng.randrange(2) for _ in range(N))
            s=tuple(sorted(rng.sample(range(N),J)))
            y=tuple(rng.randrange(2) for _ in range(J))
            t=tuple(sorted(rng.sample(range(N),rng.randrange(J+1))))
            d={i:old[i] for i in t}
            observed=count_posterior_from_public(sum(old),s,y,d)
            brute=brute_posterior(sum(old),s,sum(y),d)
            self.assertEqual(observed,brute)

    def test_corrected_old_disclosure_restores_named_bayes_exactly(self):
        rng=random.Random(171)
        for _ in range(48):
            x=tuple(rng.randrange(2) for i in range(12))
            s=tuple(sorted(rng.sample(range(12),6)))
            y=tuple(rng.randrange(2) for i in range(6))
            d={i:x[i] for i in s}
            self.assertEqual(count_posterior_from_public(sum(x),s,y,d),named_posterior(x,s,y))

    def test_partial_named_teacher_matches_direct_schedule_enumeration(self):
        rng=random.Random(319)
        for _ in range(6):
            old=tuple(rng.randrange(2) for _ in range(N))
            s=tuple(sorted(rng.sample(range(N),J)))
            y=tuple(rng.randrange(2) for _ in range(J))
            for r in range(7):
                self.assertEqual(partial_named_posterior(old,s[:r],y[:r]),
                                 brute_partial_named(old,s[:r],y[:r]))

    def test_ordered_replacement_horizon_is_enforced(self):
        with self.assertRaises(ValueError): partial_named_posterior((0,)*12,(8,), (1,))
        with self.assertRaises(ValueError): partial_named_posterior((0,)*12,(4,2), (1,1))

    def test_first_sorted_correction_is_informative_about_future_schedule(self):
        # First revealed correction is slot 4. Future corrected slots must be >4.
        old=(1,1,1,1,1,1,0,0,0,0,0,0)
        ordered=partial_named_posterior(old,(4,),(1,))
        self.assertAlmostEqual(float(ordered[0]),0.463023788,places=8)
        self.assertAlmostEqual(float(ordered[1]),0.536249636,places=8)
        self.assertAlmostEqual(float(ordered[2]),0.000726576,places=8)
        self.assertEqual(ordered,brute_partial_named(old,(4,),(1,)))

    def test_initial_and_final_named_targets(self):
        rng=random.Random(1088)
        for _ in range(32):
            x=tuple(rng.randrange(2) for _ in range(12))
            s=tuple(sorted(rng.sample(range(12),6)))
            y=tuple(rng.randrange(2) for _ in range(6))
            self.assertEqual(partial_named_posterior(x,(),()),initial_count_posterior(sum(x)))
            self.assertEqual(partial_named_posterior(x,s,y),named_posterior(x,s,y))

    def test_impossible_disclosure_and_private_fields_fail(self):
        with self.assertRaises(ValueError): count_posterior(0,0,6,6,0,0)
        with self.assertRaises(ValueError): count_posterior_from_public(5,tuple(range(6)),(1,)*6,{12:1})
        with self.assertRaises(ValueError): count_posterior_from_public(5,tuple(range(6)),(1,)*6,{0:2})
        with self.assertRaises(ValueError): count_posterior_from_public(5,tuple(range(6)),(1,)*6,{0:True})
        with self.assertRaises(ValueError): partial_named_posterior((1,)*12,(0,0),(1,1))
        with self.assertRaises(ValueError): named_posterior((0,)*12,tuple(range(6)),(0,)*5)

    def test_full_posterior_is_injective_by_accepted_count(self):
        self.assertEqual(len(set(full_posterior(i) for i in range(13))),13)


class PopulationRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corrected=[exact_population_regret(r,'corrected') for r in range(7)]
        cls.random=[exact_population_regret(r,'random') for r in range(7)]

    def test_compressed_population_risk_and_complete_repair(self):
        self.assertEqual(self.corrected[0],self.random[0])
        self.assertEqual(self.corrected[-1],F(0))
        self.assertAlmostEqual(float(self.random[0]),0.06446269084478505,places=14)
        self.assertAlmostEqual(float(self.random[-1]),0.02440156846261376,places=14)

    def test_stagewise_information_gain_and_crossover(self):
        for curve in (self.corrected,self.random):
            self.assertTrue(all(curve[i] > curve[i+1] for i in range(6)))
        self.assertGreater(self.corrected[1],self.random[1])
        self.assertGreater(self.corrected[2],self.random[2])
        self.assertLess(self.corrected[3],self.random[3])
        self.assertEqual(exact_population_regret(6,'corrected'),F(0))

    def test_invalid_stage_policy(self):
        for r,policy in [(-1,'random'),(7,'corrected'),(6,'hidden')]:
            with self.assertRaises(ValueError): exact_population_regret(r,policy)


if __name__=='__main__':
    unittest.main()
