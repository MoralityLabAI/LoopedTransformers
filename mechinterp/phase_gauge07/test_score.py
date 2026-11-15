"""COE-07 exact quotient diagnostics and train/test isolation checks."""
import itertools
import unittest
import numpy as np
try:
    from .score import (VISITS, necklace, rotate, score_codes, codes_from_projection,
                        select_heads, evaluate_heads, audit_phases_equal, select_manifest,
                        SOURCE_BUNDLE_SHA256, STRATA)
except ImportError:
    from score import (VISITS, necklace, rotate, score_codes, codes_from_projection,
                       select_heads, evaluate_heads, audit_phases_equal, select_manifest,
                       SOURCE_BUNDLE_SHA256, STRATA)

class QuotientTests(unittest.TestCase):
    def test_cyclic_rotation_invariance_for_every_word(self):
        self.assertTrue(audit_phases_equal(list(range(256))))

    def test_count_cyclic_order_nesting_exhaustively(self):
        for a,b in itertools.combinations(range(256),2):
            if a.bit_count()!=b.bit_count():
                self.assertNotEqual(necklace(a),necklace(b))
            if necklace(a)!=necklace(b):
                self.assertNotEqual(a,b)

    def test_recoverable_order_and_lost_phase(self):
        words=list(range(9))
        result=score_codes({'fit':words})['pooled']
        self.assertEqual(result['denominator'],36)
        self.assertEqual(result['order_with_unknown_clock_loss'],result['ordered']-result['cyclic'])
        self.assertEqual(result['recoverable_order_only'],result['cyclic']-result['count'])

    def test_phase_blind_positive_control(self):
        # Distinct constant-weight necklaces have count collision and phase-robust order.
        words=[3,5,9,17,33,65,129,6,10]
        self.assertTrue(all(x.bit_count()==2 for x in words))
        m=score_codes({'demo':words})['pooled']
        self.assertEqual(m['count'],0)
        self.assertGreater(m['cyclic'],0)

    def test_shape_rejects_invalid_projection(self):
        with self.assertRaises(ValueError):
            codes_from_projection(np.zeros((2,9,7)),.5)

    def test_fit_only_selection_unaffected_by_heldout_state_values(self):
        rng=np.random.default_rng(8701)
        fit=rng.normal(size=(2,9,VISITS,5))
        test=rng.normal(size=(2,9,VISITS,5))
        chosen1=select_heads(fit,seed=12)
        chosen2=select_heads(fit,seed=12)
        self.assertEqual(chosen1,chosen2)
        a=evaluate_heads(fit,test,seed=12)
        b=evaluate_heads(fit,np.zeros_like(test),seed=12)
        for key in chosen1:
            self.assertEqual(a[key]['direction'],b[key]['direction'])
            self.assertEqual(a[key]['threshold'],b[key]['threshold'])
            self.assertEqual(a[key]['fit_score'],b[key]['fit_score'])
        self.assertNotEqual(a['cyclic']['heldout_codes'],b['cyclic']['heldout_codes'])

    def test_permutation_must_not_turn_cyclic_into_order(self):
        rng=np.random.default_rng(271)
        for _ in range(100):
            words=rng.integers(0,256,size=9).tolist()
            shift=int(rng.integers(0,8))
            rotated=[rotate(w,shift) for w in words]
            before=score_codes({'a':words})['pooled']
            after=score_codes({'a':rotated})['pooled']
            self.assertEqual(before['cyclic'],after['cyclic'])
            self.assertEqual(before['count'],after['count'])

    def test_source_bundle_digest_guard(self):
        with self.assertRaises(ValueError):
            select_manifest({'bundle_sha256':'wrong','items':[]})

    def test_deterministic_label_selection(self):
        rows=[]
        for g in STRATA:
            for d in range(1,10):
                q='.'*81
                a=str(d)*81
                rows.append(dict(stratum=g,item_id=f'{g}:{d}',target_idx=0,target_digit=d,question=q,answer=a))
        manifest=select_manifest({'bundle_sha256':SOURCE_BUNDLE_SHA256,'items':rows})
        self.assertEqual(list(manifest['strata']),list(STRATA))
        self.assertEqual(len(manifest['strata'][STRATA[0]]),9)

if __name__=='__main__':unittest.main()
