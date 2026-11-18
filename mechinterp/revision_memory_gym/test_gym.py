"""Standard-library preflight for LRM-REV-01."""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path
from itertools import product
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gym import (PRIORS,MATCH,make_episode,oracle,full_posterior,mass,normalize)


class RevisionGymTests(unittest.TestCase):
    def test_early_and_delayed_public_interface(self):
        ids=set()
        for j in (1,3,6):
            for i in range(50):
                e=make_episode(i,j)
                self.assertNotIn(e.source_id,ids)
                ids.add(e.source_id)
                early=e.public_encoder_input(early=True)
                late=e.public_encoder_input(early=False)
                self.assertEqual(len(early["observations"]),12-j)
                self.assertIsNone(late["schedule"])
                self.assertEqual(len(late["observations"]),12)
                self.assertEqual(early["schedule"],list(e.correction_slots))
                self.assertEqual(e.reference(j),e.full_reference())
                self.assertEqual(sum(e.reference(0)),Q(1))
                self.assertNotIn("true_regime",repr(early))
                self.assertEqual(e.verified_old_total(e.old),
                                 sum(e.old[k] for k in e.correction_slots))
                with self.assertRaises(ValueError):
                    e.verified_old_total(tuple(1-x for x in e.old))
                self.assertEqual(len(e.correction_messages(j)["old_disclosures"]),j)
                count_early=sum(e.old[k] for k in range(12) if k not in e.correction_slots)
                for idx,val in enumerate(e.replacements):
                    count_early+=val
                    self.assertEqual(full_posterior(count_early,12-j+idx+1),
                                     full_posterior(count_early,12-j+idx+1))

    def test_independent_raw_bit_oracle(self):
        # Literal old-vector integration, independent of binomial coefficient
        # calculation in the gym's posterior function.
        for n,j in ((4,1),(4,3)):
            slots=tuple(range(j))
            for old in product((0,1),repeat=n):
                for new in product((0,1),repeat=j):
                    for r in range(j+1):
                        weights=[]
                        for pi,t in zip(PRIORS,MATCH):
                            w=Q(0)
                            for candidate in product((0,1),repeat=n):
                                if sum(candidate)!=sum(old):
                                    continue
                                if candidate[:r]!=old[:r]:
                                    continue
                                p=pi
                                for i,val in enumerate(candidate):
                                    p*=Q(1,2) if i in slots else (t if val else 1-t)
                                for val in new:
                                    p*=t if val else 1-t
                                w+=p
                            weights.append(w)
                        expected=normalize(tuple(weights))
                        self.assertEqual(expected,oracle(sum(old),sum(new),
                                     sum(old[:r]),r,j,n))

    def test_bayes_old_total_repair(self):
        for j in (1,3,6):
            for i in range(20):
                e=make_episode(i,j)
                m=sum(e.old)
                v=e.verified_old_total(e.old)
                q=sum(e.replacements)
                self.assertEqual(full_posterior(m-v+q,12),e.full_reference())

    def test_collision_and_rejection(self):
        self.assertEqual(full_posterior(0,4),full_posterior(0,4))
        self.assertNotEqual(full_posterior(0,4),full_posterior(1,4))
        for call in (lambda:make_episode(0,6,6),
                     lambda:make_episode(True,3),
                     lambda:oracle(4,1,0,0,0,4),
                     lambda:oracle(0,1,1,1,3,4)):
            with self.assertRaises(ValueError):
                call()


if __name__=="__main__":
    unittest.main()
