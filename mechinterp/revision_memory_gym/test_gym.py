"""Standard-library preflight for LRM-REV-01."""
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path
from itertools import product
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gym import (PRIORS,MATCH,Episode,make_episode,oracle,full_posterior,mass,normalize)


def independent_observed_bits_posterior(bits):
    """Multiply named informative-report likelihoods without count scoring."""
    weights=[]
    for prior,t in zip(PRIORS,MATCH):
        weight=prior
        for bit in bits:
            weight*=t if bit else 1-t
        weights.append(weight)
    total=sum(weights,Q(0))
    return tuple(weight/total for weight in weights)


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
                self.assertEqual((early["n"],early["j"]),(12,j))
                self.assertEqual((late["n"],late["j"]),(12,j))
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
                observed=[event["bit"] for event in early["observations"]]
                self.assertEqual(full_posterior(sum(observed),len(observed)),
                                 independent_observed_bits_posterior(observed))
                for event in e.correction_messages()["accepted"]:
                    observed.append(event["bit"])
                    self.assertEqual(full_posterior(sum(observed),len(observed)),
                                     independent_observed_bits_posterior(observed))
                self.assertEqual(independent_observed_bits_posterior(observed),
                                 e.full_reference())

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
        a=Episode("public-collision-a",0,(1,0,0,0),(0,),(0,))
        b=Episode("public-collision-b",0,(0,1,0,0),(0,),(0,))
        self.assertEqual(sum(a.old),sum(b.old))
        self.assertEqual(a.correction_messages(),b.correction_messages())
        self.assertEqual(a.reduced_reference(0),b.reduced_reference(0))
        self.assertNotEqual(a.full_reference(),b.full_reference())
        for episode in (a,b):
            old_messages=episode.public_encoder_input(early=False)["observations"]
            public_old={event["position"]:event["bit"] for event in old_messages}
            corrections=episode.correction_messages()
            public_valid=[bit for position,bit in public_old.items()
                          if position not in corrections["schedule"]]
            public_valid.extend(event["bit"] for event in corrections["accepted"])
            self.assertEqual(independent_observed_bits_posterior(public_valid),
                             episode.full_reference())
            self.assertNotEqual(episode.reduced_reference(0),episode.full_reference())
            self.assertEqual(episode.reduced_reference(1),episode.full_reference())
            self.assertEqual(episode.reference(0),episode.reduced_reference(0))
        for call in (lambda:make_episode(0,6,6),
                     lambda:make_episode(True,3),
                     lambda:oracle(4,1,0,0,0,4),
                     lambda:oracle(0,1,1,1,3,4)):
            with self.assertRaises(ValueError):
                call()


if __name__=="__main__":
    unittest.main()

