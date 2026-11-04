"""Dependency-free proof audit, plus optional torch invariance/gradient tests."""
import copy
import random
import unittest
from proof_data import (generate_one, check_term, check_trace, legal, required,
                        structure_id, trace_from_term, lean_theorem)


def sample():
    rng=random.Random(901)
    while True:
        t=generate_one(rng)
        if t is not None: return t


class ProofTests(unittest.TestCase):
    def test_generated_terms(self):
        rng=random.Random(21); count=0
        while count<100:
            t=generate_one(rng, count%2==0)
            if t is None: continue
            self.assertEqual(check_term(t,t['term']),t['goal'])
            check_trace(t,t['trace'])
            self.assertEqual(set(trace_from_term(t['term'])),required(t))
            self.assertEqual(len(t['trace']),t['min_steps']); count+=1

    def test_missing_hypothesis_rejected(self):
        t=sample(); p=next(p for p in range(t['n']) if p not in t['facts'])
        with self.assertRaises(ValueError): check_term(t,['hyp',p])

    def test_wrong_argument_rejected(self):
        t=sample(); i=0; ps,_=t['rules'][i]
        wrong=next(p for p in t['facts'] if p!=ps[0])
        args=[['hyp',wrong]]+[ ['hyp',t['facts'][0]] for _ in ps[1:] ]
        with self.assertRaises(ValueError): check_term(t,['app',i,args])

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError): check_term(sample(),['app',999,[]])

    def test_incomplete_trace_rejected(self):
        with self.assertRaises(ValueError): check_trace(sample(),[])

    def test_repeated_rule_rejected(self):
        t=sample(); trace=t['trace']
        with self.assertRaises(ValueError): check_trace(t,[trace[0],trace[0]])

    def test_lean_export_has_no_escape_hatches(self):
        s=lean_theorem(sample(),'smoke')
        for forbidden in ['sorry','admit','axiom','native_decide']:
            self.assertNotIn(forbidden,s)
        self.assertIn('have s0',s)

    def test_alpha_invariant_fingerprint(self):
        t=sample(); u=copy.deepcopy(t); n=t['n']; rename=lambda p:n-1-p
        u['facts']=[rename(p) for p in t['facts']]
        u['goal']=rename(t['goal'])
        u['rules']=[[[rename(p) for p in ps],rename(h)] for ps,h in reversed(t['rules'])]
        self.assertEqual(structure_id(t),structure_id(u))

    def test_neural_parameter_gradients_and_equivariance(self):
        try: import torch
        except ImportError: self.skipTest('optional torch unavailable')
        from train import ProofPolicy, encode, stack, objective
        torch.set_num_threads(1); torch.manual_seed(17)
        t=sample(); b=stack([encode(t,set(t['facts']))])
        m=ProofPolicy(1); y=torch.zeros(1,24); y[0,t['trace']]=1
        objective(m(b),b,y).backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        u=copy.deepcopy(t); n=t['n']; rename=lambda p:n-1-p
        u['facts']=[rename(p) for p in t['facts']]; u['goal']=rename(t['goal'])
        u['rules']=[[[rename(p) for p in ps],rename(h)] for ps,h in reversed(t['rules'])]
        with torch.no_grad():
            m=ProofPolicy(4)
            x=m(b)[0,:len(t['rules'])]
            z=m(stack([encode(u,set(u['facts']))]))[0,:len(u['rules'])].flip(0)
            self.assertTrue(torch.allclose(x,z,atol=2e-6))


if __name__=='__main__': unittest.main()
