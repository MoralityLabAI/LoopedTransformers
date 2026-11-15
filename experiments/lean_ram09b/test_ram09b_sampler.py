"""Adversarial RAM-09B unit tests use fixture compiler observations only."""
from __future__ import annotations
import random
import unittest
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
RAM09=HERE if (HERE/"offline_graph.py").exists() else HERE.parent/"lean_ram09"
sys.path.insert(0,str(RAM09))
from collections import Counter
import torch
from offline_graph import ObservedClosureGraph,key
from test_offline_graph import case
from ram09b_balanced_cpu import eligible_examples,sample_pairs,state_metrics,ALPHA

class SamplingAudit(unittest.TestCase):
    def test_only_verified_failed_effects_are_negative(self):
        rows=[case('P','close',3,[]),case('P','fail-a',0,[]),case('P','fail-b',0,[]),
              case('P','progress-unknown',2,['UNOBSERVED'])]
        rows[0]['family']='rfl';rows[1]['family']='apply_local';rows[2]['family']='apply_local';rows[3]['family']='intro'
        graph=ObservedClosureGraph(rows)
        ex=eligible_examples(graph,torch.tensor([1.,2.,3.,100.]))
        self.assertEqual(len(ex),1)
        self.assertEqual(ex[0]['pos'],0)
        self.assertEqual(ex[0]['hard_failed_by_family'],{'apply_local':2})
        p,n,d=sample_pairs(ex,random.Random(1),50)
        self.assertTrue(all(graph.rows[i]['label']==0 for i in n.tolist()))
        self.assertEqual(set(n.tolist()),{2})
        self.assertEqual(set(p.tolist()),{0})
        self.assertNotIn(3,n.tolist())

    def test_equal_positive_family_sampling(self):
        e=[dict(pos=i,pos_family='omega',hard_failed_by_family={'apply_local':i+10}) for i in range(9)]
        e += [dict(pos=9,pos_family='exact_typed',hard_failed_by_family={'intro':21})]
        p,n,d=sample_pairs(e,random.Random(20261115),600)
        fam=Counter(e[i]['pos_family'] for i in p.tolist())
        self.assertGreater(fam['omega'],260)
        self.assertLess(fam['omega'],340)
        self.assertEqual(fam['omega']+fam['exact_typed'],600)

    def test_margin_protection(self):
        rows=[case('A','finish',3,[]),case('A','bad',0,[])]
        values=state_metrics(ObservedClosureGraph(rows),torch.tensor([1.0,0.0]),torch.tensor([.65,.35]))
        self.assertEqual(values['margin_certificate']['strict_high_margin_states'],1)
        self.assertEqual(values['margin_certificate']['preserved'],1)

if __name__=='__main__':unittest.main()