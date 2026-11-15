"""RAM-12R contracts. Set RAM08_ZIP, RAM12_ZIP, RAM11F_LABELS for full ZIP audit."""
from __future__ import annotations
import os
from pathlib import Path
import torch
import unittest
from state_cpu_pilot import (family,ALPHA,UPDATES,BATCH,SEED,RAM12ZIP,RAM12LABELS,
                             encode_rows,sha,m)

class RAM12Contracts(unittest.TestCase):
    def test_frozen_original_architecture_and_sha(self):
        self.assertEqual(UPDATES,150)
        self.assertEqual(BATCH,32)
        self.assertEqual(SEED,20261115)
        self.assertEqual(ALPHA,0.35)
        self.assertEqual(set(m.EXPECTED),{'loop32','loop64','flat64'})
    def test_source_grammar_families(self):
        for script in ['omega','constructor','rw [h]','exact h','induction xs',
                       'simp_all only [MathCan.F.f]','dsimp only [MathCan.F.f] at *']:
            self.assertIn(family(script),m.FAMILIES)
    def test_bounded_rank_protection(self):
        base=torch.tensor([1.,2.,2.8])
        change=ALPHA*torch.tanh(torch.tensor([20.,20.,-20.]))
        self.assertTrue((change.abs()<=ALPHA).all().item())
        self.assertEqual(int(base.argmax()),int((base+change).argmax()))
    @unittest.skipUnless(all(os.getenv(k) for k in ('RAM08_ZIP','RAM12_ZIP','RAM11F_LABELS')),'requires three pinned historical archives')
    def test_exact_native_to_original_prefix_join(self):
        a=Path(os.environ['RAM08_ZIP'])
        b=Path(os.environ['RAM12_ZIP'])
        c=Path(os.environ['RAM11F_LABELS'])
        self.assertEqual(sha(b.read_bytes()),RAM12ZIP)
        dataset,_=encode_rows(a,b,c)
        self.assertEqual(len(dataset),176)
        self.assertEqual(sum(sum(row['positive'] for row in rows) for rows in dataset.values()),230)
        self.assertEqual({split:sum(k[0]==split for k in dataset) for split in ('train','dev','holdback')},
                         {'train':105,'dev':24,'holdback':47})
        negatives=[r for rows in dataset.values() for r in rows if not r['positive']]
        self.assertEqual(len(negatives),1652)
        for rows in dataset.values():
            good={r['script'] for r in rows if r['positive']}
            self.assertTrue(good.isdisjoint({r['script'] for r in rows if not r['positive']}))

if __name__=='__main__': unittest.main()
