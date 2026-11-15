"""RAM-10 source split, candidate cap, contextual tactic gate and isolation tests."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from ram10_core import inventory,fair_candidates,gate,source_order_roots,verify_checkout,_lemma_bounds

HERE=Path(__file__).resolve().parent

class SourceAndSplitTests(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((HERE/'config.json').read_text())
    def test_exact_frozen_ancestry_and_splits(self):
        cfg=self.cfg
        self.assertEqual(cfg['schema'],'RAM10-v1')
        self.assertEqual(cfg['source_mathcan'],'7515eecf51a60e7f8cfcd0801fddf8a3cf039e89')
        self.assertEqual(cfg['registration_commit'],'00c6ba633d5e836a584efa58744438806e4282f7')
        self.assertEqual(tuple(map(len,(cfg['modules']['train'],cfg['modules']['dev'],cfg['modules']['test']))),(8,2,4))
        sets=[set(x) for x in cfg['modules'].values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertFalse(set().union(*sets)&set(cfg['excluded_pef_modules']))
        self.assertEqual(len(cfg['candidate_families']),20)
        self.assertEqual(cfg['training']['seeds'],[17,29,43])
        self.assertEqual(cfg['eval']['primary']['seconds'],30)
        self.assertEqual(cfg['eval']['secondary']['seconds'],60)
    def test_original_source_declarations_without_reference_proofs(self):
        src='''import Init
namespace Foo
-- theorem fraudulent : True := by trivial
theorem alpha : True := by
  trivial
lemma beta : True := by
  trivial
end Foo
'''
        self.assertEqual(inventory(src,2),[('theorem','alpha'),('lemma','beta')])
        lines=src.splitlines();self.assertEqual(_lemma_bounds(lines,'beta',lambda l,n:None),(5,5,7))
    def test_fair_all_family_actions_and_budget(self):
        names=self.cfg['candidate_families']
        candidate=[(f,f'{f}-{i}') for f in names for i in range(12)]
        fair=fair_candidates(candidate,names,limit=80,first=4)
        self.assertEqual(len(fair),80)
        self.assertTrue(set((f,f'{f}-0') for f in names).issubset(set(fair)))
        self.assertTrue(all(sum(f==name for f,s in fair)>=4 for name in names))
        self.assertEqual(fair,fair_candidates(candidate,names,80,4))
    def test_compiler_only_gate(self):
        row=lambda file,name,fam,label,part='train':dict(partition=part,source_file=file,theorem=name,
          family=fam,label=label,state_sha256='s',script=fam)
        good=[row('MathCan/A.lean','a','exact_typed',3),row('MathCan/B.lean','b','exact_local',1),
              row('MathCan/C.lean','c','rw_local',2),row('MathCan/D.lean','d','rcases_local',1)]
        self.assertTrue(gate(good,self.cfg)['admitted'])
        self.assertFalse(gate(good[:-1],self.cfg)['admitted'])
        self.assertFalse(gate([row('MathCan/A.lean','a','exact_typed',3),row('MathCan/A.lean','a','exact_local',1)] + good[2:],self.cfg)['admitted'])
        self.assertFalse(gate([dict(x,partition='dev') if x['family']=='rw_local' else x for x in good],self.cfg)['admitted'])
    def test_untrusted_censored_action_never_positive(self):
        self.assertFalse(gate([dict(partition='train',source_file='X',theorem='x',family=f,label=0,
           state_sha256='s',script='a') for f in ('exact_typed','rw_local','rcases_local')],self.cfg)['admitted'])

if __name__=='__main__':unittest.main()
