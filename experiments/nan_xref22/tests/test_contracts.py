"""Future-prefix admission, missing coverage and unchanged-arithmetic guards."""
import copy
from pathlib import Path
import unittest
from experiments.nan_xref22.common import ROOT,read,rows,sha,qualified
from experiments.nan_xref22.cache import coverage,validate_prefix
from experiments.nan_xref22.derive import transform,WORKER_RULES,AUDIT_RULES,check_coverage_source


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public=rows('experiments/nan_ctx21/fixtures/public.jsonl')

    def test_exact_explicit_derivation_preserves_all_other_arithmetic(self):
        provenance=read(ROOT/'DERIVATION.json')['files']
        for name,rules in [('worker.py',WORKER_RULES),('audit.py',AUDIT_RULES)]:
            base=Path('experiments/nan_xref19')/name;text=transform(base.read_text(encoding='utf-8'),rules)
            if name=='audit.py':
                text=text.replace('\n\nif __name__',check_coverage_source()+'\n\nif __name__')
            self.assertEqual((ROOT/name).read_text(encoding='utf-8'),text)
            record=next(r for r in provenance if r['derived']==name)
            self.assertEqual(record['parent_sha256'],sha(base));self.assertEqual(record['derived_sha256'],sha(ROOT/name))

    def test_partial_prefix_is_admitted_without_orphan_or_gap_rescue(self):
        audit=dict(status='passed',contexts_checked=2,registered_contexts=624,records=[dict(id=r['id']) for r in self.public[:2]])
        terminal=dict(completed_contexts=2)
        self.assertEqual(validate_prefix(self.public,audit,terminal),2)
        audit['records'][1]['id']=self.public[2]['id']
        with self.assertRaises(AssertionError):
            validate_prefix(self.public,audit,terminal)

    def test_every_missing_stratum_remains_explicit(self):
        meta=[self.public[0]|dict(loop=t) for t in (1,2)]
        result=coverage(self.public,meta)
        self.assertEqual(len(result),48);self.assertEqual(sum(g['registered_vectors'] for g in result),1248)
        self.assertEqual(sum(g['available_vectors'] for g in result),2);self.assertEqual(sum(g['missing_vectors'] for g in result),1246)
        live=[g for g in result if g['available_vectors']]
        self.assertEqual(len(live),2)
        self.assertTrue(all(g['available_roots']==1 and 2*g['available_roots']-g['available_vectors']==1 for g in live))

    def test_original_or_partial_fit_never_qualifies(self):
        cfg=read(ROOT/'config.json')
        audit=dict(status='passed',source_manifest_sha256=cfg['FIT16_seal'],cells=[dict(passes=2,status='passed',training=dict(optimizer_steps=16))])
        self.assertTrue(qualified(audit,2))
        audit['cells'][0]['training']['optimizer_steps']=2
        self.assertFalse(qualified(audit,2))
        audit['source_manifest_sha256']='35371bbb5323be88ae4eca4cb43677c2f583494b8a46a5f7b7aa89c2fc710139'
        with self.assertRaises(AssertionError):
            qualified(audit,2)


if __name__=='__main__':
    unittest.main()
