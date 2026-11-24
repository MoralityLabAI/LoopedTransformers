import json
import unittest
from mechinterp.nanbeige_bpd_gym.campaign_index import extract, build


class ContinuationPresentation(unittest.TestCase):
    def test_pending_fit_is_missing_not_zero_or_complete(self):
        rows = extract('IO18', {'backends': {'nf4': {'status': 'pending', 'cells': []}}}, {})
        self.assertEqual(len(rows[0]['cells']), 2)
        for cell in rows[0]['cells']:
            self.assertIsNone(cell['optimizer_steps'])
            self.assertEqual(cell['status'], 'awaiting_published_cell_audit')

    def test_amendment_preserves_historical_pending_and_marks_unrun(self):
        amendment = {'BF16_status': 'unrun_registered_supervision_amendment', 'XREF17_status': 'unrun_registered_supervision_amendment', 'supervision_commit': 'registered-before-action'}
        data = {'backends': {'bf16': {'status': 'pending', 'cells': []}}}
        row = extract('FIT16', data, amendment)[0]
        self.assertEqual(row['published_status'], 'pending'); self.assertEqual(row['status'], amendment['BF16_status'])
        self.assertEqual(row['status_amendment']['supervision_commit'], 'registered-before-action')
        data['backends']['bf16']['status'] = 'completed_audited'
        with self.assertRaises(ValueError):
            extract('FIT16', data, amendment)

    def test_missing_official_denominators_cannot_disappear(self):
        data = {'backends': {'nf4': dict(execution_status='pending', registered_contexts=96, successful_official_episodes=0, missing_graded_contexts=96, groups=[])}}
        self.assertEqual(extract('PRIME20', data, {})[0]['missing_graded_contexts'], 96)
        data['backends']['nf4']['missing_graded_contexts'] = 95
        with self.assertRaises(ValueError):
            extract('PRIME20', data, {})

    def test_source_seal_mismatch_rejected(self):
        def loader(path):
            return json.dumps({'sealed_test_accessed': False} if path.endswith('SOURCE_MANIFEST.json') else {'source_manifest_sha256': '0'*64}).encode()
        with self.assertRaises(ValueError):
            build(loader, 'immutable-input', 'producer-hash')

    def test_context_missing_denominators_and_unknown_scores_preserved(self):
        group = dict(split='development',context_budget=8192,position='late',arm='main',loop=2,
            correct_worlds=None,observed_contexts=0,registered_contexts=624,missing_contexts=624)
        item = dict(status='pending',audited_contexts=0,registered_contexts=624,missing_contexts=624,groups=[group])
        data = {'backends': {'nf4': item}}
        self.assertIsNone(extract('CTX21',data,{})[0]['groups'][0]['correct_worlds'])
        group['correct_worlds']=0
        with self.assertRaises(ValueError):
            extract('CTX21',data,{})
        group['correct_worlds']=None; item['missing_contexts']=623
        with self.assertRaises(ValueError):
            extract('CTX21',data,{})

    def test_future_cache_coverage_cannot_create_admitted_vectors(self):
        coverage = [dict(registered_vectors=26,available_vectors=0,missing_vectors=26) for _ in range(48)]
        item = dict(status='pending',registered_vectors=1248,available_vectors=None,missing_vectors=1248,coverage=coverage,cells=[])
        data = {'backends': {'nf4': item}}
        self.assertIsNone(extract('XREF22',data,{})[0]['cells'][0]['cached_vectors'])
        coverage[0]['available_vectors']=1;coverage[0]['missing_vectors']=25
        with self.assertRaises(ValueError):
            extract('XREF22',data,{})

    def test_empty_future_cache_is_never_a_completed_fit_probe(self):
        coverage = [dict(registered_vectors=26,available_vectors=0,missing_vectors=26) for _ in range(48)]
        item = dict(status='completed_audited',registered_vectors=1248,available_vectors=0,missing_vectors=1248,coverage=coverage,
            cells=[dict(passes=1,status='completed',datasets=[],geometry=[])],arithmetic_audit=[dict(passes=1,status='passed',cached_vectors=0,physical_geometry_cells=0)])
        with self.assertRaises(ValueError):
            extract('XREF22',{'backends':{'nf4':item}},{})


if __name__ == '__main__':
    unittest.main()
