"""Terminal guards and lossless presentation of partial synthetic coverage."""
import copy
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.nan_ctx21.report import groups
from mechinterp.nanbeige_bpd_gym import ctx21_results as report
from mechinterp.nanbeige_bpd_gym.ctx21_results_handoff import terminal_source


def fixture():
    """Public sources only; deliberately invented outcomes, never model replies."""
    public = [json.loads(line) for line in Path(
        'experiments/nan_ctx21/fixtures/public.jsonl').read_bytes().splitlines()]
    expected = report.layout()
    records = {}
    data = dict(experiment='NAN-CTX-21', source_manifest_sha256=report.SEAL,
        sealed_test_accessed=False, BPD_donor_transfer_attempted=False,
        changes_old_failed_gates=False, planned_formal_comparisons=0,
        actual_formal_comparisons=0, resolution_verdict='REDESIGN_OR_PILOT_ONLY', backends={})
    for backend, count, offset in (('bf16', 319, 0), ('nf4', 311, 1)):
        rows = [dict(source, correct=[int((i+offset)%4 != 0), int((i+offset)%3 != 0)],
            format_valid=[1, 1], token_ids=[i%16, (i+1)%16]) for i,source in enumerate(public[:count])]
        records[backend] = {r['id']:r for r in rows}
        data['backends'][backend] = dict(status='budget_reached_audited' if backend=='bf16' else 'capacity_stopped_audited',
            registered_contexts=624, audited_contexts=count, missing_contexts=624-count,
            physical_vectors_checked=count*40, audit_sha256='a'*64, receipt_sha256='b'*64,
            CUDA_peak_allocated_bytes=1024, CUDA_peak_reserved_bytes=2048, groups=groups(public,rows))
    shared = public[:311]
    pairs = []
    geometry = []
    for k, sources in sorted(expected.items()):
        subset = [s for s in shared if report.key(s) == k]
        root_counts = {}
        for s in subset:
            root_counts[s['root']] = root_counts.get(s['root'],0)+1
        for loop in (1,2):
            row = dict(zip(report.KEYS,k), loop=loop, registered_contexts=len(sources),
                matched_contexts=len(subset), missing_matched_contexts=len(sources)-len(subset),
                both_correct=0, BF16_only_correct=0, NF4_only_correct=0, neither_correct=0)
            for source in subset:
                b,n = [records[backend][source['id']]['correct'][loop-1] for backend in ('bf16','nf4')]
                label = 'both_correct' if b and n else 'BF16_only_correct' if b else 'NF4_only_correct' if n else 'neither_correct'
                row[label] += 1
            pairs.append(row)
        for layer in report.LAYERS:
            row = dict(zip(report.KEYS,k), layer_zero_based=layer, registered_contexts=len(sources),
                matched_contexts=len(subset), missing_contexts=len(sources)-len(subset),
                registered_roots=len({s['root'] for s in sources}), observed_roots=len(root_counts),
                partial_world_roots=sum(n!=2 for n in root_counts.values()))
            for metric in report.METRICS:
                row[metric] = dict(equal_root_mean=(0.875 if metric.endswith('cosine') else 0.12345678901234567) if subset else None,
                    defined_roots=len(root_counts), undefined_matched_contexts=0)
            geometry.append(row)
    # A legitimate undefined zero-norm metric, beside a nonzero displacement.
    row = next(g for g in geometry if g['matched_contexts'])
    for metric in ('loop1_relative_l2','loop1_cosine'):
        row[metric] = dict(equal_root_mean=None, defined_roots=0,
            undefined_matched_contexts=row['matched_contexts'])
    data['paired_physical_geometry'] = dict(status='completed_available_pairs',
        registered_contexts=624, matched_initial_prefixes=311, missing_matched_prefixes=313,
        groups=geometry, paired_correctness=pairs)
    return data,expected


class PresentationContracts(unittest.TestCase):
    def setUp(self):
        self.data,self.expected = fixture()

    def test_partial_roots_and_unknown_8K_outcomes_survive(self):
        _,_,geometry = report.validate(self.data,self.expected)
        self.assertTrue(any(r['partial_world_roots'] for r in geometry.values()))
        self.assertTrue(all(r['matched_contexts']==0 for r in geometry.values() if r['context_budget']==8192))
        md,accuracy,pairs,scalars = report.presentation(self.data,self.expected,'c'*40,'d'*64)
        self.assertEqual((len(accuracy),len(pairs),len(scalars)),(96,48,576))
        self.assertIn('unknown [0/',md)
        missing = [r for r in accuracy if r['backend']=='nf4' and r['context_budget']==8192]
        self.assertTrue(missing)
        self.assertTrue(all(r['correct_worlds'] is None and r['correct_lower_bound']==0 and
            r['correct_upper_bound']==r['registered_contexts'] for r in missing))

    def test_CSV_keeps_every_metric_at_original_precision(self):
        _,accuracy,_,scalars = report.presentation(self.data,self.expected,'c'*40,'d'*64)
        parsed = list(csv.DictReader(io.StringIO(report.csv_text(scalars))))
        self.assertEqual(len(parsed),576)
        self.assertEqual(len({tuple(r[k] for k in report.KEYS+('layer_zero_based','metric')) for r in parsed}),576)
        for source,row in zip(scalars,parsed):
            mean = source['equal_root_mean']
            self.assertEqual(row['equal_root_mean'],'unknown' if mean is None else str(mean))
            if mean is not None:
                self.assertEqual(float(row['equal_root_mean']),mean)
        scores=list(csv.DictReader(io.StringIO(report.csv_text(accuracy))))
        self.assertTrue(all(r['erased_repeat_pairs_identical']=='not_applicable' for r in scores if r['arm']!='erased'))

    def test_pending_backend_or_geometry_is_refused(self):
        for target in ('backend','geometry'):
            d = copy.deepcopy(self.data)
            if target == 'backend': d['backends']['bf16']['status']='pending'
            else: d['paired_physical_geometry']['status']='pending_both_terminal_audits'
            with self.assertRaises(AssertionError): report.validate(d,self.expected)

    def test_duplicate_missing_or_extra_stratum_is_refused(self):
        for change in ('duplicate','missing','extra'):
            d=copy.deepcopy(self.data);rows=d['paired_physical_geometry']['groups']
            if change=='duplicate': rows.append(copy.deepcopy(rows[0]))
            elif change=='missing': rows.pop()
            else: rows[0]['layer_zero_based']=22
            with self.assertRaises(AssertionError): report.validate(d,self.expected)

    def test_wrong_bounds_pairs_and_source_are_refused(self):
        for change in ('bounds','pairs','source'):
            d=copy.deepcopy(self.data)
            if change=='bounds': d['backends']['nf4']['groups'][0]['whole_roster_correct_bounds'][1]+=1
            elif change=='pairs': d['paired_physical_geometry']['paired_correctness'][0]['both_correct']+=1
            else: d['source_manifest_sha256']='0'*64
            with self.assertRaises(AssertionError): report.validate(d,self.expected)

    def test_invalid_or_fabricated_defined_geometry_is_refused(self):
        for value in (float('nan'),-0.1,0.5):
            d=copy.deepcopy(self.data)
            g=next(g for g in d['paired_physical_geometry']['groups'] if g['matched_contexts']==0)
            g['loop2_relative_l2']['equal_root_mean']=value
            with self.assertRaises(AssertionError): report.validate(d,self.expected)

    def test_real_main_writes_hash_bound_files_and_refuses_pending_before_writes(self):
        for pending in (False,True):
            data=copy.deepcopy(self.data)
            if pending:data['backends']['bf16']['status']='pending'
            source=json.dumps(data,allow_nan=False).encode()
            code=Path(report.PRODUCER).read_bytes()
            replies={('git','rev-parse','fixture^{commit}'):'c'*40+'\n',
                ('git','show','c'*40+':'+report.SOURCE):source,
                ('git','rev-parse','HEAD'):'d'*40+'\n',
                ('git','show','d'*40+':'+report.PRODUCER):code}
            def git_read(command,**kwargs):return replies[tuple(command)]
            with tempfile.TemporaryDirectory(prefix='.ctx21-presentation-test-',dir=Path.cwd()) as temporary:
                assert Path(temporary).resolve().parent==Path.cwd().resolve()
                destination=Path(temporary)/'presentation'
                argv=['ctx21_results','--source-commit','fixture','--out-dir',str(destination)]
                with patch.object(sys,'argv',argv),patch.object(report.subprocess,'check_output',side_effect=git_read),contextlib.redirect_stdout(io.StringIO()):
                    if pending:
                        with self.assertRaises(AssertionError):report.main()
                        self.assertFalse(destination.exists())
                        continue
                    report.main()
                receipt=json.loads((destination/'nan_ctx21_results_receipt.json').read_bytes())
                self.assertEqual(set(p.name for p in destination.iterdir()),{Path(p).name for p in report.OUTPUTS})
                import hashlib
                for name,digest in receipt['output_sha256'].items():
                    self.assertEqual(hashlib.sha256((destination/name).read_bytes()).hexdigest(),digest)
                self.assertEqual(receipt['geometry_metric_rows'],576)
                self.assertEqual(receipt['new_model_calls'],0)

    def test_handoff_requires_both_audits_and_post_BF16_publication(self):
        journal=[]
        for backend,commit in (('nf4','c'*40),('bf16','d'*40)):
            for name in ('worker_terminal','audit_terminal','report_terminal'):
                journal.append(dict(event=name,backend=backend,exit_code=0))
            journal.append(dict(event='aggregate_pushed',commit=commit))
        self.assertEqual(terminal_source(journal),'d'*40)
        self.assertIsNone(terminal_source(journal[:4]))
        self.assertIsNone(terminal_source(journal[:-1]))
        failed=copy.deepcopy(journal);failed[5]['exit_code']=1
        self.assertIsNone(terminal_source(failed))
        unordered=copy.deepcopy(journal);unordered[4],unordered[5]=unordered[5],unordered[4]
        self.assertIsNone(terminal_source(unordered))


if __name__=='__main__':
    unittest.main()
