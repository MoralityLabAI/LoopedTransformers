"""Lossless aggregate tables from a committed, terminal-audited CTX21 report.

No neural calls, grading, state reads, refitting, endpoint selection or new
geometry calculation. The frozen producer remains authoritative.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

PREFIX = 'reports/nanbeige_bpd_gym/'
SOURCE = PREFIX + 'nan_ctx21_progress.json'
SEAL = 'd3d0587416d0bf92e7b5d044f527d1fc6bbb630c6d3f074e5fcf0d1e03b6f4bc'
PRODUCER = 'mechinterp/nanbeige_bpd_gym/ctx21_results.py'
KEYS = ('split', 'context_budget', 'position', 'arm')
LAYERS = (5, 11, 17, 21)
METRICS = ('loop1_relative_l2', 'loop1_cosine', 'loop2_relative_l2',
           'loop2_cosine', 'recurrence_displacement_relative_l2',
           'recurrence_displacement_cosine')
OUTPUTS = tuple(PREFIX + name for name in (
    'NAN_CTX21_RESULTS.md', 'nan_ctx21_accuracy.csv',
    'nan_ctx21_paired_correctness.csv', 'nan_ctx21_geometry.csv',
    'nan_ctx21_results_receipt.json'))


def key(row):
    return tuple(row[name] for name in KEYS)


def layout():
    root = Path('experiments/nan_ctx21')
    manifest = (root/'SOURCE_MANIFEST.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == SEAL
    raw = (root/'fixtures/public.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == json.loads(manifest)['critical_files_sha256']['fixtures/public.jsonl']
    sources = [json.loads(line) for line in raw.splitlines()]
    groups = defaultdict(list)
    for source in sources:
        groups[key(source)].append(source)
    assert len(sources) == 624 and len(groups) == 24
    return dict(groups)


def indexed(rows, suffix):
    result = {key(row) + tuple(row[name] for name in suffix): row for row in rows}
    assert len(result) == len(rows), 'Duplicate aggregate stratum'
    return result


def validate(data, expected):
    assert data['experiment'] == 'NAN-CTX-21'
    assert data['source_manifest_sha256'] == SEAL
    assert data['sealed_test_accessed'] is False
    assert data['BPD_donor_transfer_attempted'] is False
    assert data['changes_old_failed_gates'] is False
    assert data['planned_formal_comparisons'] == data['actual_formal_comparisons'] == 0
    assert data['resolution_verdict'] == 'REDESIGN_OR_PILOT_ONLY'
    assert set(data['backends']) == {'bf16', 'nf4'}
    scores = {}
    wanted = {k + (t,) for k in expected for t in (1, 2)}
    for backend, summary in data['backends'].items():
        assert summary['status'] in {v + '_audited' for v in (
            'completed', 'capacity_stopped', 'budget_reached', 'failed')}, 'Terminal audit required'
        count = summary['audited_contexts']
        assert 0 <= count <= 624 and summary['registered_contexts'] == 624
        assert summary['missing_contexts'] == 624 - count
        assert summary['physical_vectors_checked'] == 40 * count
        for field in ('audit_sha256', 'receipt_sha256'):
            digest = summary[field]
            assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        for field in ('CUDA_peak_allocated_bytes', 'CUDA_peak_reserved_bytes'):
            assert isinstance(summary[field], int) and summary[field] >= 0
        assert summary['CUDA_peak_allocated_bytes'] <= summary['CUDA_peak_reserved_bytes']
        groups = indexed(summary['groups'], ('loop',))
        assert set(groups) == wanted, 'Missing or extra accuracy stratum'
        scores[backend] = groups
        for group_key, row in groups.items():
            source = expected[group_key[:-1]]
            registered = len(source); roots = len({v['root'] for v in source})
            n = row['observed_contexts']; missing = registered - n
            assert row['registered_contexts'] == registered and 0 <= n <= registered
            assert row['missing_contexts'] == missing and row['registered_roots'] == roots
            correct = row['correct_worlds']; valid = row['format_valid']
            assert (correct is None and valid is None) if not n else (
                isinstance(correct, int) and isinstance(valid, int) and 0 <= correct <= valid <= n)
            assert row['whole_roster_correct_bounds'] == [correct or 0, (correct or 0) + missing]
            complete = row['complete_world_pairs']; both = row['both_world_roots_correct']
            assert 0 <= complete <= roots and 2 * complete <= n <= 2 * complete + roots - complete
            assert both is None if not complete else isinstance(both, int) and 0 <= both <= complete
            lo, hi = row['whole_roster_both_world_bounds']
            assert lo == (both or 0) and lo <= hi <= lo + roots - complete
        for t in (1, 2):
            assert sum(g['observed_contexts'] for k, g in groups.items() if k[-1] == t) == count
    physical = data['paired_physical_geometry']
    assert physical['status'] == 'completed_available_pairs', 'Both terminal geometry inputs required'
    shared = physical['matched_initial_prefixes']
    assert shared == min(v['audited_contexts'] for v in data['backends'].values())
    assert physical['registered_contexts'] == 624 and physical['missing_matched_prefixes'] == 624 - shared
    pairs = indexed(physical['paired_correctness'], ('loop',))
    assert set(pairs) == wanted, 'Missing or extra paired correctness stratum'
    labels = ('both_correct', 'BF16_only_correct', 'NF4_only_correct', 'neither_correct')
    for group_key, row in pairs.items():
        registered = len(expected[group_key[:-1]])
        matched = row['matched_contexts']
        assert row['registered_contexts'] == registered and row['missing_matched_contexts'] == registered - matched
        assert matched == min(scores[b][group_key]['observed_contexts'] for b in scores)
        assert all(isinstance(row[k], int) and row[k] >= 0 for k in labels)
        assert sum(row[k] for k in labels) == matched
        for backend, own in (('bf16', 'BF16_only_correct'), ('nf4', 'NF4_only_correct')):
            known = row['both_correct'] + row[own]; score = scores[backend][group_key]
            assert known <= (score['correct_worlds'] or 0) <= known + score['observed_contexts'] - matched
    geometry = indexed(physical['groups'], ('layer_zero_based',))
    assert set(geometry) == {k + (layer,) for k in expected for layer in LAYERS}
    for group_key, row in geometry.items():
        registered = len(expected[group_key[:-1]])
        roots = len({v['root'] for v in expected[group_key[:-1]]})
        n = row['matched_contexts']; observed_roots = row['observed_roots']
        assert n == pairs[group_key[:-1] + (1,)]['matched_contexts']
        assert row['registered_contexts'] == registered and row['missing_contexts'] == registered - n
        assert row['registered_roots'] == roots and 0 <= observed_roots <= roots
        assert 0 <= row['partial_world_roots'] <= observed_roots
        assert 2 * observed_roots - row['partial_world_roots'] == n
        for metric in METRICS:
            item = row[metric]; mean = item['equal_root_mean']; defined = item['defined_roots']
            assert 0 <= defined <= observed_roots and 0 <= item['undefined_matched_contexts'] <= n
            assert (mean is None) == (defined == 0)
            assert defined <= n - item['undefined_matched_contexts'] <= 2 * defined
            if mean is not None:
                assert isinstance(mean, (float, int)) and math.isfinite(mean)
                assert -1.000000001 <= mean <= 1.000000001 if metric.endswith('cosine') else mean >= 0
    for layer in LAYERS:
        assert sum(g['matched_contexts'] for k, g in geometry.items() if k[-1] == layer) == shared
    return scores, pairs, geometry


def csv_text(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: ('not_applicable' if k == 'erased_repeat_pairs_identical' and r.get('arm') != 'erased'
        else 'unknown') if v is None else v for k, v in r.items()} for r in rows)
    return output.getvalue()


def number(value):
    return 'unknown' if value is None else format(value, '.7g')


def presentation(data, expected, commit, digest):
    scores, pairs, geometry = validate(data, expected)
    accuracy_rows = []
    for backend in ('bf16', 'nf4'):
        for k, g in sorted(scores[backend].items()):
            r = dict(backend=backend, **{name:g[name] for name in KEYS + ('loop',
                'registered_contexts', 'observed_contexts', 'missing_contexts', 'correct_worlds',
                'format_valid', 'registered_roots', 'complete_world_pairs', 'both_world_roots_correct',
                'erased_repeat_pairs_identical')})
            for field, prefix in (('whole_roster_correct_bounds', 'correct'),
                                  ('whole_roster_both_world_bounds', 'both_world_roots')):
                r[prefix + '_lower_bound'], r[prefix + '_upper_bound'] = g[field]
            accuracy_rows.append(r)
    pair_rows = [g for _, g in sorted(pairs.items())]
    geometry_rows = []
    for _, g in sorted(geometry.items()):
        for metric in METRICS:
            r = {name:g[name] for name in KEYS + ('layer_zero_based', 'registered_contexts',
                'matched_contexts', 'missing_contexts', 'registered_roots', 'observed_roots', 'partial_world_roots')}
            geometry_rows.append(dict(**r, metric=metric, **g[metric]))
    lines = ['# CTX21: complete matched accuracy, recurrence and memory tables', '',
        'This is a transcription of the frozen terminal-audited aggregate, with no new model calls, grades, geometry calculation or selected endpoints. Both loops, every semantic control and all missing strata are retained.', '',
        'There are 16 fresh DEV roots and four CAL roots on one pinned checkpoint. Worlds, positions, arms, loops and layers repeat within roots. Geometry averages available worlds within each root, then gives roots equal weight; partial roots remain explicit. No confidence interval, power, equivalence or causal confirmation is supplied. The original 168-case test stays sealed and the continuous goal remains active.', '',
        '## Terminal execution and resources', '',
        '| Backend | Audited terminal status | Contexts / 624 | Missing | Checked vectors | Peak allocated GiB | Peak reserved GiB |',
        '|---|---|---:|---:|---:|---:|---:|']
    for b in ('bf16', 'nf4'):
        g = data['backends'][b]
        lines.append(f"| {b} | {g['status']} | {g['audited_contexts']}/624 | {g['missing_contexts']} | {g['physical_vectors_checked']} | {g['CUDA_peak_allocated_bytes']/2**30:.6f} | {g['CUDA_peak_reserved_bytes']/2**30:.6f} |")
    lines += ['', 'NF4 keeps all 22 decoder layers on GPU; BF16 uses four resident layers plus a reusable stage. Both use the GPU head, CPU embeddings, native B1 arithmetic, no KV cache and the registered 85% allocator fraction. Capacity and timing therefore include placement and shared-machine effects. Allocator peaks differ from driver residency. Missing outcomes are unknown, not incorrect.', '',
        '## Every accuracy stratum, both loops', '',
        '| Backend | Split | Budget | Position | Arm | Loop | Correct / observed | Missing | Correct bounds / registered | Both-correct / complete roots | Root-correct bounds / registered |',
        '|---|---|---:|---|---|---:|---:|---:|---|---:|---|']
    for r in accuracy_rows:
        lines.append('| ' + ' | '.join(str(v) for v in (
            r['backend'], r['split'], r['context_budget'], r['position'], r['arm'], r['loop'],
            f"{number(r['correct_worlds'])}/{r['observed_contexts']}", r['missing_contexts'],
            f"[{r['correct_lower_bound']},{r['correct_upper_bound']}]/{r['registered_contexts']}",
            f"{number(r['both_world_roots_correct'])}/{r['complete_world_pairs']}",
            f"[{r['both_world_roots_lower_bound']},{r['both_world_roots_upper_bound']}]/{r['registered_roots']}")) + ' |')
    lines += ['', '## Exact-prefix paired correctness', '',
        'These are descriptive paired context counts, not independent root-level inferential tests. Available unpaired outcomes stay in the accuracy table and never become matched observations.', '',
        '| Split | Budget | Position | Arm | Loop | Matched / registered | Both correct | BF16 only | NF4 only | Neither | Missing matched |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in pair_rows:
        lines.append('| ' + ' | '.join(str(v) for v in (
            r['split'], r['context_budget'], r['position'], r['arm'], r['loop'],
            f"{r['matched_contexts']}/{r['registered_contexts']}", r['both_correct'],
            r['BF16_only_correct'], r['NF4_only_correct'], r['neither_correct'], r['missing_matched_contexts'])) + ' |')
    lines += ['', '## All 96 physical geometry strata and six metrics', '',
        'Physical residuals are measured before the final loop RMSNorm; displacement is h2−h1. Relative L2 uses the BF16 norm denominator. Each metric cell is `equal-root mean [defined roots / undefined matched contexts]`. Zero norms remain undefined. High state cosine alone does not establish recurrence-displacement preservation. No independently fitted SAE indices are aligned or subtracted.', '',
        '| Split | Budget | Position | Arm | Layer | Matched / registered | Roots / registered (partial) | h1 relative L2 | h1 cosine | h2 relative L2 | h2 cosine | Displacement relative L2 | Displacement cosine |',
        '|---|---:|---|---|---:|---:|---:|---|---|---|---|---|---|']
    for _, g in sorted(geometry.items()):
        values = [g['split'], g['context_budget'], g['position'], g['arm'], g['layer_zero_based'],
            f"{g['matched_contexts']}/{g['registered_contexts']}",
            f"{g['observed_roots']}/{g['registered_roots']} ({g['partial_world_roots']})"]
        values += [f"{number(g[m]['equal_root_mean'])} [{g[m]['defined_roots']}/{g[m]['undefined_matched_contexts']}]" for m in METRICS]
        lines.append('| ' + ' | '.join(str(v) for v in values) + ' |')
    lines += ['', 'The [accuracy CSV](nan_ctx21_accuracy.csv) also retains format-valid and erased-repeat counts; non-erased repeat fields are marked `not_applicable`, and other nulls are `unknown`. The [paired correctness CSV](nan_ctx21_paired_correctness.csv) retains every discordant/missing count. The [geometry CSV](nan_ctx21_geometry.csv) retains all 576 metric rows at full JSON numerical precision. Markdown numbers are display-rounded only.', '',
        'This presentation validates source/schema/coverage consistency and lossless scalar transcription; it does not independently recompute the upstream geometry. Independent terminal audits remain the source of per-case grades and physical-state checks. Original failures and uncalibrated effect/sham/power gates remain unchanged: **REDESIGN_OR_PILOT_ONLY**, formal comparisons 0.', '',
        f"Source aggregate commit `{commit}`, SHA-256 `{digest}`; source seal `{SEAL}`.", '',
        f'[Immutable source aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/{commit}/{SOURCE}) · [Frozen report](NAN_CTX21_PROGRESS.md) · [Presentation receipt](nan_ctx21_results_receipt.json)', '']
    return '\n'.join(lines), accuracy_rows, pair_rows, geometry_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--out-dir', default=PREFIX)
    args = parser.parse_args()
    from .evidence_index import below_normal
    below_normal()
    commit = subprocess.check_output(['git', 'rev-parse', args.source_commit + '^{commit}'], text=True).strip()
    raw = subprocess.check_output(['git', 'show', commit + ':' + SOURCE])
    data = json.loads(raw); digest = hashlib.sha256(raw).hexdigest()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    code = subprocess.check_output(['git', 'show', head + ':' + PRODUCER])
    assert code.replace(b'\r\n', b'\n') == Path(PRODUCER).read_bytes().replace(b'\r\n', b'\n')
    md, accuracy, pairs, geometry = presentation(data, layout(), commit, digest)
    files = dict(zip((Path(p).name for p in OUTPUTS[:-1]), (
        md, csv_text(accuracy), csv_text(pairs), csv_text(geometry))))
    receipt = dict(source_commit=commit, source_JSON_sha256=digest,
        source_manifest_sha256=SEAL, producer_commit=head,
        producer_git_blob_sha256=hashlib.sha256(code).hexdigest(),
        output_sha256={name:hashlib.sha256(text.encode('utf-8')).hexdigest() for name,text in files.items()},
        accuracy_rows=len(accuracy), paired_correctness_rows=len(pairs), geometry_metric_rows=len(geometry),
        upstream_audit_sha256={b:g['audit_sha256'] for b,g in data['backends'].items()},
        upstream_receipt_sha256={b:g['receipt_sha256'] for b,g in data['backends'].items()},
        new_model_calls=0, new_grades=0, new_geometry_calculations=0,
        sealed_test_accessed=False, formal_comparisons=0,
        verification_scope='Terminal/source/coverage/CSV transcription consistency, not independent geometry recomputation')
    files[Path(OUTPUTS[-1]).name] = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    destination = Path(args.out_dir); destination.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (destination/name).write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({k:receipt[k] for k in ('source_commit','accuracy_rows','paired_correctness_rows','geometry_metric_rows')}))


if __name__ == '__main__':
    main()
