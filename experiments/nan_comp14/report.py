"""Sanitized closed-evidence milestones and matched initial-prefix geometry."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from .common import ROOT, RUNS, events, read, rows, sha, verify, write_new

OUT = Path('reports/nanbeige_bpd_gym')
OUTPUTS = (OUT / 'NAN_COMP14_PROGRESS.md', OUT / 'nan_comp14_progress.json')


def grade_groups(records):
    groups = defaultdict(list)
    for row in records:
        for endpoint, correct, valid in [('first_token_loop1', row['first_token_correct'][0], row['first_token_valid'][0]),
                                          ('first_token_loop2', row['first_token_correct'][1], row['first_token_valid'][1]),
                                          ('bounded_loop2', row['bounded_correct'], row['bounded_valid'])]:
            groups[row['split'], row['arm'], endpoint].append((row, correct, valid))
    result = []
    for (split, arm, endpoint), values in sorted(groups.items()):
        roots = defaultdict(list)
        for row, correct, valid in values:
            roots[row['root']].append((row['world'], correct))
        paired = [v for v in roots.values() if len(v) == 2 and {x[0] for x in v} == {'A', 'B'}]
        result.append(dict(split=split, arm=arm, endpoint=endpoint, contexts=len(values),
            correct=sum(v[1] for v in values), format_valid=sum(v[2] for v in values),
            source_roots=len(roots), complete_world_pairs=len(paired),
            both_world_roots_correct=sum(all(x[1] for x in pair) for pair in paired)))
    return result


def gates(audit, groups, cfg):
    by = {(r['split'], r['arm'], r['endpoint']): r for r in groups}
    repeats = defaultdict(list)
    for row in audit['records']:
        if row['arm'].endswith('_erased'):
            repeats[row['split'], row['arm'], row['root']].append(row['generated_ids_sha256'])
    result = []
    for template in cfg['templates']:
        for endpoint in ('first_token_loop2', 'bounded_loop2'):
            checks = []
            for split, count in cfg['roots'].items():
                main = by.get((split, template, endpoint)); copy = by.get((split, 'copy', endpoint))
                erased = by.get((split, template + '_erased', endpoint))
                matched = [v for (s, arm, _), v in repeats.items() if s == split and arm == template + '_erased']
                stable = len(matched) == count and all(len(v) == 2 and v[0] == v[1] for v in matched)
                passed = bool(main and copy and erased and main['complete_world_pairs'] == copy['complete_world_pairs'] == erased['complete_world_pairs'] == count
                    and main['both_world_roots_correct'] >= cfg['minimum_both_world_roots'][split]
                    and copy['both_world_roots_correct'] == cfg['copy_both_world_roots_required'][split]
                    and erased['both_world_roots_correct'] == 0 and stable)
                checks.append(dict(split=split, main_both_correct=main['both_world_roots_correct'] if main else None,
                    main_required=cfg['minimum_both_world_roots'][split], roots=count,
                    copy_both_correct=copy['both_world_roots_correct'] if copy else None,
                    erased_both_correct=erased['both_world_roots_correct'] if erased else None,
                    erased_repeats_byte_identical=stable, passed=passed))
            result.append(dict(template=template, endpoint=endpoint, checks=checks,
                passed=audit['worker_status'] == 'completed' and all(v['passed'] for v in checks),
                scope='New source/readout-specific development screen only; does not reopen FIT07 or original algebraic gates. Bounded success cannot qualify a first-token fit.'))
    return result


def geometry(runs, cfg):
    import numpy as np
    source = rows(ROOT / 'fixtures/public.jsonl'); groups = defaultdict(list)
    for i, row in enumerate(source):
        records = [read(runs / backend / f'case_{i:04d}.json') for backend in ('bf16', 'nf4')]
        assert records[0]['initial_prompt_ids_sha256'] == records[1]['initial_prompt_ids_sha256']
        arrays = []
        for backend, record in zip(('bf16', 'nf4'), records):
            file = runs / backend / record['state_file']; assert sha(file) == record['state_file_sha256']
            with np.load(file, allow_pickle=False) as z:
                arrays.append({(v['loop'], v['layer_zero_based']): z[v['array_key']].astype(np.float64)
                    for v in record['state_index'] if v['site'] == 'residual_after_layer'})
        bf, nf = arrays
        def values(a, b, prefix):
            na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
            return {prefix + '_relative_l2': float(np.linalg.norm(b - a)) / na if na else None,
                    prefix + '_cosine': float(np.dot(a, b)) / (na * nb) if na and nb else None}
        for layer in cfg['layers_zero_based']:
            metrics = {}
            for t in (1, 2):
                metrics.update(values(bf[t, layer], nf[t, layer], 'loop' + str(t)))
            metrics.update(values(bf[2, layer] - bf[1, layer], nf[2, layer] - nf[1, layer], 'recurrence_displacement'))
            groups[row['split'], row['arm'], layer].append((row['root'], metrics))
    result = []
    for (split, arm, layer), records in sorted(groups.items()):
        output = dict(split=split, arm=arm, layer_zero_based=layer, contexts=len(records), source_roots=len({v[0] for v in records}))
        for name in records[0][1]:
            byroot = defaultdict(list)
            for root, metrics in records:
                if metrics[name] is not None:
                    byroot[root].append(metrics[name])
            means = [float(np.mean(v)) for v in byroot.values()]
            output[name] = dict(equal_root_mean=float(np.mean(means)) if means else None,
                minimum_root_mean=min(means) if means else None, maximum_root_mean=max(means) if means else None,
                defined_root_means=len(means), undefined_contexts=sum(v[1][name] is None for v in records))
        result.append(output)
    assert len(result) == 40
    return dict(status='completed', matched_initial_prefixes=240, groups=result,
        coordinate='Physical residual_after_layer before final loop RMSNorm; BF16 norm denominators; equal world-within-root then equal-root means.',
        scope='All initial prefixes only, not later divergent generated histories; ranges are descriptive, not CIs; independently fitted SAE indices are not compared.')


def report(runs=RUNS):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    verify(); runs = Path(runs); cfg = read(ROOT / 'config.json'); backends = {}
    for backend in cfg['backends']:
        base = runs / backend; receipt = base / 'receipt.json'; auditfile = runs / 'audits' / (backend + '.json')
        journal = events(base / 'events.jsonl')
        closed = sum(v['event'] == 'case_closed' for v in journal)
        terminal_closed = any(v['event'] == 'worker_receipt_closed' for v in journal)
        item = dict(status='pending', closed_contexts=closed, registered_contexts=240, audited_contexts=0,
                    groups=[], competence_screens=[], missing_contexts=240)
        if terminal_closed:
            terminal = read(receipt); item.update(status=terminal['status'] + '_audit_pending', worker_receipt_sha256=sha(receipt),
                CUDA_peak_allocated_bytes=terminal['CUDA_peak_allocated_bytes'], CUDA_peak_reserved_bytes=terminal['CUDA_peak_reserved_bytes'],
                elapsed_seconds=terminal['elapsed_seconds'], failure=terminal['failure'])
        if auditfile.exists():
            audit = read(auditfile); assert audit['status'] == 'passed'
            assert audit['receipt_sha256'] == sha(receipt)
            item.update(status=audit['worker_status'] + '_audited', audit_sha256=sha(auditfile), audited_contexts=audit['contexts_checked'],
                missing_contexts=audit['missing_contexts'], physical_vectors_checked=audit['physical_vectors_checked'],
                groups=grade_groups(audit['records']))
            item['competence_screens'] = gates(audit, item['groups'], cfg)
            item['driver_samples'] = []
            for name in ('driver_after_load.json', 'driver_terminal.json'):
                sample = read(base / name)
                item['driver_samples'].append(dict(stage=name, status=sample['status'],
                    dedicated_bytes=sum(v['DedicatedUsage'] for v in sample.get('WDDM_worker_rows', [])) if sample['status'] == 'sampled' else None,
                    shared_bytes=sum(v['SharedUsage'] for v in sample.get('WDDM_worker_rows', [])) if sample['status'] == 'sampled' else None,
                    whole_device_used_MiB=sample.get('nvidia_device_used_MiB'), source_sha256=sha(base / name)))
        elif base.exists():
            item['status'] = 'running_closed_case_counts_only' if not terminal_closed else item['status']
        backends[backend] = item
    paired = None
    if all(v['status'] == 'completed_audited' for v in backends.values()):
        cached = runs / 'paired_geometry.json'
        if not cached.exists():
            write_new(cached, geometry(runs, cfg))
        paired = read(cached)
    payload = dict(experiment='NAN-COMP-14', updated_UTC=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), backends=backends, paired_initial_prefix_geometry=paired,
        unit='Fresh source root; two counterfactual worlds, templates, controls, loops and tokens are correlated repeated measures; one checkpoint.',
        roots=cfg['roots'], contexts_per_backend=240, primary_contrast='Source-dependent both-world competence in each prespecified template; matched BF16/NF4 development description.',
        physical_dose=None, scientific_causal_floor=None, matched_sham_floor=None, estimated_causal_power='Not calibrated; competence pilot only.',
        readout_selection_exposure='Prior GYM05/FIT07/CTX08 results exposed; this fresh source/parser/gate freeze precedes all COMP14 pretrained inference. Two fixed prompt candidates, all outcomes retained.',
        planned_formal_comparisons=0, actual_formal_comparisons=0, BPD_donor_transfer_attempted=False,
        changes_old_competence_gates=False, sealed_test_accessed=False,
        privacy='Only aggregate counts, fixed criteria, geometry groups and hashes public; raw model replies/tokens/states/per-record grades stay local.')
    OUT.mkdir(parents=True, exist_ok=True)
    OUTPUTS[1].write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# NAN-COMP-14: fresh NF4/BF16 source-conditioned competence', '',
        'Eight fresh CAL and sixteen DEV roots; two permutation worlds with different answers per root. Two fixed prompt recipes, explicit copy and repeated source-erasure controls. '+
        'All 240 contexts/backend use the same pinned checkpoint and initial prompts. Full-vocabulary greedy generation stops at the first anchored answer prefix, EOS or six-token bound. '+
        'First-token and bounded endpoints stay separate; no old failed gate is changed.', '',
        '| Backend | Status | Closed contexts | Audited contexts | Missing audited coverage |', '|---|---|---:|---:|---:|']
    for b, item in backends.items():
        lines.append(f"| {b} | {item['status']} | {item['closed_contexts']}/240 | {item['audited_contexts']} | {item['missing_contexts']} |")
    lines += ['', 'Scores below appear only after independent terminal tokenizer/grammar/graph/state audits. Partial closed counts are operational evidence, not graded outcomes.', '',
        '| Backend | Split | Arm | Endpoint | Correct worlds | Format valid | Both-correct roots |', '|---|---|---|---|---:|---:|---:|']
    for b, item in backends.items():
        for g in item['groups']:
            lines.append(f"| {b} | {g['split']} | {g['arm']} | {g['endpoint']} | {g['correct']}/{g['contexts']} | {g['format_valid']}/{g['contexts']} | {g['both_world_roots_correct']}/{g['complete_world_pairs']} |")
    lines += ['', '## Fixed new-condition screens', '',
        'Each template/endpoint must solve both worlds in at least6/8 CAL and12/16 DEV roots, pass every copy pair, and retain byte-identical erased repeats with zero erased both-world successes. '+
        'These are exposed development screening criteria. Bounded success cannot qualify a first-token fit; any later fit needs its own prospective registration and matching readout. No donor transfer is enabled.', '',
        '| Backend | Template | Endpoint | New screen |', '|---|---|---|---|']
    for b, item in backends.items():
        for gate in item['competence_screens']:
            lines.append(f"| {b} | {gate['template']} | {gate['endpoint']} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines += ['', 'All first-prefix physical recurrence groups are retained in the JSON after both complete audits; later backend generation histories are not paired. '+
        'There is no causal intervention, formal comparison, equivalence claim or independent sealed confirmation. Original algebraic and FIT07 gates remain unchanged. The original168-case test stays sealed.', '',
        'NF4 retains all22 decoder layers and the unchanged BF16 head on GPU; BF16 retains four layers plus staging and the same head. CPU embeddings preserve capacity. '+
        'One large model, one numerical CPU thread and Below Normal priority; the separate math process is untouched. Allocator peaks and sampled dedicated/shared counters are distinct.', '',
        f"Source seal: `{payload['source_manifest_sha256']}`. Updated {payload['updated_UTC']}.", '',
        '[Frozen protocol](../../experiments/nan_comp14/PROTOCOL.md) · [Source seal](../../experiments/nan_comp14/SOURCE_MANIFEST.json) · [Full aggregates](nan_comp14_progress.json)', '']
    OUTPUTS[0].write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print({b: v['status'] for b, v in backends.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', default=str(RUNS)); a = p.parse_args(); report(Path(a.runs))
