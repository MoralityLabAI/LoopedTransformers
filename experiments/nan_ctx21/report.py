"""Whole-roster aggregate coverage and matched physical initial-prefix geometry."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from .common import ROOT, RUNS, read, rows, sha, verify

OUT = Path('reports/nanbeige_bpd_gym')
OUTPUTS = ('NAN_CTX21_PROGRESS.md', 'nan_ctx21_progress.json')


def groups(public, records):
    by_id = {r['id']:r for r in records}; assert len(by_id)==len(records)
    assert set(by_id) <= {r['id'] for r in public}
    strata = defaultdict(list)
    for r in public:
        strata[r['split'], r['context_budget'], r['position'], r['arm']].append(r)
    result = []
    for (split, budget, position, arm), expected in sorted(strata.items()):
        observed = [by_id[r['id']] for r in expected if r['id'] in by_id]
        byroot = defaultdict(list)
        for r in expected:
            byroot[r['root']].append(by_id.get(r['id']))
        for t in range(2):
            complete = [v for v in byroot.values() if all(x is not None for x in v)]
            both = sum(all(x['correct'][t] for x in pair) for pair in complete)
            possible = sum(any(x is None for x in pair) and all(x is None or x['correct'][t] for x in pair) for pair in byroot.values())
            correct = sum(r['correct'][t] for r in observed)
            result.append(dict(split=split, context_budget=budget, position=position, arm=arm, loop=t+1,
                registered_contexts=len(expected), observed_contexts=len(observed), missing_contexts=len(expected)-len(observed),
                correct_worlds=correct if observed else None, format_valid=sum(r['format_valid'][t] for r in observed) if observed else None,
                whole_roster_correct_bounds=[correct, correct+len(expected)-len(observed)], registered_roots=len(byroot),
                complete_world_pairs=len(complete), both_world_roots_correct=both if complete else None,
                whole_roster_both_world_bounds=[both, both+possible],
                erased_repeat_pairs_identical=sum(pair[0]['token_ids'][t]==pair[1]['token_ids'][t] for pair in complete) if arm=='erased' else None))
    return result


def geometry(public, audits, cfg):
    import numpy as np
    maps = {b:{r['id']:r for r in a['records']} for b,a in audits.items()}
    shared = [r for r in public if all(r['id'] in m for m in maps.values())]
    strata = defaultdict(list)
    for source in public:
        for layer in cfg['layers_zero_based']:
            strata[source['split'],source['context_budget'],source['position'],source['arm'],layer]
    metrics = ('loop1_relative_l2','loop1_cosine','loop2_relative_l2','loop2_cosine','recurrence_displacement_relative_l2','recurrence_displacement_cosine')
    paired_correctness = defaultdict(lambda: dict(both_correct=0, BF16_only_correct=0, NF4_only_correct=0, neither_correct=0, matched_contexts=0,registered_contexts=0))
    for source in public:
        for t in (1,2):
            key = (source['split'],source['context_budget'],source['position'],source['arm'],t)
            paired_correctness[key]['registered_contexts'] += 1
    ordinals = {r['id']:i for i,r in enumerate(public)}
    for source in shared:
        states = {}
        for backend in ('bf16','nf4'):
            audited = maps[backend][source['id']]; ordinal = ordinals[source['id']]
            casefile = RUNS/backend/f'case_{ordinal:04d}.json'; assert sha(casefile)==audited['case_sha256']
            record = read(casefile)
            assert record['initial_prompt_ids_sha256']==read(RUNS/('nf4' if backend=='bf16' else 'bf16')/f'case_{ordinal:04d}.json')['initial_prompt_ids_sha256']
            archive = RUNS/backend/record['state_file']; assert sha(archive)==audited['state_sha256']
            with np.load(archive, allow_pickle=False) as z:
                states[backend] = {(v['loop'],v['layer_zero_based']):z[v['array_key']].astype(np.float64)
                    for v in record['state_index'] if v['site']=='residual_after_layer'}
        key = (source['split'],source['context_budget'],source['position'],source['arm'])
        for t in range(2):
            b,n = [maps[k][source['id']]['correct'][t] for k in ('bf16','nf4')]
            label = 'both_correct' if b and n else 'BF16_only_correct' if b else 'NF4_only_correct' if n else 'neither_correct'
            paired_correctness[key+(t+1,)][label] += 1; paired_correctness[key+(t+1,)]['matched_contexts'] += 1
        for layer in cfg['layers_zero_based']:
            values = {}
            def compare(a,b,name):
                na,nb = float(np.linalg.norm(a)),float(np.linalg.norm(b))
                values[name+'_relative_l2'] = float(np.linalg.norm(b-a))/na if na else None
                values[name+'_cosine'] = float(np.dot(a,b))/(na*nb) if na and nb else None
            for t in (1,2):
                compare(states['bf16'][t,layer],states['nf4'][t,layer],'loop'+str(t))
            compare(states['bf16'][2,layer]-states['bf16'][1,layer],states['nf4'][2,layer]-states['nf4'][1,layer],'recurrence_displacement')
            strata[key+(layer,)].append((source['root'],values))
    result = []
    for (split,budget,pos,arm,layer), values in sorted(strata.items()):
        n = cfg['roots'][split]; root_counts = defaultdict(int)
        for root,_ in values:
            root_counts[root] += 1
        item = dict(split=split,context_budget=budget,position=pos,arm=arm,layer_zero_based=layer,
            registered_contexts=2*n,matched_contexts=len(values),missing_contexts=2*n-len(values),
            registered_roots=n,observed_roots=len(root_counts),partial_world_roots=sum(v!=2 for v in root_counts.values()))
        for name in metrics:
            byroot = defaultdict(list)
            for root,metric in values:
                if metric[name] is not None:
                    byroot[root].append(metric[name])
            means = [sum(v)/len(v) for v in byroot.values()]
            item[name] = dict(equal_root_mean=sum(means)/len(means) if means else None,defined_roots=len(means),
                undefined_matched_contexts=sum(v[name] is None for _,v in values))
        result.append(item)
    assert len(result)==96
    paired = [dict(split=k[0],context_budget=k[1],position=k[2],arm=k[3],loop=k[4],missing_matched_contexts=v['registered_contexts']-v['matched_contexts'],**v) for k,v in sorted(paired_correctness.items())]
    return dict(status='completed_available_pairs',matched_initial_prefixes=len(shared),registered_contexts=624,
        missing_matched_prefixes=624-len(shared),groups=result,paired_correctness=paired,
        coordinate='Physical residual before final loop RMSNorm; delta=h2-h1; BF16 norm denominators; zero norms undefined.',
        weighting='Available worlds within root, then equal roots. Partial world roots and every missing denominator explicit; no imputation, CI or equivalence claim.')


def main():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify(); cfg = read(ROOT/'config.json'); public = rows(ROOT/'fixtures/public.jsonl'); audits = {}; backends = {}
    for backend in cfg['backends']:
        item = dict(status='pending',audited_contexts=0,registered_contexts=624,missing_contexts=624,groups=groups(public,[]))
        auditfile = RUNS/'audits'/f'{backend}.json'
        journal = rows(RUNS/'queue_events.jsonl') if (RUNS/'queue_events.jsonl').exists() else []
        terminal_events = [v for v in journal if v.get('backend')==backend and v['event']=='worker_terminal']
        audit_events = [v for v in journal if v.get('backend')==backend and v['event']=='audit_terminal']
        if terminal_events:
            item['status'] = 'terminal_audit_pending'
        if audit_events and audit_events[-1]['exit_code']:
            item['status'] = 'terminal_audit_failed_unverified'
        if auditfile.exists():
            audit = read(auditfile); receipt = RUNS/backend/'receipt.json'
            assert audit['status']=='passed' and audit['receipt_sha256']==sha(receipt)
            terminal = read(receipt); audits[backend] = audit
            item.update(status=terminal['status']+'_audited',audited_contexts=audit['contexts_checked'],missing_contexts=audit['missing_contexts'],
                physical_vectors_checked=audit['physical_vectors_checked'],audit_sha256=sha(auditfile),receipt_sha256=sha(receipt),
                failure=terminal['failure'],elapsed_seconds=terminal['elapsed_seconds'],CUDA_peak_allocated_bytes=terminal['CUDA_peak_allocated_bytes'],
                CUDA_peak_reserved_bytes=terminal['CUDA_peak_reserved_bytes'],orphan_case_artifacts_not_scored=len(audit['orphan_case_artifacts_not_scored']),
                groups=groups(public,audit['records']))
        backends[backend] = item
    physical = geometry(public,audits,cfg) if set(audits)=={'nf4','bf16'} else dict(status='pending_both_terminal_audits',groups=[])
    payload = dict(experiment='NAN-CTX-21',updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        backends=backends,paired_physical_geometry=physical,unit='16 fresh DEV/4 CAL graph roots;one pinned checkpoint;624 repeated contexts/backend.',
        primary_contrast='Descriptive matched NF4/BF16 source-conditioned one-hop accuracy and physical recurrence at4K/8K and early/middle/late positions.',
        physical_dose='Whole-target erasure and answer-copy semantic controls, not matched-dose donor interventions.',
        scientific_causal_floor=None,matched_sham_floor=None,estimated_causal_power='Uncalibrated; exploratory development only.',resolution_verdict='REDESIGN_OR_PILOT_ONLY',
        readout_selection_exposure='Prior exposed COMP14/CTX08 and fitting results known; fresh source draws and all criteria frozen before CTX21 inference.',
        planned_formal_comparisons=0,actual_formal_comparisons=0,BPD_donor_transfer_attempted=False,changes_old_failed_gates=False,sealed_test_accessed=False)
    OUT.mkdir(exist_ok=True); (OUT/OUTPUTS[1]).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines = ['# NAN-CTX-21: fresh 4K/8K source-conditioned development challenge','',
        '624 registered contexts/backend:576 DEV from16 fresh roots,48 CAL profiles from4 roots. Native two loops; one free full-vocabulary greedy token per loop. No new competence gate, donor transfer, causal or equivalence claim. Original failures are preserved and the168-case test stays sealed.','',
        '| Backend | Status | Audited contexts | Missing | Physical vectors checked |','|---|---|---:|---:|---:|']
    for b,v in backends.items():
        lines.append(f"| {b} | {v['status']} | {v['audited_contexts']}/624 | {v['missing_contexts']} | {v.get('physical_vectors_checked','—')} |")
    lines += ['', '## Every loop-2 score and coverage bound','',
        'Pending/missing outcomes are unknown, not wrong answers. Bounds give possible correct counts over the entire registered stratum. Each complete pair contains both counterfactual worlds of one root. Loop1 and erased-repeat equality remain in the JSON.','',
        '| Backend | Split | Budget | Position | Arm | Correct / observed | Missing | Whole-roster correct bounds | Both-correct / complete roots |','|---|---|---:|---|---|---:|---:|---|---:|']
    for b,v in backends.items():
        for g in v['groups']:
            if g['loop'] != 2:
                continue
            correct = g['correct_worlds'] if g['correct_worlds'] is not None else '—'
            both = g['both_world_roots_correct'] if g['both_world_roots_correct'] is not None else '—'
            bounds = g['whole_roster_correct_bounds']
            lines.append(f"| {b} | {g['split']} | {g['context_budget']} | {g['position']} | {g['arm']} | {correct}/{g['observed_contexts']} | {g['missing_contexts']} | [{bounds[0]}, {bounds[1]}]/{g['registered_contexts']} | {both}/{g['complete_world_pairs']} |")
    lines += ['', '## Physical geometry and resources','',
        f"Paired geometry status: `{physical['status']}`. All registered split/budget/position/arm/layer strata and six state/displacement metrics are retained in the JSON, including empty strata and undefined zero norms. Only exact shared initial prefixes are paired. Available worlds are averaged within root, then roots equally; partial root coverage is explicit. Independently fitted SAE indices are never subtracted.",'',
        'NF4 retains all22 decoder layers and the unchanged BF16 head on GPU; BF16 retains four layers plus a stage and the same head. CPU embeddings, batch1, native BF16/SDPA, no KV cache,85% CUDA allocator cap and four-hour between-prefix budget/backend. OOM stops all remaining contexts of that backend; no truncation or neural retry. Allocator peaks are not driver residency or evidence of no WDDM paging. One numerical CPU thread and Below Normal priority protect the math process. Timing reflects a shared machine.','',
        '| Backend | Peak allocated GiB | Peak reserved GiB |','|---|---:|---:|']
    for b,v in backends.items():
        values = [v.get(k) for k in ('CUDA_peak_allocated_bytes','CUDA_peak_reserved_bytes')]
        lines.append('| '+b+' | '+' | '.join(f'{x/2**30:.6f}' if x is not None else '—' for x in values)+' |')
    lines += ['', 'Independent terminal audits verify textual oracles, full-target erasure, exact pinned tokenizer IDs, greedy-score bindings, all vector hashes/shapes/finite coverage, unchanged targets when complete, and full-head parity. They do not replay neural forwards. Orphan artifacts never become grades. Synthetic sources/oracles are public; generated outputs, IDs, vectors and per-case grades stay local.','',
        'Descriptive exposed development only: no calibrated causal-effect/sham floor, sealed mechanism readout or population power; planned/actual formal comparisons0. More untouched independent roots and prospective matched causal controls are needed for confirmation. No learned-fiber/SAE analysis of this cache is claimed. The broader continuous goal remains active.','',
        f"Source seal: `{payload['source_manifest_sha256']}`.",'',
        '[Protocol](../../experiments/nan_ctx21/PROTOCOL.md) · [Source manifest](../../experiments/nan_ctx21/SOURCE_MANIFEST.json) · [Every score and geometry group](nan_ctx21_progress.json)','']
    (OUT/OUTPUTS[0]).write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:v['status'] for b,v in backends.items()},flush=True)


if __name__ == '__main__':
    main()
