"""Closed-audit aggregates and physical operator comparisons, never live receipts."""
from datetime import datetime, timezone
import json
from pathlib import Path
from .common import ROOT, RUNS, read, rows, sha, verify

OUT = Path('reports/nanbeige_bpd_gym')
OUTPUTS = [OUT / 'NAN_FIT16_IO18_PROGRESS.md', OUT / 'nan_fit16_io18_progress.json']


def operator_pairs(backends):
    import numpy as np
    if not all(v['status'] == 'completed_audited' and len(v['cells']) == 2
               and all(c['status'] == 'passed' for c in v['cells']) for v in backends.values()):
        return dict(status='pending_complete_fit_audits', pairs=[])
    banks = {}
    for backend in backends:
        for passes in (1, 2):
            folder = RUNS / backend / f't{passes}_layer21_down_proj_rank8'
            receipt = read(folder / 'receipt.json'); checkpoint = folder / receipt['final_checkpoint']['file']
            assert sha(checkpoint) == receipt['final_checkpoint']['sha256']
            with np.load(checkpoint, allow_pickle=False) as z:
                banks[backend, passes] = z['V'].astype(np.float64), z['U'].astype(np.float64)
    pairs = []
    for a, b in [(('nf4', 1), ('nf4', 2)), (('bf16', 1), ('bf16', 2)), (('bf16', 1), ('nf4', 1)), (('bf16', 2), ('nf4', 2))]:
        aa = bb = ab = dd = 0.
        for start in range(0, 10752, 256):
            x = banks[a][0][start:start + 256] @ banks[a][1]
            y = banks[b][0][start:start + 256] @ banks[b][1]
            aa += float(np.sum(x*x)); bb += float(np.sum(y*y)); ab += float(np.sum(x*y)); dd += float(np.sum((y-x)**2))
        pairs.append(dict(reference_backend=a[0], reference_depth=a[1], compared_backend=b[0], compared_depth=b[1],
            relative_l2=(dd/aa)**.5 if aa else None, cosine=ab/(aa*bb)**.5 if aa and bb else None))
    return dict(status='completed', pairs=pairs, coordinate='V @ U, physical input-to-output map; float64 row chunks, reference Frobenius denominator.',
        limitation='Within-backend shared initialization and small updates; cross-backend target quantization and stochastic trajectories differ. Depth recipes jointly change losses/masks. No mechanism equivalence or isolated BPD advantage.')


def report():
    verify(); backends = {}
    for backend in ('nf4', 'bf16'):
        value = dict(status='pending', cells=[], baseline_groups=[]); path = RUNS / 'audits' / (backend + '.json')
        eligibility = RUNS / 'eligibility.json'
        if eligibility.exists() and not read(eligibility)['passed']:
            value['status'] = 'unrun_upstream_prerequisite_failed_or_missing'
        if path.exists():
            audit = read(path); receipt = RUNS / backend / 'receipt.json'; assert sha(receipt) == audit['receipt_sha256']
            terminal = read(receipt)
            value.update(status=terminal['status'] + '_audited', cells=audit['cells'], baseline_groups=audit['baseline_groups'],
                audit_sha256=sha(path), receipt_sha256=sha(receipt), failure=terminal['failure'], elapsed_seconds=terminal['elapsed_seconds'],
                CUDA_peak_allocated_bytes=terminal['CUDA_peak_allocated_bytes'], CUDA_peak_reserved_bytes=terminal['CUDA_peak_reserved_bytes'])
        backends[backend] = value
    original = dict(NF4_status='awaiting_original_terminal_receipt', BF16_status='unrun_registered_supervision_amendment',
        XREF17_status='unrun_registered_supervision_amendment', supervision_commit='9689206dc4c9c390e189cdd7ffd19bc735ceb9a2')
    old_events = Path('runs/nan_fit16/nf4/events.jsonl')
    if old_events.exists() and any(v['event'] == 'worker_receipt_closed' for v in rows(old_events)):
        previous = read('runs/nan_fit16/nf4/receipt.json')
        original.update(NF4_status=previous['status'], NF4_cells=previous['cells'], NF4_receipt_sha256=sha('runs/nan_fit16/nf4/receipt.json'))
    partial_audit = RUNS / 'original_attempt_audit.json'
    if partial_audit.exists():
        original.update(partial_update_audit_sha256=sha(partial_audit), partial_update_audit=read(partial_audit))
    physical = operator_pairs(backends)
    payload = dict(experiment='NAN-FIT-16-IO18', updated_UTC=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), superseded_attempt=original, backends=backends, physical_operator_comparison=physical,
        unit='8 reused CAL roots/16 world training prefixes;16 reused DEV roots/32 worlds. Depths, loops, routes and tokens correlated;one pinned checkpoint.',
        primary_contrast='Matched NF4/BF16 genuine factor/controller fitting and exposed fidelity diagnostics; no outcome-selected prefixes.',
        readout_selection_exposure='COMP14/GRAD15 exposed development condition; focused recipe qualified on loop2 only. T1 is a depth control, not independently established task competence.',
        scientific_causal_floor=None, matched_sham_floor=None, estimated_causal_power='Not calibrated; zero formal comparisons.',
        physical_dose='Unmatched routing changes, not a matched-dose donor intervention.',
        planned_formal_comparisons=0, actual_formal_comparisons=0, SAE_cross_reference_status='Separate subsequent registered analysis required.',
        BPD_donor_transfer_attempted=False, changes_old_competence_gates=False, sealed_test_accessed=False)
    OUT.mkdir(exist_ok=True)
    OUTPUTS[1].write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    lines = ['# NAN-FIT-16-IO18: matched actual NF4/BF16 VPD/BPD development fitting', '',
        'Execution-only repair: the original FIT16 NF4 partial updates and step-log FileExistsError are preserved; original BF16 and XREF17 were left unrun under the prospective supervision amendment. Only the repeated step-log writer changes; all sources/numerics/criteria remain identical. No failed checkpoint is warm-started. ', '',
        'Conditional on both unchanged COMP14 focused first-token screens and every GRAD15 gradient case passing. Each rank8 layer21 down_proj depth1/2 fit starts independently from its backend-specific SVD and the same CI seed, with sixteen genuine stock optimizer steps on both worlds of all eight CAL roots. No DEV/control fitting or donor transfer. T1 is a depth/recipe control; only loop2 competence was qualified.', '',
        '| Backend | Status | Depth | Cell audit/status | Optimizer steps | Factor update L2 | Controller update L2 |', '|---|---|---:|---|---:|---:|---:|']
    for b, value in backends.items():
        if not value['cells']:
            lines.append(f"| {b} | {value['status']} | — | pending/unrun | — | — | — |")
        for c in value['cells']:
            lines.append(f"| {b} | {value['status']} | {c['passes']} | {c['status']} | {c.get('training', {}).get('optimizer_steps', c.get('optimizer_steps_completed', 0))} | {c.get('factor_update_l2', '—')} | {c.get('controller_update_l2', '—')} |")
    lines += ['', 'Scores appear only after independent terminal checkpoint/controller/first-token audits. Pending/missing cells are not zeros. Initial component-only routes and every baseline split/arm/loop remain in the JSON.', '',
        '| Backend | Depth | Split | Route | Loop | Correct worlds | Both-correct roots |', '|---|---:|---|---|---:|---:|---:|']
    for b, value in backends.items():
        for c in value['cells']:
            for g in c.get('final_groups', []):
                lines.append(f"| {b} | {c['passes']} | {g['split']} | {g['mode']} | {g['loop']} | {g['correct']}/{g['contexts']} | {g['both_world_roots_correct']}/{g['source_roots']} |")
    lines += ['', '## Physical factor operators', '',
        'Compare V@U in shared physical input/output coordinates, never independently fitted component indices. All four fixed pairs are reported only after all four fit audits. Shared initialization, small updates and joint depth/recipe changes limit interpretation.', '',
        '| Reference | Compared | Relative Frobenius L2 | Cosine |', '|---|---|---:|---:|']
    for r in physical['pairs']:
        lines.append(f"| {r['reference_backend']} T{r['reference_depth']} | {r['compared_backend']} T{r['compared_depth']} | {r['relative_l2']} | {r['cosine']} |")
    lines += ['', 'One large model, one numerical CPU thread and Below Normal priority protect the separate math process. Existing offline checkpoint/software only; native NF4 kernels retained, GPU head unchanged, CPU embeddings. 2-GiB added saved-tensor CPU cap,85% CUDA allocator cap, four-hour between-stage budget per backend. Allocator and driver counters remain distinct.', '',
        'Copy/erasure are clean-baseline controls; all five learned routing modes are evaluated on all48 main prefixes, not on copy/erasure prompts. These are unmatched-dose fidelity diagnostics, without a calibrated sham/effect floor, population power, mechanism specificity or transfer claim. Original algebraic/FIT07 failures remain unchanged and the168-case test stays sealed. The continuous goal remains active; learned SAE cross-references require subsequent registered analysis.', '',
        'Synthetic prompts and host-only oracle fixtures are public frozen sources; host-only means isolated from the worker. Generated outputs, training checkpoints and per-record run grades stay local. Host audits replay controller/checkpoint arithmetic, not a second full neural backward or exact CAL-mean recomputation.', '',
        f"Source seal: `{payload['source_manifest_sha256']}`.", '',
        '[Protocol](../../experiments/nan_fit16_io18/PROTOCOL.md) · [Source seal](../../experiments/nan_fit16_io18/SOURCE_MANIFEST.json) · [Full aggregates](nan_fit16_io18_progress.json)', '']
    OUTPUTS[0].write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print({b: v['status'] for b, v in backends.items()}, flush=True)


if __name__ == '__main__':
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); report()
