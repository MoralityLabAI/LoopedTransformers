"""Publish only closed, audited real-model engineering aggregates."""
from datetime import datetime, timezone
import json
from pathlib import Path
from .common import ROOT, RUNS, read, sha, verify

OUT = Path('reports/nanbeige_bpd_gym')
OUTPUTS = [OUT / 'NAN_GRAD15_PROGRESS.md', OUT / 'nan_grad15_progress.json']


def report():
    verify(); backends = {}
    for b in ('nf4', 'bf16'):
        a = RUNS / 'audits' / (b + '.json'); value = dict(status='pending', conditions=[])
        if (RUNS / 'eligibility.json').exists() and not read(RUNS / 'eligibility.json')['passed']:
            value['status'] = 'unrun_upstream_comp14_gate_failed'
        if a.exists():
            audit = read(a); receipt = RUNS / b / 'receipt.json'; assert sha(receipt) == audit['receipt_sha256']
            worker = read(receipt); value.update(status=worker['status'] + '_audited', audit_sha256=sha(a),
                conditions=audit['conditions'], CUDA_peak_allocated_bytes=worker['CUDA_peak_allocated_bytes'],
                CUDA_peak_reserved_bytes=worker['CUDA_peak_reserved_bytes'], failure=worker['failure'],
                elapsed_seconds=worker['elapsed_seconds'])
            peaks = []
            for c in audit['conditions']:
                for i in range(c['archived_cases']):
                    r = read(RUNS / b / f"t{c['passes']}_case_{i:02d}.json")
                    peaks.append(r['saved_tensor_statistics']['peak_added_cpu_bytes'])
            value['peak_added_saved_tensor_CPU_bytes'] = max(peaks) if peaks else None
        backends[b] = value
    payload = dict(experiment='NAN-GRAD-15', updated_UTC=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), backends=backends,
        eligibility=read(RUNS / 'eligibility.json') if (RUNS / 'eligibility.json').exists() else None,
        unit='Two reused CAL roots/four fixed counterfactual prefixes; depths and gradients are correlated engineering measurements.',
        primary_contrast='Native real-checkpoint NF4/BF16 gradient and memory feasibility before optimizer fitting.',
        physical_dose='Unmatched fixed component=.35/residual=.7 masks; no donor intervention.',
        scientific_causal_floor=None, matched_sham_floor=None, estimated_causal_power='Not a causal test.',
        readout_selection_exposure='COMP14 NF4 aggregates already observed; focused recipe fixed before GRAD15. No success-based prefix selection.',
        planned_formal_comparisons=0, actual_formal_comparisons=0, optimizer_steps=0, factors_trained=False,
        BPD_donor_transfer_attempted=False, sealed_test_accessed=False)
    OUT.mkdir(exist_ok=True)
    OUTPUTS[1].write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# NAN-GRAD-15: real-checkpoint NF4/BF16 gradient preflight', '',
        'Conditional on both COMP14 focused first-token screens passing. Four fixed CAL prefixes from two roots; native depth1/2, rank8 layer21 down_proj, fixed SVD initialization. '+
        'Zero optimizer steps: this is not trained VPD/BPD or donor transfer.', '',
        '| Backend | Status | Depth | Gradient cases passed | Registered cases |', '|---|---|---:|---:|---:|']
    for b, v in backends.items():
        if not v['conditions']:
            lines.append(f"| {b} | {v['status']} | — | — | 8 |")
        for c in v['conditions']:
            lines.append(f"| {b} | {c['condition_status']} | {c['passes']} | {c['successful_gradient_cases']} | 4 |")
    lines += ['', 'Exact native all-one closure, finite/nonzero V/U gradients, archived-logit KL and resource checks are separate from gradient correctness against a full backward replay. '+
        'Fixed masks are not matched-dose causal interventions. Target weights and factor values must remain unchanged. Failed/unrun cases remain visible; no neural retry.', '',
        'One large model at a time, one numerical CPU thread/Below Normal,2-GiB added saved-tensor CPU cap,85% CUDA cap. The separate math process is untouched. '+
        'Original FIT07/algebraic failures remain unchanged. The168-case test stays sealed. The broader goal remains active for actual factor fitting and later registered stages.', '',
        f"Source seal: `{payload['source_manifest_sha256']}`.", '',
        '[Frozen protocol](../../experiments/nan_grad15/PROTOCOL.md) · [Source seal](../../experiments/nan_grad15/SOURCE_MANIFEST.json) · [Full aggregates](nan_grad15_progress.json)', '']
    OUTPUTS[0].write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print({b: v['status'] for b, v in backends.items()}, flush=True)


if __name__ == '__main__':
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); report()
