"""Publish audited aggregate-only observational cross-references."""
from datetime import datetime, timezone
import json
from pathlib import Path
from .common import ROOT, RUNS, read, rows, sha, verify

OUT = Path('reports/nanbeige_bpd_gym')
OUTPUTS = [OUT / 'NAN_XREF17_PROGRESS.md', OUT / 'nan_xref17_progress.json']


def report():
    verify(); backends = {}; journal = rows(RUNS / 'queue_events.jsonl') if (RUNS / 'queue_events.jsonl').exists() else []
    for backend in ('nf4', 'bf16'):
        path = RUNS / 'audits' / (backend + '.json'); value = dict(status='pending', cells=[])
        failures = [r for r in journal if r.get('backend') == backend and r['event'] in ('worker_terminal', 'audit_terminal') and r['exit_code']]
        if failures:
            value.update(status='failed_host_attempt_preserved', failures=failures)
        if path.exists():
            audit = read(path); file = RUNS / backend / 'receipt.json'; assert sha(file) == audit['receipt_sha256']
            terminal = read(file); value.update(status='completed_audited', cells=terminal['cells'], arithmetic_audit=audit['cells'], audit_sha256=sha(path))
        backends[backend] = value
    payload = dict(experiment='NAN-XREF-17', source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), updated_UTC=datetime.now(timezone.utc).isoformat(),
        backends=backends, unit='Cached source root/family, reused development data; SAE features, worlds, loops and routing components correlated.',
        primary_contrast='CAL-only fixed SAE towers and actual audited FIT16 factors versus their SVD/Haar controls on every cached context.',
        scientific_causal_floor=None, matched_sham_floor=None, estimated_causal_power='Not calibrated; observational cross-reference only.',
        planned_formal_comparisons=0, actual_formal_comparisons=0, new_model_calls=0, optimizer_steps=0,
        BPD_donor_transfer_attempted=False, sealed_test_accessed=False,
        limitations='Cached clean inputs only, no online edited state/readout and no new correctness/transfer evidence. Backend SAE indices never subtracted.')
    OUT.mkdir(exist_ok=True)
    OUTPUTS[1].write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    lines = ['# NAN-XREF-17: actual learned factors and fixed CAL-only SAE towers', '',
        'Conditional on each passed FIT16 checkpoint audit. No model forward/backward, fitting or donor transfer. Each eligible cell retains all1706 cached vectors:644 GYM05,480 COMP14 and582 CTX08, both loops. Eight existing CAL-only dictionaries span both backends and layers5/11/17/21; learned component geometry is restricted to the layer21 down_proj output identity bridge.', '',
        '| Backend | Status | Fitted depth | Cell | Cached vectors | Physical geometry cells |', '|---|---|---:|---|---:|---:|']
    for b, v in backends.items():
        if not v['cells']:
            lines.append(f"| {b} | {v['status']} | — | pending/missing | — | — |")
        for c in v['cells']:
            lines.append(f"| {b} | {v['status']} | {c['passes']} | {c['status']} | {sum(d['vectors'] for d in c.get('datasets', []))} | {len(c.get('geometry', []))} |")
    lines += ['', 'All128 SAE features and all8 components are retained, with complete split/arm/context/position/loop strata in the JSON. Root means average repeated worlds/variants within source before averaging sources. Constant columns or fewer than3 roots produce undefined correlations, retained as null, not zero. The context families are reused and not asserted independent population replications.', '',
        'Initial random dictionaries, initial SVD factors, matched-column/row-norm Haar factors, all out-of-fit-depth applications and all failed/unrun cells remain explicit. Sparse-code loop displacements compare identical prefixes within one dictionary. Independently fitted BF16/NF4 SAE feature indices are never subtracted.', '',
        'Pure component module reconstruction and centered routed-energy correlations use clean cached inputs. Energy is a proxy, not the actual sphere-rescaled online write. The layer21 down_proj output is added directly to the physical pre-final-norm residual; no identity bridge is claimed for another layer or module. Physical weight geometry is observational, not reasoning specificity or equivalence.', '',
        'One numerical CPU thread/Below Normal and serialized analyses protect the separate math process. The finite host queue waits for the actual FIT16 supervisor to terminate before analysis. Source/target hashes and independently audited fit checkpoints are required. Generated replies, per-record codes/routes and checkpoints stay local; only aggregate geometry/statistics/hashes are public.', '',
        'The original168-case test stays sealed; earlier failed competence gates remain unchanged. There is no calibrated causal/sham floor or power, zero formal tests and no donor-transfer claim. The broader continuous goal remains active.', '',
        f"Source seal: `{payload['source_manifest_sha256']}`.", '',
        '[Frozen protocol](../../experiments/nan_xref17/PROTOCOL.md) · [Full aggregates](nan_xref17_progress.json)', '']
    OUTPUTS[0].write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print({b: v['status'] for b, v in backends.items()}, flush=True)


if __name__ == '__main__':
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); report()
