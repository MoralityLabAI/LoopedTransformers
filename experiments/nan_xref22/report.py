"""Publish only independently audited future-cache probes and complete coverage."""
from datetime import datetime,timezone
import json
from pathlib import Path
from .common import ROOT,RUNS,read,rows,sha,verify
from .cache import coverage

OUT=Path('reports/nanbeige_bpd_gym')
OUTPUTS=('NAN_XREF22_PROGRESS.md','nan_xref22_progress.json')


def report():
    verify();backends={};journal=rows(RUNS/'queue_events.jsonl') if (RUNS/'queue_events.jsonl').exists() else []
    public=rows('experiments/nan_ctx21/fixtures/public.jsonl')
    for backend in ('nf4','bf16'):
        value=dict(status='pending',cells=[],registered_vectors=1248,available_vectors=None,missing_vectors=1248,coverage=coverage(public,[]))
        failures=[r for r in journal if r.get('backend')==backend and r['event'] in ('worker_terminal','audit_terminal') and r['exit_code']]
        if failures:
            value.update(status='failed_host_attempt_preserved',failures=failures)
        admissionfile=RUNS/'admission'/f'{backend}.json'
        if admissionfile.exists():
            admission=read(admissionfile);value.update(context_admission_status=admission['status'],admission_sha256=sha(admissionfile))
            if admission['status']=='admitted':
                value['upstream_context_worker_status']=admission['worker_status']
                assert sha(admission['audit'])==admission['audit_sha256']
                ctx=read(admission['audit']);meta=[r|dict(loop=t) for r in ctx['records'] for t in (1,2)]
                value.update(available_vectors=len(meta),missing_vectors=1248-len(meta),coverage=coverage(public,meta))
        path=RUNS/'audits'/f'{backend}.json'
        if path.exists():
            audit=read(path);receipt=RUNS/backend/'receipt.json';assert audit['status']=='passed' and sha(receipt)==audit['receipt_sha256']
            terminal=read(receipt);assert terminal['admission_sha256']==sha(admissionfile)
            value.update(status='completed_audited',cells=terminal['cells'],arithmetic_audit=audit['cells'],audit_sha256=sha(path))
        backends[backend]=value
    payload=dict(experiment='NAN-XREF-22',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),updated_UTC=datetime.now(timezone.utc).isoformat(),
        backends=backends,unit='CTX21 fresh16 DEV/four CAL roots;one checkpoint;624 correlated prefixes,1248 vectors/backend/layer when complete.',
        primary_contrast='Fixed CAL-only SAE towers and actual IO18 factors/controllers on future4K/8K caches;all controls/features/coverage retained.',
        readout_selection_exposure='Prior COMP14/CTX08/IO18 results known;CTX21 outcomes unknown at XREF22 registration.',
        physical_dose='Clean cached input proxy and static weight directions;no online intervention dose.',
        scientific_causal_floor=None,matched_sham_floor=None,estimated_causal_power='Uncalibrated;observational only.',resolution_verdict='REDESIGN_OR_PILOT_ONLY',
        planned_formal_comparisons=0,actual_formal_comparisons=0,new_model_calls=0,new_grades=0,optimizer_steps=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False,
        limitations='Cached clean inputs,not actual sphere-rescaled online writes/task mediation. Independent backend SAE indices never subtracted. No feature selection or refit.')
    OUT.mkdir(exist_ok=True);(OUT/OUTPUTS[1]).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# NAN-XREF-22: prospective SAE/fiber probes of fresh4K/8K contexts','',
        'All fixed dictionaries and every passed IO18 depth fit are applied to every available independently audited CTX21 prefix. Full roster:624 prefixes/two loops=1248 vectors per backend/layer,16 DEV/four CAL roots. No new model calls, optimizer steps, grades, donor transfer or repaired old competence gate. Pending outcomes remain unknown.','',
        '| Backend | Host status | Available/registered vectors | Missing vectors | Fit depth | Cell | Analyzed vectors | Geometry cells |','|---|---|---:|---:|---:|---|---:|---:|']
    for b,v in backends.items():
        available=v['available_vectors'] if v['available_vectors'] is not None else '—'
        if not v['cells']:
            lines.append(f"| {b} | {v['status']} | {available}/1248 | {v['missing_vectors']} | — | pending/missing | — | — |")
        for c in v['cells']:
            count=sum(d['vectors'] for d in c.get('datasets',[])) if c['status']=='completed' else '—'
            lines.append(f"| {b} | {v['status']} | {available}/1248 | {v['missing_vectors']} | {c['passes']} | {c['status']} | {count} | {len(c.get('geometry',[])) if c['status']=='completed' else '—'} |")
    lines+=['','## Complete coverage accounting','',
        'Every registered stratum has a coverage slot, even when empty. Missing vectors are unknown, not zero errors/activations. Available worlds are averaged within each root before roots are weighted equally. Partial-world roots equal2×available_roots−available_vectors in a single-loop stratum; these are explicit below.','',
        '| Backend | Split | Budget | Position | Arm | Loop | Admitted/registered | Missing | Admitted/registered roots | Partial-world roots |','|---|---|---:|---|---|---:|---:|---:|---:|---:|']
    for b,v in backends.items():
        for g in v['coverage'] or []:
            lines.append(f"| {b} | {g['split']} | {g['context_budget']} | {g['position']} | {g['arm']} | {g['loop']} | {g['available_vectors']}/{g['registered_vectors']} | {g['missing_vectors']} | {g['available_roots']}/{g['registered_roots']} | {2*g['available_roots']-g['available_vectors']} |")
    lines+=['','## Fixed instruments and physical meaning','',
        'All eight original CAL-only SAE dictionaries remain fixed:layers5/11/17/21 per backend,128 features/top4,1000 original CAL steps on32 roots/256 vectors per dictionary. Random initial dictionaries, initial SVD and deterministic matched-norm Haar controls remain explicit. No new feature/task selection, DEV fitting or model query. All feature activation fractions, split/budget/position/arm/loop strata and same-prefix loop gaps remain in the JSON.','',
        'Each learned factor/controller requires a passed IO18 terminal audit and16 real optimizer steps. T1 at loop2 is outside fit depth. Every128-by8 feature/routed-energy correlation uses root means; fewer than3 roots or exact constant columns produce undefined/null cells. Four CAL roots do not calibrate a population null tail. Cached centered energy is a proxy, not the actual sphere-rescaled reached-state write.','',
        'Physical encoder/decoder/sensitivity geometry uses only layer21 down_proj output addition to pre-final-norm residual coordinates. Six application-loop/basis cells reuse static dictionary/factor weights and are correlated bookkeeping, not six independent mechanism measurements. Earlier-layer SAE states are never assigned an invented identity bridge from layer21. Independently fitted BF16/NF4 feature indices are never subtracted. Zero-direction cosine conventions retain explicit zero-norm counts.','',
        'The independent host auditor shares the admitted-cache decoder but separately checks code/top4 constraints, all root means, controller equations, loop gaps, correlations, matched-norm controls and physical direction matrices. Missing/failed context audits or failed fits stay unrun; zero admitted vectors stay unrun, without metric imputation. A zero native-output denominator is an explicit host failure. Orphan cache records are never rescued.','',
        'One numerical CPU thread/Below Normal priority and serialized backend analyses/audits protect the math process. The queue waits on the actual CTX21 supervisor, including both model/audit/publication paths. No GPU/model load. One-hour between-dataset host budget/backend, no automatic host retry. Raw codes, masks, vectors and per-record derivatives stay local; only aggregate statistics/geometry and hashes are public.','',
        'Observational exposed development only: uncalibrated causal/sham floor and power, REDESIGN_OR_PILOT_ONLY, planned/actual formal comparisons0. No task mediation, specificity, equivalence or donor transfer. Additional untouched roots and prospective calibrated causal controls are needed for confirmation. The original168-case test stays sealed. The continuous goal remains active.','',
        f"Source seal: `{payload['source_manifest_sha256']}`.",'',
        '[Protocol](../../experiments/nan_xref22/PROTOCOL.md) · [Full aggregate groups/geometry](nan_xref22_progress.json) · [Upstream context results](NAN_CTX21_PROGRESS.md)','']
    (OUT/OUTPUTS[0]).write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:v['status'] for b,v in backends.items()},flush=True)


if __name__=='__main__':
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();report()
