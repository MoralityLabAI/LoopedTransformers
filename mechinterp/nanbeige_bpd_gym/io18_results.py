"""Readable presentation of one committed IO18 audit snapshot; no run-file reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

SOURCE = 'reports/nanbeige_bpd_gym/nan_fit16_io18_progress.json'
SEAL = '7bb76c5c895848cf12643312609a584ca0797bc87e9a05e320b8cfa2d04f4830'
DEFAULT_COMMIT = '8336514e632ba2f37a2143498f8f1403bbd044ce'
ROUTES = ('all_one','component_only','static_CAL_mean','clean_informed','reached_state')
METRICS = ('full_vocabulary_KL','actual_last_token_module_write_relative_l2',
    'local_intervention_l2_at_actual_reached_input','local_intervention_relative_l2_at_actual_reached_input',
    'reached_input_relative_l2','reached_vs_clean_mask_max_abs')


def validate(data):
    assert data['source_manifest_sha256']==SEAL and data['sealed_test_accessed'] is False
    assert data['actual_formal_comparisons']==data['planned_formal_comparisons']==0 and not data['BPD_donor_transfer_attempted']
    old=data['superseded_attempt']
    assert old['BF16_status']==old['XREF17_status']=='unrun_registered_supervision_amendment'
    assert len(old['NF4_cells'])==2 and {c['passes'] for c in old['NF4_cells']}=={1,2}
    assert all(c['status']=='failed' and c['optimizer_steps_completed']==2 for c in old['NF4_cells'])
    for backend,item in data['backends'].items():
        assert backend in ('nf4','bf16')
        if item['status']=='pending':
            assert not item['cells'] and not item['baseline_groups']
            continue
        assert item['status'].endswith('_audited')
        assert len({c['passes'] for c in item['cells']})==len(item['cells'])
        for cell in item['cells']:
            passes=cell['passes']; assert passes in (1,2)
            if cell['status']!='passed':
                assert cell['status'] in ('failed','budget_reached')
                continue
            assert cell['training']['optimizer_steps']==16 and cell['training']['DEV_or_control_training_uses']==0
            assert cell['training']['CAL_roots']==8 and cell['training']['CAL_worlds']==16
            assert cell['factor_update_l2']>0 and cell['controller_update_l2']>0
            assert cell['gauge_diagnostic']['stable_at_1e_minus5']
            mem=cell['saved_tensor_statistics']; assert 0 <= mem['peak_added_cpu_bytes'] <= mem['maximum_added_cpu_bytes']==2**31
            for field,modes in [('initial_groups',('component_only',)),('final_groups',ROUTES)]:
                expected={(split,mode,t) for split in ('calibration','development') for mode in modes for t in range(1,passes+1)}
                groups=cell[field]
                assert len(groups)==len(expected) and {(g['split'],g['mode'],g['loop']) for g in groups}==expected
                for g in groups:
                    n=8 if g['split']=='calibration' else 16
                    assert g['contexts']==2*n and g['source_roots']==n and g['arm']=='focused'
                    assert 0 <= g['correct'] <= g['format_valid'] <= 2*n and 0 <= g['both_world_roots_correct'] <= n
                    for name in METRICS:
                        v=g[name]; assert 0 <= v['defined_roots'] <= n and 0 <= v['undefined_contexts'] <= 2*n
                        if name=='reached_vs_clean_mask_max_abs' and g['mode']!='reached_state':
                            assert v['equal_root_mean'] is None and v['defined_roots']==0 and v['undefined_contexts']==2*n
                        else:
                            assert v['defined_roots']==n and v['undefined_contexts']==0 and math.isfinite(v['equal_root_mean'])
                            assert v['minimum_root_mean'] <= v['equal_root_mean'] <= v['maximum_root_mean']
        if item['status']=='completed_audited':
            assert {c['passes'] for c in item['cells']}=={1,2} and all(c['status']=='passed' for c in item['cells'])
    physical=data['physical_operator_comparison']
    if physical['status']=='completed':
        assert len(physical['pairs'])==4 and all(v['status']=='completed_audited' for v in data['backends'].values())
    else:
        assert not physical['pairs']


def number(value):
    return '—' if value is None else f'{value:.6g}'


def render(data, commit, source_sha):
    validate(data)
    lines=['# IO18: audited native factor fitting and routing fidelity','',
        'This is a readable presentation of one immutable published audit snapshot. It introduces no model calls, grades, paired-outcome tests, feature selection or new raw-run analysis. Pending backends remain unknown. The continuous goal remains active and the original168-case test stays sealed.','',
        f'Snapshot `{commit}`; aggregate SHA256 `{source_sha}`; source seal `{SEAL}`.','',
        '## Execution and training','',
        'Each backend/depth starts fresh from its own fixed rank8 SVD at layer21 down_proj and the same controller seed. Sixteen stock optimizer steps use all16 worlds of8 CAL roots;16 DEV roots/32 worlds are diagnostic only. T1 and T2 also differ in registered ablation/rescaling/loss recipes, so depth is not an isolated causal contrast. Only loop2 task competence was qualified; T1 is a depth/recipe control.','',
        '| Backend | Published status | Depth | Cell audit | Real optimizer steps | Factor update L2 | Controller update L2 |','|---|---|---:|---|---:|---:|---:|']
    passed=[]
    for b,item in data['backends'].items():
        if not item['cells']:
            lines.append(f"| {b} | {item['status']} | — | unknown | — | — | — |")
        for c in item['cells']:
            steps=c.get('training',{}).get('optimizer_steps',c.get('optimizer_steps_completed'))
            lines.append(f"| {b} | {item['status']} | {c['passes']} | {c['status']} | {steps if steps is not None else '—'} | {number(c.get('factor_update_l2'))} | {number(c.get('controller_update_l2'))} |")
            if c['status']=='passed':
                passed.append((b,c))
    lines += ['', '## Native baseline and source controls','',
        'These baselines retain the original source-conditioned one-token task and copy/whole-source-erased controls. Copy/erasure are not used to train the fibers and are not rerun under the learned routes. Loop1 and loop2 remain distinct.','',
        '| Backend | Split | Arm | Loop | Correct worlds | Valid format | Both-correct roots |','|---|---|---|---:|---:|---:|---:|']
    for b,item in data['backends'].items():
        for g in item['baseline_groups']:
            lines.append(f"| {b} | {g['split']} | {g['arm']} | {g['loop']} | {g['correct']}/{g['contexts']} | {g['format_valid']}/{g['contexts']} | {g['both_world_roots_correct']}/{g['source_roots']} |")
    lines += ['', '## Every learned-route score','',
        'All-one preserves the complete native module: component and residual masks are both1. Component-only keeps every fitted component with residual mask0. Static CAL mean uses the fixed average CAL controller mask. Clean-informed masks are computed on each clean source trajectory. Reached-state masks are recomputed from the input actually reached under preceding edits. All reduced routes have residual mask0 and use their registered depth-specific mean/rescaling recipe.','',
        '| Backend | Fit depth | Split | Loop | Route | Correct worlds | Valid format | Both-correct roots |','|---|---:|---|---:|---|---:|---:|---:|']
    for b,c in passed:
        for g in c['final_groups']:
            lines.append(f"| {b} | {c['passes']} | {g['split']} | {g['loop']} | {g['mode']} | {g['correct']}/{g['contexts']} | {g['format_valid']}/{g['contexts']} | {g['both_world_roots_correct']}/{g['source_roots']} |")
    lines += ['', 'At T2 DEV loop2, the fixed routes give:','']
    for b,c in passed:
        if c['passes']==2:
            selected={g['mode']:g for g in c['final_groups'] if g['split']=='development' and g['loop']==2}
            lines.append('- '+b+': '+', '.join(f"{mode} {selected[mode]['correct']}/32; both-world roots {selected[mode]['both_world_roots_correct']}/16" for mode in ROUTES)+'.')
    lines += ['', 'Identical aggregate counts do not mean identical successful cases or trajectories. No paired discordances are reconstructed from these aggregates. Positive factor/controller updates establish actual fitting, not improved task competence, an isolated adaptive-routing advantage or causal mediation.','',
        '### Initial component-only reference','',
        'Every registered initial diagnostic is shown; initial/final equal counts alone do not establish operator equivalence.','',
        '| Backend | Fit depth | Split | Loop | Correct worlds | Valid format | Both-correct roots |','|---|---:|---|---:|---:|---:|---:|']
    for b,c in passed:
        for g in c['initial_groups']:
            lines.append(f"| {b} | {c['passes']} | {g['split']} | {g['loop']} | {g['correct']}/{g['contexts']} | {g['format_valid']}/{g['contexts']} | {g['both_world_roots_correct']}/{g['source_roots']} |")
    lines += ['', '## Fidelity and physical dose diagnostics','',
        'All rows below are equal-root means: two worlds averaged within each root, then roots equally. CAL has8 roots, DEV16. Every defined metric has all those roots; absent mask-gap metrics remain undefined. Full root-mean ranges, defined-root counts and undefined-context counts remain in the immutable JSON; they are not confidence intervals.','',
        'KL is the registered float32 full-vocabulary KL(clean || changed), using natural logs at the phase readout. “Write vs clean” is ||changed module output at reached input − clean output at clean input|| / ||clean output||. “Local abs/rel” instead compares the changed output with the native module output at that SAME reached input, with the latter as denominator. These quantities differ when prior edits change the input. “Input gap” normalizes reached−clean input by the clean input norm. “Mask gap” is the maximum absolute difference between reached-state and clean-informed masks for that prefix/pass. Denominators use the registered1e-12 floor.','',
        '| Backend | Depth | Split | Loop | Route | KL(clean‖changed) | Write vs clean | Local abs | Local rel | Input gap | Mask gap |','|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for b,c in passed:
        for g in c['final_groups']:
            values=' | '.join(number(g[name]['equal_root_mean']) for name in METRICS)
            lines.append(f"| {b} | {c['passes']} | {g['split']} | {g['loop']} | {g['mode']} | {values} |")
    lines += ['', 'T1 reached-state and clean-informed routing see the same input by construction; this is a closure control. At T2, nonzero mask changes demonstrate state-conditioned execution, not a task-specific causal advantage. Routes have unmatched physical doses and different information availability. No sham-calibrated mechanism floor or population power is established.','',
        '## Resources and gauge scope','',
        'CUDA peaks below are cumulative within the backend from load through the named cell; they are not isolated cell increments. The saved-tensor CPU peak belongs to that cell’s owner and measures added activation snapshots under the2-GiB cap, not total RAM. Allocator reserved/allocated counters are distinct from driver residency and WDDM shared-memory samples.','',
        '| Backend | Through depth | Cumulative allocated GiB | Cumulative reserved GiB | Added saved-tensor CPU peak GiB | CPU cap GiB | Controlled gauge-write rel L2 |','|---|---:|---:|---:|---:|---:|---:|']
    for b,c in passed:
        mem=c['saved_tensor_statistics']
        lines.append(f"| {b} | {c['passes']} | {c['CUDA_peak_allocated_bytes']/2**30:.6f} | {c['CUDA_peak_reserved_bytes']/2**30:.6f} | {mem['peak_added_cpu_bytes']/2**30:.6f} | {mem['maximum_added_cpu_bytes']/2**30:.1f} | {number(c['gauge_diagnostic']['controlled_write_relative_l2'])} |")
    lines += ['', 'The finite float32 gauge diagnostic covers coordinated positive diagonal scaling and component permutation; it does not certify arbitrary rotations or hard-mask feature identity. Cross-backend physical V@U comparisons appear only when every required terminal fit audit passes; independent component/SAE indices are never subtracted.','',
        '| Reference | Compared | Physical operator relative L2 | Cosine |','|---|---|---:|---:|']
    physical=data['physical_operator_comparison']
    for pair in physical['pairs']:
        lines.append(f"| {pair['reference_backend']} T{pair['reference_depth']} | {pair['compared_backend']} T{pair['compared_depth']} | {number(pair['relative_l2'])} | {number(pair['cosine'])} |")
    lines += ['', f"Operator comparison status: `{physical['status']}`. Shared initialization, small updates, backend quantization and joint depth/recipe changes limit interpretation.",'',
        '## Preserved failures and evidence limits','',
        'Original FIT16 NF4 T1/T2 each failed after2 actual updates at the exclusive step-log writer; each log contains1 loss and1 missing loss. Those partial checkpoints are not successes, are never warm-started and supply no reconstructed missing loss. Original BF16 and XREF17 remain unrun under the prospectively pushed supervision amendment. IO18 changes only the repeated step-log writer and retains all numerical/source/gate/optimizer/resource criteria.','',
        'Unit:8 reused CAL roots and16 reused DEV roots on one pinned checkpoint. Worlds, routes, depths, loops, features and updates are correlated. Selection exposure: this COMP14 development condition and prior outcomes are already visible. Physical doses are unmatched; scientific causal floor/sham floor and causal power remain uncalibrated. Planned/actual formal comparisons0; resolution class remains pilot/diagnostic only. More untouched roots, prospective causal contrasts and calibrated shams are needed for confirmation. Original algebraic/FIT07 failures remain unchanged; no donor transfer or equivalence claim.','',
        'Terminal audits checked target hashes, exact native all-one closure, finite/nonzero factor gradients and updates, optimizer counters, all loss terms, saved-tensor resources, checkpoint/controller arithmetic, gauge diagnostics and every registered grade/route. They did not replay a full neural backward or exact CAL mean computation. Synthetic prompts/oracles are public; generated outputs, checkpoints and per-record grades remain local. NF4 retains all22 decoder layers and the unchanged BF16 head on GPU, BF16 four layers plus a stage; CPU embeddings. One large model at a time, one numerical CPU thread and Below Normal priority protect the separate math process.','',
        f'[Immutable aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/{commit}/{SOURCE}) · [Complete fitting stream](NAN_FIT16_IO18_PROGRESS.md) · [Frozen protocol](../../experiments/nan_fit16_io18/PROTOCOL.md) · [XREF19 registration](../../experiments/nan_xref19/PROTOCOL.md) · [Current evidence index](CONTINUATION_INDEX.md)','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-commit',default=DEFAULT_COMMIT); a=p.parse_args()
    from .evidence_index import below_normal
    below_normal(); commit=subprocess.check_output(['git','rev-parse',a.source_commit],text=True).strip()
    raw=subprocess.check_output(['git','show',commit+':'+SOURCE]); digest=hashlib.sha256(raw).hexdigest(); data=json.loads(raw)
    seal_path='experiments/nan_fit16_io18/SOURCE_MANIFEST.json'
    assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+seal_path])).hexdigest()==SEAL
    producer='mechinterp/nanbeige_bpd_gym/io18_results.py'; producer_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    committed=subprocess.check_output(['git','show',producer_commit+':'+producer])
    assert committed.replace(b'\r\n',b'\n')==Path(producer).read_bytes().replace(b'\r\n',b'\n'),'Commit presentation producer first'
    out=Path('reports/nanbeige_bpd_gym'); (out/'NAN_FIT16_IO18_RESULTS.md').write_text(render(data,commit,digest),encoding='utf-8',newline='\n')
    receipt=dict(source_commit=commit,source_file=SOURCE,source_sha256=digest,source_manifest_sha256=SEAL,
        producer_commit=producer_commit,producer_sha256=hashlib.sha256(committed).hexdigest(),
        published_backend_status={b:v['status'] for b,v in data['backends'].items()},
        evidence_class='Host-only presentation of committed audits; zero new model calls, grades, fitting, raw-run analysis or formal comparisons.',sealed_test_accessed=False)
    (out/'nan_fit16_io18_results_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(receipt)


if __name__=='__main__':
    main()
