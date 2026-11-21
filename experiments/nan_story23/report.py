"""Sanitized complete-coverage milestone; never reads live SDK/progress JSON."""
from pathlib import Path
from collections import defaultdict
import json
from .common import ROOT,RUNS,read,sha


def root_probe_groups(summaries):
    grouped=defaultdict(lambda:defaultdict(list))
    keys=('split','query_type','event_count','arm','mode')
    for row in summaries:
        for phase in row['phase_metrics']:
            key=tuple(row[k] for k in keys)+(phase['layer_zero_based'],phase['loop'])
            grouped[key][row['root']].append(phase)
    result=[]
    for key,byroot in sorted(grouped.items()):
        item=dict(zip(keys+('layer_zero_based','loop'),key));item['independent_roots']=len(byroot)
        for metric in ('normalized_MSE','random_initial_MSE','physical_reconstruction_L2'):
            item[metric+'_root_mean']=sum(sum(v[metric] for v in values)/len(values) for values in byroot.values())/len(byroot)
        item['feature_active_fraction_root_mean']=[sum(sum(v['feature_active_fraction'][i] for v in values)/len(values) for values in byroot.values())/len(byroot) for i in range(128)]
        result.append(item)
    return result


def main():
    destination=Path('reports/nanbeige_bpd_gym');cfg=read(ROOT/'config.json')
    result=dict(experiment=cfg['experiment'],source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        unit='independent generated storyworld root',n_CAL_units=3,n_DEV_units=9,source_worlds=24,
        registered_trajectories_per_backend=144,primary_endpoints=['final_room_correct','joint_trace_exact'],
        physical_dose='component/residual routing using passed frozen IO18 factors; no calibrated equal-dose sham',
        scientific_variation_floor=None,matched_sham_floor=None,estimated_resolution_power='uncalibrated development pilot',
        planned_formal_comparisons=0,actual_formal_comparisons=0,verdict='REDESIGN_OR_PILOT_ONLY',
        readout_selection_exposure='prior studies exposed; new source and methods frozen before STORY23 checkpoint forwards',
        sealed_test_accessed=False,BPD_donor_transfer_attempted=False,backends={})
    lines=['# NAN-STORY-23: storyworld traces, online BPD and fixed SAE towers','',
        'Fresh source-only development pilot; original 168-case test remains sealed. Continuous Nanbeige goal remains active.','',
        'Twelve independent roots (three CAL, nine DEV), two counterfactual stories each. Tasks track nested objects, first-order recorded beliefs and second-order recorded beliefs through 32/64/96 events. Prompts reach 3,522 tokens; verified canonical traces require at most 52 of the 64 allowed generated tokens.','',
        'Each backend registers 144 trajectories: 72 native main/copy/erased controls and 72 main trajectories across component-only, fixed CAL-mean-mask and reached-state routes. Additional routes require an independently passed sixteen-step IO18 T2 fit. Unqualified/missing outcomes remain explicit.','',
        'Grade final room, three intermediate room states, final containment path, format and joint trace separately. Model-emitted traces are checkable explanations, not proof of faithful internal reasoning. Record all 40 physical tower vectors at every generated prefix and apply all four backend-specific CAL-only SAE dictionaries with their random initial controls. T1/T2 BPD shadow traces retain all eight components; T1 loop2 is outside fit depth.','',
        'NF4 keeps all decoder weights and the unchanged BF16 output head on the 3050. BF16 uses four resident decoder layers plus a streamed stage and the same head. CPU embedding weights use checked GPU activation lookup reuse. Every generated prefix recomputes both loops without a KV cache. Pointwise BPD computation uses bounded 256-token chunks as a distinct registered execution condition.','',
        '| Backend | Status | Graded / registered | Captured prefixes | Tower vectors audited |','|---|---|---:|---:|---:|']
    for backend in cfg['backends']:
        file=RUNS/'audits'/(backend+'.json');probe=RUNS/'probes'/backend/'aggregate.json'
        if file.exists():
            value=read(file);result['backends'][backend]=dict(audit=value,audit_sha256=sha(file))
            lines.append(f"| {backend} | {value['worker_status']} | {value['graded_trajectories']} / 144 | {value['captured_prefixes']} | {value['physical_vectors_checked']} |")
            if probe.exists():
                p=read(probe);result['backends'][backend]['probes']={k:v for k,v in p.items() if k!='trajectory_summaries'}
                result['backends'][backend]['probes']['SAE_phase_root_groups']=root_probe_groups(p['trajectory_summaries'])
                result['backends'][backend]['probe_aggregate_sha256']=sha(probe)
        else:
            result['backends'][backend]=dict(status='frozen_queued_not_started',graded_trajectories=None,model_outcomes_available=False)
            lines.append(f'| {backend} | Frozen; queued | pending / 144 | pending | pending |')
    geometry=RUNS/'paired_geometry.json'
    if geometry.exists():result['paired_geometry']=read(geometry);result['paired_geometry_sha256']=sha(geometry)
    lines+=['','The GPU queue starts after actual CTX21 termination. STORY23 can run while the one registered XREF22 host analysis finishes; substantial STORY23 audits wait for XREF22 and serialize. One model and one host analysis maximum, one CPU numerical thread each, Below Normal priority. The separate math process is unchanged.','',
        'Limits: 85% CUDA allocator fraction; 24 hours/backend checked between prefixes; one hour/episode; 6 GiB artifacts/backend; minimum 10 GiB free disk. Preserve every stopped/missing/orphan slot. No neural retries or lost-response rescue. Live supervision uses authenticated health, actual process handles and append-only journals.','',
        '**Evidence verdict: REDESIGN_OR_PILOT_ONLY.** Zero formal inferential comparisons; scientific/sham causal floors and power are uncalibrated. Pair backend/route geometry only at identical prompt and generated-history hashes; token indices after divergence are not semantic alignment. Prior failures remain failed. No transfer, equivalence or causal mechanism conclusion.','',
        f"Source manifest SHA-256: `{result['source_manifest_sha256']}`. Public generated sources and host oracle fixtures are frozen; generated replies, tokens, states, masks, codes and individual grades stay local.",'',
        'Full coverage and all registered group endpoints: [machine-readable report](nan_story23_progress.json). Methods: [frozen protocol](../../experiments/nan_story23/PROTOCOL.md).']
    for name,value in [('nan_story23_progress.json',json.dumps(result,indent=2,allow_nan=False)+'\n'),('NAN_STORY23_PROGRESS.md','\n'.join(lines)+'\n')]:
        (destination/name).write_text(value,encoding='utf-8',newline='\n')
    print(dict(report='NAN_STORY23_PROGRESS.md',sealed_test_accessed=False))


if __name__=='__main__':main()
