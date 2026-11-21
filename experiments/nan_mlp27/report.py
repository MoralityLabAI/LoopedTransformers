"""Sanitized closed-artifact coverage, grades, root units, timing and failures."""
import argparse
from collections import defaultdict
from pathlib import Path
from .common import read,rows,sha,ROOT,RUNS,load_sources,intervention_rows,source_group

def run(artifact,out_dir='reports/nanbeige_bpd_gym'):
    base=Path(artifact);host=base/'host';groups=defaultdict(list);coverage={};timing={}
    qualification=read(RUNS/'qualification.json') if (RUNS/'qualification.json').exists() else dict(status='pending_FULL26_actual_termination',run_both_backends=False)
    if (host/'source_groups.jsonl').exists():
        import json
        with (host/'source_groups.jsonl').open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):continue
                entry=json.loads(line);metadata=host/entry['file'];assert sha(metadata)==entry['sha256'];m=read(metadata)
                file=host/m['audit_file'];assert sha(file)==m['audit_sha256'];r=read(file);r['source_group']=m['source_group']
                groups[r['backend'],r['stage'],r['split'],r['lane'],r['mode'],json.dumps(r['source_group'],sort_keys=True)].append(r)
    for backend in ('nf4','bf16'):
        file=base/backend/'receipt.json';coverage[backend]=dict(status='unrun_or_live_no_terminal_claim')
        journal=base/backend/'events.jsonl'
        if journal.exists():
            events=[]
            with journal.open('rb') as stream:
                for line in stream:
                    if line.endswith(b'\n'):
                        import json
                        v=json.loads(line)
                        if v['event']=='prefix_batch_closed':events.append(v)
                        elif v['event']=='worker_terminal':coverage[backend]=read(file)
            total_cuda=sum(v['cuda_elapsed_ms'] for v in events)/1000;wall=sum(v['wall_seconds'] for v in events)
            timing[backend]=dict(closed_generated_prefix_batches=len(events),generated_tokens=sum(v['batch_size'] for v in events),
                two_row_batches=sum(v['batch_size']==2 for v in events),CUDA_event_seconds=total_cuda,forward_wall_seconds=wall,
                CUDA_event_over_forward_wall_ratio=total_cuda/wall if wall else None,
                CUDA_peak_allocated_bytes=max((v['CUDA_peak_allocated_bytes'] for v in events),default=None))
            total=coverage[backend].get('elapsed_seconds_including_load')
            timing[backend]['generated_tokens_per_end_to_end_second']=sum(v['batch_size'] for v in events)/total if total else None
    # Preserve every predeclared source/control stratum, including entirely
    # missing ones. Labels supply task metadata only, never task selection.
    from .host import labels
    public=load_sources()[0];gold=labels();expected=defaultdict(list)
    import json
    for backend in ('nf4','bf16'):
        for r in public:
            key=(backend,r['stage'],r['split'],'baseline','native',json.dumps(source_group(r,gold[r['id']]),sort_keys=True))
            expected[key].append(r['root'])
        for route in read(ROOT/'config.json')['AB_routes']:
            for r in intervention_rows(public):
                for mode in ('native',route):
                    key=(backend,r['stage'],r['split'],'AB',mode,json.dumps(source_group(r,gold[r['id']],route),sort_keys=True))
                    expected[key].append(r['root'])
    assert set(groups)<=set(expected),'An observed group is outside frozen scope'
    summaries=[]
    for key,registered_roots in sorted(expected.items()):
        items=groups.get(key,[])
        roots={}
        for r in items:roots.setdefault(r['root'],[]).append(r)
        summaries.append(dict(backend=key[0],stage=key[1],split=key[2],lane=key[3],mode=key[4],source_group=__import__('json').loads(key[5]),
            closed_episodes=len(items),roots=len(roots),format_valid=sum(r['grade']['format_valid'] for r in items),
            final_correct=sum(r['grade']['final_correct'] for r in items),generated_prefixes=sum(r['prefixes'] for r in items),
            registered_episodes=len(registered_roots),registered_roots=len(set(registered_roots)),missing_episodes=len(registered_roots)-len(items),
            whole_roster_correct_bounds=[sum(r['grade']['final_correct'] for r in items),sum(r['grade']['final_correct'] for r in items)+len(registered_roots)-len(items)],
            all_observed_worlds_correct_roots=sum(all(r['grade']['final_correct'] for r in v) for v in roots.values())))
    paired=None;host_terminal=None
    if (base/'host_lane_terminal.json').exists():
        host_terminal=read(base/'host_lane_terminal.json')
        if (host/'paired_geometry.json').exists():paired=read(host/'paired_geometry.json')
    result=dict(experiment='NAN-MLP-27',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),baseline_registered_per_backend=1549,
        AB_pairs_registered_per_backend=216,terminal_coverage=coverage,timing=timing,groups=summaries,
        qualification=qualification,MLP_token_chunk_rows=256,source26_replies_reused=0,
        host_terminal=host_terminal,exact_input_physical_geometry=paired,
        evidence_class='Exposed development descriptive trajectories; reasoning text is not internal faithfulness.',
        verdict='REDESIGN_OR_PILOT_ONLY',formal_comparisons=0,sealed_test_accessed=False)
    for backend,t in timing.items():
        t['independently_audited_prefixes']=sum(r['generated_prefixes'] for r in summaries if r['backend']==backend)
        t['generated_prefixes_without_closed_episode_audit']=t['generated_tokens']-t['independently_audited_prefixes']
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    import json
    (out/'nan_mlp27_progress.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# NAN-MLP-27: full development trajectories and batched A/B','',
        'Conditional new capacity condition: gate/up token chunks256, one full physical down projection, padded reached-state capacity probes. FULL26 remains separate with all failures and grades preserved. Scope: 1,549 baseline contexts per backend: GYM05 322, COMP14 240, CTX08 291, CTX21 624, STORY23 72. '
        'After full baseline generation: 216 clean/intervention pairs per backend (all 24 story worlds and 48 focused competence worlds × three routes).',
        '', 'Native full-prefix generation, two recurrent loops, full vocabulary greedy sampling. Native clean row A and edited row B share one real decoder batch with bounded gate/up activation workspaces; one model is resident. '
        'Capacity-only preflight governs batch two versus one for baseline contexts. A/B requires batch two; no serial substitution. '
        'All 40 physical vectors are saved at every generated prefix, with exact BF16 storage. Fixed SAE/random controls and independent BPD mask/write audits stream on one CPU thread.',
        '', 'This is a new bounded-MLP and batched execution condition on the identical frozen FULL26 prompts. Earlier one-token scores are not its matched control. '
        'Cross-precision accuracy differences include batch-shape differences wherever capacity admission differs; geometry requires identical histories and shapes. '
        'Generated prefixes without a closed audited episode retain unknown grades and are not replayed. '
        'All 168 original test cases remain sealed. Repeated roots, tokens, routes and features are correlated measures. '
        'No transfer, equivalence or causal confirmation is claimed; effect floors, dose/sham calibration and power remain uncalibrated. Verdict: REDESIGN_OR_PILOT_ONLY.', '']
    lines += ['Qualification: '+qualification['status']+'. No task correctness selects admission; a resource failure runs BOTH matched precisions fresh. No old reply is replayed.', '']
    for b in ('nf4','bf16'):
        c=coverage[b];lines += [f"{b.upper()}: {c['status']}. Closed baseline {c.get('baseline_completed','unknown')}/1549; A/B {c.get('AB_pairs_completed','unknown')}/216."]
        if b in timing:lines += ['Timing of closed generation batches: '+json.dumps(timing[b],sort_keys=True)+'. CUDA events measure forward intervals including synchronization gaps; this is not a kernel occupancy measurement.']
    lines += ['', 'All registered task/control strata are shown. Missing outcomes are unknown; zero correct among zero observed replies is not a zero accuracy.', '', '|Backend|Stage|Split|Lane|Mode|Stratum|Closed|Roots|Format|Correct|Prefixes|Missing|Correct bounds / registered|', '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summaries:lines.append('|'+ '|'.join(str(r[k]) for k in ('backend','stage','split','lane','mode'))+'|'+json.dumps(r['source_group'],sort_keys=True)+'|'+'|'.join(str(r[k]) for k in ('closed_episodes','roots','format_valid','final_correct','generated_prefixes','missing_episodes'))+'|'+str(r['whole_roster_correct_bounds'])+'/'+str(r['registered_episodes'])+'|')
    if paired:
        lines += ['', 'Matched physical geometry admits '+str(paired['exact_history_and_batch_shape_prefix_pairs'])+' identical prompt/history/batch-shape prefixes. '
            'All values below average within each independent root first. No fitted alignment or equivalence claim.', '',
            '|Stage|Split|Lane|Mode|Stratum|Layer|Roots|Loop1 cosine|Loop2 cosine|Displacement cosine|Displacement relative L2|',
            '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|']
        for r in paired['root_level_geometry']:
            lines.append('|'+ '|'.join(str(r[k]) for k in ('stage','split','lane','mode'))+'|'+json.dumps(r['source_group'],sort_keys=True)+'|'+str(r['layer'])+'|'+str(r['independent_roots'])+'|'+'|'.join(f'{v:.6g}' for v in r['mean_of_root_means'])+'|')
        lines += ['', 'Descriptive A/B outcomes use only closed matched trajectories. Gains/losses are not formal tests or dose-matched causal effects.', '',
            '|Backend|Stage|Split|Route|Stratum|Paired trajectories|Roots|A correct|B correct|Gains|Losses|Root mean B-A|',
            '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in paired['descriptive_AB_behavior']:
            lines.append('|'+ '|'.join(str(r[k]) for k in ('backend','stage','split','route'))+'|'+json.dumps(r['source_group'],sort_keys=True)+'|'+'|'.join(str(r[k]) for k in ('paired_trajectories','independent_roots','A_correct','B_correct','gains','losses','root_macro_B_minus_A'))+'|')
    (out/'NAN_MLP27_PROGRESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',required=True);run(p.parse_args().artifact_root)
