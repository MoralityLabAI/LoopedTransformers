"""Synthetic closed-artifact exercise of the frozen FULL26 host and report.

Real fixed dictionaries and qualified CAL-fit factors are read, but no
Nanbeige weights or neural responses are created, replayed or graded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from experiments.nan_full26.common import ROOT,THREADS,append,load_sources,read,sha,verify,write


def run(out,attempt):
    for key in THREADS:os.environ[key]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from experiments.nan_full26 import host,report
    from experiments.nan_story23.common import fit_asset
    from experiments.nan_story23.host import bf16_round,local_replay
    from experiments.nan_fit07.pointwise import masks
    from mechinterp.nanbeige_bpd_gym.probe_towers import load_npz
    assert attempt>=2
    repo=Path.cwd().resolve();folder=repo/f'runs/nan_full26_host_engineering/attempt{attempt}'
    assert folder.is_relative_to(repo);folder.mkdir(parents=True,exist_ok=False)
    public,_=load_sources();gold=host.labels()
    comp=[r for r in public if r['stage']=='nan_comp14' and r['split']=='calibration' and r['arm']=='focused' and r['world']=='A']
    story=next(r for r in public if r['stage']=='nan_story23' and r['split']=='calibration' and r['arm']=='main')
    copy=next(r for r in public if r['stage']=='nan_story23' and r['source_case_id']==story['source_case_id'] and r['arm']=='copy')
    gym=next(r for r in public if r['stage']=='nan_gym05' and r['split']=='calibration' and r['suite']=='pointer' and not r['id'].endswith(':copy'))
    cases=[(comp[0],'native',3,'correct'),(comp[1],'native',1,'wrong'),
        (comp[0],'component_only',1,'correct'),(comp[0],'static_CAL_mean',1,'correct'),
        (comp[0],'reached_state',1,'correct'),(story,'native',2,'correct'),
        (copy,'native',1,'invalid'),(gym,'native',1,'correct')]
    expected_grades={};state_files=0
    for backend in ('nf4','bf16'):
        base=folder/backend;base.mkdir();asset=fit_asset(backend,2);assert asset
        bank=load_npz(asset['checkpoint']);cfg=asset['receipt']['final_checkpoint']['CI_config']
        for case,(r,mode,count,answer_kind) in enumerate(cases):
            lane='baseline' if mode=='native' else 'AB';closed_name=f'{lane}_{case:04d}_r0_closed.json'
            if r['stage']=='nan_story23':text=gold[r['id']]['canonical_trace'] if answer_kind=='correct' else 'SYNTHETIC_MALFORMED'
            else:
                answer=gold[r['id']]['answer']
                if answer_kind=='wrong':answer='A' if answer!='A' else 'B'
                text=json.dumps(dict(trace=['SYNTHETIC host fixture'],answer=answer))+';END'
            expected_grades[backend,closed_name]=answer_kind
            fragments=[]
            for step in range(count):
                rng=np.random.default_rng(26+case*100+step);arrays={};index=[];native=[];actual_masks=[]
                for phase in (1,2):
                    X=bf16_round(rng.normal(0,.1,10752).astype(np.float32))
                    N=bf16_round(rng.normal(0,.1,3072).astype(np.float32));native.append(N)
                    M=masks(X[None],bank,cfg)[0] if mode=='reached_state' else np.asarray(asset['static_CAL_mask'])[phase-1] if mode=='static_CAL_mean' else np.ones(8)
                    actual_masks.append(M)
                    Y=N if mode=='native' else local_replay(X,N,M,bank,asset['receipt']['training_contract']['ablation'],phase-1)
                    # Analytically known root-weighting signal. Root0 has
                    # three prefixes with displacement difference1; root1
                    # has one prefix with difference3. Root macro is2, not1.5.
                    H=np.zeros(3072,dtype=np.float32);H[0]=1
                    if phase==2:H[1]=1 if backend=='bf16' else 4 if case==1 else 2
                    for layer in (5,11,17,21):
                        vectors=dict(residual_after_layer=H,down_proj_input=X,down_proj_output=Y,
                            o_proj_input=np.zeros(6144,dtype=np.float32),o_proj_output=np.zeros(3072,dtype=np.float32))
                        for site,value in vectors.items():
                            key='v'+str(len(index));rounded=bf16_round(value)
                            arrays[key]=(rounded.view(np.uint32)>>16).astype(np.uint16)
                            index.append(dict(loop=phase,layer=layer,site=site,key=key))
                arrays['native_at_reached_input']=(np.asarray(native,dtype=np.float32).view(np.uint32)>>16).astype(np.uint16)
                if mode!='native':arrays['actual_masks']=np.asarray(actual_masks)
                state=base/f'case{case}_prefix{step}.npz'
                with state.open('xb') as stream:np.savez(stream,**arrays)
                state_files+=1;prefix=base/f'case{case}_prefix{step}.json'
                history='fixture_history_'+str(step)
                if backend=='nf4' and case==5 and step==1:history='SYNTHETIC_DIFFERENT_HISTORY'
                write(prefix,dict(state_file=state.name,state_sha256=sha(state),state_index=index,
                    history_sha256=hashlib.sha256(history.encode()).hexdigest(),
                    prefix_ids_sha256=hashlib.sha256((r['id']+str(step)).encode()).hexdigest(),
                    batch_size=2,padded_length=34 if backend=='nf4' and case==6 else 33))
                fragments.append(dict(file=prefix.name,sha256=sha(prefix)))
            closed=base/closed_name
            write(closed,dict(source_id=r['id'],mode=mode,lane=lane,text=text,fragments=fragments,
                fixture_only=True,not_a_neural_reply=True))
            append(base/'closed.jsonl',dict(file=closed.name,sha256=sha(closed)))
        # A truncated append has no complete record and must not be consumed.
        with (base/'closed.jsonl').open('ab') as stream:stream.write(b'{"file":')
        for _ in range(11):append(base/'events.jsonl',dict(event='prefix_batch_closed',batch_size=1,cuda_elapsed_ms=0.,wall_seconds=.1,CUDA_peak_allocated_bytes=0))
        write(base/'receipt.json',dict(status='SYNTHETIC_COMPLETED',baseline_completed=5,AB_pairs_completed=3,elapsed_seconds_including_load=2.))
        append(base/'events.jsonl',dict(event='worker_terminal',status='SYNTHETIC_COMPLETED'))
    write(folder/'model_lane_terminal.json',dict(fixture_only=True))
    host.run(SimpleNamespace(artifact_root=str(folder)))
    closed=read(folder/'host/receipt.json');assert closed['closed_episodes_audited']==16
    checked=0
    for line in (folder/'host/events.jsonl').read_bytes().splitlines():
        entry=json.loads(line);r=read(folder/'host'/entry['file']);kind=expected_grades[r['backend'],r['closed_file']]
        assert r['grade']['format_valid']==(kind!='invalid')
        assert r['grade']['final_correct']==(kind=='correct')
        assert r['physical_vectors']==40*r['prefixes']
        if r['stage']=='nan_story23':assert r['grade']['joint_correct']==(kind=='correct')
        if r['mode']!='native':assert r['maximum_mask_abs_error']<=1e-5 and r['maximum_local_write_relative_L2_error']<=.03
        checked+=r['physical_vectors']
    geometry=read(folder/'host/paired_geometry.json')
    assert geometry['exact_history_and_batch_shape_prefix_pairs']==9
    assert geometry['shared_positions_excluded_for_history_or_shape']==2
    group=[r for r in geometry['root_level_geometry'] if r['stage']=='nan_comp14' and r['lane']=='baseline']
    assert len(group)==4
    assert all(r['independent_roots']==2 and r['mean_of_root_means'][3]==2. for r in group)
    write(folder/'host_lane_terminal.json',dict(exit_code=0,fixture_only=True))
    # Isolate the frozen report's fixed relative output directory. Only its
    # ROOT data pointer is made absolute; the report function is unchanged.
    original_root=report.ROOT
    (folder/'reports').mkdir()
    try:
        report.ROOT=ROOT.resolve();os.chdir(folder);report.run(folder)
    finally:os.chdir(repo);report.ROOT=original_root
    rendered=read(folder/'reports/nanbeige_bpd_gym/nan_full26_progress.json')
    assert len(rendered['groups'])==12
    assert all(v['generated_prefixes_without_closed_episode_audit']==0 for v in rendered['timing'].values())
    assert rendered['exact_input_physical_geometry']['exact_history_and_batch_shape_prefix_pairs']==9
    result=dict(status='passed',synthetic_only=True,neural_replies_created_or_replayed=0,pretrained_model_forwards=0,GPU_allocations=0,
        synthetic_episodes_audited=16,physical_vectors_checked=checked,state_files=state_files,
        correct_wrong_malformed_packets_checked=True,three_real_fixed_BPD_route_replays=True,
        truncated_journal_records_ignored=True,exact_prefix_pairs=9,excluded_history_or_shape_pairs=2,
        unequal_root_prefix_counts=[3,1],root_macro_displacement_relative_L2=2.,incorrect_token_weighted_result=1.5,
        current_report_rendered_in_isolated_fixture_directory=True,real_report_files_untouched=True,
        frozen_source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),CPU_threads=1,priority='BelowNormal',sealed_test_accessed=False,
        source_sha256={str(ROOT/name):sha(ROOT/name) for name in ('host.py','paired.py','report.py')},
        fixture_source_sha256=sha(__file__),
        preserved_prior_fixture_failure=read(repo/'runs/nan_full26_host_engineering/attempt1/fixture_failure.json'))
    write(out,result);print({k:result[k] for k in ('status','synthetic_episodes_audited','physical_vectors_checked','exact_prefix_pairs','root_macro_displacement_relative_L2')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--attempt',type=int,default=2)
    a=p.parse_args();run(a.out,a.attempt)
