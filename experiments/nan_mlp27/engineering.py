"""Execute real new preflight policy and new host/report on synthetic artifacts."""
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from .common import ROOT,THREADS,read,sha,write

def run():
    for k in THREADS:os.environ[k]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    torch.set_num_threads(1)
    from .engine import Engine
    from . import host,report
    repo=Path.cwd().resolve();base=repo/'runs/nan_mlp27_engineering';base.mkdir(parents=True,exist_ok=True)
    assert base.is_relative_to(repo)
    controls=[]
    with tempfile.TemporaryDirectory(prefix='preflight_',dir=base) as tmp:
        assert Path(tmp).resolve().is_relative_to(base)
        for deny_long_B1 in (False,True):
            e=Engine.__new__(Engine);e.t=torch;e.cfg=read(ROOT/'config.json');e.out=Path(tmp)/str(deny_long_B1);e.out.mkdir()
            e.workspace=SimpleNamespace(enabled=True);e.ids=lambda r:[10,11,12]
            e.embedding=SimpleNamespace(reset=lambda:None);e.loop=SimpleNamespace(run_pairs=lambda x:x,device=torch.device('cpu'))
            e.old=e.loop.run_pairs;e.install_instrument=lambda:None;e.head=SimpleNamespace(weight=torch.ones(16,4,dtype=torch.bfloat16))
            e.forward_count=0;e.capture_attempts=0;calls=[]
            def capture(sequences,modes,observe=True):
                e.capture_attempts+=1;B=len(sequences);P=max(map(len,sequences));calls.append(dict(batch=B,lengths=list(map(len,sequences)),modes=modes))
                if P==9216 and (B==2 or deny_long_B1):raise torch.cuda.OutOfMemoryError('MOCK_ONLY CUDA capacity')
                e.forward_count+=1
                state={(b,p,l,s):torch.full((4,),float(p),dtype=torch.bfloat16) for b in range(B) for p in (1,2) for l in (5,11,17,21)
                    for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
                boundaries=[torch.ones(B,4,dtype=torch.bfloat16)*p for p in (1,2)]
                logits=[torch.nn.functional.linear(h,e.head.weight).float() for h in boundaries]
                return dict(states=state,boundaries=boundaries,logits=logits,cuda_elapsed_ms=0.)
            e.capture=capture
            try:e.preflight({})
            except RuntimeError as err:
                assert deny_long_B1 and str(err)=='registered_padded_capacity_gate_failed_before_scored_inference'
            else:assert not deny_long_B1
            r=read(e.out/'preflight.json');assert r['all_full_roster_inputs_and_short_AB_admitted']==(not deny_long_B1)
            assert r['capacity'][-1]['status']=='capacity_failed'
            assert r['attempted_preflight_captures']==9
            assert r['completed_preflight_captures']==(7 if deny_long_B1 else 8)
            assert calls[-1]==dict(batch=2,lengths=[9216,9215],modes=['native','reached_state'])
            assert not e.admitted(2,9000)
            controls.append(dict(required_capacity_denied=deny_long_B1,capture_calls=len(calls),status=r['status']))
    # Reuse only the CLOSED synthetic files from the earlier host fixture.
    # No original neural artifact is copied or regraded.
    previous=repo/'runs/nan_full26_host_engineering/attempt2';assert read(previous/'host/receipt.json')['closed_episodes_audited']==16
    artifact=base/'synthetic_host_v3';artifact.mkdir(exist_ok=False)
    from .common import load_sources,intervention_rows,append
    public=load_sources()[0];byid={r['id']:i for i,r in enumerate(public)};AB_source=intervention_rows(public)
    for backend in ('nf4','bf16'):
        import json
        source=previous/backend;dest=artifact/backend;dest.mkdir()
        for p in source.glob('case*_prefix*'):shutil.copyfile(p,dest/p.name)
        entries=[]
        for line in (source/'closed.jsonl').read_bytes().splitlines(keepends=True):
            if line.endswith(b'\n'):entries.append(json.loads(line))
        baseline=read(source/'baseline_0000_r0_closed.json');AB_native=[]
        for entry in entries:
            record=read(source/entry['file'])
            if record['lane']=='baseline':name=f"baseline_{byid[record['source_id']]:04d}_r0_closed.json"
            else:
                route=('component_only','static_CAL_mean','reached_state').index(record['mode'])
                index=next(i for i,r in enumerate(AB_source) if r['id']==record['source_id'])
                ordinal=route*72+index;name=f'AB_{ordinal:04d}_r1_closed.json'
                if route==2:
                    packet=json.loads(record['text'][:-4]);packet['answer']='A' if packet['answer']!='A' else 'B';record['text']=json.dumps(packet)+';END'
                clean=dict(baseline,lane='AB',mode='native',fragments=baseline['fragments'][:1])
                if route==1:
                    packet=json.loads(clean['text'][:-4]);packet['answer']='A' if packet['answer']!='A' else 'B';clean['text']=json.dumps(packet)+';END'
                AB_native.append((f'AB_{ordinal:04d}_r0_closed.json',clean))
            write(dest/name,record);append(dest/'closed.jsonl',dict(file=name,sha256=sha(dest/name)))
        for name,record in AB_native:
            write(dest/name,record);append(dest/'closed.jsonl',dict(file=name,sha256=sha(dest/name)))
        with (dest/'closed.jsonl').open('ab') as stream:stream.write(b'{"file":')
        shutil.copyfile(source/'receipt.json',dest/'receipt.json')
        for _ in range(14):append(dest/'events.jsonl',dict(event='prefix_batch_closed',batch_size=1,cuda_elapsed_ms=0.,wall_seconds=.1,CUDA_peak_allocated_bytes=0))
        append(dest/'events.jsonl',dict(event='worker_terminal',status='SYNTHETIC_COMPLETED'))
    write(artifact/'model_lane_terminal.json',dict(synthetic_only=True))
    original_verify=host.verify
    try:
        from experiments.nan_full26.common import verify as source26_verify
        host.verify=source26_verify;host.run(SimpleNamespace(artifact_root=str(artifact)))
    finally:host.verify=original_verify
    assert read(artifact/'host/receipt.json')['closed_episodes_audited']==22
    geometry=read(artifact/'host/paired_geometry.json');assert geometry['exact_history_and_batch_shape_prefix_pairs']==12
    behavior=geometry['descriptive_AB_behavior'];assert len(behavior)==6
    assert sum(x['gains'] for x in behavior)==2 and sum(x['losses'] for x in behavior)==2
    (artifact/'reports').mkdir();write(artifact/'host_lane_terminal.json',dict(exit_code=0,synthetic_only=True))
    # The new manifest does not exist yet: give this CPU fixture a local
    # placeholder seal. Final freeze separately binds every real source byte.
    roots=artifact/'fixture_source';roots.mkdir();write(roots/'SOURCE_MANIFEST.json',dict(synthetic_only=True))
    shutil.copyfile(ROOT/'config.json',roots/'config.json')
    old_root,old_runs=report.ROOT,report.RUNS
    try:
        report.ROOT=roots;report.RUNS=roots;report.run(artifact,out_dir=artifact/'reports/nanbeige_bpd_gym')
    finally:os.chdir(repo);report.ROOT,report.RUNS=old_root,old_runs
    rendered=read(artifact/'reports/nanbeige_bpd_gym/nan_mlp27_progress.json')
    assert rendered['experiment']=='NAN-MLP-27'
    assert sum(x['closed_episodes'] for x in rendered['groups'])==22
    assert sum(x['registered_episodes'] for x in rendered['groups'])==3962
    assert sum(x['missing_episodes'] for x in rendered['groups'])==3940
    assert sum(x['closed_episodes']>0 for x in rendered['groups'])==20
    assert not (artifact/'reports/nanbeige_bpd_gym/NAN_FULL26_PROGRESS.md').exists()
    write(ROOT/'engineering_receipt.json',dict(status='passed',synthetic_only=True,pretrained_model_forwards=0,GPU_allocations=0,
        preflight_controls=controls,closed_synthetic_host_episodes=22,geometry_pairs=12,source26_report_untouched=True,
        paired_behavior_groups=6,synthetic_gains=2,synthetic_losses=2,all_registered_coverage_slots=3962,
        complete_original_criteria_reused=True,CPU_threads=1,priority='BelowNormal',sealed_test_accessed=False,
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    print(dict(status='passed',preflight_controls=2,host_episodes=22,source26_replies_replayed=0))

if __name__=='__main__':run()
