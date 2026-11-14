"""Post-hoc FC-13 regression; no new theorem-family selection or training."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import model_data as m
from train_local import load_checkpoint, write_json, append_json
from resources import ResourceGuard, guarded_process
from bounded_multigoal_search import LeanCompiler,solve

MATHCAN_SHA='c84e2025334e9a967c1297425a490f98d5b3d189'
GYM_SHA='e8a780254b5cf41d479095f4f22bcfc4aa76ed72'
GOALS_SHA='3df327272bc4768eac2fdfaa60a2cdbead06443e48153781683ee03944568554'

class ModelRanker:
    def __init__(self,model): self.model=model
    @torch.no_grad()
    def rank(self,goal,candidates):
        if not candidates:return []
        rows=[dict(goal_text=goal,family=f,script=s) for f,s in candidates]
        logits=torch.cat([self.model(m.encode(rows[i:i+16])) for i in range(0,len(rows),16)])
        p=logits.softmax(-1); utility=2*p[:,3]+p[:,1]+p[:,2]
        return [candidates[i] for i in sorted(range(len(rows)),key=lambda i:(-float(utility[i]),i))]

class FrequencyRanker:
    def __init__(self,frequency): self.frequency=frequency
    def rank(self,goal,candidates):
        return sorted(candidates,key=lambda c:-self.frequency.get(c[0],0.))

def preflight(compiler,out):
    good='import Init\nnamespace RAM02Smoke\ntheorem toy (P : Prop) (h : P) : P := by\n  exact h\nend RAM02Smoke\n'
    bad='import Init\nnamespace RAM02Smoke\naxiom poisoned : False\ntheorem toy : False := by\n  exact poisoned\nend RAM02Smoke\n'
    a=compiler(good,'toy',['exact h'],False,15.,out/'good')
    b=compiler(bad,'toy',['exact poisoned'],False,15.,out/'bad')
    assert a['accepted'] and a['axiom_gate_passed'],a
    assert b['returncode']==0 and not b['accepted'] and not b['axiom_gate_passed'],b
    branch='import Init\nnamespace RAM02Smoke\ntheorem toy (P Q : Prop) (h : P) (k : Q) : P ∧ Q := by\n  constructor\n  exact h\n  exact k\nend RAM02Smoke\n'
    actions=['constructor','exact h','exact k']
    probes=[compiler(branch,'toy',actions[:i],True,15.,out/f'branch{i}') for i in range(1,4)]
    assert all(x['accepted'] for x in probes),probes
    assert [x['goal_count'] for x in probes]==[2,1,0],probes
    final=compiler(branch,'toy',actions,False,15.,out/'branch-final')
    assert final['accepted'],final
    replay=compiler.compile_source(Path(final['source_file']).read_text(encoding='utf8'),False,15.,out/'branch-replay')
    assert replay['accepted'] and replay['source_sha256']==final['source_sha256']
    receipt=dict(good=a,unauthorized_axiom=b,branch_probes=probes,branch_final=final,replay=replay,
                 counted_as_regression_proofs=False)
    write_json(out/'preflight.json',receipt)
    return receipt

def prepare_cases(mathcan,out,compiler,guard):
    from research.refutation_bench.fc13 import test_data
    actual=subprocess.check_output(['git','-C',str(mathcan),'rev-parse','HEAD'],text=True).strip()
    if actual!=MATHCAN_SHA:raise ValueError('MathCan revision differs')
    # The source oracle is separate from policy search and never feeds actions to rankers.
    test_data.emit(mathcan,out)
    for filename in test_data.TEST_MODULES:
        stem=Path(filename).stem
        result=guarded_process([compiler.executable,str((out/(stem+'.lean')).resolve())],out/(stem+'.log'),guard,120.)
        if result['returncode'] or 'error:' in result['log'] or 'sorryAx' in result['log']:
            raise AssertionError('original source oracle failed: '+filename)
    test_data.extract(out)
    raw=(out/'actual_goal_states.jsonl').read_bytes()
    lf=raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(lf).hexdigest()!=GOALS_SHA:
        raise AssertionError('FC-13 initial compiler states differ from frozen evidence; independent audit required')
    rows=m.read_rows(out/'actual_goal_states.jsonl')
    initial={(r['source_file'],r['theorem']):r['actual_lean_goal_state'] for r in rows if r['stage_index']==0}
    cases=[(f,n) for f,names in test_data.TEST_MODULES.items() for n in names]
    assert len(cases)==16 and set(cases)==set(initial)
    sources={f:(mathcan/f).read_text(encoding='utf8') for f in test_data.TEST_MODULES}
    write_json(out/'ram02_source_audit.json',dict(mathcan_sha=actual,original_oracle_proofs=16,
               raw_goal_file_sha256=hashlib.sha256(raw).hexdigest(),lf_goal_file_sha256=hashlib.sha256(lf).hexdigest(),
               original_source_hashes={f:hashlib.sha256(s.encode()).hexdigest() for f,s in sources.items()}))
    return cases,initial,sources

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mathcan',type=Path,required=True)
    p.add_argument('--gym',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lean',required=True)
    p.add_argument('--lake')
    p.add_argument('--build',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if subprocess.check_output(['git','-C',str(args.gym),'rev-parse','HEAD'],text=True).strip()!=GYM_SHA:
        raise ValueError('gym revision differs')
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(experiment='RAM02_posthoc_FC13_regression',published_set=True,models=[],status='running',
                mathcan_sha=MATHCAN_SHA,gym_sha=GYM_SHA,lean_version=subprocess.check_output([args.lean,'--version'],text=True).strip())
    if not report['lean_version'].startswith('Lean (version 4.35.0,'):raise ValueError('Lean version differs')
    training=json.loads((args.weights/'training_report.json').read_text(encoding='utf8'))
    assert training['status']=='complete' and len(training['models'])>=6
    search_config=training['config']['search']
    report.update(config_sha256=training['config_sha256'],training_report_sha256=m.digest(args.weights/'training_report.json'),search=search_config)
    guard=ResourceGuard(args.out/'resources.jsonl',torch)
    try:
        with guard:
            compiler=LeanCompiler(args.lean,guard)
            report['preflight']=preflight(compiler,args.out/'preflight')
            if args.preflight_only:
                report['status']='preflight_complete';write_json(args.out/'regression_report.json',report);return
            if args.build:
                if not args.lake:raise ValueError('--lake required for --build')
                build=guarded_process([args.lake,'build','MathCan'],args.out/'lake-build.log',guard,900.,cwd=args.mathcan)
                assert build['returncode']==0 and not build['timed_out'],build['log'][-2000:]
                report['build']={k:v for k,v in build.items() if k!='log'}
            cases,states,sources=prepare_cases(args.mathcan,args.out/'oracle',compiler,guard)
            policies=[('native',None,None),('frequency',FrequencyRanker(training['data_audit']['frequency_weights']),None)]
            policies += [(f"{r['name']}_s{r['seed']}",r,r['checkpoints']['best.pt']) for r in training['models']]
            for name,policy,checkpoint in policies:
                checkpoint_hash=None
                if checkpoint is not None:
                    path=Path(checkpoint['path'])
                    assert m.digest(path)==checkpoint['sha256']
                    checkpoint_hash=checkpoint['sha256']
                    policy=ModelRanker(load_checkpoint(path))
                trajectories=[]
                started=time.perf_counter()
                for index,(filename,theorem) in enumerate(cases):
                    row=solve(sources[filename],theorem,[states[filename,theorem]],policy,args.out/'evidence'/name/f'{index:02d}',compiler,
                        seconds=search_config['seconds'],max_calls=search_config['lean_subprocesses'],
                        max_transitions=search_config['transitions'],per_call_seconds=search_config['per_call_seconds'])
                    row.update(model=name,source_file=filename,theorem=theorem,checkpoint_sha256=checkpoint_hash)
                    # Independent replay uses the exact final source bytes, not a search result cache.
                    if row['accepted']:
                        final=row['final_certificate']
                        source=Path(final['source_file']).read_text(encoding='utf8')
                        replay=compiler.compile_source(source,False,30.,args.out/'replay'/name/f'{index:02d}')
                        if not replay['accepted'] or replay['source_sha256']!=final['source_sha256']:
                            raise AssertionError('independent exact-source replay failed')
                        row['independent_replay']=replay
                    trajectories.append(row)
                    append_json(args.out/(name+'.jsonl'),row)
                    print('RAM02_PROOF',json.dumps({k:row[k] for k in ('model','theorem','accepted','reason','lean_subprocesses','elapsed_seconds')}),flush=True)
                solved=[f"{r['source_file']}:{r['theorem']}" for r in trajectories if r['accepted']]
                seconds=sum(r['elapsed_seconds'] for r in trajectories)
                replay_seconds=sum(r.get('independent_replay',{}).get('wall_seconds',0.) for r in trajectories)
                summary=dict(model=name,certified=len(solved),cases=len(cases),solved=solved,
                    search_seconds=seconds,replay_seconds=replay_seconds,
                    total_seconds_per_independently_certified_proof=(seconds+replay_seconds)/len(solved) if solved else None,
                    actual_lean_calls=sum(r['lean_subprocesses'] for r in trajectories),
                    replay_lean_calls=len(solved),reason_counts=dict(Counter(r['reason'] for r in trajectories)),
                    checkpoint_sha256=checkpoint_hash,trajectory_sha256=m.digest(args.out/(name+'.jsonl')),
                    wall_seconds_including_logging=time.perf_counter()-started,
                    successful_multigoal_transitions=sum(r['diagnostics'].get('multigoal_transitions',0) for r in trajectories))
                report['models'].append(summary)
                write_json(args.out/'regression_report.json',report)
            native=set(report['models'][0]['solved'])
            for result in report['models']:
                result['model_only_vs_native']=sorted(set(result['solved'])-native)
                result['native_only_vs_model']=sorted(native-set(result['solved']))
            report['status']='complete'
        report['resources']=guard.receipt()
    except BaseException as exc:
        report.update(status='aborted',error=repr(exc),resources=guard.receipt())
        write_json(args.out/'regression_report.json',report)
        raise
    write_json(args.out/'regression_report.json',report)

if __name__=='__main__':main()
