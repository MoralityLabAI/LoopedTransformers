"""Audit completed campaign evidence and export measured summaries."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import time
import torch
import model_data as m
from train_local import load_checkpoint,checkpoint_path,metrics,split_data,write_json
from resources import ResourceGuard

@torch.no_grad()
def pass_benchmark(model,rows,passes):
    model.config=m.Config(model.config.name,model.config.width,passes)
    chunks=[m.encode(rows[i:i+16]) for i in range(0,len(rows),16)]
    def forward():
        for batch in chunks:model(batch)
    for _ in range(5):forward()
    trials=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(20):forward()
        trials.append((time.perf_counter()-start)/20)
    return dict(passes=passes,median_forward_seconds_per_dev_set=statistics.median(trials),
                timing_trials_seconds=trials,examples=len(rows),microbatch=16,
                metrics=metrics(model,rows))

def family(script):
    if script in m.FAMILIES[:6]:return script
    for prefix,name in [('exact ','exact_local'),('apply ','apply_local'),('rw ','rw_local'),
                        ('rcases ','rcases_local'),('induction ','induction_list'),('simpa ','simpa_local')]:
        if script.startswith(prefix):return name
    raise ValueError('unknown selected script '+script)

def audit_regression(root,report):
    summaries=[]
    for summary in report['models']:
        path=root/(summary['model']+'.jsonl')
        assert m.digest(path)==summary['trajectory_sha256']
        rows=m.read_rows(path)
        assert len(rows)==16
        by_family=defaultdict(Counter)
        accepted=[]
        diagnostics=Counter()
        for row in rows:
            assert row['lean_subprocesses']==len(row['calls'])<=12
            assert row['max_path_transitions']<=3
            diagnostics.update(row['diagnostics'])
            for call in row['calls']:
                assert m.digest(Path(call['source_file']))==call['source_sha256']
                assert m.digest(Path(call['log_file']))==call['log_sha256']
                if call['probe']:
                    f=by_family[call['family']]
                    f['attempts']+=1
                    f['compiler_accepted']+=int(call['accepted'])
                    f['created_branches']+=int(call.get('creates_branches',False))
                    f['reduced_goals']+=int(call.get('reduces_goal_count',False))
                    f['scratch_closed']+=int(call.get('all_goals_closed',False))
            if row['accepted']:
                assert row['lean_subprocesses']+1<=12
                assert row['elapsed_seconds']<=20
                final=row['final_certificate']; replay=row['independent_replay']
                assert final['accepted'] and replay['accepted'] and final['axiom_gate_passed'] and replay['axiom_gate_passed']
                assert final['source_sha256']==replay['source_sha256']==m.digest(Path(replay['source_file']))
                assert set(final['axioms'])<={'propext','Classical.choice','Quot.sound'}
                proof_source=Path(final['source_file']).read_text(encoding='utf8')
                assert 'all_goals sorry' not in proof_source
                accepted.append(dict(theorem=row['theorem'],source_file=row['source_file'],
                    actions=row['chosen_actions'],source_sha256=final['source_sha256'],
                    replay_sha256=replay['source_sha256'],axioms=final['axioms']))
                for f in set(family(a) for a in row['chosen_actions']):by_family[f]['certified_proofs_using_family']+=1
        summaries.append(dict(**summary,per_family={f:dict(c) for f,c in by_family.items()},
            diagnostics=dict(diagnostics),certificates=accepted))
    return summaries

def prune_metrics(value):
    result=json.loads(json.dumps({k:v for k,v in value.items() if k!='choices'}))
    for cls in result['classes'].values():
        if cls['support']==0:cls['recall']=None
    for item in result['per_family'].values():
        if item['positives']==0:item['positive_recall']=None
    result['choices']={k:{a:b for a,b in v.items() if a!='selected'} for k,v in value['choices'].items()}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--regression',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    training=json.loads((args.training/'training_report.json').read_text(encoding='utf8'))
    regression=json.loads((args.regression/'regression_report.json').read_text(encoding='utf8'))
    assert training['status']==regression['status']=='complete'
    assert len(training['models'])==9 and len(regression['models'])==11
    train,dev,data_audit=split_data(args.data)
    result=dict(experiment='RAM02_completed_campaign_analysis',config_sha256=training['config_sha256'],
        source_training_sha256=m.digest(args.training/'training_report.json'),
        source_regression_sha256=m.digest(args.regression/'regression_report.json'),
        calibration=training['calibration'],training_resources=training['resources'],
        regression_resources=regression['resources'],training_models=[],
        regression_models=audit_regression(args.regression,regression),data_audit=data_audit,
        total_optimizer_steps=sum(r['optimizer_steps'] for r in training['models']),
        sum_fit_seconds=sum(r['elapsed_seconds'] for r in training['models']),
        certified_instances=sum(r['certified'] for r in regression['models']),
        distinct_certified_theorems=sorted(set(t for r in regression['models'] for t in r['solved'])))
    fit_times={f"{r['name']}_s{r['seed']}":r['elapsed_seconds'] for r in training['models']}
    for policy in result['regression_models']:
        policy['fit_seconds']=fit_times.get(policy['model'],0.)
        policy['fit_search_replay_seconds_per_certified_instance']=(policy['fit_seconds']+policy['search_seconds']+policy['replay_seconds'])/policy['certified'] if policy['certified'] else None
        policy['cost_per_model_only_theorem']=None if not policy['model_only_vs_native'] else (policy['fit_seconds']+policy['search_seconds']+policy['replay_seconds'])/len(policy['model_only_vs_native'])
    with ResourceGuard(args.out/'resources.jsonl',torch) as guard:
        for row in training['models']:
            checkpoint=checkpoint_path(args.training,row)
            assert m.digest(checkpoint)==row['checkpoints']['best.pt']['sha256']
            assert m.digest(checkpoint_path(args.training,row,'last.pt'))==row['checkpoints']['last.pt']['sha256']
            model=load_checkpoint(checkpoint)
            original=metrics(model,dev)
            assert abs(original['loss']-row['dev']['loss'])<1e-7
            ablations=[]
            if row['name']!='flat64':
                for passes in (1,2,4,8):
                    guard.check()
                    score=pass_benchmark(model,dev,passes)
                    score['metrics']=prune_metrics(score['metrics'])
                    ablations.append(score)
            result['training_models'].append(dict(name=row['name'],seed=row['seed'],
                parameters=row['parameters'],optimizer_steps=row['optimizer_steps'],best_step=row['best_step'],
                elapsed_seconds=row['elapsed_seconds'],updates_per_second=row['updates_per_second'],
                train=prune_metrics(row['train']),dev=prune_metrics(row['dev']),
                checkpoints=row['checkpoints'],cpu_reload_exact=row['cpu_reload_exact'],
                inference_only_pass_ablations=ablations))
        result['analysis_resources']=guard.receipt()
    result['limitations']=[
        'FC-13 is already published and this is post-hoc; no fresh theorem-family claim.',
        'One hardware calibration and one sequential proof run per policy; timing variation is not a robust speed comparison.',
        'Pass ablations change inference depth on four-pass-trained weights, not training depth.',
        'The source-module development split has only 12 states; exact and rcases have one positive training probe each.',
        'Development contains zero positive multi-goal, exact, rcases or induction outcomes; their positive recall is undefined.',
        'Syntactic state signatures do not implement semantic equivalence or complete alpha-equivalence.',
        'The grammar is unchanged and the three-transition/20-second budget can prevent completing branches.',
        'Resource peaks are sampled; torch also records its allocator calibration peak.']
    write_json(args.out/'metrics.json',result)
    certificates=args.out/'certificates'
    certificates.mkdir()
    certificate_index=[]
    for policy in regression['models']:
        for row in m.read_rows(args.regression/(policy['model']+'.jsonl')):
            if row['accepted']:
                final=row['final_certificate']
                source=Path(final['source_file']).read_bytes()
                (certificates/(final['source_sha256']+'.lean')).write_bytes(source)
                certificate_index.append(dict(model=policy['model'],source_file=row['source_file'],
                    theorem=row['theorem'],actions=row['chosen_actions'],source_sha256=final['source_sha256'],
                    axioms=final['axioms'],independent_replay_sha256=row['independent_replay']['source_sha256']))
    write_json(certificates/'index.json',certificate_index)
    manifest={}
    for kind,root in [('training',args.training),('regression',args.regression)]:
        for path in sorted(root.rglob('*')):
            if path.is_file():manifest[kind+'/'+str(path.relative_to(root)).replace('\\','/')]=dict(path=str(path),bytes=path.stat().st_size,sha256=m.digest(path))
    write_json(args.out/'artifact_manifest.json',manifest)
    write_json(args.out/'checkpoints.json',{f"{r['name']}_s{r['seed']}":r['checkpoints'] for r in training['models']})
    print('RAM02_EVIDENCE_AUDIT_COMPLETE',len(manifest),result['certified_instances'],flush=True)

if __name__=='__main__':main()
