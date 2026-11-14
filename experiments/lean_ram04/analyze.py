"""Audit RAM-04 weights, all scored trajectories and complete Lean certificates."""
import argparse,json,hashlib,shutil
from pathlib import Path
from collections import Counter,defaultdict
import torch
import ram04_core as c
from runtime import ResourceGuard

def stream_digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('data','training','evaluation','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    train_report=json.loads((a.training/'training_report.json').read_text(encoding='utf8'))
    evaluation=json.loads((a.evaluation/'evaluation_report.json').read_text(encoding='utf8'))
    audit=json.loads((a.data/'independent_audit.json').read_text(encoding='utf8'))
    assert train_report['status']==evaluation['status']=='complete' and audit['status']=='passed'
    assert audit['dataset_sha256']==train_report['data']['dataset_sha256']==c.digest(a.data/'transitions.jsonl')
    assert audit['checked']==len(c.read_rows(a.data/'transitions.jsonl'))
    certificates=a.out/'certificates';certificates.mkdir();index=[];diagnostics=[];families=defaultdict(Counter)
    result=dict(campaign='RAM04',training=train_report,evaluation=evaluation,data_audit=audit,
        collection=json.loads((a.data/'collection_report.json').read_text(encoding='utf8')))
    with ResourceGuard(a.out/'resources.jsonl',torch) as guard:
        from train_local import load_data,load_checkpoint,data_tensors,metrics
        train,dev,data=load_data(a.data,train_report['config'],train_report['config_sha256']);weights=data_tensors(train)[3]
        checkpoints={}
        for model in train_report['models']:
            best=Path(model['checkpoint']);last=best.with_name('last.pt')
            assert stream_digest(best)==model['checkpoint_sha256'] and stream_digest(last)==model['last_checkpoint_sha256']
            assert abs(metrics(load_checkpoint(best),dev,weights)['selection_loss']-model['dev']['selection_loss'])<1e-6
            checkpoints[f"{model['name']}_s{model['seed']}"]=dict(best=str(best),best_sha256=stream_digest(best),last=str(last),last_sha256=stream_digest(last))
        trajectories=0;solved=set();counts=Counter()
        paths=sorted(a.evaluation.glob('*__prospective__*.jsonl'));assert len(paths)==7
        for path in paths:
            experiment,suite,policy=path.stem.split('__');rows=c.read_rows(path);assert len(rows)==12
            events=Counter();reasons=Counter();seconds=0.;verification=0.
            for r in rows:
                guard.check();trajectories+=1;budget=r['budget'];reasons[r['reason']]+=1;events.update(r['diagnostics'])
                assert len(r['calls'])==r['logical_evaluations']<=budget['evaluations']
                assert r['max_path_transitions']<=budget['transitions'] and r['actual_lean_processes']<=5
                seconds+=r['elapsed_seconds'];verification+=sum((r.get(k) or {}).get('wall_seconds',0.) for k in ('certificate','replay'))
                for call in r['calls']:
                    f=families[(experiment,policy,call['family'])];f['attempts']+=1;f['progress']+=int(call.get('outcome',0)>0)
                    f['branches']+=int(call.get('creates_branches',False));f['goal_reductions']+=int(call.get('closes_goals',False))
                if not r['accepted']:continue
                assert r['elapsed_seconds']<=budget['seconds'];cert,replay=r['certificate'],r['replay']
                assert cert['accepted'] and replay['accepted'] and cert['axiom_gate'] and replay['axiom_gate']
                assert set(cert['axioms'])<={'propext','Classical.choice','Quot.sound'}
                assert cert['source_sha256']==replay['source_sha256']
                for stage in (cert,replay):
                    assert stream_digest(stage['source_file'])==stage['source_sha256'] and stream_digest(stage['log_file'])==stage['log_sha256']
                text=Path(cert['source_file']).read_text(encoding='utf8');assert 'ProbeServer' not in text and 'all_goals sorry' not in text
                shutil.copyfile(cert['source_file'],certificates/(cert['source_sha256']+'.lean'))
                theorem=r['source_file']+':'+r['theorem'];solved.add(theorem);counts[experiment,policy]+=1
                index.append(dict(experiment=experiment,policy=policy,theorem=theorem,actions=r['actions'],axioms=cert['axioms'],source_sha256=cert['source_sha256']))
            diagnostics.append(dict(experiment=experiment,policy=policy,reasons=dict(reasons),events=dict(events),seconds=seconds,verification_seconds=verification))
        assert trajectories==84
        manifest=a.out/'artifact_manifest.jsonl';files=0;size=0
        with manifest.open('w',encoding='utf8',newline='\n') as handle:
            for kind,root in [('data',a.data),('training',a.training),('evaluation',a.evaluation)]:
                for path in sorted(root.rglob('*')):
                    if not path.is_file():continue
                    guard.check();entry=dict(kind=kind,relative_path=path.relative_to(root).as_posix(),absolute_path=str(path),bytes=path.stat().st_size,sha256=stream_digest(path))
                    handle.write(json.dumps(entry,ensure_ascii=False)+'\n');files+=1;size+=entry['bytes']
        result.update(trajectories=trajectories,certified_instances=len(index),unique_theorems=sorted(solved),certificates=index,
          family_search=[dict(experiment=k[0],policy=k[1],family=k[2],**v) for k,v in families.items()],search_diagnostics=diagnostics,
          artifact_files=files,artifact_bytes=size,artifact_manifest_sha256=stream_digest(manifest),analysis_resources=guard.receipt(),
          total_updates=sum(r['updates'] for r in train_report['models']),fit_seconds=sum(r['seconds'] for r in train_report['models']))
    c.write_json(a.out/'metrics.json',result);c.write_json(a.out/'checkpoints.json',checkpoints);c.write_json(certificates/'index.json',index)
    print('RAM04_FINAL_AUDIT',files,trajectories,len(index),len(solved),flush=True)
if __name__=='__main__':main()
