"""Audit completed RAM-03 observations, weights and independent Lean certificates."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import torch
import ram03_core as c
from runtime import ResourceGuard

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repair-audit',type=Path,help='Additional complete corrected-source replay audit, if applicable')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    training=json.loads((a.training/'training_report.json').read_text())
    evaluation=json.loads((a.evaluation/'evaluation_report.json').read_text())
    audit=json.loads((a.data/'independent_audit.json').read_text())
    assert training['status']==evaluation['status']=='complete' and audit['status']=='passed'
    assert audit['checked']==len(c.read_rows(a.data/'transitions.jsonl')) and audit['dataset_sha256']==c.digest(a.data/'transitions.jsonl')
    repair_audit=None
    if a.repair_audit:
        repair_audit=json.loads(a.repair_audit.read_text())
        assert repair_audit['status']=='passed' and repair_audit['checked']==audit['checked']
        assert repair_audit['dataset_sha256']==audit['dataset_sha256']
    result=dict(experiment='RAM03',training=training,evaluation=evaluation,data_audit=audit,corrected_source_audit=repair_audit,
      collection=json.loads((a.data/'collection_report.json').read_text()),
      provenance=dict(protocol_commit='1e791ce',model_search_commit='f0e0fac',selection_commit='2756a59',
                      evaluation_import_fix_commit='90520b1',
                      evaluation_implementation_repair='Source-option placement and per-request heartbeat isolation; evaluation-01 invalidated; all weights, data bytes and policy budgets frozen'),
      training_report_sha256=c.digest(a.training/'training_report.json'),
      evaluation_report_sha256=c.digest(a.evaluation/'evaluation_report.json'),
      total_training_updates=sum(r['updates'] for r in training['models']),
      summed_fit_seconds=sum(r['seconds'] for r in training['models']),
      certificates=[],family_search_counts={},source_manifest='artifact_manifest.jsonl')
    certificates=a.out/'certificates';certificates.mkdir()
    with ResourceGuard(a.out/'resources.jsonl',torch) as guard:
        from train_local import load_checkpoint,metrics,data_tensors,load_data
        config=training['config'];train,dev,_=load_data(a.data,config,training['config_sha256']);weights=data_tensors(train)[3]
        for model in training['models']:
            path=Path(model['checkpoint']);last=path.with_name('last.pt')
            assert c.digest(path)==model['checkpoint_sha256'] and c.digest(last)==model['last_checkpoint_sha256']
            restored=load_checkpoint(path)
            assert abs(metrics(restored,dev,weights)['selection_loss']-model['dev']['selection_loss'])<1e-6
        instances=0;trajectories=0;unique=defaultdict(set);family_counts=defaultdict(Counter);maximum_elapsed=0.
        for path in sorted(a.evaluation.glob('*__*__*.jsonl')):
            experiment,suite,model=path.stem.split('__');rows=c.read_rows(path)
            assert len(rows)==(12 if suite=='prospective' else 16)
            for row in rows:
                guard.check();trajectories+=1;budget=row['budget'];maximum_elapsed=max(maximum_elapsed,row['elapsed_seconds'])
                assert row['logical_evaluations']==len(row['calls'])<=budget['evaluations']
                assert row['max_path_transitions']<=budget['transitions']
                assert row['actual_lean_processes']<=5
                for call in row['calls']:
                    group=family_counts[(experiment,suite,model,call['family'])]
                    group['attempts']+=1;group['observed_progress']+=int(call.get('outcome',0)>0)
                    group['branch_creation']+=int(call.get('creates_branches',False))
                    group['goal_reduction']+=int(call.get('closes_goals',False))
                if not row['accepted']:continue
                assert row['elapsed_seconds']<=budget['seconds']
                final,replay=row['certificate'],row['replay']
                assert final['accepted'] and replay['accepted'] and final['axiom_gate'] and replay['axiom_gate']
                assert set(final['axioms'])<={'propext','Classical.choice','Quot.sound'}
                assert final['source_sha256']==replay['source_sha256']
                assert c.digest(final['source_file'])==final['source_sha256'] and c.digest(replay['source_file'])==replay['source_sha256']
                assert c.digest(final['log_file'])==final['log_sha256'] and c.digest(replay['log_file'])==replay['log_sha256']
                source=Path(final['source_file']).read_text(encoding='utf8')
                assert 'ProbeServer' not in source and 'all_goals sorry' not in source
                destination=certificates/(final['source_sha256']+'.lean')
                shutil.copyfile(final['source_file'],destination)
                identity=row['source_file']+':'+row['theorem'];unique[suite].add(identity);instances+=1
                result['certificates'].append(dict(experiment=experiment,suite=suite,model=model,theorem=identity,
                   actions=row['actions'],source_sha256=final['source_sha256'],replay_sha256=replay['source_sha256'],axioms=final['axioms']))
        assert trajectories==196
        result.update(trajectories=trajectories,certified_instances=instances,
          unique_theorems={k:sorted(v) for k,v in unique.items()},maximum_observed_search_seconds=maximum_elapsed,
          family_search_counts=[dict(experiment=k[0],suite=k[1],model=k[2],family=k[3],**v) for k,v in family_counts.items()],
          analysis_resources=guard.receipt())
        manifest=a.out/'artifact_manifest.jsonl';files=0;total=0
        for kind,root in [('data',a.data),('training',a.training),('evaluation',a.evaluation)]:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    guard.check();entry=dict(kind=kind,relative_path=path.relative_to(root).as_posix(),
                      absolute_path=str(path),bytes=path.stat().st_size,sha256=c.digest(path))
                    c.append_json(manifest,entry);files+=1;total+=entry['bytes']
        result['artifact_files']=files;result['artifact_bytes']=total;result['artifact_manifest_sha256']=c.digest(manifest)
    c.write_json(a.out/'metrics.json',result);c.write_json(certificates/'index.json',result['certificates'])
    c.write_json(a.out/'checkpoints.json',{f"{r['name']}_s{r['seed']}":dict(best=r['checkpoint'],best_sha256=r['checkpoint_sha256'],
       last=str(Path(r['checkpoint']).with_name('last.pt')),last_sha256=r['last_checkpoint_sha256']) for r in training['models']})
    print('RAM03_FINAL_AUDIT',files,trajectories,instances,flush=True)
if __name__=='__main__':main()
