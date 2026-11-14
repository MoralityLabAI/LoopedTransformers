"""Frozen prospective and post-hoc RAM-04 comparison against matched native search."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import subprocess
import torch
import ram04_core as c
from runtime import ResourceGuard,guarded_process
from search import search
from train_local import load_checkpoint,load_data

class Frequency:
    def __init__(self,rows):
        groups=defaultdict(list)
        for r in rows:
            if not r['development']:
                reward={0:-.6,1:1.5,2:.3,3:3}[r['label']]+.4*r['delta']
                groups[len(r['before']['goals'])>1,r['family']].append(reward)
        self.table={k:sum(v)/len(v) for k,v in groups.items()}
    def rank(self,state,pool):return sorted(pool,key=lambda p:-self.table.get((len(state['goals'])>1,p[0]),-.6))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mathcan',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lean',required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--experiment',choices=('both','primary','secondary'),default='both',help='Run an entire registered budget comparison')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text());training=json.loads((a.training/'training_report.json').read_text())
    assert training['status']=='complete' and training['config_sha256']==c.digest(a.config)
    assert subprocess.check_output(['git','-C',str(a.mathcan),'rev-parse','HEAD'],text=True).strip()=='c84e2025334e9a967c1297425a490f98d5b3d189'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    suites={'prospective':config['prospective']}
    models={};fingerprints={}
    for name,receipt in training['selected'].items():
        checkpoint=Path(receipt['checkpoint'])
        assert c.digest(checkpoint)==receipt['checkpoint_sha256']
        models[name]=load_checkpoint(checkpoint);fingerprints[name]=dict(seed=receipt['seed'],sha256=receipt['checkpoint_sha256'])
    frequency_train,_,_=load_data(a.data,config,c.digest(a.config))
    frequency=Frequency(frequency_train)
    freeze=json.loads((a.training/'freeze.json').read_text(encoding='utf8'))
    assert freeze['selected_models']==fingerprints and freeze['dataset_sha256']==training['data']['dataset_sha256']
    for filename,sha in freeze['files'].items():assert c.digest(Path(__file__).with_name(filename))==sha,('Frozen implementation changed',filename)
    report=dict(experiment='RAM04',config_sha256=c.digest(a.config),dataset_sha256=training['data']['dataset_sha256'],
      selected_models=fingerprints,status='running',summaries=[],original_oracles={},source_hashes={},requested_experiment=a.experiment,pre_evaluation_freeze=freeze)
    trajectories=defaultdict(list)
    try:
        with ResourceGuard(a.out/'resources.jsonl',torch) as guard:
            # These originals are read only after the model/search freeze.
            sources={}
            for suite,modules in suites.items():
                for filename,names in modules.items():
                    source=(a.mathcan/filename).read_text(encoding='utf8');sources[filename]=source
                    if suite=='prospective':
                        declared=re.findall(r'^theorem\s+([A-Za-z_0-9\']+)',source,re.M)
                        assert [declared[i] for i in config['prospective_positions']]==list(names),'prospective source-order selection changed'
                    report['source_hashes'][filename]=dict(normalized_sha256=c.sha(source),raw_sha256=c.digest(a.mathcan/filename))
                    result=guarded_process([a.lean,'-j','2','-DElab.async=false',str((a.mathcan/filename).resolve())],a.out/'oracle'/Path(filename).with_suffix('.log'),guard,120.)
                    if result['returncode'] or result['timed_out'] or 'sorryAx' in result['log'] or 'error:' in result['log']:
                        raise AssertionError('original source oracle failed '+filename)
                report['original_oracles'][suite]=sum(map(len,modules.values()))
            c.write_json(a.out/'evaluation_report.json',report)
            for experiment in (('primary','secondary') if a.experiment=='both' else (a.experiment,)):
                policies={'native':None,'loop64':models['loop64']}
                if experiment=='primary':policies={'native':None,'frequency':frequency,**models}
                for suite,modules in suites.items():
                    cases=[(f,n) for f,names in modules.items() for n in names]
                    for index,(filename,theorem) in enumerate(cases):
                        names=list(policies);offset=index%len(names);names=names[offset:]+names[:offset]
                        for name in names:
                            row=search(a.lean,sources[filename],theorem,policies[name],
                              a.out/'evidence'/experiment/suite/f'{index:02d}'/name,guard,config[experiment])
                            row.update(experiment=experiment,suite=suite,model=name,source_file=filename,theorem=theorem,
                                       checkpoint=fingerprints.get(name))
                            key=experiment+'__'+suite+'__'+name
                            trajectories[key].append(row);c.append_json(a.out/(key+'.jsonl'),row)
                            print('RAM04_PROOF',json.dumps({k:row[k] for k in ('experiment','suite','model','theorem','accepted','reason','logical_evaluations','actual_lean_processes','elapsed_seconds')}),flush=True)
                report['summaries']=summarize(trajectories)
                c.write_json(a.out/'evaluation_report.json',report)
            report['summaries']=summarize(trajectories);report['status']='complete';report['resources']=guard.receipt()
    except BaseException as error:
        report.update(status='aborted',error=repr(error),summaries=summarize(trajectories))
        c.write_json(a.out/'evaluation_report.json',report);raise
    c.write_json(a.out/'evaluation_report.json',report)

def summarize(trajectories):
    result=[]
    for key,rows in trajectories.items():
        experiment,suite,model=key.split('__');native=trajectories.get(experiment+'__'+suite+'__native',[])
        native_solved={r['source_file']+':'+r['theorem'] for r in native if r['accepted']}
        solved={r['source_file']+':'+r['theorem'] for r in rows if r['accepted']}
        elapsed=sum(r['elapsed_seconds'] for r in rows)
        result.append(dict(experiment=experiment,suite=suite,model=model,cases=len(rows),certified=len(solved),
           solved=sorted(solved),model_only_vs_native=sorted(solved-native_solved),native_only_vs_model=sorted(native_solved-solved),
           total_seconds=elapsed,seconds_per_independent_certificate=elapsed/len(solved) if solved else None,
           logical_evaluations=sum(r['logical_evaluations'] for r in rows),actual_lean_processes=sum(r['actual_lean_processes'] for r in rows),
           reasons=dict(Counter(r['reason'] for r in rows)),multi_goal_transitions=sum(r['diagnostics'].get('multi_goal_transitions',0) for r in rows)))
    return result
if __name__=='__main__':main()
