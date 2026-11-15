"""RAM-06 first exact curriculum transfer versus native and strong typed symbolic controls."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import subprocess
import torch
import ram08_core as c
from runtime import ResourceGuard,guarded_process
from search import search
from train_local import load_checkpoint,load_data

from shape_frequency import Frequency, ShapeFrequency

EXACT_CASES={
    "MathCan/CoarseStateDependence.lean":(
        "two_parent_covariance_numerator","joint_exceeds_product_marginal",
        "shared_critical_linear_growth"),
    "MathCan/OrderedJet.lean":(
        "scalar_order_invisible","adjacent_weighted_numerator",
        "adjacent_coefficient_numerator"),
    "MathCan/TacticFibers.lean":(
        "exactly_two_matching_orbit","diagonal_is_legal","shifted_is_legal"),
    "MathCan/TreeReconstruction.lean":(
        "weights_sum_to_fifteen","contrast_transfer_eigenvalue",
        "moment_gap_identity"),
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mathcan',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lean',required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--experiment',choices=('both','primary','secondary'),default='both',help='Run an entire registered budget comparison')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text());training=json.loads((a.training/'training_report.json').read_text())
    assert training['status']=='complete' and training['config_sha256']==c.digest(a.config)
    assert subprocess.check_output(['git','-C',str(a.mathcan),'rev-parse','HEAD'],text=True).strip()=='fca46bfb3430ac85ffca16f6a6a83bb0b0231e41'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

    assert config['source_test_commit']=='fca46bfb3430ac85ffca16f6a6a83bb0b0231e41'
    assert config['source_train_commit']=='c84e2025334e9a967c1297425a490f98d5b3d189'
    assert config['registration']=="e2a717aca927221f21fd3df9b47520b6f74f1f7c"
    assert {f:tuple(v) for f,v in config['prospective'].items()}==EXACT_CASES
    suites={'prospective_ram08':EXACT_CASES}
    models={};fingerprints={}
    for name,receipt in training['selected'].items():
        # CI training and heldout evaluation use separate runners. Preserve
        # recorded checkpoint BYTE hashes while re-rooting from the old
        # absolute Windows/Linux training path to the downloaded artifact.
        checkpoint=a.training/f"{name}_s{receipt['seed']}"/'best.pt'
        assert c.digest(checkpoint)==receipt['checkpoint_sha256']
        models[name]=load_checkpoint(checkpoint)
        fingerprints[name]=dict(seed=receipt['seed'],
                                sha256=receipt['checkpoint_sha256'])
    frequency_train,_,_=load_data(a.data,config,c.digest(a.config))
    assert all(not row['development'] for row in frequency_train)
    frequency=Frequency(frequency_train)
    shape_frequency=ShapeFrequency(frequency_train,prior_weight=8)
    report=dict(experiment='RAM08',config_sha256=c.digest(a.config),dataset_sha256=training['data']['dataset_sha256'],
      selected_models=fingerprints,status='running',summaries=[],original_oracles={},source_hashes={},requested_experiment=a.experiment)
    trajectories=defaultdict(list)
    try:
        with ResourceGuard(a.out/'resources.jsonl',torch) as guard:
            # These originals are read only after the model/search freeze.
            sources={}
            for suite,modules in suites.items():
                for filename,names in modules.items():
                    source=(a.mathcan/filename).read_text(encoding='utf8');sources[filename]=source
                    if suite=='prospective_ram08':
                        declared=re.findall(r'^theorem\s+([A-Za-z_0-9\']+)',source,re.M)
                        assert list(declared[:3])==list(names),'prospective source-order selection changed'
                    report['source_hashes'][filename]=dict(normalized_sha256=c.sha(source),raw_sha256=c.digest(a.mathcan/filename))
                    result=guarded_process([a.lean,str((a.mathcan/filename).resolve())],a.out/'oracle'/Path(filename).with_suffix('.log'),guard,120.)
                    if result['returncode'] or result['timed_out'] or 'sorryAx' in result['log'] or 'error:' in result['log']:
                        raise AssertionError('original source oracle failed '+filename)
                report['original_oracles'][suite]=sum(map(len,modules.values()))
            c.write_json(a.out/'evaluation_report.json',report)
            for experiment in (('primary','secondary') if a.experiment=='both' else (a.experiment,)):
                policies={'native':None,'frequency':frequency,
                          'shape_frequency':shape_frequency,'loop64':models['loop64']}
                if experiment=='primary':
                    policies={'native':None,'frequency':frequency,
                              'shape_frequency':shape_frequency,**models}
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
                            print('RAM08_PROOF',json.dumps({k:row[k] for k in ('experiment','suite','model','theorem','accepted','reason','logical_evaluations','actual_lean_processes','elapsed_seconds')}),flush=True)
                report['summaries']=summarize(trajectories)
                c.write_json(a.out/'evaluation_report.json',report)
            report['summaries']=summarize(trajectories);report['status']='complete';report['resources']=guard.receipt()
            primary={x['model']:x for x in report['summaries']
                     if x['experiment']=='primary' and x['suite']=='prospective_ram08'}
            assert set(primary)=={'native','frequency','shape_frequency',
                                  'loop32','loop64','flat64'}
            symbolic=set().union(*(set(primary[name]['solved'])
                   for name in ('native','frequency','shape_frequency')))
            recurrent=set().union(*(set(primary[name]['solved'])
                   for name in ('loop32','loop64')))
            report['registered_model_only_success']=bool(recurrent-symbolic)
            report['registered_recurrence_specific_success']=bool(
                   recurrent-(symbolic|set(primary['flat64']['solved'])))
            report['new_recurrent_theorems']=sorted(recurrent-symbolic)
            report['strongest_non_neural_certified_count']=max(
                 primary[name]['certified']
                 for name in ('native','frequency','shape_frequency'))
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
