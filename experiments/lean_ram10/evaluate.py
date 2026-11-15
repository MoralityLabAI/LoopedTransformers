"""RAM-10 original TEST-only complete Lean theorem search, with matched controls.

Must be executed in a separate job *after* verifying nine sealed TRAIN fits.
The first access to new heldout MathCan sources is below verify_frozen().
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
import torch
from ram10_core import (sha_file,sha_bytes,rows,stable_json,append_json,source_order_roots,
                        verify_checkout,registered_ram08_core,RAM08)

PHASE_POLICIES={'primary':['native','frequency','shape_frequency','loop32','loop64','flat64'],
                'secondary':['native','frequency','shape_frequency','loop64']}
UNKNOWN={'startup_or_request_timeout','request_timeout','deadline_shutdown_overrun'}


def verify_frozen(config_path,data,training):
    config=json.loads(Path(config_path).read_text());report=json.loads((training/'training_report.json').read_text())
    audit=json.loads((data/'independent_audit.json').read_text())
    gate=json.loads((data/'training_gate.json').read_text())
    if report['status']!='complete' or not gate['admitted'] or audit['status']!='passed':
        raise AssertionError('training/audit not frozen and successful')
    if len(report['models'])!=9 or set(report['selected'])!={'loop32','loop64','flat64'}:
        raise AssertionError('missing new model/seeds before test')
    if report['config_sha256']!=sha_file(config_path) or report['dataset_sha256']!=sha_file(data/'transitions.jsonl'):
        raise AssertionError('new training config/data digest differs')
    if gate['dataset_sha256']!=report['dataset_sha256'] or audit['dataset_sha256']!=report['dataset_sha256']:
        raise AssertionError('compiler observation audit changed')
    fingerprints={}
    for name,info in report['selected'].items():
        path=training/info['file']
        if sha_file(path)!=info['sha256']:
            raise AssertionError('selected CPU checkpoint byte hash changed')
        fingerprints[name]=dict(seed=info['seed'],sha256=info['sha256'],file=str(path))
    return config,report,fingerprints


def scored_summaries(traces):
    table=[]
    for (phase,policy),observations in traces.items():
        native=traces.get((phase,'native'),[])
        native_solved={r['source_file']+':'+r['theorem'] for r in native if r['accepted']}
        successful={r['source_file']+':'+r['theorem'] for r in observations if r['accepted']}
        table.append(dict(phase=phase,policy=policy,cases=len(observations),
               certified=len(successful),solved=sorted(successful),
               new_over_native=sorted(successful-native_solved),
               native_over_model=sorted(native_solved-successful),
               compiler_requests=sum(r['logical_evaluations'] for r in observations),
               lean_processes=sum(r['actual_lean_processes'] for r in observations),
               wall_seconds=sum(r['elapsed_seconds'] for r in observations),
               reasons=dict(Counter(r['reason'] for r in observations))))
    return table


def run(mathcan,data,training,config_path,lean,out):
    mathcan=Path(mathcan);data=Path(data);training=Path(training);out=Path(out)
    if out.exists():raise ValueError('original heldout output already exists; preserve first run')
    config,frozen,fingerprints=verify_frozen(config_path,data,training)
    verify_checkout(mathcan,config,'test')
    core=registered_ram08_core(config)
    if str(RAM08) not in sys.path:sys.path.insert(0,str(RAM08))
    from train import original_model
    from train_local import load_checkpoint as old_load_checkpoint
    from shape_frequency import Frequency,ShapeFrequency
    from search import search
    from runtime import ResourceGuard,guarded_process
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    models={}
    for name,rec in fingerprints.items():
        original_ckpt=torch.load(rec['file'],map_location='cpu',weights_only=True)
        if original_ckpt['schema']!='RAM10-v1' or original_ckpt['seed']!=rec['seed']:
            raise AssertionError('selected checkpoint invalid')
        from model import Policy,Config
        model=Policy(Config(**original_ckpt['model_config']))
        model.load_state_dict(original_ckpt['state_dict']);models[name]=model.eval()
    train_rows=[dict(r,development=False) for r in rows(data/'transitions.jsonl') if r['partition']=='train']
    if not train_rows:raise AssertionError('TRAIN empirical symbolic control missing')
    if any(Path(r['source_file']).stem in config['modules']['test'] for r in train_rows):
        raise AssertionError('new TEST entered source-only baseline')
    frequency=Frequency(train_rows)
    shape=ShapeFrequency(train_rows,prior_weight=config['eval']['frequency_shape_prior'])
    # Only now may this job open any heldout source file, after every hash and
    # frozen checkpoint was independently checked.
    tests=source_order_roots(mathcan,config,'test')
    expected_files=len(config['modules']['test'])
    if len({r['source_file'] for r in tests})!=expected_files:
        raise AssertionError('some registered prospective source module has no original declarations')
    out.mkdir(parents=True,exist_ok=False)
    stable_json(out/'test_manifest.json',dict(config_sha256=sha_file(config_path),
       original_mathcan_commit=config['source_mathcan'],tests=tests,
       selection='first two original top-level theorem/lemma declarations per frozen new module',
       original_dev_in_training=False,source_reference_scripts_as_model_inputs=False))
    traces=defaultdict(list)
    receipt=dict(campaign='RAM10_fresh_original_mathcan_heldout_theorem_search',status='scoring',
         config_sha256=sha_file(config_path),data_sha256=frozen['dataset_sha256'],
         source_commit=config['source_mathcan'],test_count=len(tests),
         selected_models={k:{'seed':v['seed'],'sha256':v['sha256']} for k,v in fingerprints.items()},
         reference_oracles={},source_hashes={},summaries=[],certificates=0)
    stable_json(out/'evaluation_report.json',receipt)
    with ResourceGuard(out/'resources.jsonl',torch) as guard:
        sources={}
        for root in tests:
            fn=root['source_file']
            if fn in sources:continue
            text=(mathcan/fn).read_text(encoding='utf8')
            if sha_bytes(text.encode())!=root['source_sha256']:
                raise AssertionError('raw new source changed after manifest construction')
            sources[fn]=text
            receipt['source_hashes'][fn]=dict(normalized_sha256=core.sha(text),raw_sha256=sha_file(mathcan/fn),
                                              original_git_blob=root['git_blob_sha'])
            original=guarded_process([lean,str((mathcan/fn).resolve())],out/'oracle'/Path(fn).with_suffix('.log'),guard,120.)
            log=original['log']
            if original['returncode']!=0 or original['timed_out'] or 'sorryAx' in log or 'error:' in log:
                raise AssertionError('original registered MathCan TEST source oracle failed '+fn)
            receipt['reference_oracles'][fn]=True
        stable_json(out/'evaluation_report.json',receipt)
        for phase,policies in PHASE_POLICIES.items():
            options={'native':None,'frequency':frequency,'shape_frequency':shape,
                     'loop32':models['loop32'],'loop64':models['loop64'],'flat64':models['flat64']}
            for i,root in enumerate(tests):
                names=list(policies)
                offset=i%len(names);names=names[offset:]+names[:offset]
                for policy in names:
                    evidence=out/'evidence'/phase/f'{i:02d}'/policy
                    observation=search(lean,sources[root['source_file']],root['theorem'],options[policy],
                          evidence,guard,config['eval'][phase])
                    observation.update(phase=phase,policy=policy,source_file=root['source_file'],
                          theorem=root['theorem'],declaration_kind=root['declaration_kind'],
                          checkpoint=receipt['selected_models'].get(policy))
                    append_json(out/f'{phase}__{policy}.jsonl',observation)
                    traces[phase,policy].append(observation)
                    print('RAM10_TEST_PROOF',json.dumps(dict(phase=phase,policy=policy,
                           theorem=root['theorem'],accepted=observation['accepted'],
                           reason=observation['reason'],evaluations=observation['logical_evaluations'])),flush=True)
            receipt['summaries']=scored_summaries(traces)
            stable_json(out/'evaluation_report.json',receipt)
        receipt['summaries']=scored_summaries(traces)
        primary={name:traces['primary',name] for name in PHASE_POLICIES['primary']}
        def good(policy):return {r['source_file']+':'+r['theorem'] for r in primary[policy] if r['accepted']}
        symbolic=set().union(*(good(p) for p in ('native','frequency','shape_frequency')))
        recurrent=good('loop32')|good('loop64')
        candidates=recurrent-symbolic
        uncertain_baseline={r['source_file']+':'+r['theorem']
            for name in ('native','frequency','shape_frequency')
            for r in primary[name] if r['reason'] in UNKNOWN}
        receipt['candidate_recurrent_advantage']=sorted(candidates)
        receipt['certified_recurrent_advantage']=sorted(candidates-uncertain_baseline)
        receipt['registered_primary_met']=bool(candidates-uncertain_baseline)
        receipt['recurrence_specific_advantage']=sorted((candidates-uncertain_baseline)-good('flat64'))
        receipt['baseline_unknown_cases']=sorted(uncertain_baseline)
        receipt['status']='complete'
        receipt['resource_report']=guard.receipt()
    stable_json(out/'evaluation_report.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mathcan',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--lean',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.mathcan,a.data,a.training,a.config,a.lean,a.out)
