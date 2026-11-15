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
    """Verify every one of nine fitted model pairs before first TEST access.

    Original RAM-10 reports store both best and last full CPU checkpoints per
    seed. This audit rehashes all eighteen files, verifies their source/data
    identity and retained optimizer/RNG metadata, and reconstructs the DEV
    selection from nine original TRAIN reports. No TEST file is needed here.
    """
    config_path=Path(config_path);data=Path(data);training=Path(training)
    config=json.loads(config_path.read_text())
    report=json.loads((training/'training_report.json').read_text())
    audit=json.loads((data/'independent_audit.json').read_text())
    gate=json.loads((data/'training_gate.json').read_text())
    collection=json.loads((data/'collection_report.json').read_text())
    dataset_sha=sha_file(data/'transitions.jsonl')
    config_sha=sha_file(config_path)
    if report['status']!='complete' or not gate['admitted'] or audit['status']!='passed':
        raise AssertionError('training/audit not frozen and successful')
    if report.get('frozen_model_count')!=9 or len(report['models'])!=9 or set(report['selected'])!={'loop32','loop64','flat64'}:
        raise AssertionError('missing original nine model fits or fixed model families')
    if collection['status']!='complete' or any(
        c['dataset_sha256']!=dataset_sha for c in (report,audit,gate,collection)):
        raise AssertionError('source compiler dataset digest is inconsistent')
    if any(c['config_sha256']!=config_sha for c in (report,gate,collection)):
        raise AssertionError('original preregistered config digest changed')
    if report.get('original_mathcan_source')!=config['source_mathcan'] or not report.get('development_only_selection'):
        raise AssertionError('original MathCan source or DEV selection receipt changed')
    if report.get('fresh_test_proof_cases_scored')!=0:
        raise AssertionError('TEST goal states were accessed before RAM-10 freeze')
    required={(spec['name'],seed) for spec in config['training']['models'] for seed in config['training']['seeds']}
    observed={(r['name'],r['seed']) for r in report['models']}
    if observed!=required or len(observed)!=len(report['models']):
        raise AssertionError('missing, duplicate or unregistered original CPU fit')
    expected_models={spec['name']:spec for spec in config['training']['models']}
    by_family={name:[] for name in expected_models}
    # Import torch only after source/data/checkpoint manifest checks. The
    # first frozen test preflight can run before the external compiler gym
    # checkout; the original model state dict will be checked again on load.
    for row in report['models']:
        name=row['name'];seed=row['seed']
        prefix=f'{name}_s{seed}'
        relative=f'{prefix}/best.pt'
        if row['relative_best_file']!=relative:
            raise AssertionError('noncanonical original checkpoint path')
        if row['best_step']%config['training']['val_interval'] or row['last_step']%config['training']['val_interval']:
            raise AssertionError('checkpoint at unregistered optimizer step')
        if not (0<row['best_step']<=row['last_step']<=config['training']['updates']):
            raise AssertionError('invalid original optimizer step range')
        if row['parameters']>=250000 or not isinstance(row['best_score'],(float,int)):
            raise AssertionError('invalid compact policy fit metadata')
        best_path=training/relative
        last_path=training/prefix/'last.pt'
        if sha_file(best_path)!=row['best_sha256'] or sha_file(last_path)!=row['last_sha256']:
            raise AssertionError('one or more of nine original best/last checkpoints changed')
        for kind,path,step in [('best',best_path,row['best_step']),('last',last_path,row['last_step'])]:
            ckpt=torch.load(path,map_location='cpu',weights_only=True)
            if ckpt.get('schema')!='RAM10-v1' or ckpt.get('seed')!=seed or ckpt.get('step')!=step:
                raise AssertionError('frozen '+kind+' checkpoint model/step differs')
            if ckpt.get('model_config')!=expected_models[name]:
                raise AssertionError('original frozen model architecture mismatch')
            if ckpt.get('config_sha256')!=config_sha or ckpt.get('data_sha256')!=dataset_sha:
                raise AssertionError('original CPU checkpoint provenance mismatch')
            if ckpt.get('original_mathcan_sha')!=config['source_mathcan']:
                raise AssertionError('original Lean TRAIN source differs in checkpoint')
            if not isinstance(ckpt.get('optimizer'),dict) or ckpt.get('sampler_rng') is None or ckpt.get('torch_rng') is None:
                raise AssertionError('frozen CPU checkpoint lacks original optimizer/RNG evidence')
            if not isinstance(ckpt.get('state_dict'),dict) or not ckpt['state_dict']:
                raise AssertionError('empty CPU policy tensor weights')
        history=[h for h in row['history'] if h['step']==row['best_step']]
        if len(history)!=1 or history[0]['dev']['selection_loss']!=row['best_score']:
            raise AssertionError('original best optimizer step not supported by DEV history')
        by_family[name].append(row)
    fingerprints={}
    for name in expected_models:
        winner=min(by_family[name],key=lambda x:(x['best_score'],x['seed']))
        rec=report['selected'][name]
        if rec!={'seed':winner['seed'],'step':winner['best_step'],
                 'sha256':winner['best_sha256'],'file':winner['relative_best_file']}:
            raise AssertionError('RAM-10 checkpoint selected using changed or non-DEV evidence')
        fingerprints[name]=dict(seed=rec['seed'],sha256=rec['sha256'],
                                file=str(training/rec['file']))
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
