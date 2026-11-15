"""AF-06 preregistered scientific judgment, separate from engineering audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import experiment as ex

ELIGIBILITY=.80
MIN_GAIN=.05
MIN_LOCALITY=.90
MIN_OTHER=.90
MAX_CLEAN_DROP=.03
PRIMARY=('held_depth4','held_depth5')


def eligible_metrics(metrics):
    return all(metrics[split]['goal_accuracy']>=ELIGIBILITY and
               metrics[split]['subgoal_exact']>=ELIGIBILITY for split in ('train','dev'))


def judge(entry):
    fit=entry['training']
    warm=eligible_metrics(fit['warmup']['fit_metrics'])
    qualified={arm:eligible_metrics(fit[arm]['fit_metrics']) for arm in ex.ARMS}
    eligibility=warm and qualified['status'] and qualified['ramp']
    result={'eligible':eligibility,'warmup_qualified':warm,
            'continuation_qualified':qualified,'families':{},'verdict':'inconclusive'}
    for split in PRIMARY:
        baseline=entry['patches']['status'][split]
        causal=entry['patches']['ramp'][split]
        abrupt=entry['patches']['abrupt'][split]
        clean_status=entry['scores']['status'][split]
        clean_ramp=entry['scores']['ramp'][split]
        diff=causal['informative_goal_accuracy']-baseline['informative_goal_accuracy']
        conditions={
            'informative_patch_gain':diff>=MIN_GAIN-1e-9,
            'clean_goal_preserved':clean_ramp['goal_accuracy']>=clean_status['goal_accuracy']-MAX_CLEAN_DROP-1e-9,
            'untouched_subgoal':causal['untouched_subgoal_accuracy']>=MIN_LOCALITY-1e-9,
            'other_patch_retention':causal['other_patch_goal_retention']>=MIN_OTHER-1e-9,
            'false_goal_attempts':clean_ramp['false_goal_attempts']<=clean_status['false_goal_attempts']}
        result['families'][split]={
            'status_informative_goal':baseline['informative_goal_accuracy'],
            'ramp_informative_goal':causal['informative_goal_accuracy'],
            'abrupt_informative_goal':abrupt['informative_goal_accuracy'],
            'ramp_minus_status':diff,
            'ramp_minus_abrupt':causal['informative_goal_accuracy']-abrupt['informative_goal_accuracy'],
            'conditions':conditions,'passed':all(conditions.values()),
            'clean_status_goal':clean_status['goal_accuracy'],
            'clean_ramp_goal':clean_ramp['goal_accuracy'],
            'untouched_ramp':causal['untouched_subgoal_accuracy'],
            'other_patch_goal_ramp':causal['other_patch_goal_retention']}
    if eligibility:
        result['verdict']='pass' if all(v['passed'] for v in result['families'].values()) else 'fail'
    return result


def aggregate(root):
    root=Path(root);expected={(seed,width) for seed in ex.SEEDS for width in ex.WIDTHS}
    seen={};source=None
    for path in root.rglob('evaluation.json'):
        item=json.loads(path.read_text());seed,width=item['seed'],item['width']
        if (seed,width) not in expected or (seed,width) in seen or item['smoke']:
            raise ValueError('duplicate, smoke or unexpected run: '+str(path))
        auditpath=path.parent/'independent_audit.json'
        if not auditpath.exists():raise ValueError('missing independent replay evidence')
        audit=json.loads(auditpath.read_text())
        if not audit['passed'] or audit['evaluation_sha256']!=ex.sha_file(path):
            raise ValueError('detached or failed independent replay')
        if audit['clean_rows']!=item['clean_rows'] or audit['counterfactual_rows']!=item['patch_rows']:
            raise ValueError('mismatched row counts')
        if source is None:source=item['source']
        if item['source']!=source or source!=ex.source_receipt():
            raise ValueError('source hash mismatch across seed matrix')
        if set(item['training'])!=set((ex.STAGE_WARM,)+ex.ARMS):
            raise ValueError('training phase roster incomplete')
        if set(item['scores'])!=set(ex.ARMS) or set(item['patches'])!=set(ex.ARMS):
            raise ValueError('score arm roster incomplete')
        for arm in ex.ARMS:
            if item['training'][arm]['steps']!=ex.CONT_STEPS:
                raise ValueError('nonregistered continuation budget')
            addressed=item['training'][arm]['addressed_rows']
            if arm=='status' and addressed!=0:raise ValueError('status baseline addressed patches')
            if arm=='abrupt' and addressed!=ex.CONT_STEPS*ex.BATCH_SIZE:
                raise ValueError('abrupt arm patch budget')
            if arm=='ramp' and not 0<addressed<ex.CONT_STEPS*ex.BATCH_SIZE:
                raise ValueError('ramp did not vary treatment')
        if item['training']['warmup']['steps']!=ex.WARM_STEPS:
            raise ValueError('nonregistered warmup budget')
        if any(set(item['scores'][arm])!=set(('held_composition','held_layout')+PRIMARY) for arm in ex.ARMS):
            raise ValueError('heldout family missing')
        seen[seed,width]=(item,audit)
    if set(seen)!=expected:raise ValueError('incomplete 3x2 matrix: '+str(expected-set(seen)))
    response={'experiment':'AF-06','source':source,'pretraining_steps':ex.WARM_STEPS,
              'continuation_steps':ex.CONT_STEPS,'primary_width':2,
              'scientific_cutoffs':{'goal_and_joint_q':ELIGIBILITY,'informative_patch_gain':MIN_GAIN,
                   'untouched_Q':MIN_LOCALITY,'other_patch_goal':MIN_OTHER,'max_clean_goal_drop':MAX_CLEAN_DROP},
              'runs':{},'clean_rows':0,'patch_rows':0,'trained_models':0,'shared_warmups':0,
              'independent_audit_passed':True}
    primary=[]
    for (seed,width),(item,audit) in sorted(seen.items()):
        judgment=judge(item)
        response['runs'][f's{seed}_w{width}']={
            'seed':seed,'width':width,'judgment':judgment,
            'warmup_dev':item['training']['warmup']['fit_metrics']['dev'],
            'continuation_dev':{arm:item['training'][arm]['fit_metrics']['dev'] for arm in ex.ARMS},
            'heldout':item['scores'],'patches':item['patches'],'audit':audit}
        response['clean_rows']+=item['clean_rows'];response['patch_rows']+=item['patch_rows']
        response['trained_models']+=len(ex.ARMS);response['shared_warmups']+=1
        if width==2:primary.append(judgment['verdict'])
    response['primary_verdict']='fail' if 'fail' in primary else ('pass' if all(j=='pass' for j in primary) else 'inconclusive')
    return response


def markdown(data):
    out=['# AF-06 registered staged-intervention results','',
         f"**Primary width-2 verdict: {data['primary_verdict'].upper()}**. Scientific verdict distinct from CI integrity.",'',
         'Warm-up status supervision is a shared, fixed checkpoint. All three continuations start from identical saved weights and get the same 160-step forward-call budget. The causal-ramp objective uses progressively more addressed semantic interventions; the abrupt arm starts fully addressed.','',
         '| Seed | Width | Warm-up dev goal/Q exact | Status dev goal/Q exact | Ramp dev goal/Q exact | Depth4 status/ramp/abrupt | Depth5 status/ramp/abrupt | Gate |',
         '|---:|---:|---|---|---|---|---|---|']
    for item in data['runs'].values():
        j=item['judgment'];w=item['warmup_dev'];d=item['continuation_dev'];
        def acc(x):return f"{x['goal_accuracy']:.3f}/{x['subgoal_exact']:.3f}"
        def scores(f):return '/'.join(f"{f[x+'_informative_goal']:.3f}" for x in ('status','ramp','abrupt'))
        out.append(f"| {item['seed']} | {item['width']} | {acc(w)} | {acc(d['status'])} | {acc(d['ramp'])} | {scores(j['families']['held_depth4'])} | {scores(j['families']['held_depth5'])} | {j['verdict']} |")
    out.extend(['','The evaluator records all original-goal and inferred-Q predictions and each selected-donor patch. An independent scorer reconstructs both Q statuses and counterfactual goal truth with the fixed finite Horn-closure oracle, checks model checkpoint arrays, and tests whole-state donor replay.','',
      f"Retained scored rows: {data['clean_rows']} clean and {data['patch_rows']} counterfactual. Final models: {data['trained_models']}, shared warm-up checkpoints: {data['shared_warmups']}.",'',
      'The generated Lean Boolean certificate checks the synthetic proof-state oracle; neural proof generation is outside this experiment. An ineligible warm-up or continuation makes its seed result inconclusive.'])
    return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();result=aggregate(a.root)
    ex.json_write(a.out,result);a.report.write_text(markdown(result))
    print(json.dumps({k:result[k] for k in ('primary_verdict','clean_rows','patch_rows','trained_models')},sort_keys=True),flush=True)

if __name__=='__main__':main()
