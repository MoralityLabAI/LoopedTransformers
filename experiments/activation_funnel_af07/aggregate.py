"""AF-07 frozen scientific gates, isolated from numerical and oracle integrity.

The registered result is separate from the GitHub Actions engineering verdict.
Locality scores are independently reconstructed by verify.py before aggregation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import experiment as ex

ELIGIBILITY=0.80
MIN_Q_GAIN=0.10
MIN_STEP_GAIN=0.05
MIN_OTHER=0.90
MAX_CLEAN_DROP=0.03
MAX_PATCH_DROP=0.03
MAX_EXTENDED_DROP=0.02
PRIMARY=('held_depth4','held_depth5')


def eligible_metrics(fit):
    return all(fit[split]['goal_accuracy']>=ELIGIBILITY and
               fit[split]['subgoal_exact']>=ELIGIBILITY for split in ('train','dev'))


def judge(run):
    fit=run['training']
    qualifies={arm:eligible_metrics(fit[arm]['fit_metrics'])
               for arm in ('warmup',)+ex.ARMS}
    eligible=all(qualifies.values())
    result={'eligible':eligible,'qualified_arms':qualifies,'families':{},
            'verdict':'inconclusive'}
    for family in PRIMARY:
        st=run['scores']['status'][family]
        ramp=run['scores']['ramp'][family]
        loc=run['scores']['locality'][family]
        rp=run['patches']['ramp'][family]
        lp=run['patches']['locality'][family]
        r3=run['trajectories']['ramp'][family]['3']
        l3=run['trajectories']['locality'][family]['3']
        l5=run['trajectories']['locality'][family]['5']
        q_gain=lp['untouched_subgoal_accuracy']-rp['untouched_subgoal_accuracy']
        intermediate_gain=sum(l3['protected_q_step_accuracy'])/3-sum(r3['protected_q_step_accuracy'])/3
        conditions={
           'final_untouched_Q_gain':q_gain >= MIN_Q_GAIN-1e-9,
           'mean_stepwise_untouched_Q_gain':intermediate_gain >= MIN_STEP_GAIN-1e-9,
           'clean_goal_retained':loc['goal_accuracy']>=ramp['goal_accuracy']-MAX_CLEAN_DROP-1e-9,
           'addressed_goal_retained':lp['informative_goal_accuracy']>=rp['informative_goal_accuracy']-MAX_PATCH_DROP-1e-9,
           'other_patch_retention':lp['other_patch_goal_retention']>=MIN_OTHER-1e-9 and
                   lp['other_patch_goal_retention']>=rp['other_patch_goal_retention']-0.02-1e-9,
           'extended_downstream_Q_stability':l5['protected_q_step_accuracy'][-1]>=
                   l3['protected_q_step_accuracy'][-1]-MAX_EXTENDED_DROP-1e-9,
           'false_goal_attempts':loc['false_goal_attempts']<=ramp['false_goal_attempts'],
        }
        result['families'][family]={
             'status_clean_goal':st['goal_accuracy'],
             'ramp_clean_goal':ramp['goal_accuracy'],'locality_clean_goal':loc['goal_accuracy'],
             'ramp_untouched_Q':rp['untouched_subgoal_accuracy'],
             'locality_untouched_Q':lp['untouched_subgoal_accuracy'],
             'locality_minus_ramp_untouched_Q':q_gain,
             'locality_minus_ramp_mean_step_Q':intermediate_gain,
             'ramp_informative_goal':rp['informative_goal_accuracy'],
             'locality_informative_goal':lp['informative_goal_accuracy'],
             'locality_irrelevant_goal_retention':lp['other_patch_goal_retention'],
             'locality_depth3_post_steps':l3['protected_q_step_accuracy'],
             'locality_depth5_post_steps':l5['protected_q_step_accuracy'],
             'locality_code_drift':l3['mean_protected_code_l2_by_step'],
             'conditions':conditions,'passed':all(conditions.values())}
    if eligible:
        result['verdict']='pass' if all(f['passed'] for f in result['families'].values()) else 'fail'
    return result


def aggregate(root):
    root=Path(root)
    expected={(seed,width) for seed in ex.SEEDS for width in ex.WIDTHS}
    seen={};source=None
    for p in root.rglob('evaluation.json'):
        item=json.loads(p.read_text());key=(item['seed'],item['width'])
        if key not in expected or key in seen or item['smoke']:
            raise ValueError('unexpected, repeated or smoke result '+str(p))
        audit_path=p.parent/'independent_audit.json'
        if not audit_path.exists():raise ValueError('missing independent replay')
        audit=json.loads(audit_path.read_text())
        if not audit['passed'] or audit['evaluation_sha256']!=ex.sha_file(p) or \
           item['clean_rows']!=audit['clean_rows'] or \
           item['patch_rows']!=audit['counterfactual_rows'] or \
           item['trajectory_rows']!=audit['recurrence_trace_rows']:
            raise ValueError('detached/failed independent record audit')
        if source is None:source=item['source']
        if source!=item['source'] or source!=ex.source_receipt():
            raise ValueError('source differs across runs or from repository')
        if set(item['training'])!=set(('warmup',)+ex.ARMS) or \
           set(item['scores'])!=set(ex.ARMS) or \
           set(item['patches'])!=set(ex.ARMS) or \
           set(item['trajectories'])!=set(ex.ARMS):
            raise ValueError('missing trained arm or scores')
        if item['training']['warmup']['steps']!=ex.WARM_STEPS:
            raise ValueError('nonregistered warmup budget')
        for arm in ex.ARMS:
            fit=item['training'][arm]
            if fit['steps']!=ex.CONT_STEPS or fit['total_patch_rows']!=ex.CONT_STEPS*ex.BATCH_SIZE:
                raise ValueError('nonregistered continuation or forward-call budget')
            if arm=='status' and fit['addressed_rows']!=0:
                raise ValueError('status control was addressed')
            if arm in ('ramp','locality') and fit['addressed_rows']!=ex.CONT_STEPS*ex.BATCH_SIZE//2:
                raise ValueError('ramp/locality did not receive the same exposure budget')
            if set(item['scores'][arm])!=set(('held_composition','held_layout')+PRIMARY):
                raise ValueError('missing heldout family')
            for split in item['scores'][arm]:
                expected_h={'3','5'} if split in PRIMARY else {'3'}
                if set(item['trajectories'][arm][split])!=expected_h:
                    raise ValueError('incomplete traced recurrence horizons')
        seen[key]=(item,audit)
    if set(seen)!=expected:
        raise ValueError('missing seed-width pairs '+str(expected-set(seen)))
    out={'experiment':'AF-07','source':source,'frozen_protocol':'PROTOCOL.json',
         'warm_steps':ex.WARM_STEPS,'continuation_steps':ex.CONT_STEPS,
         'primary_width':2,'scientific_cutoffs':{
             'eligibility_goal_Q':ELIGIBILITY,'untouched_Q_gain':MIN_Q_GAIN,
             'step_Q_gain':MIN_STEP_GAIN,'irrelevant_goal_retention':MIN_OTHER,
             'max_clean_drop':MAX_CLEAN_DROP,'max_patch_drop':MAX_PATCH_DROP,
             'extra_recurrence_Q_drop':MAX_EXTENDED_DROP},
         'runs':{},'clean_rows':0,'patch_rows':0,'trajectory_rows':0,
         'models':0,'warmups':0,'independent_audit_passed':True}
    primary=[]
    for key,(item,audit) in sorted(seen.items()):
        verdict=judge(item)
        out['runs'][f's{key[0]}_w{key[1]}']={
          'seed':key[0],'width':key[1],'judgment':verdict,
          'warmup_dev':item['training']['warmup']['fit_metrics']['dev'],
          'dev':{arm:item['training'][arm]['fit_metrics']['dev'] for arm in ex.ARMS},
          'heldout':item['scores'],'patches':item['patches'],
          'trace_summaries':item['trajectories'],'independent_audit':audit}
        out['clean_rows']+=item['clean_rows'];out['patch_rows']+=item['patch_rows']
        out['trajectory_rows']+=item['trajectory_rows'];out['models']+=len(ex.ARMS);out['warmups']+=1
        if key[1]==2:primary.append(verdict['verdict'])
    out['primary_verdict']='fail' if 'fail' in primary else (
         'pass' if len(primary)==len(ex.SEEDS) and all(v=='pass' for v in primary)
         else 'inconclusive')
    return out


def markdown(data):
    lines=['# AF-07 depth-locality scientific report', '',
       f"**Registered width-2 verdict: {data['primary_verdict'].upper()}**. Engineering integrity and scientific success are separate checks.",
       '',
       'The status and ramp controls receive the same source weights, minibatches and number of native recurrent forwards as the locality arm. The latter adds recurrence-wise constraints on the semantically unaffected Q. The additional KL/MSE computation is reported and is not equal FLOP cost.',
       '',
       '| Seed | Width | Warmup dev goal/Q | Ramp dev goal/Q | Locality dev goal/Q | Depth4 Q ramp → locality | Depth5 Q ramp → locality | Gate |',
       '|---:|---:|---|---|---|---|---|---|']
    for r in data['runs'].values():
        j=r['judgment'];families=j['families']
        fmt=lambda a:f"{a['goal_accuracy']:.3f}/{a['subgoal_exact']:.3f}"
        qfmt=lambda f:f"{f['ramp_untouched_Q']:.3f} → {f['locality_untouched_Q']:.3f}"
        lines.append(f"| {r['seed']} | {r['width']} | {fmt(r['warmup_dev'])} | {fmt(r['dev']['ramp'])} | {fmt(r['dev']['locality'])} | {qfmt(families['held_depth4'])} | {qfmt(families['held_depth5'])} | {j['verdict']} |")
    lines.extend(['', f"Evidence: {data['models']} continuation checkpoints, {data['warmups']} shared warmups, {data['clean_rows']} clean predictions, {data['patch_rows']} original boundary counterfactuals, and {data['trajectory_rows']} recurrence-resolved counterfactual traces.",
       '', 'Unqualified runs remain in the evidence and never count as positive causal results. The Horn oracle and Lean synthetic Boolean proofs certify targets rather than neural theorem proving.'])
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();data=aggregate(a.root)
    ex.json_write(a.out,data);a.report.write_text(markdown(data))
    print(json.dumps({k:data[k] for k in ('primary_verdict','clean_rows','patch_rows','trajectory_rows','models')}),flush=True)

if __name__=='__main__':main()
