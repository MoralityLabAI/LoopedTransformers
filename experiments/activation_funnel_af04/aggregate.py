"""AF-04 cross-seed strict scientific gates, independent of CI integrity status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import gym
from . import experiment as ex

ELIGIBILITY=.80
MIN_GAIN=.05
MIN_LOCALITY=.90
MIN_OTHER=.90
MAX_TASK_DROP=.03
PRIMARY=('held_composition','held_depth')


def eligible(entry):
    return all(entry['training'][arm]['fit_metrics'][split]['goal_accuracy']>=ELIGIBILITY and
               entry['training'][arm]['fit_metrics'][split]['subgoal_exact']>=ELIGIBILITY
               for arm in ('status','causal') for split in ('train','dev'))


def judge(entry):
    result={'eligible':eligible(entry),'families':{},'verdict':'inconclusive'}
    for split in PRIMARY:
        b=entry['patches']['status'][split]
        c=entry['patches']['causal'][split]
        task_b=entry['scores']['status'][split]
        task_c=entry['scores']['causal'][split]
        delta=c['informative_goal_accuracy']-b['informative_goal_accuracy']
        conditions={
            'counterfactual_gain':delta>=MIN_GAIN-1e-9,
            'clean_goal_preservation':task_c['goal_accuracy']>=task_b['goal_accuracy']-MAX_TASK_DROP-1e-9,
            'untouched_subgoal':c['untouched_subgoal_accuracy']>=MIN_LOCALITY-1e-9,
            'other_patch_goal_retention':c['other_patch_goal_retention']>=MIN_OTHER-1e-9,
            'false_goal_attempts':task_c['false_goal_attempts']<=task_b['false_goal_attempts']
        }
        result['families'][split]={'status_informative_goal':b['informative_goal_accuracy'],
                                  'causal_informative_goal':c['informative_goal_accuracy'],
                                  'gain':delta,'status_clean_goal':task_b['goal_accuracy'],
                                  'causal_clean_goal':task_c['goal_accuracy'],
                                  'causal_untouched':c['untouched_subgoal_accuracy'],
                                  'conditions':conditions,
                                  'passed':all(conditions.values())}
    if result['eligible']:
        result['verdict']='pass' if all(x['passed'] for x in result['families'].values()) else 'fail'
    return result


def aggregate(root):
    root=Path(root)
    expected={(seed,width) for seed in ex.SEEDS for width in ex.WIDTHS}
    seen={};source=None
    for path in root.rglob('evaluation.json'):
        entry=json.loads(path.read_text())
        seed,width=entry['seed'],entry['width']
        if (seed,width) not in expected or (seed,width) in seen or entry['smoke']:
            raise ValueError('unregistered, duplicated, or smoke evaluation found: '+str(path))
        auditpath=path.parent/'independent_audit.json'
        if not auditpath.exists():raise ValueError('missing independent audit: '+str(path))
        audit=json.loads(auditpath.read_text())
        if not audit['passed'] or audit['evaluation_sha256']!=ex.sha_file(path):
            raise ValueError('failed or disconnected independent audit')
        if audit['clean_rows']!=entry['clean_rows'] or audit['counterfactual_rows']!=entry['patch_rows']:
            raise ValueError('scored row count does not match independent verifier')
        if source is None:source=entry['source']
        if source!=entry['source'] or source!=ex.source_receipt():
            raise ValueError('source-hash disagreement across matrix')
        if set(entry['scores'])!=set(ex.ARMS) or set(entry['patches'])!=set(ex.ARMS):
            raise ValueError('missing arm')
        if any(set(entry['scores'][a])!=set(PRIMARY+('held_layout',)) for a in ex.ARMS):
            raise ValueError('missing heldout split')
        seen[(seed,width)]=(entry,audit)
    if set(seen)!=expected:raise ValueError('incomplete 3x2 evidence matrix '+str(set(expected)-set(seen)))
    summary={'experiment':'AF-04','source':source,'corpus_manifest':ex.sha_file(ex.PINNED_MANIFEST),
             'scientific_cutoffs':{'eligibility_goal_and_q_exact':ELIGIBILITY,
                 'informative_goal_gain':MIN_GAIN,'untouched_Q':MIN_LOCALITY,
                 'other_patch_goal':MIN_OTHER,'max_clean_goal_drop':MAX_TASK_DROP},
             'runs':{},'independent_audit_passed':True,
             'clean_rows':0,'patch_rows':0,'training_models':0,
             'registered_primary_width':2}
    primary=[]
    for seed,width in sorted(seen):
        entry,audit=seen[(seed,width)]
        jud=judge(entry)
        summary['runs'][f's{seed}_w{width}']={'seed':seed,'width':width,
             'judgment':jud,'development':{a:entry['training'][a]['fit_metrics']['dev'] for a in ex.ARMS},
             'heldout':{a:entry['scores'][a] for a in ex.ARMS},
             'patches':{a:entry['patches'][a] for a in ex.ARMS},
             'audit':audit}
        if width==2:primary.append(jud['verdict'])
        summary['clean_rows']+=entry['clean_rows'];summary['patch_rows']+=entry['patch_rows']
        summary['training_models']+=len(ex.ARMS)
    summary['primary_verdict']='fail' if 'fail' in primary else ('pass' if all(x=='pass' for x in primary) else 'inconclusive')
    assert len(primary)==3
    return summary


def markdown(summary):
    lines=['# AF-04 final registered proof-state interchange results','',
           '**Engineering:** independent scored-row and model replay passed for all six seed/width configurations.',
           f"**Primary tenet (width 2): {summary['primary_verdict'].upper()}**; experimental outcome judged separately from CI.",
           '', '| Seed | Width | Dev status goal/Q | Dev causal goal/Q | Composition patch baseline/causal | Depth3 patch baseline/causal | Verdict |',
           '|---:|---:|---|---|---|---|---|']
    for item in summary['runs'].values():
        j=item['judgment'];d=item['development'];
        def f(x):return f'{x:.3f}'
        status=d['status'];causal=d['causal']
        comp=j['families']['held_composition'];depth=j['families']['held_depth']
        lines.append(f"| {item['seed']} | {item['width']} | {f(status['goal_accuracy'])}/{f(status['subgoal_exact'])} | {f(causal['goal_accuracy'])}/{f(causal['subgoal_exact'])} | {f(comp['status_informative_goal'])}/{f(comp['causal_informative_goal'])} | {f(depth['status_informative_goal'])}/{f(depth['causal_informative_goal'])} | {j['verdict']} |")
    lines.extend(['','The semantic oracle computes intermediate proof derivability by Horn least fixed point from public initial facts and declared rules. During the second phase the model receives only downstream dependencies. Counterfactual labels do(Q_j := donor Q_j) are recomputed from recipient other-state and downstream AND/OR rules.','',
        'This is a synthetic forward-Horn proof-state study, with generated Boolean theorems checked by Lean; it is distinct from a claim of original Lean theorem proving by the model.',
        '',f"Rows: {summary['clean_rows']} clean, {summary['patch_rows']} counterfactual, across {summary['training_models']} trained models.",'',
        'The gate requires two eligible arms on both training and development for each seed. Eligible failures make the aggregate fail; otherwise missing eligibility makes the result inconclusive.'] )
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    answer=aggregate(a.root)
    ex.json_write(a.out,answer)
    a.report.write_text(markdown(answer))
    print(json.dumps({k:answer[k] for k in ('primary_verdict','clean_rows','patch_rows','training_models')},sort_keys=True),flush=True)

if __name__=='__main__':main()
