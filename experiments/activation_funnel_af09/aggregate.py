"""AF-09 preregistered cross-seed scientific scorer and evidence reconciler."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import experiment as ex

TOL=1e-10
FAMILIES=('held_depth4','held_depth5')


def aggregate(root):
    root=Path(root)
    runs={};all_sources=None;clean=patch=stage=margin=0
    for seed in ex.SEEDS:
        folder=root/f's{seed}'
        scored=json.loads((folder/'evaluation.json').read_text())
        audit=json.loads((folder/'independent_audit.json').read_text())
        if not audit['verified'] or audit['seed']!=seed or scored['seed']!=seed:
            raise ValueError('AF-09 incomplete independent seed evidence')
        if audit['evaluation_sha256']!=ex.sha_file(folder/'evaluation.json'):
            raise ValueError('AF-09 source-bound audit changed')
        if scored['source_hashes']!=ex.source_receipt():
            raise ValueError('AF-09 frozen source differs')
        if all_sources is None:all_sources=scored['source_hashes']
        elif all_sources!=scored['source_hashes']:raise ValueError('AF-09 cross-seed source mismatch')
        if set(scored['fits'])!=set(ex.ARMS):raise ValueError('AF-09 arm roster missing')
        gates={arm:ex.qualified(scored['fits'][arm]['fit_metrics']) for arm in ex.ARMS}
        if not ex.qualified(scored['qualified_parent_metrics']):
            raise ValueError('AF-09 pretrained source eligibility failure')
        eligible=all(gates.values())
        comparisons={}
        judgment='inconclusive'
        if eligible:
            passing=True
            for family in FAMILIES:
                line=scored['trajectories']
                loc=line['locality'][family]['3']['protected_q_step_accuracy'][-1]
                non=line['semantic_margin'][family]['3']['protected_q_step_accuracy'][-1]
                cm=scored['certificates']['semantic_margin'][family]['3']
                cb=scored['certificates']['locality'][family]['3']
                if cm['false_certificates'] or cb['false_certificates']:
                    raise ValueError('AF-09 false affine certificate in model data')
                cert_gain=(cm['positive_certificate_rate_by_step'][-1] -
                           cb['positive_certificate_rate_by_step'][-1])
                ps=scored['patches']['semantic_margin'][family]
                pstatus=scored['patches']['status'][family]
                clean_=scored['scores']['semantic_margin'][family]
                clean_baseline=scored['scores']['locality'][family]
                checks={
                   'untouched_gain':non-loc,
                   'gain_gate':non-loc>=.05-TOL,
                   'positive_certificate_rate':cm['positive_certificate_rate_by_step'][-1],
                   'certificate_gain':cert_gain,
                   'certificate_gain_gate':cert_gain>=.05-TOL,
                   'certificate_integrity_gate':cm['false_certificates']==0,
                   'absolute_untouched_q_gate':non>=.90-TOL,
                   'irrelevant_patch_goal_retention':ps['other_patch_goal_retention'],
                   'negative_control_gate':ps['other_patch_goal_retention']>=.90-TOL,
                   'clean_goal_drop':clean_['goal_accuracy']-clean_baseline['goal_accuracy'],
                   'clean_goal_gate':clean_['goal_accuracy']>=clean_baseline['goal_accuracy']-.03-TOL,
                   'false_goal_gate':clean_['false_goal_attempts']<=clean_baseline['false_goal_attempts'],
                   'informative_goal_advantage':ps['informative_goal_accuracy']-scored['patches']['locality'][family]['informative_goal_accuracy'],
                   'status_informative_goal':pstatus['informative_goal_accuracy'],
                }
                passing=passing and all(checks[k] for k in ('gain_gate','absolute_untouched_q_gate',
                        'negative_control_gate','clean_goal_gate','false_goal_gate',
                        'certificate_gain_gate','certificate_integrity_gate'))
                comparisons[family]=checks
            judgment='pass' if passing else 'fail'
        runs[f's{seed}']={'dev':{arm:scored['fits'][arm]['fit_metrics']['dev'] for arm in ex.ARMS},
                         'qualified_parent':True,'continuation_qualifies':gates,
                         'eligible':eligible,'verdict':judgment,'comparisons':comparisons,
                         'scores':scored['scores'],'patches':scored['patches'],
                         'trajectories':scored['trajectories'],'separation':scored['separation'],'certificates':scored['certificates'],
                         'evidence_sha256':audit['evaluation_sha256'],
                         'source_zip_sha256':ex.SOURCE_ARTIFACTS[seed]['sha256']}
        clean+=audit['clean_rows'];patch+=audit['patch_rows'];stage+=audit['stage_rows'];margin+=audit['margin_rows']
    verdicts=[x['verdict'] for x in runs.values()]
    primary='fail' if 'fail' in verdicts else ('pass' if all(v=='pass' for v in verdicts) else 'inconclusive')
    return {'experiment':'AF-09','scientific_verdict':primary,'engineering_verification':True,
            'source':all_sources,'parent':ex.PARENT_COMMIT,'parent_artifact_ids':{str(k):v['id'] for k,v in ex.SOURCE_ARTIFACTS.items()},
            'seeds':list(ex.SEEDS),'width':ex.WIDTH,'arms':list(ex.ARMS),
            'additional_training_steps_per_arm':ex.STEPS,'models':len(ex.SEEDS)*len(ex.ARMS),
            'clean_rows':clean,'patch_rows':patch,'stage_rows':stage,'margin_rows':margin,'runs':runs}


def report(r):
    lines=['# AF-09: source-qualified, signed semantic-margin causal locality',
           '',f"**Registered primary scientific verdict: {r['scientific_verdict'].upper()}.**",
           f"**Evidence:** {r['models']} continuation models; {r['clean_rows']} clean logits rows, {r['patch_rows']} donor patches, {r['stage_rows']} stage-trace rows, {r['margin_rows']} affine certificate rows.",
           '', 'Both source checkpoints were independently qualified AF-06 shared warm-ups. No new warm-up fitting occurred in AF-09.',
           '', '| Seed | Eligible | Depth4 locality → semantic_margin protected Q | Depth5 locality → semantic_margin protected Q | Judgment |',
           '|---:|:---:|---|---|---|']
    for seed in r['seeds']:
        x=r['runs'][f's{seed}']
        def val(f):
            p=x['trajectories'];a=p['locality'][f]['3']['protected_q_step_accuracy'][-1];b=p['semantic_margin'][f]['3']['protected_q_step_accuracy'][-1]
            return f'{a:.4f} → {b:.4f}'
        lines.append(f"| {seed} | {x['eligible']} | {val('held_depth4')} | {val('held_depth5')} | {x['verdict']} |")
    lines += ['', 'The semantic_margin arm adds true-class signed margin lower-bound and patched-state softplus losses to the same stepwise semantic and locality objectives. Matched model calls and steps do not imply matched total backpropagation FLOPs.',
              '', 'Lean certificates bound the original synthetic Horn oracle only. Model predictions and intervention claims are scored separately against that oracle. An ineligible continuation yields inconclusive seed evidence.',
              '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    r=aggregate(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    ex.json_write(a.out,r)
    a.report.write_text(report(r))
    print(json.dumps({k:r[k] for k in ('scientific_verdict','models','clean_rows','patch_rows','stage_rows')},sort_keys=True))

if __name__=='__main__':main()
