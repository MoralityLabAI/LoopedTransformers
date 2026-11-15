"""AF-12 fixed cross-seed registered verdict; engineering success is separate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex


def qualified(fit,arm):
    return all(fit[arm][split]['goal_accuracy']>=.90 and
               fit[arm][split]['joint_q_accuracy']>=.90
               for split in ('train','dev'))


def decide(root,out):
    root=Path(root);seeds={};total_clean=total_patch=0
    for seed in ex.SEEDS:
        tr=root/f's{seed}'/'training'/'train_receipt.json'
        ev=root/f's{seed}'/'evaluation'/'evaluation.json'
        au=root/f's{seed}'/'evaluation'/'independent_audit.json'
        if not all(x.is_file() for x in (tr,ev,au)):
            raise FileNotFoundError('AF12 missing source or independent audit')
        r=json.loads(tr.read_text());s=json.loads(ev.read_text());v=json.loads(au.read_text())
        if ((r['seed'],s['seed'],v['seed'])!=(seed,seed,seed) or not v['passed'] or
            v['training_sha256']!=ex.sha(tr) or v['evaluation_sha256']!=ex.sha(ev)):
            raise ValueError('AF12 cross-seed provenance mismatch')
        if r['fresh_manifest_sha256']!=ex.verify_manifest():raise ValueError('AF12 corpus changed')
        eligible=all(qualified(r['fit'],arm) for arm in ('relay_final','closure_rollout'))
        outcomes={};scientific=[]
        for family in ('fresh_depth4','fresh_depth5'):
            a=s['scores']['relay_final'][family];b=s['scores']['closure_rollout'][family]
            q_gain=b['joint_q_accuracy']-a['joint_q_accuracy']
            relay_gain=b['relay_final_accuracy']-a['relay_final_accuracy']
            # Source Q inference is the primary target. Preserve the actual
            # host goal, and require counterfactual downstream behavior stable.
            passed=(q_gain>=.05-1e-9 and relay_gain>=.05-1e-9 and
                b['goal_accuracy']>=a['goal_accuracy']-.03 and
                b['informative_flip_accuracy']>=a['informative_flip_accuracy']-.03 and
                b['false_goal']<=a['false_goal'] and b['irrelevant_retention']>=.90)
            scientific.append(passed)
            outcomes[family]={'q_baseline':a['joint_q_accuracy'],
                 'q_closure':b['joint_q_accuracy'],'q_gain':q_gain,
                 'relay_baseline':a['relay_final_accuracy'],
                 'relay_closure':b['relay_final_accuracy'],'relay_gain':relay_gain,
                 'baseline_goal':a['goal_accuracy'],'closure_goal':b['goal_accuracy'],
                 'baseline_informative_flip':a['informative_flip_accuracy'],
                 'closure_informative_flip':b['informative_flip_accuracy'],
                 'closure_negative_goal_retention':b['irrelevant_retention'],
                 'closure_false_goal':b['false_goal'],'baseline_false_goal':a['false_goal'],
                 'gate':bool(passed) if eligible else None}
        verdict='inconclusive' if not eligible else ('pass' if all(scientific) else 'fail')
        seeds[str(seed)]={'eligible':bool(eligible),'verdict':verdict,'families':outcomes,
                    'fit':r['fit'],'source_training_sha256':ex.sha(tr),
                    'independent_replays':v['sampled_forward_replays']}
        total_clean+=s['clean_rows'];total_patch+=s['patched_rows']
    statuses=[x['verdict'] for x in seeds.values()]
    overall='fail' if 'fail' in statuses else ('pass' if all(x=='pass' for x in statuses) else 'inconclusive')
    result={'experiment':'AF-12','registered_verdict':overall,'engineering_pass':True,
       'seed_results':seeds,'clean_rows':total_clean,'patch_rows':total_patch,
       'qualified_original_source':'AF-06 seed17/29 original warm-ups',
       'claim_scope':'fresh physical permutations of old Boolean Horn templates; training depths 2/3',
       'protocol_sha256':ex.sha(ex.PROTO),'fresh_manifest_sha256':ex.verify_manifest()}
    ex.write_json(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(decide(a.root,a.out)))
if __name__=='__main__':main()
