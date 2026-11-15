"""AF-11 independent two-seed scientific verdict from held-out scoring receipts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from . import experiment as ex


def decide(root,out):
    root=Path(root)
    seeds={};all_ok=True
    total=0
    for seed in ex.SEEDS:
        folder=root/f's{seed}'
        train=folder/'training'/'train_receipt.json'
        evaluation=folder/'evaluation'/'evaluation.json'
        audit=folder/'evaluation'/'independent_audit.json'
        # CI artifacts are stored at predictable prefixed paths.
        for path in (train,evaluation,audit):
            if not path.exists():raise FileNotFoundError(path)
        a=json.loads(audit.read_text());t=json.loads(train.read_text());s=json.loads(evaluation.read_text())
        if not a['verified'] or (a['seed'],t['seed'],s['seed'])!=(seed,seed,seed):
            raise ValueError('AF-11 wrong verified seed receipt')
        if a['source_receipt_sha256']!=ex.sha_file(train) or a['evaluation_sha256']!=ex.sha_file(evaluation):
            raise ValueError('AF-11 tampered source-bound aggregate')
        if s['counterfactual_rows']!=sum(len(ex.data.cases_for(f))*2 for f in ex.FAMILIES)*len(ex.ARMS):
            raise ValueError('AF-11 held-out roster incomplete')
        total+=s['counterfactual_rows']
        eligible=s['shared_cut_qualified'] and all(s['training_qualified'][a] for a in
                 ('counterfactual_cut','influence_cut'))
        family={}
        for name in ('fresh_depth4','fresh_depth5'):
            x=s['scores']['counterfactual_cut'][name]
            y=s['scores']['influence_cut'][name]
            rows=y['informative']
            if rows<32 or y['prefix_correct_informative']<8:
                raise ValueError('AF-11 insufficient informative and conditional roster')
            advantage=y['informative_flip_accuracy']-x['informative_flip_accuracy']
            # Original registered primary threshold frozen before held-out scoring.
            pass_gate=(advantage>=.05-1e-9 and y['informative_flip_accuracy']>=.80
                 and y['irrelevant_patch_goal_retention']>=.90
                 and y['clean_goal_accuracy']>=x['clean_goal_accuracy']-.03
                 and y['false_goal']<=x['false_goal'])
            family[name]={'baseline_informative_goal_flip':x['informative_flip_accuracy'],
                 'influence_informative_goal_flip':y['informative_flip_accuracy'],
                 'gain':advantage,'influence_prefix_q_exact':y['prefix_joint_q_accuracy'],
                 'influence_conditional_flip':y['prefix_conditioned_flip_accuracy'],
                 'influence_irrelevant_goal_retention':y['irrelevant_patch_goal_retention'],
                 'influence_clean_goal':y['clean_goal_accuracy'],
                 'counterfactual_clean_goal':x['clean_goal_accuracy'],
                 'influence_false_goal':y['false_goal'],
                 'baseline_false_goal':x['false_goal'],
                 'science_gate':bool(pass_gate) if eligible else None}
        verdict='inconclusive' if not eligible else ('pass' if all(f['science_gate'] for f in family.values()) else 'fail')
        seeds[str(seed)]={'eligible':bool(eligible),'verdict':verdict,'families':family,
                         'source_receipt_sha256':a['source_receipt_sha256'],
                         'counterfactual_rows':s['counterfactual_rows'],
                         'comparators':list(ex.ARMS)}
    v=[r['verdict'] for r in seeds.values()]
    overall='fail' if 'fail' in v else ('pass' if all(x=='pass' for x in v) else 'inconclusive')
    result={'experiment':'AF-11','registered_verdict':overall,'engineering_pass':True,
            'seed_results':seeds,'counterfactual_rows':total,
            'new_families':list(ex.FAMILIES),'train_splits':['train','dev'],
            'claim_scope':'fresh physical graph permutations of existing Boolean-Horn templates',
            'source':'fixed two-seed, four-arm, native TypedLoop comparator'}
    ex.json_write(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(decide(a.root,a.out)))
if __name__=='__main__':main()
