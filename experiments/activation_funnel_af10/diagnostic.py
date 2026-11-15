"""AF-10 source-bound causal-cut mechanism evaluation.

Diagnostic replay of AF-09 frozen seeds 17/29, no model optimization. All saved
negative results are descriptive because these existing heldouts informed the
AF-10 architecture; formal structural invariance is tested independently.
"""
from __future__ import annotations
import argparse, hashlib, io, json, tempfile, zipfile
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.experiment import Dataset,EVAL_LOOPS,sha_file
from experiments.activation_funnel_af09 import experiment as af09
from .core import paired,exact_prefix_factorization,cross_subgoal_jacobian

ARTIFACTS={
 17:{'id':14148947456,'sha256':'c29f63fec6c66dc1ae62b820bb7fa12a12e744bd49933de397c4e7bace16c488'},
 29:{'id':14148897762,'sha256':'ef4fb7b2a45fd643e25959b7d1d20142d5063dfe86d8da19fb069dd01a0bcc5c'},
}
FROZEN_SOURCE='643fd9085c4801960a228a16ddbe49a8a58dc750'
ARMS=('status','locality','semantic_margin')
SPLITS=('dev','held_composition','held_layout','held_depth4','held_depth5')


def load_original(archive:Path,seed:int):
    """Verify original artifact, embedded .pt and .npz, source hash, source receipt."""
    if seed not in ARTIFACTS or sha_file(archive)!=ARTIFACTS[seed]['sha256']:
        raise ValueError('AF-09 original artifact checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='af10-source-') as tmp:
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('original ZIP CRC failed')
            roster=z.namelist()
            if len(set(roster))!=len(roster) or any(n.startswith('/') or '..' in Path(n).parts for n in roster):
                raise ValueError('ZIP traversal or repeated path')
            if z.read('source_revision.txt').decode().strip()!=FROZEN_SOURCE:
                raise ValueError('AF-09 source revision mismatch')
            for member in roster:
                if member.startswith('train/'):
                    target=Path(tmp)/member
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(member))
        receipt,models=af09.load_models(Path(tmp)/'train',seed,smoke=False)
        # load_models rechecks the original source ledger and full weight/export roster.
        if set(models)!=set(ARMS):raise ValueError('missing original arm')
        for arm,model in models.items():
            if not af09.qualified(receipt['arms'][arm]['fit_metrics']):
                raise ValueError('original model development eligibility failure')
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError('nonfinite original state')
        return models,{'original_artifact_sha256':sha_file(archive),
                       'original_source_commit':FROZEN_SOURCE,
                       'original_training_receipt_sha256':sha_file(Path(tmp)/'train/train_receipt.json'),
                       'original_model_exports':receipt['files']}


def run(archive:Path,seed:int,out:Path,*,splits=SPLITS,chunk=32):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh output folder required')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    models,source=load_original(archive,seed)
    all_rows=[];summary={'experiment':'AF-10','seed':seed,'source':source,'models':{},'scope':'exploratory_frozen_model_diagnostic'}
    for arm,model in models.items():
        model.eval();summary['models'][arm]={}
        for split in splits:
            ds=Dataset(split);loops=EVAL_LOOPS[split];post=5 if split.startswith('held_depth') else 3
            counts={'pairs':0,'prefix_protected_correct':0,'prefix_joint_correct':0,
                    'ordinary_protected_correct':0,'cut_protected_correct':0,
                    'ordinary_composed_goal_correct':0,'cut_composed_goal_correct':0,
                    'prefix_pair_correct':0,'cut_goal_correct_given_prefix_pair':0,
                    'cut_control_goal_retention':0,'ordinary_control_goal_retention':0,
                    'cut_noninterference_error_max':0.}
            with torch.no_grad():
                for first in range(0,len(ds.cases)*2,chunk):
                    ids=list(range(first,min(first+chunk,len(ds.cases)*2)))
                    case_ix=[k//2 for k in ids];selected=[k%2 for k in ids]
                    rx=ds.get(case_ix)
                    donor,composed=ds.pair(case_ix,selected)
                    which=torch.tensor(selected,dtype=torch.long)
                    ordinary=paired(model,rx,donor,which,loops,post,cut=False)
                    cut=paired(model,rx,donor,which,loops,post,cut=True)
                    factors=exact_prefix_factorization(cut,rx.targets,composed)
                    pro=(1-which)[:,None]
                    o_p=ordinary.patched.q.argmax(-1).gather(1,pro).squeeze(1)
                    c_p=cut.patched.q.argmax(-1).gather(1,pro).squeeze(1)
                    protected_target=rx.targets[:,:2].gather(1,1-which[:,None]).squeeze(1)
                    pre_other=ordinary.prefix_clean.az.gather(1,rx.read_slots[:,:2].gather(1,1-which[:,None]).unsqueeze(-1).expand(-1,-1,model.code_width)).squeeze(1)
                    pre_selected=ordinary.prefix_donor.az.gather(1,rx.read_slots[:,:2].gather(1,which[:,None]).unsqueeze(-1).expand(-1,-1,model.code_width)).squeeze(1)
                    d_pre,_=model._read(ordinary.prefix_donor.a,rx.read_slots)
                    r_pre,_=model._read(ordinary.prefix_clean.a,rx.read_slots)
                    pair_ok=(r_pre.argmax(-1).gather(1,(1-which)[:,None]).squeeze(1)==protected_target)& \
                            (d_pre.argmax(-1).gather(1,which[:,None]).squeeze(1)==composed[:,:2].gather(1,which[:,None]).squeeze(1))
                    protected_ordinary=(o_p==protected_target)
                    protected_cut=(c_p==protected_target)
                    goal_ordinary=(ordinary.patched.goal.argmax(-1)==composed[:,2])
                    goal_cut=(cut.patched.goal.argmax(-1)==composed[:,2])
                    control_ordinary=ordinary.control.goal.argmax(-1)==rx.targets[:,2]
                    control_cut=cut.control.goal.argmax(-1)==rx.targets[:,2]
                    delta=factors['protected_code_max_difference_by_step']
                    if bool(delta.any()):raise ArithmeticError('latched protected code changed')
                    if not torch.equal(cut.patched.q_steps.gather(2,(1-which)[:,None,None,None].expand(-1,post,1,2)),
                                       cut.clean.q_steps.gather(2,(1-which)[:,None,None,None].expand(-1,post,1,2))):
                        raise ArithmeticError('noninterference exact readout violated')
                    # Factorization theorem: if both prefix Q labels correct, then the
                    # transplanted Q pair must be exactly correct under the causal cut.
                    if bool((pair_ok & ~(cut.patched.q.argmax(-1)==composed[:,:2]).all(-1)).any()):
                        raise ArithmeticError('prefix semantic factorization violated')
                    for key,tensor in [('prefix_protected_correct',factors['prefix_protected_correct']),
                                       ('prefix_joint_correct',factors['prefix_joint_correct']),
                                       ('ordinary_protected_correct',protected_ordinary),
                                       ('cut_protected_correct',protected_cut),
                                       ('ordinary_composed_goal_correct',goal_ordinary),
                                       ('cut_composed_goal_correct',goal_cut),
                                       ('prefix_pair_correct',pair_ok),
                                       ('cut_goal_correct_given_prefix_pair',pair_ok & goal_cut),
                                       ('ordinary_control_goal_retention',control_ordinary),
                                       ('cut_control_goal_retention',control_cut)]:
                        counts[key]+=int(tensor.sum())
                    counts['pairs']+=len(ids)
                    for i,k in enumerate(ids):
                        all_rows.append({'seed':seed,'arm':arm,'split':split,'recipient_id':ds.cases[k//2].id,
                             'donor_id':ds.cases[ds.donors[k//2,k%2][0]].id,
                             'selected':k%2,'truth':rx.targets[i].tolist(),
                             'composed_truth':composed[i].tolist(),
                             'pre_clean_q_logits':r_pre[i].tolist(),
                             'pre_donor_q_logits':d_pre[i].tolist(),
                             'ordinary_patch_q_logits':ordinary.patched.q[i].tolist(),
                             'cut_patch_q_logits':cut.patched.q[i].tolist(),
                             'ordinary_patch_goal_logits':ordinary.patched.goal[i].tolist(),
                             'cut_patch_goal_logits':cut.patched.goal[i].tolist(),
                             'ordinary_control_goal_logits':ordinary.control.goal[i].tolist(),
                             'cut_control_goal_logits':cut.control.goal[i].tolist(),
                             'cut_protected_code_difference_by_step':delta[i].tolist(),
                             'cut_prefix_protected_code':pre_other[i].tolist(),
                             'donor_selected_code':pre_selected[i].tolist(),
                             'cut_prefix_pair_correct':bool(pair_ok[i])})
            counts['prefix_pair_fraction']=counts['prefix_pair_correct']/counts['pairs']
            for k in tuple(counts):
                if k.endswith('_correct') or k.endswith('_retention') or k in ('prefix_protected_correct','ordinary_protected_correct','cut_protected_correct'):
                    counts[k+'_rate']=counts[k]/counts['pairs']
            counts['cut_goal_accuracy_conditional_on_correct_prefix_pair']=(
                    counts['cut_goal_correct_given_prefix_pair']/counts['prefix_pair_correct']
                    if counts['prefix_pair_correct'] else None)
            summary['models'][arm][split]=counts
            print(json.dumps({'arm':arm,'split':split,'pairs':counts['pairs'],
                              'prefix_q':counts['prefix_protected_correct_rate'],
                              'ordinary_q':counts['ordinary_protected_correct_rate'],
                              'cut_q':counts['cut_protected_correct_rate']}),flush=True)
    rows=out/'cut_replay.jsonl'
    rows.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in all_rows))
    summary['rows']=len(all_rows);summary['row_sha256']=sha_file(rows)
    (out/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    (out/'summary.sha256').write_text(sha_file(out/'summary.json')+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-zip',required=True,type=Path)
    p.add_argument('--seed',required=True,type=int,choices=ARTIFACTS)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.source_zip,a.seed,a.out,
                          splits=('dev',) if a.smoke else SPLITS,chunk=32)
if __name__=='__main__':main()
