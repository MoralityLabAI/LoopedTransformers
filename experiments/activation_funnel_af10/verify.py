"""AF-10 independent semantic and affine readout verifier for retained data."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from experiments.activation_funnel_af05.experiment import Dataset,EVAL_LOOPS,sha_file
from .diagnostic import load_original,ARMS,SPLITS
from .core import paired


def check_rows(archive:Path,seed:int,folder:Path,replay:bool=True):
    folder=Path(folder)
    if sha_file(folder/'summary.json')!=(folder/'summary.sha256').read_text().strip():
        raise ValueError('summary digest mismatch')
    summary=json.loads((folder/'summary.json').read_text())
    raw=(folder/'cut_replay.jsonl').read_text()
    if hashlib.sha256(raw.encode()).hexdigest()!=summary['row_sha256']:
        raise ValueError('row digest mismatch')
    rows=[json.loads(s) for s in raw.splitlines()]
    models,src=load_original(archive,seed)
    if summary['seed']!=seed or summary['source']!=src or summary['rows']!=len(rows):
        raise ValueError('wrong frozen source identity')
    seen=set();by_group={};max_affine_error=0.;count=0
    for row in rows:
        group=(row['arm'],row['split'])
        if group[0] not in ARMS or group[1] not in SPLITS and group[1]!='dev':
            raise ValueError('unregistered result arm/split')
        key=(*group,row['recipient_id'],row['selected'])
        if key in seen:raise ValueError('duplicate scored intervention')
        seen.add(key);by_group.setdefault(group,[]).append(row)
    for (arm,split),observed in by_group.items():
        ds=Dataset(split);model=models[arm]
        if len(observed)!=2*len(ds.cases):raise ValueError('partial donor roster')
        target_map={(c.id,j):(i,ds.donors[i,j][0],ds.donors[i,j][1])
                    for i,c in enumerate(ds.cases) for j in (0,1)}
        arrays=model.state_dict()
        wd=arrays['codec.atom.decoder.weight'].detach().numpy().astype(np.float64)
        bd=arrays['codec.atom.decoder.bias'].detach().numpy().astype(np.float64)
        wr=arrays['subgoal_head.weight'].detach().numpy().astype(np.float64)
        br=arrays['subgoal_head.bias'].detach().numpy().astype(np.float64)
        scores={'pairs':0,'prefix_protected_correct':0,'prefix_joint_correct':0,
                    'ordinary_protected_correct':0,'cut_protected_correct':0,
                    'ordinary_composed_goal_correct':0,'cut_composed_goal_correct':0,
                    'prefix_pair_correct':0,'cut_goal_correct_given_prefix_pair':0,
                    'cut_control_goal_retention':0,'ordinary_control_goal_retention':0,
                    'cut_noninterference_error_max':0.}
        for row in observed:
            selected=row['selected'];k=(row['recipient_id'],selected)
            if k not in target_map:raise ValueError('unknown source case')
            i,d,composed=target_map[k]
            r=ds.cases[i];donor=ds.cases[d]
            if row['donor_id']!=donor.id or row['truth']!=list(ds.batch.targets[i].tolist()) or row['composed_truth']!=list(composed):
                raise ValueError('independent Horn source or donor identity mismatched')
            protected=1-selected;truth=row['truth'][protected]
            def get_q(a):return int(np.argmax(np.asarray(a)))
            pre_p=get_q(row['pre_clean_q_logits'][protected]);pre_d=get_q(row['pre_donor_q_logits'][selected])
            pair_ok=(pre_p==truth and pre_d==composed[selected])
            if row['cut_prefix_pair_correct']!=pair_ok:raise ValueError('tampered semantic factorization')
            patch_q=row['cut_patch_q_logits']
            # Independently reconstruct the *protected* readout from two stored floats.
            pcode=np.asarray(row['cut_prefix_protected_code'],dtype=np.float64)
            selected_code=np.asarray(row['donor_selected_code'],dtype=np.float64)
            if pcode.shape!=(model.code_width,) or selected_code.shape!=(model.code_width,):
                raise ValueError('changed carried-code dimensions')
            computed_protected=wr @ (wd@pcode+bd)+br
            computed_selected=wr @ (wd@selected_code+bd)+br
            aff0=np.max(np.abs(computed_protected-np.asarray(patch_q[protected])))
            aff1=np.max(np.abs(computed_selected-np.asarray(patch_q[selected])))
            pre=np.max(np.abs(computed_protected-np.asarray(row['pre_clean_q_logits'][protected])))
            max_affine_error=max(max_affine_error,float(aff0),float(aff1),float(pre))
            if max(aff0,aff1,pre)>2e-5:raise ValueError('independent affine cut readout mismatch')
            if any(v!=0. for v in row['cut_protected_code_difference_by_step']):
                raise ValueError('noninterference violated in retained row')
            if get_q(patch_q[protected])!=pre_p:
                raise ValueError('cut did not preserve protected prefix classification')
            if pair_ok and [get_q(q) for q in patch_q]!=list(composed[:2]):
                raise ValueError('composed prefix Q theorem violated')
            def inc(k,truth):scores[k]+=int(bool(truth))
            inc('prefix_protected_correct',pre_p==truth)
            inc('prefix_joint_correct',[get_q(q) for q in row['pre_clean_q_logits']]==list(row['truth'][:2]))
            inc('ordinary_protected_correct',get_q(row['ordinary_patch_q_logits'][protected])==truth)
            inc('cut_protected_correct',get_q(patch_q[protected])==truth)
            inc('ordinary_composed_goal_correct',get_q(row['ordinary_patch_goal_logits'])==composed[2])
            inc('cut_composed_goal_correct',get_q(row['cut_patch_goal_logits'])==composed[2])
            inc('prefix_pair_correct',pair_ok)
            inc('cut_goal_correct_given_prefix_pair',pair_ok and get_q(row['cut_patch_goal_logits'])==composed[2])
            inc('ordinary_control_goal_retention',get_q(row['ordinary_control_goal_logits'])==row['truth'][2])
            inc('cut_control_goal_retention',get_q(row['cut_control_goal_logits'])==row['truth'][2])
            scores['pairs']+=1
        old=summary['models'][arm][split]
        for k,val in scores.items():
            if old[k]!=val:raise ValueError('tampered aggregate field: '+k)
        if replay:
            torch.set_num_threads(2)
            for i in sorted(set([0,len(ds.cases)//2,len(ds.cases)-1])):
                for selected in (0,1):
                    r=ds.get([i]);d,h=ds.pair([i],[selected])
                    fresh=paired(model,r,d,torch.tensor([selected]),EVAL_LOOPS[split],5 if split.startswith('held_depth') else 3,cut=True)
                    recorded=next(x for x in observed if x['recipient_id']==ds.cases[i].id and x['selected']==selected)
                    for key,tensor in [('cut_patch_q_logits',fresh.patched.q[0]),('cut_patch_goal_logits',fresh.patched.goal[0]),('cut_control_goal_logits',fresh.control.goal[0])]:
                        if not np.allclose(np.asarray(recorded[key]),tensor.detach().numpy(),atol=2e-5,rtol=0):
                            raise ValueError('independent checkpoint forward replay mismatch: '+key)
                    count+=1
    return {'experiment':'AF-10','seed':seed,'verified':True,'rows':len(rows),
            'model_groups':len(by_group),'independent_affine_max_error':max_affine_error,
            'checkpoints_replayed':count,'source_sha256':src['original_artifact_sha256'],
            'retained_rows_sha256':sha_file(folder/'cut_replay.jsonl'),
            'summary_sha256':sha_file(folder/'summary.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-zip',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=check_rows(a.source_zip,a.seed,a.evaluation)
    a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
