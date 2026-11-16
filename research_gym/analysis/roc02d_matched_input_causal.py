"""ROC-02D, prereg 714f149: dev-only legal/soft-input and hidden perturbation pilot.

The only source files this program opens are train_inputs.jsonl and
calibration_inputs.jsonl. The sealed source family is never loaded.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import torch
try:
    from research_gym.analysis.roc02_delayed_rule_source import rule_table, symbolic_execute, canonical_hash
    from research_gym.analysis.roc02_tied_train import tensors, rows
    from research_gym.analysis.roc02b_calibration_card import load_and_verify, stage_model, suffix, tangent, SEEDS, DEPTHS, WIDTH
except ModuleNotFoundError:
    from roc02_delayed_rule_source import rule_table, symbolic_execute, canonical_hash
    from roc02_tied_train import tensors, rows
    from roc02b_calibration_card import load_and_verify, stage_model, suffix, tangent, SEEDS, DEPTHS, WIDTH

INDICES=(2,3,4,5,6,7)
RADIUS=.2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter(path,index,seed,depth,ck_hash,source_dir):
    expected=next(x for x in index['adapter_receipts'] if x['seed']==seed and x['depth']==depth)
    if digest(path)!=expected['adapter_sha256']:
        raise RuntimeError('saved ROC-02C observer adapter checksum mismatch')
    v=torch.load(path,map_location='cpu',weights_only=True)
    if (v['seed'],v['depth'],v['source_checkpoint_sha256'])!=(seed,depth,ck_hash):
        raise RuntimeError('adapter identity mismatch')
    for k,split in [('source_data_train_sha256','train'),('source_data_calibration_sha256','calibration')]:
        if v[k]!=digest(Path(source_dir)/'data'/f'{split}_inputs.jsonl'):
            raise RuntimeError('adapter source data mismatch')
    for name in ('task_only','joint'):
        U=v[f'{name}_U'].double()
        if U.shape!=(WIDTH,8) or float(torch.linalg.norm(U.T@U-torch.eye(8,dtype=torch.float64)))>1e-8:
            raise AssertionError('saved observer not orthonormal')
    return v


def changed_legal_binding(case):
    if case['family'] not in (4,5):
        raise AssertionError('dev case requires source family 4 or 5')
    q=copy.deepcopy(case)
    other=5 if case['family']==4 else 4
    if q['events'][0]['op']!='bind' or q['events'][0]['src']!='A':
        raise AssertionError('first event must bind A')
    q['family']=other
    q['events'][0]['table']=rule_table(other)
    answer=symbolic_execute(q['events'])
    q['label']=answer['label']
    q['id']=canonical_hash(q)
    if answer['status']!='ok' or q['label']==case['label']:
        raise AssertionError('input flip does not change exact oracle label')
    return q


def predict_mid(adapter,arm,h):
    U=adapter[f'{arm}_U'].double()
    head=adapter[f'{arm}_head'].double()
    v=adapter['mean_y'].double()+(h.double()-adapter['mean_h'].double())@U@head
    return v


def output(model,h,adapter,arm):
    with torch.no_grad():
        final=suffix(model,h.view(1,-1).float())[0]
        logits=model.readout(final).double()
        dec=predict_mid(adapter,arm,h)
    return logits,dec


def margin(logits,label):
    a=logits[label]
    rest=torch.cat((logits[:label],logits[label+1:]))
    return float(a-rest.max())


def seeded_vector(checkpoint_sha,case_i,arm,obs_arm,U):
    key=f'{checkpoint_sha}|roc02d|{case_i}|{obs_arm}|{arm}'
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63-1)
    g=torch.Generator(device='cpu').manual_seed(seed)
    if arm=='observed':
        a=U@torch.randn(8,dtype=torch.float64,generator=g)
    else:
        a=torch.randn(WIDTH,dtype=torch.float64,generator=g)
        if arm=='dark':a=a-U@(U.T@a)
    if float(torch.linalg.norm(a))<1e-12:raise ValueError('degenerate random direction')
    return a/torch.linalg.norm(a)


def mid_for(model,features,mask):
    with torch.no_grad():
        return stage_model(model,features,mask)[0][0].double()


def interpolate_input(model,x,m,xc,mid0,midfull):
    full_displacement=float(torch.linalg.norm(midfull-mid0))
    if full_displacement < RADIUS:
        return None,full_displacement,None
    lo,hi=0.,1.
    # For a regular local path, find the smallest admissible t on a fixed grid;
    # then bisect the first crossing to avoid assuming global monotonicity.
    grid=[i/64 for i in range(65)]
    distances=[]
    for t in grid:
        y=mid_for(model,x+t*(xc-x),m)
        distances.append(float(torch.linalg.norm(y-mid0)))
    ix=next((i for i,v in enumerate(distances) if v>=RADIUS),None)
    if ix is None:return None,full_displacement,None
    lo,hi=grid[ix-1],grid[ix]
    for _ in range(24):
        t=(lo+hi)/2
        dist=float(torch.linalg.norm(mid_for(model,x+t*(xc-x),m)-mid0))
        if dist>=RADIUS:hi=t
        else:lo=t
    z=mid_for(model,x+hi*(xc-x),m)
    err=abs(float(torch.linalg.norm(z-mid0))-RADIUS)
    if err>2e-5:raise AssertionError('matched input radius error')
    return z,full_displacement,hi


def cell(seed,depth,source_dir,ckdir,adapter_dir,adapter_index,outdir):
    torch.set_num_threads(1)
    ck,cksha,model=load_and_verify(seed,depth,source_dir,ckdir)
    adapter_path=Path(adapter_dir)/f'roc02c_adapter_R{depth}_S{seed}.pt'
    adapter=load_adapter(adapter_path,adapter_index,seed,depth,cksha,source_dir)
    devfile=Path(source_dir)/'data'/'calibration_inputs.jsonl'
    cases=rows(devfile)
    observations=[]
    for idx in INDICES:
        case=cases[idx]
        cf=changed_legal_binding(case)
        x,m,y=tensors([case]);xc,mc,yc=tensors([cf])
        if not torch.equal(m,mc) or x.shape!=xc.shape:
            raise AssertionError('input causal edit changed timing/shape')
        mid0=mid_for(model,x,m)
        midfull=mid_for(model,xc,mc)
        midpoint_soft,full_dist,t=interpolate_input(model,x,m,xc,mid0,midfull)
        T=tangent(model,mid0)
        R=model.readout.weight.detach().double()
        with torch.no_grad():
            for obs_arm in ('task_only','joint'):
                U=adapter[f'{obs_arm}_U'].double()
                logit0,dec0=output(model,mid0,adapter,obs_arm)
                result_points=[]
                def record(arm,mid,role,valid=True,notes=''):
                    if mid is None:
                        result_points.append({'arm':arm,'available':False,'notes':notes})
                        return
                    logit1,dec1=output(model,mid,adapter,obs_arm)
                    shift=mid-mid0
                    prediction=R@(T@shift)
                    result_points.append({
                        'arm':arm,'available':True,'kind':role,
                        'midpoint_shift_norm':float(torch.linalg.norm(shift)),
                        'original_logits_change_norm':float(torch.linalg.norm(logit1-logit0)),
                        'observer_mid_readout_change_norm':float(torch.linalg.norm(dec1-dec0)),
                        'future_logits_tangent_residual_norm':float(torch.linalg.norm((logit1-logit0)-prediction)),
                        'true_original_margin_change':margin(logit1,int(y[0]))-margin(logit0,int(y[0])),
                        'argmax_changed':int(logit1.argmax())!=int(logit0.argmax()),
                        'original_label_probability_direction':float((logit1-logit0)[int(yc[0])]- (logit1-logit0)[int(y[0])]),
                        'notes':notes})
                record('full_legal_flip',midfull,'discrete legal binding',notes='unmatched physical norm')
                record('input_soft_radius',midpoint_soft,'continuous input embedding',notes='not a valid discrete symbolic binding' if midpoint_soft is not None else 'no matched input radius available')
                induced=midfull-mid0
                record('input_induced_midpoint_radius',mid0+(RADIUS/torch.linalg.norm(induced))*induced,'midpoint direction induced by legal input edit',notes='fixed-radius injected tangent direction, not a direct input edit')
                for arm in ('observed','dark','isotropic'):
                    v=seeded_vector(cksha,idx,arm,obs_arm,U)
                    record('hidden_'+arm+'_radius',mid0+RADIUS*v,'hidden midpoint intervention')
                observations.append({'seed':seed,'depth':depth,'case_index':idx,'observer_arm':obs_arm,
                                     'calibration_only':True,'checkpoint_sha256':cksha,
                                     'adapter_sha256':digest(adapter_path),
                                     'source_family':case['family'],'flipped_family':cf['family'],
                                     'label_original':int(y[0]),'label_flipped':int(yc[0]),
                                     'full_discrete_flip_midpoint_L2':full_dist,'soft_t':t,
                                     'base_true_margin':margin(logit0,int(y[0])),
                                     'base_argmax':int(logit0.argmax()),
                                     'effects':result_points})
    outdir.mkdir(parents=True,exist_ok=True)
    receipt={'study':'ROC-02D-calibration-only-input-versus-hidden-perturbation-v1',
             'seed':seed,'depth':depth,'frozen_model_checkpoint_sha256':cksha,
             'frozen_adapter_sha256':digest(adapter_path),'calibration_only':True,
             'sealed_test_opened':False,'case_indices':list(INDICES),'L2_radius':RADIUS,
             'rows':observations,
             'claim_scope':'Legal discrete input flip changes exact label; softened input is not legal, and output changes measure a weak trained model on exposed calibration data.'}
    path=Path(outdir)/f'roc02d_R{depth}_S{seed}.json'
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'depth':depth,'rows':len(observations),
                      'input_match_eligible':sum(x['soft_t'] is not None for x in observations),
                      'receipt_sha256':digest(path)},sort_keys=True),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--depth',type=int,choices=DEPTHS,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--adapter-dir',type=Path,required=True)
    p.add_argument('--adapter-index',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    cell(args.seed,args.depth,args.source_dir,args.checkpoint_dir,args.adapter_dir,
         json.loads(args.adapter_index.read_text()),args.out_dir)

if __name__=='__main__':main()
