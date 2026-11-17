"""ROC-02E: calibration-only legal source-rule vs late-query matched causal audit.

Registered at 8fc97c97d9ed37a6a312933bc2040f5cc603d0c6 before measurement.
Original ROC-02 model and source SHA, ROC-02C per-cell receipts are verified.
Source families 6/7 are NEVER generated, read, labeled or inspected here.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import io
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import math
import torch

from research_gym.analysis.roc02_delayed_rule_source import (
    rule_table, symbolic_execute, canonical_hash)
from research_gym.analysis.roc02_tied_train import tensors, rows
from research_gym.analysis.roc02b_calibration_card import (
    load_and_verify, stage_model, suffix, tangent, observer_bases, SEEDS, DEPTHS, WIDTH)
from research_gym.analysis.roc02c_equal_budget import (
    data_matrices, score_components, optimize, evaluate, b_fraction_squared)

RANKS=(7,8)
RADII=(.05,.10,.20)
NCASE=256
BATCH=512
MAX_ERROR=2e-5
# 17 Nov CI portability note: 80-update QR/BLAS replay differs by 1e-7
# across CPU runners. Reject any scalar drift >1e-5; store all metric
# deltas in the model receipt. Exact source/checkpoint SHA gates remain strict.
REPLAY_METRIC_ATOL=1e-5
SEP='|'


def checksum(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()


def registered_pairs(orig: list[dict]):
    if len(orig)!=NCASE:raise ValueError('expected exactly 256 calibration cases')
    ctr=[]
    for r in orig:
        assert r['family'] in (4,5), 'unregistered calibration family'
        assert r['events'][0]['op']=='bind' and r['events'][0]['src']=='A'
        assert r['events'][-1]['op']=='query' and r['events'][-1]['src']=='A'
        assert r['events'][0]['table']==rule_table(r['family'])
        assert r['id']==canonical_hash(r)
        assert symbolic_execute(r['events'])['label']==r['label']
        a=copy.deepcopy(r); a['family']=5 if r['family']==4 else 4
        a['events'][0]['table']=rule_table(a['family'])
        a['label']=symbolic_execute(a['events'])['label']; a['id']=canonical_hash(a)
        b=copy.deepcopy(r); b['events'][-1]['value']=(r['events'][-1]['value']+4)%8
        b['label']=symbolic_execute(b['events'])['label']; b['id']=canonical_hash(b)
        assert a['label']==b['label']==(r['label']+4)%8
        assert a['events'][1:]==r['events'][1:] and b['events'][:-1]==r['events'][:-1]
        ctr.append((a,b))
    return [x[0] for x in ctr],[x[1] for x in ctr]


def midpoint(model,x,mask):
    with torch.no_grad():return stage_model(model,x,mask)[0].detach()


def forward_logits(model,h):
    with torch.no_grad():
        res=[]
        for j in range(0,len(h),BATCH):
            z=model.readout(suffix(model,h[j:j+BATCH].float()))
            res.append(z.double().cpu())
        return torch.cat(res,dim=0)


def matching_soft_states(model,x0,xe,mask,base,full):
    """Batch 64-point first crossing and 24 bisections, with abstain flags.

    Returns {radius:{'mid':Bxd,'ok':B bool,'t':B,'error':B}}.
    Every interpolated feature vector is off the legal symbolic data manifold.
    """
    with torch.no_grad():
        n=len(x0); delta=xe-x0; base=base.float()
        full_norm=torch.linalg.vector_norm((full-base).double(),dim=1)
        previous=torch.zeros(n,dtype=torch.float64)
        lo={r:torch.zeros(n,dtype=torch.float64) for r in RADII}
        hi={r:torch.ones(n,dtype=torch.float64) for r in RADII}
        found={r:torch.zeros(n,dtype=torch.bool) for r in RADII}
        # All 64 grid points are evaluated even if every case crosses early.
        for j in range(1,65):
            t=j/64
            h=midpoint(model,x0+t*delta,mask)
            dist=torch.linalg.vector_norm((h-base).double(),dim=1)
            for r in RADII:
                hit=(~found[r])&(dist>=r)
                lo[r][hit]=(j-1)/64
                hi[r][hit]=t
                found[r] |= hit
        results={}
        # Combine all radii into 3n batches for each bisection iteration.
        lows=torch.cat([lo[r] for r in RADII]); highs=torch.cat([hi[r] for r in RADII])
        xb=x0.repeat(len(RADII),1,1)
        dx=delta.repeat(len(RADII),1,1)
        mb=mask.repeat(len(RADII),1)
        baseb=base.repeat(len(RADII),1).double()
        for _ in range(24):
            tm=(lows+highs)/2
            hm=midpoint(model,xb+tm[:,None,None].float()*dx,mb)
            d=torch.linalg.vector_norm((hm.double()-baseb),dim=1)
            target=torch.cat([torch.full((n,),r,dtype=torch.float64) for r in RADII])
            crossed=d>=target
            highs=torch.where(crossed,tm,highs)
            lows=torch.where(crossed,lows,tm)
        hfin=midpoint(model,xb+highs[:,None,None].float()*dx,mb).double()
        err=(torch.linalg.vector_norm(hfin-baseb,dim=1)-target).abs()
        for idx,r in enumerate(RADII):
            s=slice(idx*n,(idx+1)*n)
            valid=found[r]&(full_norm>=r)&(err[s]<=MAX_ERROR)
            results[r]={'mid':hfin[s].clone(),'ok':valid,'t':highs[s].clone(),
                        'error':err[s].clone(),'full_norm':full_norm.clone(),
                        'num_nonconverged':int((found[r]&(full_norm>=r)&(err[s]>MAX_ERROR)).sum())}
        return results


def make_med_table(receipt,rank):
    z=[v for v in receipt['null_medians'] if v['rank']==rank]
    assert sorted(x['state_index'] for x in z)==list(range(6))
    assert all(x['null_128_distinct'] and x['median_b']>0 for x in z)
    return z


def replay_observers(model,cksha,train_h,train_y,cal_h,cal_y,archived,depth,seed):
    """Reconstruct chosen U, train-only head; verify both observed and dark B."""
    mats=data_matrices(train_h,train_y)
    bases=observer_bases(train_h,train_y)
    with torch.no_grad():
        Tlist=[tangent(model,cal_h[j]) for j in range(6)]
        Ds=[t-torch.eye(WIDTH,dtype=torch.float64) for t in Tlist]
    receipt=next(a for a in archived if a['depth']==depth and a['model_seed']==seed)
    assert cksha==receipt['checkpoint_sha256']
    obs={}
    for rank in RANKS:
        medians=make_med_table(receipt,rank)
        cellrows=[a for a in receipt['rows'] if a['rank']==rank]
        assert len(cellrows)==4
        selected=next(a for a in cellrows if a['selected'])
        task=next(a for a in cellrows if a['gamma']==0)
        U0=bases[('supervised',rank)].double()
        for typ,row in (('task_only',task),('joint',selected)):
            gamma=float(row['gamma'])
            U=optimize(U0,*mats[:3],len(train_h),Ds,medians,gamma)
            m=evaluate(U,train_h,train_y,cal_h,cal_y,mats,Ds,medians,Tlist,model,(2,3))
            match={'selection_calibration_MSE':'calibration_MSE',
                   'selection_B_null_squared':'B_null_squared',
                   'selection_H_future_logit_capture':'H_future_logit_capture',
                   'selection_calibration_accuracy':'calibration_accuracy',
                   'selection_train_MSE':'train_MSE'}
            metric_deltas={old:abs(float(row[old])-float(m[new])) for old,new in match.items()}
            for old,delta_abs in metric_deltas.items():
                if not math.isfinite(delta_abs) or delta_abs>REPLAY_METRIC_ATOL:
                    raise AssertionError(f'ROC02C replay mismatch {depth}/{seed}/{rank}/{typ} {old}: delta={delta_abs} tolerance={REPLAY_METRIC_ATOL}')
            with torch.no_grad():
                _,W=score_components(U,*mats[:3],len(train_h))
            obs[(rank,typ)]={'U':U,'W':W.detach().clone(),'gamma':gamma,
                             'calibration_MSE':m['calibration_MSE'],
                             'B_null_squared':m['B_null_squared'],
                             'H_future_logit_capture':m['H_future_logit_capture'],
                             'replay_metric_max_delta':max(metric_deltas.values()),
                             'replay_metric_deltas':metric_deltas}
    return obs,mats,receipt


def seeded_unit(seed,depth,cksha,case,rank,radius,edit,kind,U=None):
    digest=hashlib.sha256(f'{cksha}|{case}|{depth}|{rank}|{radius:.2f}|{edit}|{kind}'.encode()).digest()
    key=int.from_bytes(digest[:8],'big')%(2**63-1)
    g=torch.Generator(device='cpu').manual_seed(key)
    v=torch.randn((U.shape[1] if U is not None else WIDTH),dtype=torch.float64,generator=g)
    if U is not None:v=U@v
    return v/torch.linalg.vector_norm(v)


def normalize_direction(v,r):
    norm=torch.linalg.vector_norm(v,dim=1)
    ok=(norm>1e-12)&torch.isfinite(norm)
    result=(v/norm.clamp_min(1e-12)[:,None])*r
    return result,ok


def directional_gain(logits,base,y,yp):
    yp=yp.long();y=y.long()
    a=logits.gather(1,yp[:,None])[:,0]-logits.gather(1,y[:,None])[:,0]
    b=base.gather(1,yp[:,None])[:,0]-base.gather(1,y[:,None])[:,0]
    return a-b


def replay_mid_scores(h,obs,mats):
    _,_,_,hmean,ymean=mats
    return ymean+(h.double()-hmean)@obs['U']@obs['W']


def observer_margin_scale(train_h,train_y,obs,mats):
    pred=replay_mid_scores(train_h,obs,mats)
    yp=(train_y+4)%8
    margin=pred.gather(1,yp[:,None])[:,0]-pred.gather(1,train_y[:,None])[:,0]
    sd=float(margin.std(unbiased=False))
    if sd<=1e-12:raise ValueError('invalid observer output score scale')
    return sd


def evaluate_cell(seed,depth,source_dir,checkpoints,archived,out):
    torch.set_num_threads(1)
    ck,cksha,model=load_and_verify(seed,depth,source_dir,checkpoints)
    train=tensors(rows(source_dir/'data'/'train_inputs.jsonl'))
    original=rows(source_dir/'data'/'calibration_inputs.jsonl')
    rule,cue=registered_pairs(original)
    cal=tensors(original); xx,mask,y=cal
    train_h=midpoint(model,train[0],train[1])
    mid=midpoint(model,xx,mask)
    ruled=tensors(rule); cued=tensors(cue)
    assert torch.equal(mask,ruled[1]) and torch.equal(mask,cued[1])
    assert torch.equal(ruled[2],cued[2])
    yp=ruled[2]
    obs,mats,receipt=replay_observers(model,cksha,train_h,train[2],mid,y,archived,depth,seed)
    orig_logits=forward_logits(model,mid)
    mid_full={'rule':midpoint(model,ruled[0],mask).double(),
              'cue':midpoint(model,cued[0],mask).double()}
    full_logits={e:forward_logits(model,h) for e,h in mid_full.items()}
    soft={e:matching_soft_states(model,xx,(ruled[0] if e=='rule' else cued[0]),
                                 mask,mid,full) for e,full in mid_full.items()}
    scales={k:observer_margin_scale(train_h,train[2],o,mats) for k,o in obs.items()}
    base_dec={k:replay_mid_scores(mid,o,mats) for k,o in obs.items()}
    out.mkdir(parents=True,exist_ok=True)
    output=out/f'roc02e_R{depth}_S{seed}.jsonl.gz'
    perkey=defaultdict(list)
    invalid=defaultdict(int)
    nrows=0
    with output.open('wb') as sink, gzip.GzipFile(filename='', mode='wb', fileobj=sink, compresslevel=9, mtime=0) as packed, io.TextIOWrapper(packed, encoding='utf-8') as f:
        for edit in ('rule','cue'):
            legal=mid_full[edit]
            delta=legal-mid.double()
            for r in RADII:
                matched=soft[edit][r]
                delta_input,ok_input_dir=normalize_direction(delta,r)
                # The identical original-model logit effects of input controls are
                # written once per observer for transparent paired normalization.
                matched_logits=forward_logits(model,matched['mid'])
                induced_logits=forward_logits(model,mid.double()+delta_input)
                for rank in RANKS:
                    for typ in ('task_only','joint'):
                        o=obs[(rank,typ)]; U=o['U']; Pdelta=(delta@U)@U.T
                        Ddelta=delta-Pdelta
                        obsdir,ok_ob=normalize_direction(Pdelta,r)
                        darkdir,ok_dark=normalize_direction(Ddelta,r)
                        randoms=torch.stack([seeded_unit(seed,depth,cksha,i,rank,r,edit,'random_observed',U)
                                             for i in range(NCASE)],dim=0)
                        rand_iso=torch.stack([seeded_unit(seed,depth,cksha,i,rank,r,edit,'isotropic')
                                              for i in range(NCASE)],dim=0)
                        destinations={
                           'full_legal_unmatched':(legal,torch.ones(NCASE,dtype=torch.bool)),
                           'soft_input_matched':(matched['mid'],matched['ok']),
                           'input_induced_hidden':(mid.double()+delta_input,ok_input_dir),
                           'observed_input_projection':(mid.double()+obsdir,ok_ob),
                           'dark_input_projection':(mid.double()+darkdir,ok_dark),
                           'random_observed':(mid.double()+r*randoms,torch.ones(NCASE,dtype=torch.bool)),
                           'isotropic':(mid.double()+r*rand_iso,torch.ones(NCASE,dtype=torch.bool)),
                        }
                        for arm,(new_h,valid) in destinations.items():
                            if arm=='full_legal_unmatched': logits=full_logits[edit]
                            elif arm=='soft_input_matched':logits=matched_logits
                            elif arm=='input_induced_hidden':logits=induced_logits
                            else:logits=forward_logits(model,new_h)
                            dz=directional_gain(logits,orig_logits,y,yp)
                            margin=replay_mid_scores(new_h,o,mats)
                            od=directional_gain(margin,base_dec[(rank,typ)],y,yp)
                            actual_norm=torch.linalg.vector_norm((new_h-mid.double()),dim=1)
                            for i in range(NCASE):
                                v=bool(valid[i]);norm=float(actual_norm[i]);
                                if arm!='full_legal_unmatched' and v and abs(norm-r)>5e-5:
                                    raise AssertionError(f'physical dose mismatch {seed} {depth} {arm} {norm} {r}')
                                row={'seed':seed,'depth':depth,'rank':rank,'observer':typ,'gamma':o['gamma'],
                                     'case_index':i,'family':original[i]['family'],'delay':original[i]['delay'],
                                     'edit':edit,'arm':arm,'radius':r,'valid':v,'actual_midpoint_l2':norm,
                                     'original_label':int(y[i]),'counterfactual_label':int(yp[i]),
                                     'original_case_id':original[i]['id'],
                                     'source_is_calibration':True,'sealed_test_evaluated':False,
                                     'logit_directional_gain':float(dz[i]) if v else None,
                                     'observer_directional_gain':float(od[i]) if v else None,
                                     'observer_normalized_gain':float(od[i])/scales[(rank,typ)] if v else None,
                                     'logit_delta_l2':float(torch.linalg.vector_norm(logits[i]-orig_logits[i])) if v else None,
                                     'future_argmax_target':bool(logits[i].argmax()==yp[i]) if v else None}
                                if arm=='soft_input_matched':
                                    row['interpolation_t']=float(matched['t'][i]) if v else None
                                    row['radius_error']=float(matched['error'][i]);
                                f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
                                if v:perkey[(rank,typ,edit,r,arm)].append((i,row['logit_directional_gain'],
                                             row['observer_directional_gain'],row['observer_normalized_gain']))
                                else:invalid[(rank,typ,edit,r,arm)]+=1
                                nrows+=1
    summaries=[]
    for (rank,typ,edit,r,arm),vals in sorted(perkey.items()):
        gids=[x[0] for x in vals]
        summaries.append({'seed':seed,'depth':depth,'rank':rank,'observer':typ,'gamma':obs[(rank,typ)]['gamma'],
             'edit':edit,'radius':r,'arm':arm,'n_valid':len(vals), 'n_invalid':invalid[(rank,typ,edit,r,arm)],
             'mean_logit_gain':sum(v[1] for v in vals)/len(vals),
             'mean_observer_gain':sum(v[2] for v in vals)/len(vals),
             'mean_normalized_observer_gain':sum(v[3] for v in vals)/len(vals),
             'full_legal_input_scale':(arm=='full_legal_unmatched'),
             'calibration_only':True})
    info={'seed':seed,'depth':depth,'checkpoint_sha256':cksha,'rows':nrows,
          'per_case_file':output.name,'per_case_sha256':checksum(output),
          'per_case_content_sha256':hashlib.sha256(gzip.decompress(output.read_bytes())).hexdigest(),
          'observers':{f'{k[0]}:{k[1]}':{'gamma':v['gamma'],'calibration_MSE':v['calibration_MSE'],
                          'B_null_squared':v['B_null_squared'],'H_capture':v['H_future_logit_capture'],
                          'train_only_score_scale':scales[k],
                          'replay_metric_max_delta':v['replay_metric_max_delta'],
                          'replay_metric_deltas':v['replay_metric_deltas']} for k,v in obs.items()},
          'original_model_calibration_accuracy':float((orig_logits.argmax(1)==y).double().mean()),
          'original_full_legal_correct_target_accuracy':{e:float((l.argmax(1)==yp).double().mean()) for e,l in full_logits.items()},
          'summary_rows':summaries,'sealed_test_evaluated':False}
    (out/f'roc02e_R{depth}_S{seed}.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'depth':depth,'seed':seed,'cases':NCASE,'row_count':nrows,
        'mean_legal_midpoint_norm':{e:float(torch.linalg.vector_norm(mid_full[e]-mid.double(),dim=1).mean()) for e in mid_full},
        'selection_gamma':{str(k):v['gamma'] for k,v in obs.items()},
        'original_acc':info['original_model_calibration_accuracy']}),flush=True)
    return info


def summarize(files,out):
    results=[json.loads(Path(p).read_text()) for p in files]
    assert len(results)==12 and all(not x['sealed_test_evaluated'] for x in results)
    summary=[z for d in results for z in d['summary_rows']]
    csv_file=out/'roc02e_seed_arm_summary.csv'
    with csv_file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]))
        w.writeheader();w.writerows(summary)
    by_group=defaultdict(dict)
    for x in summary:
        by_group[(x['seed'],x['depth'],x['rank'],x['edit'],x['radius'],x['arm'])][x['observer']]=x
    comparisons=[]
    for k,v in by_group.items():
        if set(v)!={'task_only','joint'}:raise AssertionError(f'missing paired controls {k}')
        a,b=v['task_only'],v['joint']
        comparisons.append(dict(zip(('seed','depth','rank','edit','radius','arm'),k),
          n_joint=b['n_valid'],n_task=a['n_valid'],
          task_mean=a['mean_logit_gain'],joint_mean=b['mean_logit_gain'],
          joint_minus_task=b['mean_logit_gain']-a['mean_logit_gain'],
          task_observer_margin=a['mean_observer_gain'],joint_observer_margin=b['mean_observer_gain'],
          joint_minus_task_normalized=b['mean_normalized_observer_gain']-a['mean_normalized_observer_gain']))
    cmp_path=out/'roc02e_paired_seed_comparisons.csv'
    with cmp_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    primary=[x for x in comparisons if x['depth']==16 and x['rank']==7 and x['edit']=='rule'
             and x['radius']==.2 and x['arm']=='observed_input_projection']
    assert len(primary)==6
    diffs=[r['joint_minus_task'] for r in primary]
    observed=sum(diffs)/len(diffs)
    exact=[sum(s*v for s,v in zip(signs,diffs))/6 for signs in itertools.product((-1,1),repeat=6)]
    p=sum(x>=observed-1e-14 for x in exact)/64
    report={'study':'ROC-02E-calibration-only-rule-vs-cue-causal-v1','n_original_models':len(results),
      'cases_per_original_model':256,'all_rank_observer_replays_verified':True,
      'n_per_case_effect_rows':sum(r['rows'] for r in results),
      'calibration_only':True,'sealed_test_evaluated':False,
      'replay_metric_max_delta':max(v['replay_metric_max_delta'] for x in results for v in x['observers'].values()),
      'replay_metric_atol':REPLAY_METRIC_ATOL,
      'primary':{'endpoint':'original frozen eight-logit target margin gain joint minus task-only; depth16 rank7 rule edit observed projection radius 0.2',
        'seed_differences':{str(x['seed']):x['joint_minus_task'] for x in primary},
        'mean_joint_minus_task':observed,'count_positive':sum(x>0 for x in diffs),
        'one_sided_sign_flip_p_64':p,'n_model_seeds':6,
        'warning':'Calibration-source labels reused for gamma selection; exact sign-flip is descriptive, not a prospective independent-source p-value'},
      'group_summaries':[],
      'file_sha256':{'roc02e_seed_arm_summary.csv':checksum(csv_file),
                     'roc02e_paired_seed_comparisons.csv':checksum(cmp_path)}}
    for depth in DEPTHS:
        for rank in RANKS:
            for edit in ('rule','cue'):
                for arm in ('full_legal_unmatched','soft_input_matched','input_induced_hidden','observed_input_projection','dark_input_projection','random_observed','isotropic'):
                    q=[x for x in comparisons if x['depth']==depth and x['rank']==rank and x['edit']==edit
                         and x['radius']==.2 and x['arm']==arm]
                    if len(q)!=6:raise AssertionError((depth,rank,edit,arm,'wrong count'))
                    report['group_summaries'].append({'depth':depth,'rank':rank,'edit':edit,'radius':.2,'arm':arm,
                         'mean_task_gain':sum(v['task_mean'] for v in q)/6,
                         'mean_joint_gain':sum(v['joint_mean'] for v in q)/6,
                         'mean_paired_diff':sum(v['joint_minus_task'] for v in q)/6,
                         'positive_seed_count':sum(v['joint_minus_task']>0 for v in q),
                         'total_invalid_task':sum(256-v['n_task'] for v in q),
                         'total_invalid_joint':sum(256-v['n_joint'] for v in q)})
    dest=out/'roc02e_summary.json'
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['primary'],indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir',type=Path,default=Path('original_models'))
    ap.add_argument('--checkpoint-dir',type=Path,default=Path('original_models/checkpoints'))
    ap.add_argument('--receipt-file',type=Path,default=Path('experiments/roc02_delayed_rule_v1/roc02c_calibration/roc02c_full_receipts.jsonl'))
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--depth',type=int,choices=DEPTHS)
    ap.add_argument('--summarize',action='store_true')
    args=ap.parse_args()
    if args.summarize:
        files=[args.out_dir/f'roc02e_R{d}_S{s}.json' for d in DEPTHS for s in SEEDS]
        summarize(files,args.out_dir)
        return
    if args.seed is None or args.depth is None:ap.error('require seed and depth for one model')
    archived=[json.loads(line) for line in args.receipt_file.read_text().splitlines() if line.strip()]
    if len(archived)!=12:raise AssertionError('need twelve original ROC-02C raw receipts')
    evaluate_cell(args.seed,args.depth,args.source_dir,args.checkpoint_dir,archived,args.out_dir)

if __name__=='__main__':main()