"""PEF-09R: registered post-ablation linear recovery, conditional on PEF-09.

The PEF-09R protocol was committed before heldout recovery scores were seen.
All PCA bases and ridge coefficients are computed on the 768 probe states.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from research.pef09 import pef09 as p9
from research.pef08 import pef08 as p8

EXPECTED_SELECTION_HASH='f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157'
REGISTRATION_COMMIT='4ea181286bea7d3e935f2585a2280f14e1b90117'
RANK_SWEEP=(1,2,4,8)


def fit_for_states(state,y):
    from research.pef07 import pef07 as p7
    return p7.fit_ridge(state,y)


def predict(state,fit):
    from research.pef07 import pef07 as p7
    return p7.apply_ridge(state,fit)


def accuracy(pred,truth):
    return {'correct':int((pred==truth).sum()),'total':int(len(truth)),
            'recall_0':float((pred[truth==0]==0).float().mean()),
            'recall_1':float((pred[truth==1]==1).float().mean())}


def marshal_fit(fit):
    return [v.tolist() for v in fit]


def unmarshal_fit(fit):
    return tuple(torch.tensor(x,dtype=torch.float64) for x in fit)


@torch.no_grad()
def patch_to_horizons(model,bits,trace,t,mu,u):
    patch=p9.projector(trace[t],mu,u)
    return p8.replay(model,bits,patch,t,12),p8.replay(model,bits,patch,t,14)


def rank_project(state,mean,axes):
    """Remove top-k probe PCA span from all rows without focal gating."""
    if axes.ndim!=2 or axes.shape[1]!=state.shape[1]:
        raise ValueError('axes shape mismatch')
    gram=axes.double()@axes.double().T
    if not torch.allclose(gram,torch.eye(len(axes),dtype=torch.float64),atol=1e-7):
        raise AssertionError('nonorthonormal axes')
    a=axes.to(state.dtype)
    center=mean.to(state.dtype)
    return state-(((state-center)@a.T)@a)


def pca_geometry(probe):
    center=probe.double().mean(0)
    x=probe.double()-center
    _,singular,right=torch.linalg.svd(x,full_matrices=False)
    energy=singular.square()
    norm=float(energy.sum())
    fractions={str(k):float(energy[:k].sum()/energy.sum()) for k in RANK_SWEEP}
    eff_rank=float(norm**2/float(energy.square().sum()))
    return center,right,{'top_k_variance_fraction':fractions,
                         'effective_rank':eff_rank,
                         'singular_values':singular.tolist(),
                         'top_eigenvalue_over_second':float(energy[0]/energy[1])}


def cosines(selected):
    data={}
    for t in (8,12):
        a=next(row for row in selected['rows'] if row['loop']==t and row['pair']==[0,1])
        b=next(row for row in selected['rows'] if row['loop']==t and row['pair']==[2,3])
        v=torch.tensor(a['conditions'][0]['unit'])
        w=torch.tensor(b['conditions'][0]['unit'])
        g=torch.tensor(a['conditions'][2]['unit'])
        ratios=[x['target_probe_energy_ratio'] for x in a['conditions'][3:]]
        data[str(t)]={'pair_abs_cosine':float(abs(v@w)),
                      'pair01_abs_cosine_with_global_pc':float(abs(v@g)),
                      'pair23_abs_cosine_with_global_pc':float(abs(w@g)),
                      'matched_random_energy_ratio_range':[min(ratios),max(ratios)]}
    return data


@torch.no_grad()
def prepare_probe(model,selection,probe_bits):
    task=selection['task']
    trace=p8.states_by_loop(model,probe_bits)
    y=p8.labels(probe_bits,task)
    natural={'h12':marshal_fit(fit_for_states(trace[12],y)),
             'h14':marshal_fit(fit_for_states(trace[14],y))}
    center, axes, geometry=pca_geometry(trace[12])
    rank=[]
    for k in RANK_SWEEP:
        residual=rank_project(trace[12],center,axes[:k])
        rank.append({'k':k,'ridge':marshal_fit(fit_for_states(residual,y))})
    tests=[]
    for t in (8,12):
        for pair in ((0,1),(2,3)):
            row=next(item for item in selection['rows'] if item['loop']==t and item['pair']==list(pair))
            mu=torch.tensor(row['center'],dtype=torch.float64)
            for name in ('target_pair_svd','global_pc'):
                cond=next(x for x in row['conditions'] if x['name']==name)
                vector=torch.tensor(cond['unit'],dtype=torch.float64)
                h12,h14=patch_to_horizons(model,probe_bits,trace,t,mu,vector)
                tests.append({'loop':t,'pair':list(pair),'condition':name,
                              'ridge_h12':marshal_fit(fit_for_states(h12,y)),
                              'ridge_h14':marshal_fit(fit_for_states(h14,y))})
    return {'seed':selection['seed'],'task':task,
            'checkpoint_sha256':selection['checkpoint_sha256'],
            'pca_center':center.tolist(),
            'pca_top8_axes':axes[:8].tolist(),
            'pca_geometry':geometry,
            'pair_cosines':cosines(selection),
            'natural_ridges':natural,'rank_sweep':rank,'tests':tests}


@torch.no_grad()
def evaluate_heldout(model,selection,prepared,held_bits):
    y=p8.labels(held_bits,selection['task'])
    trace=p8.states_by_loop(model,held_bits)
    result={'seed':selection['seed'],'task':selection['task'],'geometry':prepared['pca_geometry'],
            'pair_cosines':prepared['pair_cosines'],
            'natural':{},'rank_sweep':[],'conditions':[]}
    for name,idx in (('h12',12),('h14',14)):
        refit=unmarshal_fit(prepared['natural_ridges'][name])
        result['natural'][name]=accuracy(predict(trace[idx],refit),y)
    center=torch.tensor(prepared['pca_center'],dtype=torch.float64)
    fullaxes=torch.tensor(prepared['pca_top8_axes'],dtype=torch.float64)
    for r in prepared['rank_sweep']:
        k=r['k']
        x=rank_project(trace[12],center,fullaxes[:k])
        score=accuracy(predict(x,unmarshal_fit(r['ridge'])),y)
        result['rank_sweep'].append({'k':k,'new_ridge':score})
    for r in prepared['tests']:
        t=r['loop']; pair=r['pair']; name=r['condition']
        row=next(item for item in selection['rows'] if item['loop']==t and item['pair']==pair)
        cond=next(item for item in row['conditions'] if item['name']==name)
        mu=torch.tensor(row['center'],dtype=torch.float64)
        direction=torch.tensor(cond['unit'],dtype=torch.float64)
        h12,h14=patch_to_horizons(model,held_bits,trace,t,mu,direction)
        native=(p9.native_score(model,h12)>=0).long()
        frozen_h14=(p8.ridge_margin(h14,p9.ridge_from_selection(selection))>=0).long()
        result['conditions'].append({'loop':t,'pair':pair,'condition':name,
            'frozen_native_h12':accuracy(native,y),
            'frozen_ridge_h14':accuracy(frozen_h14,y),
            'new_ridge_h12':accuracy(predict(h12,unmarshal_fit(r['ridge_h12'])),y),
            'new_ridge_h14':accuracy(predict(h14,unmarshal_fit(r['ridge_h14'])),y)})
    return result


def run(evidence,selection_ledger,output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence,selection_ledger,output=map(Path,(evidence,selection_ledger,output))
    output.mkdir(parents=True,exist_ok=True)
    if hashlib.sha256(selection_ledger.read_bytes()).hexdigest()!=EXPECTED_SELECTION_HASH:
        raise AssertionError('PEF-09 frozen probe selection ledger has changed')
    records=json.loads(selection_ledger.read_text())
    if not records.get('all_selections_frozen_before_heldout'):
        raise AssertionError('probe-only selection was not sealed')
    cat,sig,manifest=p8.source_data()
    _,receipts=p9.verify_original_evidence(evidence,cat,sig,manifest)
    models_to_score=[]
    prepared=[]
    for rec in receipts:
        sel=next(x for x in records['selections'] if x['seed']==rec['seed'] and x['task']==rec['task'])
        model=p8.checked_model(evidence,rec)
        record=prepare_probe(model,sel,cat['probe'])
        prepared.append(record)
        models_to_score.append((rec,sel))
        print('PEF09R_RIDGES_FITTED',rec['seed'],rec['task'],flush=True)
    freeze={'campaign':'PEF-09R','protocol_commit':REGISTRATION_COMMIT,
            'source_sha256':p9.SIGNATURE,
            'pef09_selection_ledger_sha256':EXPECTED_SELECTION_HASH,
            'all_new_ridges_probe_only':True,'all_ridges_frozen_before_heldout':True,
            'models':prepared}
    (output/'frozen_ridge_ledger.json').write_bytes(p9.canonical(freeze))
    ridge_sha=hashlib.sha256((output/'frozen_ridge_ledger.json').read_bytes()).hexdigest()
    print('PEF09R_FROZEN_RIDGES_SHA256',ridge_sha,flush=True)
    evaluated=[]
    for ((rec,sel),prepared_one) in zip(models_to_score,prepared,strict=True):
        model=p8.checked_model(evidence,rec)
        result=evaluate_heldout(model,sel,prepared_one,cat['heldout'])
        evaluated.append(result)
        print('PEF09R_EVALUATED',rec['seed'],rec['task'],flush=True)
    decisions=[]
    for seed in p9.SEEDS:
        model=next(x for x in evaluated if x['seed']==seed and x['task']=='coalition')
        targeted=[x for x in model['conditions'] if x['loop']==8 and x['condition']=='target_pair_svd']
        yes=all(x['new_ridge_h12']['correct']>=576 for x in targeted)
        decisions.append({'seed':seed,'both_pairs_recover':yes,
                          'scores':[{'pair':x['pair'],'h12_new_ridge':x['new_ridge_h12']['correct']}
                                    for x in targeted]})
    count=sum(x['both_pairs_recover'] for x in decisions)
    result={'campaign':'PEF-09R','protocol_commit':REGISTRATION_COMMIT,
            'frozen_ridge_ledger_sha256':ridge_sha,
            'residual_linear_recovery_decision':('criterion_met' if count>=2 else 'criterion_not_met'),
            'recovery_seeds_met':count,'per_seed':decisions,
            'models':evaluated}
    (output/'summary.json').write_bytes(p9.canonical(result))
    print(json.dumps({'decision':result['residual_linear_recovery_decision'],'seeds':decisions},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',required=True,type=Path)
    p.add_argument('--selection-ledger',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    run(args.evidence,args.selection_ledger,args.out)


if __name__=='__main__':
    main()
