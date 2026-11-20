"""Exposed development comparison of both complete frozen native conditions.

Aggregate publication only. Dictionaries are separately fitted: never subtract
BF16/NF4 feature-index codes or interpret weight SVD as learned VPD/BPD.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.nan_gym05.common import read,write,sha
from experiments.nan_gym05.grade import grade
from .probe_towers import layer_states,load_npz,encoding


def physical_metrics(A,B):
    if A.shape!=B.shape or A.ndim!=2 or not np.isfinite(A).all() or not np.isfinite(B).all():raise ValueError('Matched finite physical coordinates required')
    na=np.linalg.norm(A,axis=1);nb=np.linalg.norm(B,axis=1)
    # Undefined cosines/relative changes remain NaN locally and are excluded
    # with an explicit count at aggregation, never silently coerced to zero.
    cosine=np.divide(np.sum(A*B,axis=1),na*nb,out=np.full(len(A),np.nan),where=(na*nb)>0)
    relative=np.divide(np.linalg.norm(B-A,axis=1),na,out=np.full(len(A),np.nan),where=na>0)
    return {'BF16_norm':na,'NF4_norm':nb,'cosine':cosine,'NF4_minus_BF16_relative_l2':relative}


def grouped(meta,metrics,*,phases=True):
    groups=defaultdict(list)
    for i,r in enumerate(meta):
        key=(r['split'],r['suite'],r['copy_control'])+((r['loop'],) if phases else ())
        groups[key].append(i)
    out=[]
    for key,ii in sorted(groups.items()):
        byroot=defaultdict(list)
        for i in ii:byroot[meta[i]['root']].append(i)
        row={'split':key[0],'suite':key[1],'copy_control':key[2],'source_roots':len(byroot),'vectors':len(ii)}
        if phases:row['loop']=key[3]
        for name,values in metrics.items():
            finite=[float(np.mean(values[jj][np.isfinite(values[jj])])) for jj in byroot.values() if np.isfinite(values[jj]).any()]
            row['root_macro_'+name]=float(np.mean(finite)) if finite else None
            row[name+'_undefined_vectors']=int(sum(not np.isfinite(values[i]) for i in ii))
        out.append(row)
    return out


def compare(runs,out):
    summaries,private=grade(runs);assert not summaries['partial']
    audits={b:read(runs/'audit'/(b+'_independent.json')) for b in ('bf16','nf4')}
    for b,a in audits.items():
        assert a['status']=='passed' and a['execution_receipt_sha256']==sha(runs/b/'receipt.json')
        assert len(read(runs/b/'sae_tower/manifest.json')['receipts'])==4
    paired=[];p={b:{r['id']:r for r in private if r['backend']==b} for b in ('bf16','nf4')}
    assert set(p['bf16'])==set(p['nf4']) and len(p['bf16'])==322
    groups=defaultdict(list)
    for key,a in p['bf16'].items():
        b=p['nf4'][key];condition='copy' if a['copy_control'] else ('hop'+str(a['hops']) if a['hops'] else 'main')
        groups[(a['split'],a['suite'],condition)].append((a,b))
    for key,rr in sorted(groups.items()):
        paired.append({'split':key[0],'suite':key[1],'condition':key[2],'records':len(rr),'roots':len({a['root'] for a,b in rr}),
            'both_correct':sum(a['correct'] and b['correct'] for a,b in rr),'BF16_only_correct':sum(a['correct'] and not b['correct'] for a,b in rr),
            'NF4_only_correct':sum(b['correct'] and not a['correct'] for a,b in rr),'both_wrong':sum(not a['correct'] and not b['correct'] for a,b in rr),
            'same_accepted_prediction':sum(a['format_valid'] and b['format_valid'] and a['prediction']==b['prediction'] for a,b in rr)})
    layers=[]
    for layer in (5,11,17,21):
        A,ma=layer_states(runs/'bf16',layer);B,mb=layer_states(runs/'nf4',layer)
        keys=('id','root','suite','split','copy_control','loop')
        assert [{k:r[k] for k in keys} for r in ma]==[{k:r[k] for k in keys} for r in mb]
        dictionaries={b:load_npz(runs/b/'sae_tower'/f'layer{layer}.npz') for b in ('bf16','nf4')}
        metrics=physical_metrics(A,B)
        for state_name,H in (('BF16',A),('NF4',B)):
            norm=np.linalg.norm(H,axis=1)
            for dictionary,sae in dictionaries.items():
                _,_,error,_=encoding(H,sae,4)
                metrics[state_name+'_state_'+dictionary+'_dictionary_relative_reconstruction_l2']=error/norm
        phase={}
        for i,r in enumerate(ma):phase.setdefault(r['id'],{})[r['loop']]=i
        pairs=[(v[1],v[2]) for v in phase.values()];indices=[i for i,j in pairs]
        da=np.array([A[j]-A[i] for i,j in pairs]);db=np.array([B[j]-B[i] for i,j in pairs])
        layer_row={'layer_zero_based':layer,'matched_physical_states':len(A),'states':grouped(ma,metrics),
            'loop_displacements':grouped([ma[i] for i in indices],physical_metrics(da,db),phases=False),
            'dictionary_byte_sha256':{b:sha(runs/b/'sae_tower'/f'layer{layer}.npz') for b in dictionaries}}
        layers.append(layer_row)
    subspaces=[]
    for layer in (5,11,17,21):
        for module in ('down_proj','o_proj'):
            factors={}
            for b in ('bf16','nf4'):
                manifest=read(runs/b/'linear_baselines/manifest.json')
                row=next(r for r in manifest['receipts'] if r['layer_zero_based']==layer and r['module']==module)
                file=runs/b/'linear_baselines'/row['file'];assert sha(file)==row['file_sha256'];factors[b]=load_npz(file)
            for rank in (8,16):
                q={b:np.linalg.qr(f['U'+str(rank)].T,mode='reduced')[0] for b,f in factors.items()}
                cos=np.linalg.svd(q['bf16'].T@q['nf4'],compute_uv=False)
                subspaces.append({'layer_zero_based':layer,'module':module,'rank':rank,'principal_cosines':cos.tolist(),
                    'scope':'backend-effective SVD module-output subspaces; not trained VPD/BPD or an o_proj-to-SAE bridge'})
    write(out,{'experiment':'NAN-GYM-05','status':'passed','evidence_class':'exposed descriptive paired diagnostics; one checkpoint, shared source roots',
        'analysis_timing':'post-registration helper designed after BF16 results, before completed NF4; no inference/fit changes or task filtering',
        'formal_inferential_comparisons':0,'accuracy_discordance':paired,'layers':layers,'weight_SVD_subspaces':subspaces,
        'audits':{b:{k:v for k,v in a.items() if k!='artifact_sha256'} for b,a in audits.items()},
        'audit_file_sha256':{b:sha(runs/'audit'/(b+'_independent.json')) for b in audits},
        'BPD_transfer_attempted':False,'sealed_test_accessed':False,
        'privacy':'aggregate only; no routing text, per-record predictions/IDs, tokens, hidden states or codes'})
    print({'status':'passed','matched_contexts':322,'paired_physical_states':2576,'layers':len(layers),'weight_rank_cells':len(subspaces)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_gym05');p.add_argument('--out',default='runs/nan_gym05/paired_aggregate.json');a=p.parse_args()
    compare(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
