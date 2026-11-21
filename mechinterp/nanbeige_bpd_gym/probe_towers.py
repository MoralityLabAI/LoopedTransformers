"""Post-registration observational probes of the fixed CAL-only SAE tower.

No feature/task selection, causal edits, refitting or hypothesis tests. Full
per-record codes stay local; publish only aggregate metrics and CAL-fit weight
geometry. SVD is not relabelled as trained VPD/BPD.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.nan_gym05.common import read,rows,sha,write
from mechinterp.sae_causal_transfer.adapter import to_physical_dictionary
from .crossref import probe_and_crossref,REVISION

def encoding(H,sae,k):
    E,b,D=to_physical_dictionary(sae['encoder'],sae['bias'],sae['decoder'],sae['mean'],sae['scale'])
    mean=np.asarray(sae['mean'],dtype=np.float64);scale=np.asarray(sae['scale'],dtype=np.float64)
    bias=np.asarray(sae['decoder_bias'],dtype=np.float64)
    pre=H@E.T+b
    normalized=((H-mean)/scale)@np.asarray(sae['encoder'],dtype=np.float64).T+np.asarray(sae['bias'],dtype=np.float64)
    err=float(np.max(np.abs(pre-normalized)))
    if err>1e-7*(1+float(np.max(np.abs(normalized)))):raise ValueError('Physical/normalized encoder coordinate mismatch')
    indices=np.argsort(-pre,axis=1,kind='stable')[:,:k];Z=np.zeros_like(pre)
    np.put_along_axis(Z,indices,np.maximum(np.take_along_axis(pre,indices,axis=1),0),axis=1)
    reconstructed=mean+scale*bias+Z@D
    mse=np.mean(((H-reconstructed)/scale)**2,axis=1)
    physical_l2=np.linalg.norm(H-reconstructed,axis=1)
    return Z,mse,physical_l2,err

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def layer_states(folder,layer):
    vectors=[];meta=[]
    for file in sorted(folder.glob('root_*.json')):
        r=read(file);statefile=folder/r['state_file'];assert sha(statefile)==r['state_file_sha256']
        with np.load(statefile,allow_pickle=False) as z:
            for o in r['observations']:
                if o['layer_zero_based']!=layer or o['site']!='residual_after_layer':continue
                vectors.append(z[o['array_key']]);meta.append({'id':o['task_id'],'root':r['root'],'suite':r['suite'],'split':r['split'],
                    'copy_control':o['task_id'].endswith(':copy'),'loop':o['loop'],'state_file_sha256':r['state_file_sha256']})
    if len(vectors)!=644:raise ValueError('Require all 322 contexts at both loops')
    H=np.asarray(vectors,dtype=np.float64)
    if H.shape!=(644,3072) or not np.isfinite(H).all():raise ValueError('Invalid physical state matrix')
    return H,meta

def aggregates(meta,Z,mse,random_mse,physical_l2,H):
    groups=defaultdict(list)
    for i,r in enumerate(meta):groups[(r['split'],r['suite'],r['copy_control'],r['loop'])].append(i)
    result=[]
    for key,indices in sorted(groups.items()):
        root_indices=defaultdict(list)
        for i in indices:root_indices[meta[i]['root']].append(i)
        root=lambda a:float(np.mean([np.mean(a[ii]) for ii in root_indices.values()]))
        result.append({'split':key[0],'suite':key[1],'copy_control':key[2],'loop':key[3],'root_groups':len(root_indices),'vectors':len(indices),
            'root_macro_normalized_MSE':root(mse),'root_macro_random_initial_MSE':root(random_mse),'root_macro_MSE_improvement':root(random_mse-mse),
            'root_macro_physical_reconstruction_l2':root(physical_l2),'root_macro_state_l2':root(np.linalg.norm(H,axis=1)),
            'root_macro_channel2279':root(H[:,2279]),
            'feature_active_fraction_root_macro':np.mean([np.mean(Z[ii]>0,axis=0) for ii in root_indices.values()],axis=0).tolist()})
    return result

def probe(runs,backend,out):
    cfg=read('experiments/nan_gym05/config.json');seal=read('experiments/nan_gym05/SOURCE_MANIFEST.json');folder=runs/backend
    for n,h in seal['observational_bridge_sha256'].items():
        if sha(n)!=h:raise ValueError('Registered physical bridge changed: '+n)
    assert read(folder/'receipt.json')['status']=='completed'
    audited=read(runs/'audit'/(backend+'_independent.json'))
    assert audited['status']=='passed' and audited['execution_receipt_sha256']==sha(folder/'receipt.json')
    fitted=read(folder/'sae_tower/manifest.json');assert len(fitted['receipts'])==4
    linear=read(folder/'linear_baselines/manifest.json');results=[];geometry=[]
    for layer in cfg['layers_zero_based']:
        fitted_receipt=next(r for r in fitted['receipts'] if r['layer_zero_based']==layer)
        base=folder/'sae_tower';trained_file=base/f'layer{layer}.npz';initial_file=base/f'layer{layer}_random_initial.npz';fit_file=base/f'layer{layer}_fit_manifest.json'
        assert sha(trained_file)==fitted_receipt['artifact_sha256'] and sha(initial_file)==fitted_receipt['random_initial_sha256'] and sha(fit_file)==fitted_receipt['fit_manifest_sha256']
        fit=read(fit_file);assert fit['calibration_only'] and fit['calibration_roots']==32 and fit['calibration_vectors']==256 and fit['settings']==cfg['SAE']
        assert fit['source_manifest_sha256']==sha('experiments/nan_gym05/SOURCE_MANIFEST.json')
        H,meta=layer_states(folder,layer);sae=load_npz(trained_file);initial=load_npz(initial_file)
        Z,mse,l2,err=encoding(H,sae,cfg['SAE']['top_k']);_,random_mse,_,random_err=encoding(H,initial,cfg['SAE']['top_k'])
        cal=np.array([r['split']=='calibration' and not r['copy_control'] for r in meta]);assert int(cal.sum())==256
        if abs(float(mse[cal].mean())-fitted_receipt['normalized_CAL_MSE'])>1e-3:raise ValueError('Fitted Torch and physical probe reconstruction disagree')
        per_layer={'backend':backend,'layer_zero_based':layer,'dictionary_sha256':sha(trained_file),'fit_manifest_sha256':sha(fit_file),
            'physical_vs_normalized_pre_max_abs':err,'random_initial_coordinate_error':random_err,'calibration_vectors':int(cal.sum()),
            'CAL_feature_active_fraction':np.mean(Z[cal]>0,axis=0).tolist(),'groups':aggregates(meta,Z,mse,random_mse,l2,H)}
        # Pair loops at the SAME source prefix. No clean-correct selection.
        paired=defaultdict(dict)
        for i,r in enumerate(meta):paired[r['id']][r['loop']]=i
        gap_groups=defaultdict(list)
        for key,v in paired.items():
            i,j=v[1],v[2];r=meta[i];gap_groups[(r['split'],r['suite'],r['copy_control'])].append((r['root'],float(np.linalg.norm(H[j]-H[i])),float(np.linalg.norm(Z[j]-Z[i]))))
        per_layer['loop_gaps']=[]
        for key,rr in sorted(gap_groups.items()):
            byroot=defaultdict(list)
            for root,h,z in rr:byroot[root].append((h,z))
            means=np.array([np.mean(x,axis=0) for x in byroot.values()])
            per_layer['loop_gaps'].append({'split':key[0],'suite':key[1],'copy_control':key[2],'root_groups':len(byroot),'contexts':len(rr),
                'root_macro_physical_state_l2':float(means[:,0].mean()),'root_macro_sparse_code_l2':float(means[:,1].mean())})
        baseline=next(r for r in linear['receipts'] if r['layer_zero_based']==layer and r['module']=='down_proj')
        factors_file=folder/'linear_baselines'/baseline['file'];assert sha(factors_file)==baseline['file_sha256'];factors=load_npz(factors_file)
        first=next(i for i,r in enumerate(meta) if r['split']=='calibration' and not r['copy_control'] and r['loop']==1)
        site={'checkpoint_revision':REVISION,'backend':backend,'layer_zero_based':layer,'loop':1,'site':'residual_after_layer','width':3072,'artifact_sha256':meta[first]['state_file_sha256']}
        rng=np.random.default_rng(cfg['source_seed']+layer);Q=np.linalg.qr(rng.normal(size=(3072,16)),mode='reduced')[0].T
        for rank in (8,16):
            U=factors['U'+str(rank)];assert U.shape==(rank,3072)
            weights=probe_and_crossref(H[first],sae,U,state_meta=site,sae_meta=site|{'artifact_sha256':sha(trained_file),'fit_split':'calibration','fit_manifest_sha256':sha(fit_file)},
                vpd_meta=site|{'artifact_sha256':sha(factors_file),'factor_role':'physical_output_directions'})
            random=Q[:rank]*np.linalg.norm(U,axis=1)[:,None]
            controlfile=out/'local'/f'{backend}_layer{layer}_rank{rank}_Haar_control.npz';controlfile.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(controlfile,output_directions=random)
            null=probe_and_crossref(H[first],sae,random,state_meta=site,sae_meta=site|{'artifact_sha256':sha(trained_file),'fit_split':'calibration','fit_manifest_sha256':sha(fit_file)},
                vpd_meta=site|{'artifact_sha256':sha(controlfile),'factor_role':'physical_output_directions'})
            names=('encoder_vpd_cosine','decoder_vpd_cosine','encoder_vpd_preactivation_sensitivity','zero_norm_counts')
            geometry.append({'backend':backend,'layer_zero_based':layer,'rank':rank,'factor_kind':'SVD_baseline_not_trained_VPD_or_BPD',
                'site_bridge':'down_proj output added immediately to physical residual_after_layer; identity coordinates only, not a causal assertion',
                'loop_basis_shared':[1,2],'dictionary_sha256':sha(trained_file),'factor_sha256':sha(factors_file),'Haar_control_file_sha256':sha(controlfile),
                'SVD':{n:weights[n] for n in names},'matched_norm_Haar_output_control':{n:null[n] for n in names}})
        # Derived per-record vectors, including routing, MUST remain local.
        local=out/'local';local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/f'{backend}_layer{layer}_codes.npz',codes=Z,normalized_MSE=mse,random_initial_MSE=random_mse)
        write(local/f'{backend}_layer{layer}_index.json',meta);results.append(per_layer)
    write(out/(backend+'_aggregate_probes.json'),{'evidence_class':'observational, one checkpoint; no causal or equivalence test','formal_inferential_comparisons':0,
        'layers':results,'BPD_transfer_attempted':False,'sealed_test_accessed':False})
    write(out/(backend+'_weight_geometry.json'),{'evidence_class':'CAL-fitted dictionary versus weight-only SVD/Haar geometry; not learned VPD/BPD','comparisons':geometry})
    print({'backend':backend,'layers':len(results),'rank_geometry_cells':len(geometry),'routing_vectors_published':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_gym05');p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--out',default='runs/nan_gym05/probes');a=p.parse_args()
    probe(Path(a.runs),a.backend,Path(a.out))

if __name__=='__main__':main()
