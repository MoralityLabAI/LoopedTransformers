"""Frozen post-fit learned-factor/SAE cross-reference on all earlier contexts.

All routing derivatives remain local. Published files contain only aggregates
and weight geometry, with no source record IDs, predictions or vector arrays.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from mechinterp.nanbeige_bpd_gym.probe_towers import load_npz,encoding
from mechinterp.nanbeige_bpd_gym.crossref import probe_and_crossref,REVISION
from .io import read,sha,write
from .pointwise import masks


def cached_states(backend,audit):
    H=[];X=[];Y=[];meta=[];folder=Path('runs/nan_gym05')/backend
    for file in sorted(folder.glob('root_*.json')):
        r=read(file);statefile=folder/r['state_file'];assert sha(statefile)==r['state_file_sha256']
        assert sha(file)==audit['artifact_sha256'][file.name] and sha(statefile)==audit['artifact_sha256'][statefile.name]
        with np.load(statefile,allow_pickle=False) as z:
            by=defaultdict(dict)
            for o in r['observations']:
                if o['layer_zero_based']==21:by[o['task_id'],o['loop']][o['site']]=o['array_key']
            for (key,loop),sites in sorted(by.items()):
                H.append(z[sites['residual_after_layer']]);X.append(z[sites['down_proj_input']]);Y.append(z[sites['down_proj_output']])
                meta.append({'id':key,'root':r['root'],'loop':loop,'split':r['split'],'suite':r['suite'],'copy_control':key.endswith(':copy'),'state_file_sha256':sha(statefile)})
    assert len(H)==644
    return np.asarray(H,dtype=np.float64),np.asarray(X,dtype=np.float64),np.asarray(Y,dtype=np.float64),meta


def root_correlation(a,b):
    a=a-a.mean(0);b=b-b.mean(0);den=np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0)[None,:]
    return np.divide(a.T@b,den,out=np.full(den.shape,np.nan),where=den>0)


def nullable(matrix):
    return [[float(v) if np.isfinite(v) else None for v in row] for row in matrix]


def probe(runs,backend,passes):
    root=Path(__file__).parent;cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen fitting/probing code changed: '+name)
    folder=runs/backend/f't{passes}_layer21_down_proj_rank8';receipt=read(folder/'receipt.json');assert receipt['status']=='completed'
    assert receipt['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
    final=folder/receipt['final_checkpoint']['file'];assert sha(final)==receipt['final_checkpoint']['sha256']
    initial=folder/receipt['initial_checkpoint']['file'];assert sha(initial)==receipt['initial_checkpoint']['sha256']
    bank=load_npz(final);old=load_npz(initial);ci=receipt['final_checkpoint']['CI_config']
    replay=load_npz(folder/'CI_replay_local.npz');replayed=masks(replay['X'],bank,ci)
    np.testing.assert_allclose(replayed,replay['masks'],rtol=2e-5,atol=1e-5)
    frozen=seal['probe_cache'][backend]
    assert sha(Path('runs/nan_gym05/audit')/(backend+'_independent.json'))==frozen['independent_audit_sha256']
    sae_file=Path('runs/nan_gym05')/backend/'sae_tower/layer21.npz';assert sha(sae_file)==frozen['sae_dictionary_sha256']
    old_audit=read(Path('runs/nan_gym05/audit')/(backend+'_independent.json'))
    H,X,Y,meta=cached_states(backend,old_audit);sae=load_npz(sae_file);Z,_,_,_=encoding(H,sae,4)
    gate=masks(X,bank,ci);acts=X@bank['V'];learned=acts@bank['U'];initial_pred=(X@old['V'])@old['U']
    rng=np.random.default_rng(cfg['source_seed']+passes)
    rv=np.linalg.qr(rng.normal(size=bank['V'].shape),mode='reduced')[0]*np.linalg.norm(bank['V'],axis=0)
    ru=np.linalg.qr(rng.normal(size=bank['U'].T.shape),mode='reduced')[0].T*np.linalg.norm(bank['U'],axis=1)[:,None]
    random_pred=(X@rv)@ru
    group=defaultdict(list)
    for i,r in enumerate(meta):group[r['split'],r['suite'],r['copy_control'],r['loop']].append(i)
    summaries=[]
    for key,indices in sorted(group.items()):
        byroot=defaultdict(list)
        for i in indices:byroot[meta[i]['root']].append(i)
        def macro(values):return float(np.mean([np.mean(values[ii]) for ii in byroot.values()]))
        norm=np.linalg.norm(Y,axis=1);relative=lambda pred:np.linalg.norm(pred-Y,axis=1)/np.maximum(norm,1e-12)
        mu=bank['mean_comp_acts'][key[3]-1] if passes==2 else np.zeros(cfg['rank'])
        energy=np.abs(gate*(acts-mu))*np.linalg.norm(bank['U'],axis=1)
        root_z=np.array([Z[ii].mean(0) for ii in byroot.values()]);root_energy=np.array([energy[ii].mean(0) for ii in byroot.values()])
        corr=root_correlation(root_z,root_energy) if len(byroot)>=3 else np.full((128,8),np.nan)
        summaries.append({'split':key[0],'suite':key[1],'copy_control':key[2],'application_loop':key[3],
            'out_of_fit_depth':key[3]>passes,'source_root_groups':len(byroot),'vectors':len(indices),
            'learned_pure_component_module_relative_l2':macro(relative(learned)),
            'initial_SVD_pure_component_module_relative_l2':macro(relative(initial_pred)),
            'matched_norm_Haar_pure_component_module_relative_l2':macro(relative(random_pred)),
            'pointwise_gate_root_macro':np.mean([gate[ii].mean(0) for ii in byroot.values()],axis=0).tolist(),
            'SAE_feature_vs_centered_routed_component_energy_root_correlation':nullable(corr),
            'undefined_correlation_cells':int(np.isnan(corr).sum()),
            'scope':'clean cached input application, not an online intervention; root summaries are not asserted independent replications'})
    public_geometry=[];controlfile=folder/'probe_local_Haar.npz'
    with controlfile.open('wb') as f:np.savez_compressed(f,V=rv,U=ru)
    for loop in (1,2):
        first=next(i for i,r in enumerate(meta) if r['split']=='calibration' and not r['copy_control'] and r['loop']==loop)
        site={'checkpoint_revision':REVISION,'backend':backend,'layer_zero_based':21,'loop':loop,'site':'residual_after_layer','width':3072}
        sae_meta=site|{'artifact_sha256':sha(sae_file),'fit_split':'calibration','fit_manifest_sha256':sha(sae_file.with_name('layer21_fit_manifest.json'))}
        named={'learned_VPD_T1' if passes==1 else 'learned_BPD_T2':(bank['U'],sha(final)),
            'initial_SVD_not_trained':(old['U'],sha(initial)),'matched_norm_Haar_not_trained':(ru,sha(controlfile))}
        for name,(U,digest) in named.items():
            result=probe_and_crossref(H[first],sae,U,state_meta=site|{'artifact_sha256':meta[first]['state_file_sha256']},sae_meta=sae_meta,
                vpd_meta=site|{'artifact_sha256':digest,'factor_role':'physical_output_directions'})
            public_geometry.append({'basis':name,'application_loop':loop,'out_of_fit_depth':loop>passes,
                'basis_artifact_sha256':digest,'SAE_dictionary_sha256':sha(sae_file),
                'bridge':'down_proj output identity addition to pre-final-norm residual_after_layer only',
                **{k:result[k] for k in ('encoder_vpd_cosine','decoder_vpd_cosine','encoder_vpd_preactivation_sensitivity','zero_norm_counts')}})
    output=runs/'probes';write(output/f'{backend}_t{passes}_aggregate.json',{'status':'passed','backend':backend,'fit_passes':passes,
        'checkpoint_sha256':sha(final),'fit_receipt_sha256':sha(folder/'receipt.json'),'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
        'pointwise_replay_max_abs':float(np.max(np.abs(replayed-replay['masks']))),
        'evidence_class':'learned factor/SAE observational cross-reference; not a causal or equivalence test','groups':summaries,'geometry':public_geometry,
        'formal_inferential_comparisons':0,'BPD_donor_transfer_attempted':False,'sealed_test_accessed':False,
        'privacy':'aggregate geometry/root statistics only; all per-record routing derivatives stay local'})
    # Preserve full derivative evidence locally, including routing.
    with (folder/'probe_local_codes.npz').open('wb') as f:np.savez_compressed(f,SAE_codes=Z,component_reads=acts,controller_masks=gate)
    write(folder/'probe_local_index.json',meta)
    print({'status':'passed','backend':backend,'fit_passes':passes,'matched_cached_vectors':len(meta),'routing_derivatives_published':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('nf4','bf16'),required=True);p.add_argument('--passes',type=int,choices=(1,2),required=True)
    p.add_argument('--runs',default='runs/nan_fit07');a=p.parse_args();probe(Path(a.runs),a.backend,a.passes)


if __name__=='__main__':main()
