"""Streaming complete-prefix fixed SAE and BPD probes; one CPU thread only."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
from .common import ROOT,RUNS,fit_asset,public_rows,read,sha,verify,write


def bf16_round(x):
    import numpy as np
    value=np.asarray(x,dtype=np.float32).copy();bits=value.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def local_replay(x,native,mask,checkpoint,ablation,phase):
    """Independent pointwise equations, then registered BF16 write rounding."""
    import numpy as np
    acts=x.astype(np.float64)@checkpoint['V'].astype(np.float64)
    mean=checkpoint['mean_comp_acts'][phase];mean_delta=checkpoint['mean_delta_out'][phase]
    write=(mask*acts+(1-mask)*mean)@checkpoint['U'].astype(np.float64)+mean_delta
    if ablation['sphere_rescale']:
        eps=ablation['eps'];frac=np.mean(1-np.clip(mask,0,1));rms=np.sqrt(np.mean(write**2)+eps)
        desired=(1-frac)*rms+frac*max(float(checkpoint['mean_write_rms'][phase]),eps)
        if np.mean(write**2)>eps:write*=desired/rms
    return bf16_round(write)


def run(backend):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();cfg=read(ROOT/'config.json');start=time.monotonic()
    import numpy as np
    from mechinterp.nanbeige_bpd_gym.probe_towers import encoding,load_npz
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_xref19.cache import correlations,nullable
    audit=read(RUNS/'audits'/(backend+'.json'));assert audit['status']=='passed'
    file=RUNS/'audits'/(backend+'_prefixes_local.json');assert sha(file)==audit['local_index_sha256']
    admitted=read(file);out=RUNS/'probes'/backend;out.mkdir(parents=True,exist_ok=False)
    source={r['id']:r for r in public_rows()};bycase=defaultdict(list)
    for p in admitted['prefixes']:bycase[p['source_id']].append(p)
    inventory=read('experiments/nan_xref19/CACHE_INVENTORY.json')['backends'][backend]
    dictionaries={}
    for layer in cfg['layers_zero_based']:
        names=inventory['dictionaries'][str(layer)]
        for description in names.values():assert sha(description['path'])==description['sha256']
        dictionaries[layer]=tuple(load_npz(names[n]['path']) for n in (f'layer{layer}.npz',f'layer{layer}_random_initial.npz'))
    assets={t:fit_asset(backend,t) for t in (1,2)}
    banks={t:load_npz(v['checkpoint']) for t,v in assets.items() if v}
    summaries=[];geometry=[];case_values=[];covered=0;write_errors=[];mask_errors=[]
    from mechinterp.sae_causal_transfer.adapter import to_physical_dictionary
    E,b,D=to_physical_dictionary(*[dictionaries[21][0][n] for n in ('encoder','bias','decoder','mean','scale')])
    svdfile=Path('runs/nan_gym05')/backend/'linear_baselines/layer21_down_proj.npz'
    # Its exact file pin is inherited from the original CAL-only tower source.
    baseline_manifest=read(svdfile.parent/'manifest.json')
    svd_description=next(v for v in baseline_manifest['receipts'] if v['layer_zero_based']==21 and v['module']=='down_proj')
    assert sha(svdfile)==svd_description['file_sha256'];svd=load_npz(svdfile)['U8']
    for depth,bank in banks.items():
        rng=np.random.default_rng(cfg['source_seed']+depth)
        random=np.linalg.qr(rng.normal(size=(3072,8)),mode='reduced')[0].T*np.linalg.norm(bank['U'],axis=1)[:,None]
        control=out/f'T{depth}_matched_norm_Haar.npz'
        with control.open('xb') as stream:np.savez_compressed(stream,U=random)
        for label,directions in (('learned_IO18',bank['U']),('CAL_SVD_baseline',svd),('matched_norm_Haar',random)):
            V=np.asarray(directions,dtype=np.float64)
            def cosine(A):
                denominator=np.linalg.norm(A,axis=1)[:,None]*np.linalg.norm(V,axis=1)[None,:]
                return nullable(np.divide(A@V.T,denominator,out=np.full(denominator.shape,np.nan),where=denominator>0))
            geometry.append(dict(passes=depth,factor_kind=label,layer_zero_based=21,site_bridge='down_proj output to residual_after_layer: identity coordinates only',
                encoder_output_cosine=cosine(E),decoder_output_cosine=cosine(D),encoder_preactivation_sensitivity=(E@V.T).tolist(),
                zero_norm_output_directions=int(np.sum(np.linalg.norm(V,axis=1)==0)),
                qualified_checkpoint_sha256=assets[depth]['checkpoint_sha256'],SVD_file_sha256=sha(svdfile),Haar_file_sha256=sha(control)))
    # Never assemble the full campaign's 40-vector tensors in host memory.
    for caseid,prefixes in bycase.items():
        assert time.monotonic()-start<cfg['maximum_host_seconds_per_backend'],'Host budget preserves closed partial case artifacts'
        r=source[caseid];meta=[];vectors=defaultdict(list);actual_masks=[];native=[]
        for p in prefixes:
            record=read(RUNS/backend/p['record_file']);assert sha(RUNS/backend/p['record_file'])==p['record_sha256']
            state=RUNS/backend/p['state_file'];assert sha(state)==p['state_sha256']
            with np.load(state,allow_pickle=False) as z:
                coords={(i['loop'],i['layer_zero_based'],i['site']):z[i['array_key']].copy() for i in record['state_index']}
                for phase in (1,2):
                    meta.append(dict(step=p['step'],loop=phase,prefix_ids_sha256=p['prefix_ids_sha256']))
                    for layer in cfg['layers_zero_based']:
                        vectors['H'+str(layer)].append(coords[phase,layer,'residual_after_layer'])
                    vectors['X'].append(coords[phase,21,'down_proj_input']);vectors['Y'].append(coords[phase,21,'down_proj_output'])
                    native.append(z['native_output_at_reached_input'][phase-1].copy())
                    if r['mode']!='native':actual_masks.append(z['actual_mask'][phase-1].copy())
        arrays={k:np.asarray(v,dtype=np.float64) for k,v in vectors.items()};native=np.asarray(native,dtype=np.float64)
        codes={};case_layers=[];case_bpd=[];phase_metrics=[];local={'native_output_at_reached_input':native}
        for layer in cfg['layers_zero_based']:
            H=arrays['H'+str(layer)];sae,random=dictionaries[layer]
            Z,mse,l2,error=encoding(H,sae,4);_,rmse,_,rerror=encoding(H,random,4);codes[layer]=Z
            local['layer'+str(layer)+'_codes']=Z;local['layer'+str(layer)+'_normalized_MSE']=mse
            local['layer'+str(layer)+'_random_initial_MSE']=rmse
            dis=H[1::2]-H[::2];zd=Z[1::2]-Z[::2]
            case_layers.append(dict(layer_zero_based=layer,vectors=len(H),physical_coordinate_error=error,random_coordinate_error=rerror,
                normalized_MSE=float(mse.mean()),random_initial_MSE=float(rmse.mean()),physical_reconstruction_L2=float(l2.mean()),
                feature_active_fraction=np.mean(Z>0,axis=0).tolist(),
                recurrent_state_displacement_L2=float(np.linalg.norm(dis,axis=1).mean()),recurrent_code_displacement_L2=float(np.linalg.norm(zd,axis=1).mean())))
            for phase in (1,2):
                ii=slice(phase-1,None,2)
                phase_metrics.append(dict(layer_zero_based=layer,loop=phase,vectors=len(H[ii]),
                    normalized_MSE=float(mse[ii].mean()),random_initial_MSE=float(rmse[ii].mean()),
                    physical_reconstruction_L2=float(l2[ii].mean()),feature_active_fraction=np.mean(Z[ii]>0,axis=0).tolist()))
        for depth,bank in banks.items():
            config=assets[depth]['receipt']['final_checkpoint']['CI_config'];X=arrays['X']
            acts=X@bank['V'];shadow=masks(X,bank,config)
            local[f'T{depth}_component_activations']=acts;local[f'T{depth}_shadow_masks_at_actual_inputs']=shadow
            case_bpd.append(dict(passes=depth,vectors=len(X),loop2_outside_fit_depth=depth==1,
                mask_mean=shadow.mean(0).tolist(),component_activation_mean=acts.mean(0).tolist()))
            if depth==2 and r['mode']!='native':
                actual=np.asarray(actual_masks,dtype=np.float64)
                expected=shadow if r['mode']=='reached_state' else np.tile(np.asarray(assets[2]['static_CAL_mask']), (len(prefixes),1)) if r['mode']=='static_CAL_mean' else np.ones_like(shadow)
                np.testing.assert_allclose(actual,expected,rtol=cfg['independent_mask_rtol'],atol=cfg['independent_mask_atol'])
                mask_errors.append(float(np.max(np.abs(actual-expected))))
                local['actual_mask']=actual;local['local_write_L2']=np.linalg.norm(arrays['Y']-native,axis=1)
                errors=[]
                for i,m in enumerate(meta):
                    predicted=local_replay(X[i],native[i],actual[i],bank,assets[2]['receipt']['training_contract']['ablation'],m['loop']-1)
                    error=float(np.linalg.norm(predicted-arrays['Y'][i])/max(float(np.linalg.norm(arrays['Y'][i])),1e-12));errors.append(error)
                assert max(errors)<=cfg['independent_local_write_relative_L2_tolerance']
                write_errors.extend(errors);local['independent_local_write_relative_L2_error']=np.asarray(errors)
        localfile=out/f"case_{prefixes[0]['ordinal']:04d}_codes_bpd.npz"
        with localfile.open('xb') as stream:np.savez_compressed(stream,**local)
        write(out/f"case_{prefixes[0]['ordinal']:04d}_index.json",dict(source_id=caseid,root=r['root'],prefixes=prefixes,vector_index=meta,
            local_file_sha256=sha(localfile),layers=case_layers,BPD=case_bpd))
        summaries.append(dict(root=r['root'],split=r['split'],query_type=r['query_type'],event_count=r['event_count'],arm=r['arm'],mode=r['mode'],
            captured_prefixes=len(prefixes),closed_neural_reply=all(p['closed_neural_reply'] for p in prefixes),layers=case_layers,phase_metrics=phase_metrics,BPD=case_bpd))
        # One mean per root later; no per-token pseudo-replication.
        for phase in (1,2):
            value=dict(root=r['root'],split=r['split'],query_type=r['query_type'],arm=r['arm'],mode=r['mode'],loop=phase,
                code_mean=codes[21][phase-1::2].mean(0).tolist(),masks={str(t):local[f'T{t}_shadow_masks_at_actual_inputs'][phase-1::2].mean(0).tolist() for t in banks})
            case_values.append(value)
        covered+=len(prefixes)
    # All 128x8 feature/controller correlations, root-level and nullable.
    corr=[];grouped=defaultdict(list)
    for r in case_values:grouped[r['split'],r['query_type'],r['arm'],r['mode'],r['loop']].append(r)
    for key,items in sorted(grouped.items()):
        byroot=defaultdict(list)
        for r in items:byroot[r['root']].append(r)
        A=np.asarray([np.mean([v['code_mean'] for v in vv],axis=0) for vv in byroot.values()])
        for depth in banks:
            B=np.asarray([np.mean([v['masks'][str(depth)] for v in vv],axis=0) for vv in byroot.values()])
            corr.append(dict(split=key[0],query_type=key[1],arm=key[2],mode=key[3],loop=key[4],passes=depth,independent_roots=len(A),
                feature_controller_correlation=nullable(correlations(A,B))))
    assert covered==audit['captured_prefixes']
    result=dict(status='completed' if covered else 'unrun_no_audited_vectors',backend=backend,captured_prefixes=covered,
        physical_vectors_audited=40*covered,SAE_residual_vectors_probed=8*covered,SAE_dictionaries=4,random_initial_controls=4,all_features=128,top_k=4,
        qualified_fit_depths=sorted(banks),trajectory_summaries=summaries,root_correlations=corr,all_registered_weight_geometry=geometry,
        max_independent_mask_abs_error=max(mask_errors,default=None),max_independent_local_write_relative_L2_error=max(write_errors,default=None),
        audit_sha256=sha(RUNS/'audits'/(backend+'.json')),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        causal_or_equivalence_claim=False,formal_inferential_comparisons=0,sealed_test_accessed=False)
    write(out/'aggregate.json',result);print(dict(backend=backend,captured_prefixes=covered,qualified_fit_depths=sorted(banks)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=('bf16','nf4'));a=p.parse_args();run(a.backend)
