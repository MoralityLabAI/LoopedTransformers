"""Post-registration CPU implementation of CTX08's fixed observational probes.

Implemented after NF4 outcomes were exposed. No fitting, selection, inference,
online edit or formal test; complete case coverage is required for each backend.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,rows,sha,write
from experiments.nan_fit07.pointwise import masks
from .probe_towers import encoding,load_npz
from experiments.nan_fit07.probe import root_correlation,nullable

ROOT=Path('experiments/nan_ctx08')
KEYS=('split','context_budget','position','task','copy_control','loop')


def groups(meta,keys=KEYS):
    result=defaultdict(list)
    for i,r in enumerate(meta):result[tuple(r[k] for k in keys)].append(i)
    return result


def root_macro(values,indices,meta):
    byroot=defaultdict(list)
    for i in indices:byroot[meta[i]['root']].append(i)
    return np.mean([np.mean(values[ii],axis=0) for ii in byroot.values()],axis=0)


def describe(key,indices,meta,keys=KEYS):
    return dict(zip(keys,key))|{'source_root_groups':len({meta[i]['root'] for i in indices}),'vectors':len(indices)}


def cached(folder,public,manifest):
    receipt=read(folder/'receipt.json')
    assert receipt['status']=='completed' and receipt['completed_contexts']==len(public)==291
    assert receipt['source_manifest_sha256']==manifest
    matrices={l:defaultdict(list) for l in (5,11,17,21)};meta=[];evidence=[]
    for i,source in enumerate(public):
        file=folder/f'case_{i:04d}.json';r=read(file)
        assert all(r[k]==source[k] for k in ('id','root','split','context_budget','position','task','world','copy_control'))
        state=folder/r['state_file'];assert sha(state)==r['state_file_sha256']
        index={(v['loop'],v['layer_zero_based'],v['site']):v['array_key'] for v in r['state_index']}
        expected={(t,l,s) for t in (1,2) for l in (5,11,17,21) for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
        assert len(r['state_index'])==40 and set(index)==expected
        with np.load(state,allow_pickle=False) as z:
            assert set(z.files)==set(index.values())
            for t in (1,2):
                meta.append({k:source[k] for k in ('id','root','split','context_budget','position','task','world','copy_control')}|{'loop':t})
                for l in matrices:
                    for site in ('residual_after_layer','down_proj_input','down_proj_output'):
                        v=z[index[t,l,site]];assert v.shape==((10752,) if site=='down_proj_input' else (3072,))
                        assert v.dtype==np.float32 and np.isfinite(v).all();matrices[l][site].append(v)
        evidence.append({'record_sha256':sha(file),'state_sha256':sha(state)})
    return {l:{s:np.asarray(v,dtype=np.float64) for s,v in sites.items()} for l,sites in matrices.items()},meta,evidence


def displacements(H,Z,meta,kind):
    if kind=='loop':
        keys=('id',);axis='loop';left,right=1,2
        outkeys=('split','context_budget','position','task','copy_control')
    else:
        keys=('root','context_budget','position','task','copy_control','loop');axis='world';left,right='a','b';outkeys=KEYS
    matched=defaultdict(dict)
    for i,r in enumerate(meta):matched[tuple(r[k] for k in keys)][r[axis]]=i
    values=[];paired=[]
    for pair in matched.values():
        if left not in pair or right not in pair:continue
        i,j=pair[left],pair[right];paired.append(meta[i])
        values.append([np.linalg.norm(H[j]-H[i]),np.linalg.norm(Z[j]-Z[i])])
    values=np.asarray(values);out=[]
    for key,ii in sorted(groups(paired,outkeys).items()):
        v=root_macro(values,ii,paired)
        out.append(describe(key,ii,paired,outkeys)|{'root_macro_physical_delta_l2':float(v[0]),'root_macro_sparse_code_delta_l2':float(v[1])})
    return out


def trained_vpd(runs,H,X,Y,Z,meta,backend):
    parent=Path('experiments/nan_fit07');seal=read(parent/'SOURCE_MANIFEST.json')
    assert sha(parent/'SOURCE_MANIFEST.json')=='8890f915d7b1ba05dc611dfb62e2c1d40abc2cc5bb71351e7b338f640b6f104e'
    folder=Path('runs/nan_fit07')/backend/'t1_layer21_down_proj_rank8'
    if not (folder/'receipt.json').exists() or read(folder/'receipt.json')['status']!='completed':
        return {'status':'missing','reason':'No completed FIT07 T1 learned asset; original competence/fitting gate retained'}
    receipt=read(folder/'receipt.json');assert receipt['source_manifest_sha256']==sha(parent/'SOURCE_MANIFEST.json')
    audit=read(Path('runs/nan_fit07/audit')/(backend+'_independent.json'))
    # The independent backend report contains the complete audit of each successful cell.
    def binds(value):
        if isinstance(value,dict):
            if value.get('status')=='passed' and value.get('checkpoint_sha256')==receipt['final_checkpoint']['sha256']:return True
            return any(binds(v) for v in value.values())
        return isinstance(value,list) and any(binds(v) for v in value)
    assert binds(audit)
    final=folder/receipt['final_checkpoint']['file'];initial=folder/receipt['initial_checkpoint']['file']
    assert sha(final)==receipt['final_checkpoint']['sha256'] and sha(initial)==receipt['initial_checkpoint']['sha256']
    bank=load_npz(final);old=load_npz(initial)
    replay=folder/'CI_replay_local.npz';assert sha(replay)==receipt['CI_replay_sha256']
    replayed=load_npz(replay);np.testing.assert_allclose(masks(replayed['X'],bank,receipt['final_checkpoint']['CI_config']),replayed['masks'],rtol=2e-5,atol=1e-5)
    gate=masks(X,bank,receipt['final_checkpoint']['CI_config']);acts=X@bank['V']
    rng=np.random.default_rng(read(parent/'config.json')['source_seed']+1)
    rv=np.linalg.qr(rng.normal(size=bank['V'].shape),mode='reduced')[0]*np.linalg.norm(bank['V'],axis=0)
    ru=np.linalg.qr(rng.normal(size=bank['U'].T.shape),mode='reduced')[0].T*np.linalg.norm(bank['U'],axis=1)[:,None]
    predictions={'learned_VPD_T1':acts@bank['U'],'initial_SVD_not_trained':(X@old['V'])@old['U'],'matched_norm_Haar_not_trained':(X@rv)@ru}
    norm=np.maximum(np.linalg.norm(Y,axis=1),1e-12);energy=np.abs(gate*acts)*np.linalg.norm(bank['U'],axis=1)
    summaries=[]
    for key,ii in sorted(groups(meta).items()):
        byroot=defaultdict(list)
        for i in ii:byroot[meta[i]['root']].append(i)
        root_z=np.array([Z[jj].mean(0) for jj in byroot.values()]);root_e=np.array([energy[jj].mean(0) for jj in byroot.values()])
        corr=root_correlation(root_z,root_e) if len(byroot)>=3 else np.full((128,8),np.nan)
        summaries.append(describe(key,ii,meta)|{'out_of_fit_depth':key[-1]>1,
            'module_relative_l2':{n:float(root_macro(np.linalg.norm(p-Y,axis=1)/norm,ii,meta)) for n,p in predictions.items()},
            'pointwise_gate_root_macro':root_macro(gate,ii,meta).tolist(),
            'SAE_feature_vs_routed_component_energy_root_correlation':nullable(corr),'undefined_correlation_cells':int(np.isnan(corr).sum())})
    local=runs/'probes/local';local.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(local/f'{backend}_VPD_T1_codes.npz',controller_masks=gate,component_reads=acts,Haar_V=rv,Haar_U=ru)
    return dict(status='passed',producer='NAN-FIT-07 only; RECOMP09 excluded from original CTX08 probe plan',fit_passes=1,
        fit_receipt_sha256=sha(folder/'receipt.json'),checkpoint_sha256=sha(final),independent_fit_audit_sha256=sha(Path('runs/nan_fit07/audit')/(backend+'_independent.json')),
        groups=summaries,scope='clean cached-input module fidelity and controller/SAE correlation; no online edit, reached-state claim or donor transfer')


def probe(runs,backend):
    seal=read(ROOT/'SOURCE_MANIFEST.json');source_sha=sha(ROOT/'SOURCE_MANIFEST.json')
    assert source_sha=='7f729d76bd0b37eabbeaf4a2fd8dc0faa7621c01f8830554bbf424a63dd4d9ec'
    public_file=ROOT/'fixtures/public.jsonl';assert sha(public_file)==seal['worker_files_sha256']['fixtures/public.jsonl']
    matrices,meta,evidence=cached(runs/backend,rows(public_file),source_sha)
    base=Path('runs/nan_gym05')/backend;fitted=read(base/'sae_tower/manifest.json');result=[];Z21=None
    local=runs/'probes/local';local.mkdir(parents=True,exist_ok=True)
    for layer,sites in matrices.items():
        H=sites['residual_after_layer'];file=base/f'sae_tower/layer{layer}.npz';initial=base/f'sae_tower/layer{layer}_random_initial.npz'
        fitfile=base/f'sae_tower/layer{layer}_fit_manifest.json';r=next(r for r in fitted['receipts'] if r['layer_zero_based']==layer)
        assert sha(file)==r['artifact_sha256']==seal['SAE_dictionary_sha256'][backend][str(layer)]
        assert sha(initial)==r['random_initial_sha256'] and sha(fitfile)==r['fit_manifest_sha256']
        fit=read(fitfile);assert fit['calibration_only'] and fit['calibration_roots']==32 and fit['shared_dictionary_for_loops']==[1,2]
        Z,mse,l2,err=encoding(H,load_npz(file),4);ZI,mi,li,ei=encoding(H,load_npz(initial),4)
        summary=[]
        for key,ii in sorted(groups(meta).items()):
            summary.append(describe(key,ii,meta)|{
                'normalized_MSE':float(root_macro(mse,ii,meta)),'random_initial_normalized_MSE':float(root_macro(mi,ii,meta)),
                'physical_reconstruction_l2':float(root_macro(l2,ii,meta)),
                'random_initial_physical_reconstruction_l2':float(root_macro(li,ii,meta)),
                'physical_state_l2':float(root_macro(np.linalg.norm(H,axis=1),ii,meta)),
                'feature_active_fraction_root_macro':root_macro((Z>0).astype(float),ii,meta).tolist(),
                'random_initial_feature_active_fraction_root_macro':root_macro((ZI>0).astype(float),ii,meta).tolist()})
        result.append(dict(layer_zero_based=layer,dictionary_sha256=sha(file),initial_dictionary_sha256=sha(initial),fit_manifest_sha256=sha(fitfile),
            physical_coordinate_max_abs=err,random_initial_coordinate_max_abs=ei,groups=summary,
            loop_displacements=displacements(H,Z,meta,'loop'),world_displacements=displacements(H,Z,meta,'world'),
            random_initial_loop_displacements=displacements(H,ZI,meta,'loop'),random_initial_world_displacements=displacements(H,ZI,meta,'world')))
        np.savez_compressed(local/f'{backend}_layer{layer}_codes.npz',SAE_codes=Z,random_initial_codes=ZI,normalized_MSE=mse,physical_states=H)
        if layer==21:Z21=Z
    write(local/f'{backend}_index.json',meta);write(local/f'{backend}_artifact_index.json',evidence)
    sites=matrices[21];vpd=trained_vpd(runs,sites['residual_after_layer'],sites['down_proj_input'],sites['down_proj_output'],Z21,meta,backend)
    output=dict(status='passed',backend=backend,updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=source_sha,
        worker_receipt_sha256=sha(runs/backend/'receipt.json'),implementation_sha256=sha(Path(__file__)),
        implementation_timing='After all NF4 behavior and partial BF16 were exposed; observational operations predeclared in CTX08 protocol; no new inference or fit',
        contexts=291,physical_residual_vectors_per_layer=582,layers=result,learned_VPD=vpd,
        learned_BPD={'status':'missing','reason':'FIT07 T2 failed; separate RECOMP09 condition is outside this registered producer set'},
        formal_inferential_comparisons=0,sealed_test_accessed=False,BPD_donor_transfer_attempted=False,
        unit='four DEV source families, one CAL resource-profile family, one checkpoint; contexts/worlds/lengths/loops are repeated measures',
        privacy='all per-record codes, controller masks, IDs and states remain local; aggregate root-macro metrics and hashes only')
    write(runs/f'probes/{backend}_context_aggregate.json',output)
    print({'backend':backend,'status':'passed','layers':4,'contexts':291,'learned_VPD':vpd['status']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--runs',default='runs/nan_ctx08');a=p.parse_args()
    probe(Path(a.runs),a.backend)


if __name__=='__main__':main()
