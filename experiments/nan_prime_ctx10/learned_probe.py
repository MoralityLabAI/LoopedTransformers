"""Declared CTX10 cached-input BPD/VPD/SAE probe, never an online intervention."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import public_rows,read,sha,write
from mechinterp.nanbeige_bpd_gym.context_probes import cached,groups,root_macro,describe
from mechinterp.nanbeige_bpd_gym.probe_towers import encoding,load_npz
from mechinterp.nanbeige_bpd_gym.crossref import probe_and_crossref,REVISION
from experiments.nan_fit07.pointwise import masks
from experiments.nan_fit07.probe import root_correlation,nullable


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_prime_ctx10');a=p.parse_args();runs=Path(a.runs)
    root=Path(__file__).parent;seal=read(root/'SOURCE_MANIFEST.json');progress=read('reports/nanbeige_bpd_gym/nan_prime_ctx10_progress.json')
    bf=progress['backends']['bf16'];assert bf['neural_requests']==bf['matched_native_prefixes']==288 and bf['physical_prefix_vectors_byte_exact']==11520
    producer=Path('experiments/nan_recomp09');assert sha(producer/'SOURCE_MANIFEST.json')==seal['learned_BPD_producer_manifest_sha256']
    folder=Path('runs/nan_recomp09/bf16/t2_layer21_down_proj_rank8');receipt=read(folder/'receipt.json')
    auditfile=Path('runs/nan_recomp09/audit/bf16_independent.json');audit=read(auditfile)
    assert audit['status']=='passed' and audit['fit_receipt_sha256']==sha(folder/'receipt.json') and receipt['status']=='completed'
    final=folder/receipt['final_checkpoint']['file'];initial=folder/receipt['initial_checkpoint']['file'];assert sha(final)==audit['checkpoint_sha256']==receipt['final_checkpoint']['sha256']
    assert sha(initial)==receipt['initial_checkpoint']['sha256'];bank=load_npz(final);old=load_npz(initial)
    assert sha(final)==seal['learned_BPD_checkpoint_sha256'] and sha(auditfile)==seal['learned_BPD_audit_sha256']
    public,_=public_rows();data,meta,_=cached(Path('runs/nan_ctx08/bf16'),public,seal['CTX08_source_manifest_sha256'])
    keep=np.array([r['split']=='development' for r in meta]);meta=[r for r in meta if r['split']=='development']
    H,X,Y=(data[21][site][keep] for site in ('residual_after_layer','down_proj_input','down_proj_output'));assert len(meta)==576
    sae_file=Path('runs/nan_gym05/bf16/sae_tower/layer21.npz');assert sha(sae_file)==seal['SAE_layer21_BF16_sha256']
    sae=load_npz(sae_file);Z,_,_,_=encoding(H,sae,4);gate=masks(X,bank,receipt['final_checkpoint']['CI_config']);acts=X@bank['V']
    mu=np.array([bank['mean_comp_acts'][r['loop']-1] for r in meta]);energy=np.abs(gate*(acts-mu))*np.linalg.norm(bank['U'],axis=1)
    rng=np.random.default_rng(2026112009);rv=np.linalg.qr(rng.normal(size=bank['V'].shape),mode='reduced')[0]*np.linalg.norm(bank['V'],axis=0)
    ru=np.linalg.qr(rng.normal(size=bank['U'].T.shape),mode='reduced')[0].T*np.linalg.norm(bank['U'],axis=1)[:,None]
    predictions={'learned_BPD_T2':acts@bank['U'],'initial_SVD_not_trained':(X@old['V'])@old['U'],'matched_norm_Haar_not_trained':(X@rv)@ru}
    norm=np.maximum(np.linalg.norm(Y,axis=1),1e-12);summary=[]
    for key,ii in sorted(groups(meta).items()):
        byroot=defaultdict(list)
        for i in ii:byroot[meta[i]['root']].append(i)
        corr=root_correlation(np.array([Z[jj].mean(0) for jj in byroot.values()]),np.array([energy[jj].mean(0) for jj in byroot.values()]))
        summary.append(describe(key,ii,meta)|{'module_relative_l2':{n:float(root_macro(np.linalg.norm(v-Y,axis=1)/norm,ii,meta)) for n,v in predictions.items()},
            'pointwise_gate_root_macro':root_macro(gate,ii,meta).tolist(),'SAE_vs_centered_routed_BPD_energy_root_correlation':nullable(corr),
            'undefined_correlation_cells':int(np.isnan(corr).sum())})
    vpdfile=Path('runs/nan_fit07/bf16/t1_layer21_down_proj_rank8/final.npz');assert sha(vpdfile)==seal['learned_VPD_checkpoint_sha256'];vpd=load_npz(vpdfile)
    geometry=[]
    source_indices={r['id']:i for i,r in enumerate(public)}
    for loop in (1,2):
        i=next(i for i,r in enumerate(meta) if r['loop']==loop)
        native=read(Path('runs/nan_ctx08/bf16')/f"case_{source_indices[meta[i]['id']]:04d}.json")
        site=dict(checkpoint_revision=REVISION,backend='bf16',layer_zero_based=21,loop=loop,site='residual_after_layer',width=3072)
        result=probe_and_crossref(H[i],sae,vpd['U'],state_meta=site|{'artifact_sha256':native['state_file_sha256']},
            sae_meta=site|dict(artifact_sha256=sha(sae_file),fit_split='calibration',fit_manifest_sha256=sha(sae_file.with_name('layer21_fit_manifest.json'))),
            vpd_meta=site|dict(artifact_sha256=sha(vpdfile),factor_role='physical_output_directions'),bpd_vectors=bank['U'],
            bpd_meta=site|dict(artifact_sha256=sha(final),factor_role='physical_output_directions'))
        geometry.append(dict(application_loop=loop,VPD_out_of_fit_depth=loop>1,
            BPD_zero_norm_rows=int(np.sum(np.linalg.norm(bank['U'],axis=1)==0)),
            **{k:result[k] for k in ('encoder_vpd_cosine','decoder_vpd_cosine','encoder_bpd_cosine','decoder_bpd_cosine','bpd_vpd_cosine','zero_norm_counts')}))
    qb=np.linalg.qr(bank['U'].astype(np.float64).T,mode='reduced')[0];qv=np.linalg.qr(vpd['U'].astype(np.float64).T,mode='reduced')[0]
    assert np.linalg.matrix_rank(bank['U'].astype(np.float64))==np.linalg.matrix_rank(vpd['U'].astype(np.float64))==8
    local=runs/'probes/local';local.mkdir(parents=True,exist_ok=True);np.savez_compressed(local/'bf16_BPD_codes.npz',SAE_codes=Z,component_reads=acts,controller_masks=gate,Haar_V=rv,Haar_U=ru)
    write(local/'bf16_BPD_index.json',meta)
    result=dict(status='passed',source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),checkpoint_sha256=sha(final),fit_audit_sha256=sha(auditfile),
        contexts=288,residual_vectors=576,source_families=4,groups=summary,geometry=geometry,
        BPD_VPD_output_subspace_principal_cosines=np.linalg.svd(qb.T@qv,compute_uv=False).tolist(),
        scope='New CTX10 analysis of RECOMP09 asset on byte-matched clean cached inputs; not part of original CTX08 producer set; no online reached-state intervention',
        NF4_learned_asset='missing: original competence gate failed',formal_inferential_comparisons=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False)
    write(Path('reports/nanbeige_bpd_gym/nan_prime_ctx10_BPD_probes.json'),result)
    Path('reports/nanbeige_bpd_gym/NAN_PRIME_CTX10_BPD_PROBES.md').write_text('# CTX10 learned BPD/VPD/SAE cross-reference\n\n'+
        'The independently audited BF16 RECOMP09 T2 asset was applied to all 288 byte-matched context prefixes, both loops. '+
        'All four source families, every position/task/control/length, all 128 SAE features and SVD/Haar controls are retained. '+
        'The aggregate JSON includes module fidelity, pointwise masks, root-level feature/centered-component correlations and '+
        'all pairwise SAE/VPD/BPD output-direction cosines. T1 VPD at loop 2 is outside fitting depth. '+
        'This is clean cached-input application, not online intervention, donor transfer or causal confirmation. '+
        'NF4 has no eligible learned asset. The original 168-case test remains sealed.\n\n'+
        '[All aggregate probes](nan_prime_ctx10_BPD_probes.json)\n',encoding='utf-8',newline='\n')
    print({'status':'passed','contexts':288,'groups':len(summary),'geometry_cells':len(geometry)})


if __name__=='__main__':main()
