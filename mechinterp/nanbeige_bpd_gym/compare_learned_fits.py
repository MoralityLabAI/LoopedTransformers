"""Source-bound BF16 VPD/BPD pilot handoff; no new inference or formal test."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,sha,write


def operator_comparison(v1,u1,v2,u2):
    v1,u1,v2,u2=(np.asarray(x,dtype=np.float64) for x in (v1,u1,v2,u2))
    a=float(np.trace((v1.T@v1)@(u1@u1.T)));b=float(np.trace((v2.T@v2)@(u2@u2.T)))
    cross=float(np.trace((v1.T@v2)@(u2@u1.T)))
    return dict(VPD_component_operator_frobenius=float(np.sqrt(max(a,0))),BPD_component_operator_frobenius=float(np.sqrt(max(b,0))),
        physical_component_operator_relative_l2=float(np.sqrt(max(a+b-2*cross,0))/max(np.sqrt(a),1e-12)),
        physical_component_operator_cosine=cross/max(np.sqrt(a*b),1e-12))


def report(out):
    vf=Path('runs/nan_fit07/audit/bf16_independent.json');bf=Path('runs/nan_recomp09/audit/bf16_independent.json')
    va=read(vf);ba=read(bf);assert va['status']==ba['status']=='passed'
    v=next(c for c in va['cells'] if c.get('status')=='passed' and c.get('passes')==1)
    folders={'VPD_T1':Path('runs/nan_fit07/bf16/t1_layer21_down_proj_rank8'),'BPD_T2':Path('runs/nan_recomp09/bf16/t2_layer21_down_proj_rank8')}
    checked={'VPD_T1':v,'BPD_T2':ba};banks={};group=[];producers={}
    target_hashes=[]
    for name,folder in folders.items():
        audit=checked[name];receipt=read(folder/'receipt.json');assert sha(folder/'receipt.json')==audit['fit_receipt_sha256']
        manifest=Path('experiments/nan_fit07' if name=='VPD_T1' else 'experiments/nan_recomp09')/'SOURCE_MANIFEST.json'
        assert receipt['source_manifest_sha256']==sha(manifest);target_hashes.append(receipt['target_parameter_sha256_unchanged'])
        path=folder/receipt['final_checkpoint']['file'];assert sha(path)==audit['checkpoint_sha256']
        with np.load(path,allow_pickle=False) as z:banks[name]={k:z[k] for k in ('V','U')}
        for row in audit['final_groups']:group.append({'producer':name,**row})
        producers[name]=dict(source_manifest_sha256=sha(manifest),checkpoint_sha256=sha(path),receipt_sha256=sha(folder/'receipt.json'),optimizer_steps=receipt['optimizer_steps_completed'],
            fit_passes=receipt['passes'],factor_update_l2=audit['factor_update_l2'],controller_update_l2=audit['controller_update_l2'],
            saved_tensor_CPU_peak_bytes=audit['saved_tensor_statistics']['peak_added_cpu_bytes'],CUDA_peak_allocated_bytes=audit['CUDA_peak_allocated_bytes'])
    assert len(set(target_hashes))==1
    V,U=banks['VPD_T1']['V'],banks['VPD_T1']['U'];B,C=banks['BPD_T2']['V'],banks['BPD_T2']['U']
    assert V.shape==B.shape==(10752,8) and U.shape==C.shape==(8,3072)
    qv=np.linalg.qr(U.astype(np.float64).T,mode='reduced')[0];qb=np.linalg.qr(C.astype(np.float64).T,mode='reduced')[0]
    geometry=operator_comparison(V,U,B,C)|{'output_subspace_principal_cosines':np.linalg.svd(qv.T@qb,compute_uv=False).tolist()}
    result=dict(updated_UTC=datetime.now(timezone.utc).isoformat(),implementation_sha256=sha(Path(__file__)),
        audit_sha256={'FIT07':sha(vf),'RECOMP09':sha(bf)},producers=producers,physical_weight_geometry=geometry,all_final_groups=group,
        unit='same eight CAL/eight DEV families, one checkpoint; loops/modes and fits are repeated measures',n_units_CAL=8,n_units_DEV=8,n_cases=16,
        n_gate_contexts_including_copy_controls=32,unchanged_target_parameter_sha256=target_hashes[0],
        primary_contrast='descriptive same-site trained physical operators and fixed diagnostics; T1/T2 depth and ablation rules differ',
        physical_dose='all original per-root actual reached-input doses remain in groups; not a matched-dose causal comparison',
        scientific_variation_floor=None,matched_sham_floor=None,estimated_resolution='not calibrated; power-limited development pilot',
        readout_selection_exposure='All fitting/diagnostic/probe outputs exposed before this post-registration handoff',
        planned_formal_comparisons=0,actual_formal_comparisons=0,causal_controls='original static-CAL-mean, clean-informed and reached-state modes; no powered donor/sham discrimination',
        verdict='actual pretrained fitting/instrument evidence plus observational geometry; behavioral specificity and donor transfer unproven',
        original_NF4_failed_gate_preserved=True,original_BF16_T2_memory_failure_preserved=True,sealed_test_accessed=False,BPD_donor_transfer_attempted=False)
    write(out/'nan_learned_VPD_BPD_comparison.json',result)
    lines=['# Learned BF16 VPD/BPD pilot: results and specificity limits','',
        'Both final learned assets are real pretrained rank-8 layer-21 down_proj fits, each with eight stock optimizer '+
        'steps and independent checkpoint/controller/diagnostic audits. Each completed all 28 source/loop SAE/SVD/Haar '+
        'probe groups and six geometry cells. The original failed T2 attempt and NF4 failed competence gate remain intact.','',
        '| Asset | Fit depth | Factor update L2 | Controller update L2 | Saved-tensor CPU peak MiB |', '|---|---:|---:|---:|---:|']
    for name,r in producers.items():lines.append(f"| {name} | {r['fit_passes']} | {r['factor_update_l2']:.6f} | {r['controller_update_l2']:.6f} | {r['saved_tensor_CPU_peak_bytes']/2**20:.2f} |")
    lines+=['','| Asset | DEV loop | Mode | Correct | Full-vocabulary KL mean |', '|---|---:|---|---:|---:|']
    for r in group:
        if r['split']=='development':lines.append(f"| {r['producer']} | {r['loop']} | {r['mode']} | {r['correct']}/{r['contexts']} | {r['full_vocabulary_KL_root_mean']:.6f} |")
    lines+=['','BPD loop-2 all-one, component-only, static-CAL-mean, clean-informed and reached-state routes each '+
        'retain 7/8 DEV correctness. Static replacement also retaining correctness means this pilot does not '+
        'demonstrate behavioral specificity of learned components. Reached-state masks differ from clean-input masks '+
        'at loop 2, but that alone is not mediation. The full aggregate preserves the measured write/state doses, '+
        'mask differences and KL shifts; high label retention is not a substitute for a matched causal contrast.','',
        f"The two learned physical component operators have relative Frobenius difference {geometry['physical_component_operator_relative_l2']:.6f} "+
        f"and cosine {geometry['physical_component_operator_cosine']:.6f}. Output-subspace principal cosines are in the JSON. "+
        'These use physical V@U and subspaces, not same-column component matching. No backend SAE-index subtraction.','',
        'T1 and T2 use different recurrent fit depths and registered ablation/adversary rules; the second condition '+
        'also uses a separately registered memory implementation. Their difference is not an isolated causal BPD '+
        'advantage. Each dataset has only eight CAL/eight DEV families, with reused readouts and one checkpoint. '+
        'No formal comparison, equivalence, mechanism confirmation or donor transfer is claimed. A useful next causal '+
        'pilot must discriminate learned donors from matched sham/static controls at fixed physical doses, check future '+
        'clean competence at the target depth, and calibrate a meaningful effect floor before any sealed confirmation.','',
        'The original 168-case test remains sealed. Raw predictions, states and per-record derivatives remain local.','',
        '[Complete grouped evidence](nan_learned_VPD_BPD_comparison.json) · [Original fit/failure](NAN_FIT07_PROGRESS.md) · '+
        '[Completed BPD memory condition](NAN_RECOMP09_PROGRESS.md)','']
    (out/'NAN_LEARNED_VPD_BPD_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({'assets':2,'status':'audited','donor_transfer':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args();report(Path(a.out))


if __name__=='__main__':main()
