"""Publish audited aggregate fresh-source fit progress; raw evidence stays local."""
import argparse
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,sha,write
from .audit_fit import ROOT,audit,checkpoint,registered_sources


def factor_comparison(a,b):
    def basis(matrix):
        q,s,_=np.linalg.svd(np.asarray(matrix,dtype=np.float64),full_matrices=False)
        rank=int(np.sum(s>s[0]*max(matrix.shape)*np.finfo(float).eps)) if len(s) and s[0]>0 else 0
        return q[:,:rank]
    qa,qb=basis(a['U'].T),basis(b['U'].T)
    read_a,read_b=basis(a['V']),basis(b['V'])
    va,ua=np.asarray(a['V'],dtype=np.float64),np.asarray(a['U'],dtype=np.float64)
    vb,ub=np.asarray(b['V'],dtype=np.float64),np.asarray(b['U'],dtype=np.float64)
    norm_a=float(np.sum((va.T@va)*(ua@ua.T)));norm_b=float(np.sum((vb.T@vb)*(ub@ub.T)))
    cross=float(np.sum((va.T@vb)*(ua@ub.T)))
    return dict(output_subspace_ranks=[qa.shape[1],qb.shape[1]],
        output_principal_cosines=np.linalg.svd(qa.T@qb,compute_uv=False).tolist(),
        input_subspace_ranks=[read_a.shape[1],read_b.shape[1]],
        input_principal_cosines=np.linalg.svd(read_a.T@read_b,compute_uv=False).tolist(),
        physical_pure_component_operator_relative_Frobenius=float(np.sqrt(max(0,norm_a+norm_b-2*cross)/norm_a)) if norm_a>0 else None,
        physical_pure_component_operator_cosine=float(cross/np.sqrt(norm_a*norm_b)) if norm_a>0 and norm_b>0 else None,
        scope='physical learned pure-component operators; controller and pass-mean residual replacement excluded; no component-index alignment')


def progress(runs,out):
    runs=Path(runs);out=Path(out);seal,cfg,_,_=registered_sources();backends={};learned={}
    for backend in ('nf4','bf16'):
        folder=runs/backend
        if not (folder/'baseline.json').exists():
            backends[backend]={'status':'awaiting_baseline' if folder.exists() else 'queued','cells':[]}
            if (folder/'receipt.json').exists():
                r=read(folder/'receipt.json');backends[backend].update(status=r['status'],failure_type=(r.get('failure') or {}).get('type'),worker_receipt_sha256=sha(folder/'receipt.json'))
            continue
        checked=audit(runs,backend)
        write(runs/'audit'/(backend+'_independent.json'),checked)
        value={'status':'baseline_complete_or_fits_in_progress','independent_audit':checked,'cells':checked['cells']}
        if (folder/'receipt.json').exists():
            receipt=read(folder/'receipt.json');assert receipt['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
            value.update(status=receipt['status'],worker_receipt_sha256=sha(folder/'receipt.json'),
                failure_type=(receipt.get('failure') or {}).get('type'),elapsed_seconds=receipt['elapsed_seconds'],
                peak_cuda_allocated_bytes=receipt['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=receipt['peak_cuda_reserved_bytes'],
                head_remained_CPU=receipt['head_remained_CPU'],runtime=receipt['runtime'])
        value['SAE_cross_references']=[]
        for cell in checked['cells']:
            if cell['status']!='passed':continue
            passes=cell['passes'];directory=folder/f't{passes}_layer21_down_proj_rank8'
            description=read(directory/'receipt.json')['final_checkpoint']
            learned[backend,passes]=(checkpoint(directory/description['file']),cell)
            probe=runs/'probes'/f'{backend}_t{passes}_aggregate.json'
            if not probe.exists():continue
            data=read(probe)
            assert data['status']=='passed' and data['checkpoint_sha256']==cell['checkpoint_sha256']
            assert data['fit_receipt_sha256']==cell['fit_receipt_sha256'] and data['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
            assert data['backend']==backend and data['fit_passes']==passes
            value['SAE_cross_references'].append({'probe_file_sha256':sha(probe),**data})
        backends[backend]=value
    comparisons=[]
    for (left,(a,ac)),(right,(b,bc)) in combinations(sorted(learned.items()),2):
        comparisons.append({'left':{'backend':left[0],'fit_passes':left[1],'checkpoint_sha256':ac['checkpoint_sha256']},
            'right':{'backend':right[0],'fit_passes':right[1],'checkpoint_sha256':bc['checkpoint_sha256']},**factor_comparison(a,b)})
    result={'experiment':'NAN-FIT-07','updated_UTC':datetime.now(timezone.utc).isoformat(),
        'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'source_freeze_commit':'561394bcbab03f42d338c3bcbd03a04eccd79369',
        'publisher_source_sha256':sha(__file__),
        'analysis_timing':'Independent CPU auditor and aggregate publisher added after registration; no frozen inference/fitting/selection changes',
        'backends':backends,'learned_factor_comparisons':comparisons,
        'unit':'source-generator world; CAL 8 plus DEV 8, repeated depth/backend/mode measures, one pretrained checkpoint',
        'formal_inferential_comparisons':0,'original_algebraic_gate_unchanged':True,'BPD_donor_transfer_enabled':False,
        'sealed_test_accessed':False,'privacy':'aggregates/hashes/weight geometry only; no per-record answers, IDs, tokens, hidden vectors or routing derivatives'}
    write(out/'nan_fit07_progress.json',result)
    lines=['# NAN-FIT-07 fresh-source VPD/BPD progress','',f"Updated {result['updated_UTC']}",'',
        'The source-only pilot was frozen and pushed at `561394b` before its first pretrained inference. '+
        'Sixteen fresh source roots plus one explicit copy control per root give 32 clean contexts/backend. '+
        'No source or answer repair, prompt truncation, result filtering or parser fitting is allowed.','',
        'The fixed screen requires at least 6/8 main and 8/8 copy answers in EACH CAL/DEV split at loop 2, separately per backend. '+
        'It grades a first free token, distinct from native prefix-stop and Prime complete-reply endpoints. '+
        'Passing this screen permits fitting only; the original algebraic gate remains failed.','',
        '| Backend | Worker evidence status | Clean screen | Audited learned cells | SAE cross-references |',
        '|---|---|---|---:|---:|']
    for backend,r in backends.items():
        gate=r.get('independent_audit',{}).get('baseline',{}).get('allow_fits')
        lines.append(f"| {backend.upper()} | {r['status']} | {'pending' if gate is None else 'pass' if gate else 'fail'} | {sum(c['status']=='passed' for c in r['cells'])}/2 | {len(r.get('SAE_cross_references',[]))}/2 |")
    lines+=['','Partial worker evidence is reported as partial; a state file alone does not prove that a process is live. '+
        'The serialized GPU queue waits on the actual Prime supervisor and checks known model processes before loading a new model.','',
        '| Backend | Split | Condition | Native loop | First-token correct | Format valid | Roots |',
        '|---|---|---|---:|---:|---:|---:|']
    for backend,r in backends.items():
        for group in r.get('independent_audit',{}).get('baseline',{}).get('groups',[]):
            lines.append(f"| {backend.upper()} | {group['split']} | {'copy' if group['copy_control'] else 'main'} | {group['loop']} | {group['correct']}/{group['contexts']} | {group['format_valid']}/{group['contexts']} | {group['source_roots']} |")
    lines+=['','| Backend | Fit depth | Cell status | Steps | Factor update L2 | Controller update L2 |',
        '|---|---:|---|---:|---:|---:|']
    for backend,r in backends.items():
        for cell in r['cells']:
            if cell['status']=='passed':
                lines.append(f"| {backend.upper()} | {cell['passes']} | independently checked | {cell['training']['optimizer_steps']} | {cell['factor_update_l2']:.6g} | {cell['controller_update_l2']:.6g} |")
            else:lines.append(f"| {backend.upper()} | {cell['passes']} | {cell['status']} | {cell.get('optimizer_steps_completed',0)} | — | — |")
    lines+=['','| Backend | Fit depth | Split | Mode | Readout loop | Correct | KL root mean | Local physical dose root mean | Reached-input change root mean |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for backend,r in backends.items():
        for cell in r['cells']:
            if cell['status']!='passed':continue
            for g in cell['final_groups']:
                lines.append(f"| {backend.upper()} | {cell['passes']} | {g['split']} | {g['mode']} | {g['loop']} | {g['correct']}/8 | {g['full_vocabulary_KL_root_mean']:.6g} | {g['local_intervention_relative_l2_at_actual_reached_input_root_mean']:.6g} | {g['reached_input_relative_l2_root_mean']:.6g} |")
    lines+=['','Every completed cell uses eight actual stock four-term BPDTrainer steps on CAL main roots only, '+
        'shared rank-8 factors at layer-21 down_proj, and a fixed final checkpoint. '+
        'The auditor checks complete source coverage, host grades, exact source/step order, four-term loss accounting, '+
        'initial SVD identity, finite checkpoint/optimizer state, nonzero factor/controller changes, controller replay, '+
        'all-one readouts and physical doses, means of saved CAL tolerance estimates and all registered diagnostic modes. '+
        'It does not independently rerun full model gradients or recompute full-token CAL activation means.','',
        'T=1 uses zero ablation. T=2 uses the framework pass-mean residual replacement and sphere rescaling. '+
        'Component-only therefore denotes the residual-neutralized framework pathway, not a complete rank-8 reconstruction of the native operator. '+
        'Static, clean-informed and reached-state routes have unmatched physical doses and are fidelity diagnostics.','',
        'The JSON retains every initial/final diagnostic group, finite gauge diagnostic, physical factor-subspace comparison '+
        'and completed SAE/initial-SVD/norm-matched-Haar cross-reference. SAE observations precede final loop RMSNorm; '+
        'only down_proj has the registered immediate identity-addition bridge. T=1 application at loop 2 is outside its fit depth.','',
        'Engineering evidence is separate: random width-64 worker paths passed BF16 T=1/T=2 and NF4 T=2 '+
        'with eight optimizer steps each; these are not pretrained results.','',
        'There are zero formal inferential comparisons, no calibrated matched causal-dose/sham correctness floor and no donor transfer. '+
        'Eight independent DEV roots cannot support mechanism confirmation or equivalence. '+
        'Rank 16, other sites/layers, the o_proj intervening-MLP bridge and genuine donor-specific causal controls remain later stages. '+
        'The original 168-case test stays sealed.','',
        '[Frozen protocol](../../experiments/nan_fit07/PROTOCOL.md) · '+
        '[Source registration](../../experiments/nan_fit07/SOURCE_MANIFEST.json) · '+
        '[Full aggregate evidence](nan_fit07_progress.json) · [Engineering receipts](../../experiments/nan_fit07/receipts)','']
    (out/'NAN_FIT07_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:{'status':r['status'],'audited_fits':sum(c['status']=='passed' for c in r['cells'])} for b,r in backends.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_fit07');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    progress(a.runs,a.out)


if __name__=='__main__':main()
