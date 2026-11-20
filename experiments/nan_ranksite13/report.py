"""Closed-event source-bound rank/site audits and complete aggregate publication."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from mechinterp.nanbeige_bpd_gym.audit_fit import registered_sources,baseline_audit
from mechinterp.nanbeige_bpd_gym.probe_towers import load_npz
from .common import read,sha,write,condition,key,verify
from .cell_audit import cell_audit
from .physical_audit import physical_audit
from .resources import below_normal
from mechinterp.nanbeige_bpd_gym.compare_learned_fits import operator_comparison


def closed_events(meta):
    path=meta/'events.jsonl'
    if not path.exists():return []
    data=path.read_bytes();data=data[:data.rfind(b'\n')+1]
    return [json.loads(line) for line in data.splitlines()]


def report(runs,out,*,with_probes=False):
    root=Path(__file__).parent;seal=verify()
    for name,h in seal['host_only_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['analysis_dependencies_sha256'].items():assert sha(name)==h,name
    _,basecfg,public,gold=registered_sources()
    original,screen=baseline_audit(Path('runs/nan_fit07/bf16'),public,gold,basecfg);assert screen['allow_fits']
    assert sha('runs/nan_fit07/bf16/baseline.json')==seal['upstream_baseline_sha256']
    sae_file=Path('runs/nan_gym05/bf16/sae_tower/layer21.npz');assert sha(sae_file)==seal['probe_cache']['bf16']['sae_dictionary_sha256'];sae=load_npz(sae_file)
    cells=[]
    for index in range(6):
        cfg=condition(index);name=key(cfg);meta=runs/'metadata'/name;folder=runs/'conditions'/name;events=closed_events(meta);stages={r['event'] for r in events}
        item=dict(condition=name,module=cfg['module'],rank=cfg['rank'],fit_passes=cfg['passes'],status='queued',baseline_contexts_verified=0,
            optimizer_steps_completed=0,fit_independent_audit=None,physical_write_SAE_probe=None,cached_suite_probe=None,failure_type=None)
        baseline=None;fit=None;checked=None;worker=None
        if 'baseline_closed' in stages:
            baseline=read(meta/'baseline.json');assert baseline['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
            assert baseline['records']==original['records'] and baseline['target_parameter_sha256']==original['target_parameter_sha256']
            assert baseline['upstream_baseline_sha256']==seal['upstream_baseline_sha256'] and baseline['matched_contexts']==32
            assert len(baseline['full_head_parity'])==2 and all(p['bit_exact'] and p['full_vocabulary_logits_checked']==166144 for p in baseline['full_head_parity'])
            item.update(status='baseline_verified_fit_in_progress',baseline_contexts_verified=32)
        if 'fit_closed' in stages:
            fit=read(folder/'receipt.json');assert fit['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
            item.update(status=fit['status'],optimizer_steps_completed=fit['optimizer_steps_completed'],failure_type=fit.get('failure_type'))
            if fit['status']=='completed':
                assert baseline is not None
                auditfile=runs/'audit'/(name+'.json')
                if auditfile.exists():
                    checked=read(auditfile);assert checked['fit_receipt_sha256']==sha(folder/'receipt.json')
                    assert checked['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json') and checked['status']=='passed'
                else:
                    checked=cell_audit(folder,'bf16',cfg['passes'],cfg,public,gold,seal,baseline)
                    checked.update(source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'));write(auditfile,checked)
                item.update(status='fit_audited_physical_probe_in_progress',fit_independent_audit=checked)
        if 'worker_receipt_closed' in stages:
            worker=read(meta/'receipt.json');assert worker['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json') and worker['condition']==name
            item.update(status=worker['status'],failure_type=(worker.get('failure') or {}).get('type') or item['failure_type'],
                worker_receipt_sha256=sha(meta/'receipt.json'),CUDA_peak_allocated_bytes=worker['CUDA_peak_allocated_bytes'],CUDA_peak_reserved_bytes=worker['CUDA_peak_reserved_bytes'])
            if checked:
                physical=folder/'physical_bridge/receipt.json'
                if 'physical_probe_closed' in stages:
                    auditfile=runs/'audit'/(name+'_physical.json')
                    if auditfile.exists():
                        p=read(auditfile);assert p['physical_receipt_sha256']==sha(physical) and p['auditor_sha256']==sha(root/'physical_audit.py')
                    else:
                        p=physical_audit(folder,public,cfg,fit,sae);p.update(auditor_sha256=sha(root/'physical_audit.py'),SAE_dictionary_sha256=sha(sae_file));write(auditfile,p)
                    assert p['status']=='passed';item['physical_write_SAE_probe']=p
                cachefile=runs/'probes'/(name+'_cached.json')
                if with_probes and not cachefile.exists():
                    from .probe import probe
                    probe(runs,index)
                if cachefile.exists():
                    p=read(cachefile);assert p['checkpoint_sha256']==checked['checkpoint_sha256'] and p['fit_receipt_sha256']==sha(folder/'receipt.json')
                    assert p['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json') and p['status']=='passed'
                    item['cached_suite_probe']=p
                if worker['status']=='completed':
                    assert item['physical_write_SAE_probe'] is not None
                    item['status']='completed_and_audited' if item['cached_suite_probe'] is not None else 'fit_and_finite_probe_audited_cached_probe_pending'
        drivers=[]
        for stage,required in (('driver_after_baseline.json','baseline_closed'),('driver_terminal.json','worker_receipt_closed')):
            if required in stages and (meta/stage).exists():
                d=read(meta/stage);drivers.append(dict(stage=stage,status=d['status'],dedicated_bytes=sum(r['DedicatedUsage'] for r in d.get('WDDM_worker_rows',[])),
                    shared_bytes=sum(r['SharedUsage'] for r in d.get('WDDM_worker_rows',[])),whole_device_used_MiB=d.get('nvidia_device_used_MiB'),source_sha256=sha(meta/stage)))
        item['driver_samples']=drivers;cells.append(item)
    pairs=[]
    for module,rank in (('down_proj',16),('o_proj',8),('o_proj',16)):
        matched=[r for r in cells if r['module']==module and r['rank']==rank and r['fit_independent_audit'] is not None]
        if len(matched)!=2:continue
        banks=[]
        for r in matched:
            folder=runs/'conditions'/r['condition'];receipt=read(folder/'receipt.json');path=folder/receipt['final_checkpoint']['file']
            assert sha(path)==r['fit_independent_audit']['checkpoint_sha256'];banks.append(load_npz(path))
        a,b=banks;qa=np.linalg.qr(a['U'].astype(np.float64).T,mode='reduced')[0];qb=np.linalg.qr(b['U'].astype(np.float64).T,mode='reduced')[0]
        pairs.append(dict(module=module,rank=rank,physical_coordinate='native module output before any downstream residual/norm/MLP',
            **operator_comparison(a['V'],a['U'],b['V'],b['U']),output_subspace_principal_cosines=np.linalg.svd(qa.T@qb,compute_uv=False).tolist(),
            scope='gauge-invariant descriptive physical operator/subspace comparison; fit depth and ablation/adversary rules differ, no isolated causal effect'))
    payload=dict(experiment='NAN-RANKSITE-13',updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),cells=cells,learned_VPD_BPD_pairs=pairs,
        upstream_BF16_competence_screen=screen,NF4_status='excluded: original FIT07 DEV5/8 failed fixed gate; no new NF4 fit or competence repair',
        unit='fitting/finite probes reuse eight exposed CAL/eight DEV families; wider cached probes cover89 earlier source groups; one checkpoint, all variant/loop/dose measures correlated',
        n_CAL_roots=8,n_DEV_roots=8,n_gate_contexts=32,n_fitted_cells_registered=6,n_cached_source_groups=89,n_cached_vectors=644,
        primary_contrast='descriptive native learned rank/site/depth instrument coverage; no isolated effect of BPD versus VPD',
        physical_dose='local module-output relative L2 requests .001/.01/.1, both signs; actual rounded doses retained, not donor/next-loop causal interventions',
        scientific_causal_floor=None,matched_sham_floor=None,estimated_causal_power='not calibrated; reused development sources, no sealed confirmation readout',
        readout_selection_exposure=read(root/'config.json')['source_exposure'],planned_formal_comparisons=0,actual_formal_comparisons=0,
        BPD_donor_transfer_attempted=False,sealed_test_accessed=False,privacy='aggregates and hashes only; prompts/replies/states/per-record codes and routing derivatives stay local')
    write(out/'nan_ranksite13_progress.json',payload)
    lines=['# NAN-RANKSITE-13: native rank/site instrument extension','',f"Updated {payload['updated_UTC']}",'',
        'Six missing BF16 learned cells are registered on the same exposed FIT07 sources: down_proj rank16 and o_proj ranks8/16, each at T1/T2. '+
        'All use the same eight-step genuine trainer recipe, source order and fixed diagnostic modes, with the validated RECOMP09 memory execution. '+
        'Historical rank8 down_proj fits and every original failure remain unchanged. NF4 remains excluded by its fixed failed fitting gate.','',
        '| Module | Rank | Fit depth | Status | Baselines verified | Optimizer steps |', '|---|---:|---:|---|---:|---:|']
    for r in cells:lines.append(f"| {r['module']} | {r['rank']} | {r['fit_passes']} | {r['status']} | {r['baseline_contexts_verified']}/32 | {r['optimizer_steps_completed']}/8 |")
    lines+=['','All 32 baseline readout records must match the old BF16 condition; both native loops receive full-vocabulary CPU/resident-head parity checks. '+
        'Each successful cell retains independent checkpoint/loss/controller/gauge/calibration/diagnostic checks and every final correctness/KL/actual-write group. '+
        'The complete cached suite probe covers644 physical vectors and every source/copy/loop group, with all128-feature correlations and initial-SVD/norm-matched-Haar controls.','',
        'Separate finite-write tapes cover all16 main fitting prefixes, both native loops, all three bases/components/doses/signs. '+
        'They preserve full prefix shape and actual residual/norm/MLP arithmetic. The CPU audit verifies native BF16 additions, actual physical norms and complete coverage. '+
        'All feature responses and zero/rounded-away writes are retained. o_proj never receives an identity bridge to the layer-output SAE. '+
        'Native MLP outputs/JVPs are source-bound worker measurements; the CPU audit does not replay a 3B MLP. The derivative is not asserted equivalent to finite rounded writes.','',
        'Finite output geometry is averaged equally over roots/components within every split/loop/basis/dose/sign group, retaining all128 features. '+
        'The full component-wise tapes remain local. T1 application at loop2 is out of training depth. These are exposed development instruments; no equivalence, '+
        'semantic feature mediation, powered mechanism comparison or donor transfer. Static controls retaining accuracy remain a specificity limitation.','',
        'All completed same-module/rank T1/T2 pairs receive gauge-invariant physical operator/subspace comparisons. '+
        'Those are native module-output coordinates, not an unbridged o_proj-to-SAE direction comparison. Fit-depth/ablation/adversary differences remain confounded.','',
        'One large model at a time, one numerical CPU thread and Below Normal. Each cell has a 7200-second between-stage budget and unchanged 2-GiB added saved-tensor CPU cap; '+
        'no automatic neural retries. Allocated/reserved CUDA peaks and instantaneous dedicated/shared counters stay separate. '+
        'The separate math process is untouched. The original 168-case test stays sealed.','',
        '[Protocol](../../experiments/nan_ranksite13/PROTOCOL.md) · [Source seal](../../experiments/nan_ranksite13/SOURCE_MANIFEST.json) · '+
        '[Full aggregate evidence](nan_ranksite13_progress.json) · [Random engineering](NAN_RANKSITE13_ENGINEERING.md)','']
    (out/'NAN_RANKSITE13_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({r['condition']:r['status'] for r in cells})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_ranksite13');p.add_argument('--out',default='reports/nanbeige_bpd_gym');p.add_argument('--with-probes',action='store_true');a=p.parse_args()
    below_normal();report(Path(a.runs),Path(a.out),with_probes=a.with_probes)


if __name__=='__main__':main()
