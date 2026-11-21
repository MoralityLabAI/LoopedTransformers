"""Source-bound aggregate amended-fit audit; original FIT07 evidence untouched."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from experiments.nan_fit07.io import read,rows,sha,write
from mechinterp.nanbeige_bpd_gym.audit_fit import baseline_audit
from .cell_audit import cell_audit


def report(runs,out):
    root=Path(__file__).parent;seal=read(root/'SOURCE_MANIFEST.json');cfg=read(root/'config.json')
    for field in ('worker_files_sha256','host_only_files_sha256'):
        for name,h in seal[field].items():assert sha(root/name)==h,name
    for field in ('model_dependencies_sha256','analysis_dependencies_sha256'):
        for name,h in seal[field].items():assert sha(name)==h,name
    upstream=Path('experiments/nan_fit07');public=rows(upstream/'fixtures/public.jsonl');labels=rows(upstream/'fixtures/host_labels.jsonl')
    assert sha(upstream/'SOURCE_MANIFEST.json')==seal['upstream_source_manifest_sha256']
    assert sha(upstream/'fixtures/public.jsonl')==seal['public_source_sha256'] and sha(upstream/'fixtures/host_labels.jsonl')==seal['host_labels_sha256']
    original,screen=baseline_audit(Path('runs/nan_fit07/bf16'),public,{r['id']:r['answer'] for r in labels},cfg)
    assert screen['allow_fits'] and sha('runs/nan_fit07/bf16/baseline.json')==seal['upstream_baseline_sha256']
    folder=runs/'bf16';status='queued';checked=None;baseline=None;receipt=None
    if (folder/'baseline.json').exists():
        baseline=read(folder/'baseline.json');assert baseline['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
        assert baseline['upstream_baseline_sha256']==seal['upstream_baseline_sha256'] and baseline['records']==original['records']
        assert baseline['target_parameter_sha256']==original['target_parameter_sha256'] and baseline['matched_contexts']==32
        assert len(baseline['full_head_parity'])==2 and all(p['bit_exact'] and p['full_vocabulary_logits_checked']==166144 for p in baseline['full_head_parity'])
        status='baseline_verified_fit_in_progress'
    cell=folder/'t2_layer21_down_proj_rank8';cell_receipt=cell/'receipt.json'
    if cell_receipt.exists():
        receipt=read(cell_receipt);status=receipt['status']
        if status=='completed':
            assert baseline is not None
            checked=cell_audit(cell,'bf16',2,cfg,public,{r['id']:r['answer'] for r in labels},seal,baseline)
            stats=checked['saved_tensor_statistics'];assert stats['checkpointed_frozen_blocks']>0 and stats['recomputed_frozen_blocks']>0 and stats['resident_GPU_weight_references']>0
            write(runs/'audit/bf16_independent.json',checked)
    worker=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None
    if worker:
        assert worker['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
        status=worker['status'] if receipt is None else status
    probe_file=runs/'probes/bf16_t2_aggregate.json';probe=None
    if checked and probe_file.exists():
        probe=read(probe_file);assert probe['status']=='passed' and probe['checkpoint_sha256']==checked['checkpoint_sha256'] and probe['fit_receipt_sha256']==sha(cell_receipt)
    result=dict(experiment='NAN-RECOMP-09',updated_UTC=datetime.now(timezone.utc).isoformat(),status=status,
        source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),upstream_competence_screen=screen,
        baseline_contexts_verified=32 if baseline else 0,independent_fit_audit=checked,SAE_cross_reference=probe,
        optimizer_steps_completed=receipt.get('optimizer_steps_completed',0) if receipt else 0,
        failure_type=receipt.get('failure_type') if receipt else (worker.get('failure') or {}).get('type') if worker else None,
        saved_tensor_statistics=receipt.get('saved_tensor_statistics') if receipt else None,
        worker_receipt_sha256=sha(folder/'receipt.json') if worker else None,
        scope='BF16-only execution amendment on exposed sources; original failed T2 and NF4 failed competence gate preserved',
        formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False,
        privacy='aggregates and hashes only; no raw predictions, tokens, states, source IDs or per-record codes')
    write(out/'nan_recomp09_progress.json',result)
    lines=['# NAN-RECOMP-09: memory-bounded native BF16 BPD fit','',f"Updated {result['updated_UTC']}",'',
        f"Evidence status: **{status}**. Independently checked baseline contexts: {result['baseline_contexts_verified']}/32. Actual optimizer steps completed: {result['optimizer_steps_completed']}/8.",'',
        'The original BF16 T=2 FIT07 attempt hit its 2-GiB added saved-tensor CPU cap before an optimizer step. '+
        'This new condition retains that cap, the same exposed source roster, CAL order, initial SVD, four stock loss terms, '+
        'eight-step final-checkpoint rule, null normalization and fixed diagnostics. It recomputes frozen blocks during backward, '+
        'retains version-guarded references to immutable resident weights, and keeps the unchanged BF16 head on GPU. '+
        'The selected layer-21 hook is excluded from recomputation so pass counters and reached-state routing remain correct.','',
        'BF16 previously passed the frozen screen: CAL 7/8, DEV 7/8, copy 16/16. This is not a new competence test. '+
        'NF4 remains excluded because its original DEV screen was 5/8. The original failed run and all its partial artifacts remain intact.','',
        'Before fitting, all 32 new-execution baseline readouts must equal the original and every vocabulary logit '+
        'must agree with the CPU head on the first source at both loops. Real masked factor gradients must be finite/nonzero. '+
        'Random 22-layer/width-64 BF16/NF4 engineering already matched forward states, factor gradients and two stock optimizer steps '+
        'per path under matched global/trainer/adversary RNG. That engineering evidence is separate from the real fit.','']
    if checked:
        lines += [f"Final independent audit passed. Factor update L2: {checked['factor_update_l2']:.6g}; controller update L2: {checked['controller_update_l2']:.6g}. "+
            'The JSON retains every initial/final diagnostic group, physical-dose/loop metric, gauge check, checkpoint and memory receipt.','']
    if probe:lines += [f"SAE/initial-SVD/norm-matched-Haar cross-reference passed: {len(probe['groups'])} source/loop groups and {len(probe['geometry'])} geometry cells. Cached clean-input application is observational, not online donor transfer.",'']
    lines += ['CPU threads=1, Below Normal priority, CUDA fraction=.85, four BF16 decoder layers plus the reusable stage and resident head, '+
        '7200-second between-stage budget. One large model at a time. No DEV/copy fitting, source selection, rank selection, '+
        'causal claim, equivalence or donor transfer. Eight CAL/eight DEV source roots remain repeated measures on one checkpoint. '+
        'The original 168-case test stays sealed.','',
        '[Protocol](../../experiments/nan_recomp09/PROTOCOL.md) · [Source seal](../../experiments/nan_recomp09/SOURCE_MANIFEST.json) · [Aggregate evidence](nan_recomp09_progress.json)','']
    (out/'NAN_RECOMP09_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({'status':status,'optimizer_steps':result['optimizer_steps_completed'],'independent_fit_checked':checked is not None,'SAE_probe_checked':probe is not None})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_recomp09');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    report(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
