"""Seal all six real rank/site cells, analyses and engineering provenance once."""
from datetime import datetime,timezone
from pathlib import Path
from .common import read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Never reseal a condition')
    upstream=Path('experiments/nan_fit07');old=read(upstream/'SOURCE_MANIFEST.json')
    assert sha(upstream/'SOURCE_MANIFEST.json')=='8890f915d7b1ba05dc611dfb62e2c1d40abc2cc5bb71351e7b338f640b6f104e'
    for field in ('worker_files_sha256','host_only_files_sha256'):
        for name,h in old[field].items():assert sha(upstream/name)==h,name
    gate=Path('runs/nan_fit07/bf16/host_gate.json');g=read(gate);assert g['allow_fits'] and g['baseline_sha256']==sha('runs/nan_fit07/bf16/baseline.json')
    nf=Path('runs/nan_fit07/nf4/host_gate.json');assert not read(nf)['allow_fits']
    failed=Path('runs/nan_fit07/bf16/t2_layer21_down_proj_rank8/receipt.json');f=read(failed)
    assert f['status']=='failed' and f['failure_type']=='MemoryError' and f['optimizer_steps_completed']==0
    for name in ('random_pipeline_v1.json','postfit_random_v1.json'):
        proof=read(root/'receipts'/name);assert proof['status']=='passed' and not proof['pretrained_weights_loaded']
        for source,h in proof['source_sha256'].items():assert sha(root/source)==h,source
    fitproof=root/'receipts/random_pipeline_v1.json';postproof=root/'receipts/postfit_random_v1.json'
    a=read(root/'receipts/random_pipeline_independent_v1.json')
    assert a['status']=='passed' and a['optimizer_steps_verified']==48 and a['engineering_receipt_sha256']==sha(fitproof)
    assert a['source_sha256']==sha(root/'audit_engineering.py')
    a=read(root/'receipts/postfit_independent_v1.json')
    assert a['status']=='passed' and a['signed_finite_writes_verified']==5760 and a['engineering_receipt_sha256']==sha(postproof)
    assert a['source_sha256']==sha(root/'audit_postfit_engineering.py') and a['physical_auditor_sha256']==sha(root/'physical_audit.py')
    bridge=Path('mechinterp/nanbeige_bpd_gym/receipts/o_proj_bridge_random_v1.json');b=read(bridge)
    assert b['status']=='passed' and len(b['full_forward_signed_comparisons'])==192 and not b['pretrained_weights_loaded']
    for name,h in b['source_sha256'].items():assert sha(Path('mechinterp/nanbeige_bpd_gym')/name)==h,name
    dependencies=dict(read('experiments/nan_recomp09/SOURCE_MANIFEST.json')['model_dependencies_sha256'])
    for name in ('experiments/nan_recomp09/common.py','mechinterp/nanbeige_bpd_gym/o_proj_bridge.py'):
        dependencies[name]=sha(name)
    analysis={name:sha(name) for name in ('mechinterp/nanbeige_bpd_gym/audit_fit.py','mechinterp/nanbeige_bpd_gym/audit_native.py',
        'mechinterp/nanbeige_bpd_gym/compare_learned_fits.py')}
    linear_file=Path('runs/nan_gym05/bf16/linear_baselines/manifest.json');linear=read(linear_file);initialization={}
    for module,width in (('down_proj',10752),('o_proj',6144)):
        row=next(r for r in linear['receipts'] if r['layer_zero_based']==21 and r['module']==module)
        assert row['shape']==[3072,width] and row['ranks']['16']['closure_relative_l2']<=1e-5
        path=linear_file.parent/row['file'];assert sha(path)==row['file_sha256']
        initialization[module]=dict(file_sha256=sha(path),effective_operator_sha256=row['effective_float32_operator_sha256'],shape=row['shape'])
    assert initialization['down_proj']['file_sha256']==old['initialization']['bf16']['file_sha256']
    host=['cell_audit.py','diagnostic_audit.py','physical_audit.py','probe.py','report.py','audit_engineering.py','audit_postfit_engineering.py',
        'tests/test_contracts.py','tests/test_physical.py']
    files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=target and p.relative_to(root).as_posix() not in host)
    write(target,dict(experiment='NAN-RANKSITE-13',frozen_UTC=datetime.now(timezone.utc).isoformat(),backend='bf16',
        worker_files_sha256={n:sha(root/n) for n in files},host_only_files_sha256={n:sha(root/n) for n in host},
        model_dependencies_sha256=dependencies,analysis_dependencies_sha256=analysis,
        upstream_source_manifest_sha256=sha(upstream/'SOURCE_MANIFEST.json'),public_source_sha256=sha(upstream/'fixtures/public.jsonl'),
        host_labels_sha256=sha(upstream/'fixtures/host_labels.jsonl'),upstream_host_gate_sha256=sha(gate),upstream_baseline_sha256=sha('runs/nan_fit07/bf16/baseline.json'),
        preserved_NF4_failed_gate_sha256=sha(nf),preserved_BF16_original_T2_failure_sha256=sha(failed),initialization=initialization,
        linear_baseline_manifest_sha256=sha(linear_file),probe_cache={'bf16':old['probe_cache']['bf16']},
        tokenizer_config_sha256=old['tokenizer_config_sha256'],checkpoint_revision=old['checkpoint_revision'],
        native_bridge_engineering_receipt_sha256=sha(bridge),
        exposure='All prior FIT07/GYM05/context/Prime/cache and random engineering outcomes exposed; every registered source/condition retained',
        registered_cells=6,CAL_roots=8,DEV_roots=8,pretrained_condition_inference_started_at_freeze=False,
        formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False))
    print({'manifest_sha256':sha(target),'worker_files':len(files),'model_dependencies':len(dependencies),'registered_cells':6})


if __name__=='__main__':main()
