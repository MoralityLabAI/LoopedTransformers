"""Seal BF16's resource amendment before any pretrained amended rollout."""
from datetime import datetime,timezone
from pathlib import Path
from experiments.nan_fit07.io import read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Never reseal an existing condition')
    parent=Path('experiments/nan_fit07');upstream=read(parent/'SOURCE_MANIFEST.json')
    assert sha(parent/'SOURCE_MANIFEST.json')=='8890f915d7b1ba05dc611dfb62e2c1d40abc2cc5bb71351e7b338f640b6f104e'
    for field in ('worker_files_sha256','host_only_files_sha256'):
        for name,h in upstream[field].items():assert sha(parent/name)==h,name
    gate_file=Path('runs/nan_fit07/bf16/host_gate.json');gate=read(gate_file);assert gate['allow_fits']
    assert gate['baseline_sha256']==sha('runs/nan_fit07/bf16/baseline.json')
    original=read('runs/nan_fit07/bf16/t2_layer21_down_proj_rank8/receipt.json')
    assert original['status']=='failed' and original['optimizer_steps_completed']==0 and original['failure_type']=='MemoryError'
    engineering=read(root/'receipts/full_pipeline_random_v2.json');assert engineering['status']=='passed' and not engineering['pretrained_weights_loaded']
    assert engineering['recomputation_source_sha256']==sha('mechinterp/nanbeige_bpd_gym/recomputed_native.py')
    for name,h in engineering['source_sha256'].items():assert sha(root/name)==h,name
    dependencies=dict(upstream['model_dependencies_sha256'])
    for name in ('experiments/nan_fit07/worker.py','experiments/nan_fit07/pointwise.py','experiments/nan_fit07/io.py',
                 'mechinterp/nanbeige_bpd_gym/resident_head.py','mechinterp/nanbeige_bpd_gym/recomputed_native.py'):
        dependencies[name]=sha(name)
    host=['cell_audit.py','probe.py','report.py','tests/test_contracts.py']
    files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=target and p.relative_to(root).as_posix() not in host)
    write(target,dict(experiment='NAN-RECOMP-09',frozen_UTC=datetime.now(timezone.utc).isoformat(),backend='bf16',
        worker_files_sha256={name:sha(root/name) for name in files},host_only_files_sha256={name:sha(root/name) for name in host},
        model_dependencies_sha256=dependencies,analysis_dependencies_sha256={'mechinterp/nanbeige_bpd_gym/audit_fit.py':sha('mechinterp/nanbeige_bpd_gym/audit_fit.py')},
        upstream_source_manifest_sha256=sha(parent/'SOURCE_MANIFEST.json'),public_source_sha256=sha(parent/'fixtures/public.jsonl'),
        host_labels_sha256=sha(parent/'fixtures/host_labels.jsonl'),upstream_host_gate_sha256=sha(gate_file),
        upstream_baseline_sha256=sha('runs/nan_fit07/bf16/baseline.json'),original_failed_cell_receipt_sha256=sha('runs/nan_fit07/bf16/t2_layer21_down_proj_rank8/receipt.json'),
        initialization={'bf16':upstream['initialization']['bf16']},probe_cache={'bf16':upstream['probe_cache']['bf16']},
        tokenizer_config_sha256=upstream['tokenizer_config_sha256'],checkpoint_revision=upstream['checkpoint_revision'],
        source_exposure='All FIT07 readouts and previous BF16 T2 resource failure exposed before this execution amendment',
        pretrained_amended_inference_started_at_freeze=False,formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False))
    print({'manifest_sha256':sha(target),'worker_files':len(files),'model_dependencies':len(dependencies)})


if __name__=='__main__':main()
