"""Seal all worker/SDK/analysis sources and existing evidence before inference."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from .common import public_rows,read,sha,write,CTX_SEAL,SDK_COMMIT


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);a=p.parse_args()
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Never reseal a condition')
    public_rows();engineering=read(root/'receipts/official_mock_v1.json');assert engineering['status']=='passed' and engineering['model_rollouts']==0
    for name,h in engineering['source_sha256'].items():assert sha(root/name)==h,name
    assert sha('nanbeige_context_suite.py')==engineering['entry_point_sha256']
    source=read('experiments/nan_ctx08/SOURCE_MANIFEST.json');deps=dict(source['model_dependencies_sha256'])
    deps['experiments/nan_prime06/wire.py']=sha('experiments/nan_prime06/wire.py')
    external=Path(a.compactification_root)/'research/algernon/algebraic_reasoning_gym/arg02'
    for name in ('runner.py','settings.py','nanbeige.py','recurrence.py','config/checkpoint_lock.json'):
        file=external/name;deps[file.as_posix()]=sha(file)
    host=['taskset.py','validate_linux.py','report.py','learned_probe.py','tests/test_contracts.py']
    files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=target and p.relative_to(root).as_posix() not in host)
    evidence=['reports/nanbeige_bpd_gym/'+n for n in ('nan_ctx08_progress.json','nan_ctx08_tower_probes.json','nan_ctx08_comparison.json')]
    evidence+=['runs/nan_ctx08/'+b+'/receipt.json' for b in ('nf4','bf16')]
    for b in ('nf4','bf16'):assert read('runs/nan_ctx08/'+b+'/receipt.json')['status']=='completed'
    bpd=read('runs/nan_recomp09/audit/bf16_independent.json');assert bpd['status']=='passed'
    bpdr=read('runs/nan_recomp09/bf16/t2_layer21_down_proj_rank8/receipt.json');assert bpdr['status']=='completed' and bpdr['optimizer_steps_completed']==8
    runtime='reports/nanbeige_bpd_gym/prime_persistent_runtime.json';assert read(runtime)['status']=='passed' and read(runtime)['sdk_commit']==SDK_COMMIT
    hostdeps=['nanbeige_context_suite.py','mechinterp/nanbeige_bpd_gym/audit_native.py','mechinterp/nanbeige_bpd_gym/prime_evidence.py',
        'mechinterp/nanbeige_bpd_gym/context_probes.py','mechinterp/nanbeige_bpd_gym/probe_towers.py','mechinterp/nanbeige_bpd_gym/crossref.py',
        'mechinterp/sae_causal_transfer/adapter.py','experiments/nan_fit07/pointwise.py','experiments/nan_fit07/probe.py']
    snap=Path(a.snapshot)
    write(target,dict(experiment='NAN-PRIME-CTX-10',frozen_UTC=datetime.now(timezone.utc).isoformat(),worker_files_sha256={n:sha(root/n) for n in files},
        host_only_files_sha256={n:sha(root/n) for n in host},model_dependencies_sha256=deps,host_dependencies_sha256={n:sha(n) for n in hostdeps},
        CTX08_source_manifest_sha256=CTX_SEAL,native_evidence_sha256={n:sha(n) for n in evidence},
        learned_BPD_producer_manifest_sha256=sha('experiments/nan_recomp09/SOURCE_MANIFEST.json'),learned_BPD_checkpoint_sha256=bpd['checkpoint_sha256'],
        learned_BPD_audit_sha256=sha('runs/nan_recomp09/audit/bf16_independent.json'),
        learned_VPD_checkpoint_sha256=sha('runs/nan_fit07/bf16/t1_layer21_down_proj_rank8/final.npz'),
        SAE_layer21_BF16_sha256=source['SAE_dictionary_sha256']['bf16']['21'],
        tokenizer_config_sha256=sha(snap/'tokenizer_config.json'),tokenizer_json_sha256=sha(snap/'tokenizer.json'),
        SDK_commit=SDK_COMMIT,persistent_runtime_receipt_sha256=sha(runtime),
        selection_exposure='Complete CTX08 outcomes/tower probes and completed FIT07/RECOMP09 fit/probe outcomes exposed; no independent replication',
        pretrained_condition_inference_started_at_freeze=False,formal_inferential_comparisons=0,BPD_transfer_enabled=False,sealed_test_accessed=False))
    print({'manifest_sha256':sha(target),'worker_files':len(files),'model_dependencies':len(deps)})


if __name__=='__main__':main()
