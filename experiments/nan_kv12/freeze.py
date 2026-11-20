"""Seal a distinct NF4 arithmetic condition before real-checkpoint execution."""
from datetime import datetime,timezone
from pathlib import Path
from .common import public_rows,read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Never reseal a condition')
    public_rows();base=Path('mechinterp/nanbeige_bpd_gym');proof=base/'receipts/effective_nf4_random_v1.json';r=read(proof)
    assert r['status']=='passed' and not r['pretrained_weights_loaded']
    assert all(sha(base/n)==h for n,h in r['source_sha256'].items())
    assert [v['policy'] for v in r['results']]==['bf16','nf4_default','nf4_explicit_BF16']
    assert all(v['status']=='passed' and v['target_parameters_unchanged'] for v in r['results'])
    previous=read('experiments/nan_kv11/SOURCE_MANIFEST.json');deps=dict(previous['model_dependencies_sha256'])
    assert sha('experiments/nan_kv11/SOURCE_MANIFEST.json')=='bf9cb5d06ffa94d57de11f54dc4d4d4b2ab473c4ccf9f22715fb8d6d7960de67'
    for name in ('mechinterp/nanbeige_bpd_gym/effective_nf4.py','experiments/nan_kv11/SOURCE_MANIFEST.json'):
        deps[name]=sha(name)
    host=['report.py','tests/test_contracts.py']
    files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=target and p.relative_to(root).as_posix() not in host)
    write(target,dict(experiment='NAN-KV-12',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        worker_files_sha256={n:sha(root/n) for n in files},host_only_files_sha256={n:sha(root/n) for n in host},
        model_dependencies_sha256=deps,GYM05_source_manifest_sha256=sha('experiments/nan_gym05/SOURCE_MANIFEST.json'),
        public_roster_sha256=sha('experiments/nan_gym05/fixtures/public.jsonl'),
        kernel_engineering_receipt_sha256=sha(proof),tokenizer_config_sha256=previous['tokenizer_config_sha256'],
        selection_exposure='All GYM05/CTX08/official SDK and partial KV11 outcomes exposed; random explicit-effective fixture exposed; all public sources retained',
        source_families=41,registered_contexts=130,native_loops=2,pretrained_condition_inference_started_at_freeze=False,
        formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False))
    print({'manifest_sha256':sha(target),'worker_files':len(files),'model_dependencies':len(deps)})


if __name__=='__main__':main()
