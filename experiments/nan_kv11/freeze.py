"""Seal the complete wider-suite cache diagnostic before pretrained execution."""
from datetime import datetime,timezone
from pathlib import Path
from .common import public_rows,read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Never reseal a condition')
    public_rows();base=Path('mechinterp/nanbeige_bpd_gym');proof=base/'receipts/native_cache_random_v2.json';r=read(proof)
    assert r['status']=='passed' and not r['pretrained_weights_loaded']
    assert r['cache_implementation_sha256']==sha(base/'native_cached.py') and r['source_sha256']==sha(base/'validate_native_cached.py')
    assert r['prospective_plan_sha256']==sha(base/'KV_ENGINEERING_PLAN.md')
    assert all(c['status']=='passed' and c['MLP_intermediate_width']==128 for c in r['results'])
    previous=read('experiments/nan_prime_ctx10/SOURCE_MANIFEST.json');deps=dict(previous['model_dependencies_sha256'])
    for name in ('experiments/nan_comp03/prompt.py','experiments/nan_recomp09/common.py','mechinterp/nanbeige_bpd_gym/native_cached.py',
                 'mechinterp/dark_fiber_gyms/core.py'):
        deps[name]=sha(name)
    host=['report.py','tests/test_contracts.py'];files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=target and p.relative_to(root).as_posix() not in host)
    write(target,dict(experiment='NAN-KV-11',frozen_UTC=datetime.now(timezone.utc).isoformat(),worker_files_sha256={n:sha(root/n) for n in files},
        host_only_files_sha256={n:sha(root/n) for n in host},model_dependencies_sha256=deps,
        GYM05_source_manifest_sha256=sha('experiments/nan_gym05/SOURCE_MANIFEST.json'),public_roster_sha256=sha('experiments/nan_gym05/fixtures/public.jsonl'),
        kernel_engineering_receipt_sha256=sha(proof),tokenizer_config_sha256=previous['tokenizer_config_sha256'],
        selection_exposure='All prior GYM05/CTX08 and partial official SDK results exposed; no result-based source selection; no independent replication',
        source_families=41,registered_contexts=130,native_loops=2,pretrained_condition_inference_started_at_freeze=False,
        formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False))
    print({'manifest_sha256':sha(target),'worker_files':len(files),'model_dependencies':len(deps)})


if __name__=='__main__':main()
