"""Prospectively seal the fresh-source fitting pilot; never reseal an attempt."""
from datetime import datetime,timezone
from pathlib import Path
from .io import read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Preserve the registered fitting condition')
    previous=read('experiments/nan_gym05/SOURCE_MANIFEST.json')
    dependencies=dict(previous['model_dependencies_sha256'])
    names=['experiments/nan_gym05/'+n+'.py' for n in ('common','sources','parser','grade')]
    names += ['mechinterp/nanbeige_bpd_gym/'+n for n in ('native_components.py','native_saved_tensors.py','BPD_FRAMEWORK_LOCK.json','probe_towers.py','crossref.py')]
    names += ['mechinterp/sae_causal_transfer/adapter.py']
    for name in names:dependencies[name]=sha(name)
    host=['gate.py','fixtures/host_labels.jsonl','sources.py','tests/test_contracts.py']
    files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p!=target and p.relative_to(root).as_posix() not in host)
    initialization={};cache={}
    for backend in ('bf16','nf4'):
        folder=Path('runs/nan_gym05')/backend
        bank=folder/'linear_baselines/layer21_down_proj.npz'
        manifest=read(folder/'linear_baselines/manifest.json')
        record=next(r for r in manifest['receipts'] if r['layer_zero_based']==21 and r['module']=='down_proj')
        assert sha(bank)==record['file_sha256']
        initialization[backend]={'file_sha256':sha(bank),'effective_operator_sha256':record['effective_float32_operator_sha256']}
        audit=Path('runs/nan_gym05/audit')/(backend+'_independent.json')
        assert read(audit)['status']=='passed'
        cache[backend]={'independent_audit_sha256':sha(audit),'sae_dictionary_sha256':sha(folder/'sae_tower/layer21.npz')}
    token=read(root/'fixtures/tokenizer_receipt.json')
    assert token['public_file_sha256']==sha(root/'fixtures/public.jsonl') and not token['pretrained_weights_loaded']
    write(target,{'experiment':'NAN-FIT-07','frozen_UTC':datetime.now(timezone.utc).isoformat(),
        'worker_files_sha256':{name:sha(root/name) for name in files},
        'host_only_files_sha256':{name:sha(root/name) for name in host},
        'model_dependencies_sha256':dependencies,'initialization':initialization,'probe_cache':cache,
        'tokenizer_config_sha256':token['tokenizer_config_sha256'],
        'prior_native_registration_sha256':sha('experiments/nan_gym05/SOURCE_MANIFEST.json'),
        'checkpoint_revision':previous['checkpoint_revision'],'BPD_commit':previous['BPD_source_reference_commit'],
        'pretrained_condition_inference_started_at_freeze':False,'pretrained_fitting_started_at_freeze':False,
        'formal_inferential_comparisons':0,'BPD_donor_transfer_enabled':False,'sealed_168_case_test_accessed':False})
    print({'manifest_sha256':sha(target),'worker_files':len(files),'host_only_files':len(host),'model_dependencies':len(dependencies)})


if __name__=='__main__':main()
