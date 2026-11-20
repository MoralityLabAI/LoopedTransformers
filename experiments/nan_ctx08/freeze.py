"""Prospectively seal the new resident-head context condition."""
from datetime import datetime,timezone
from pathlib import Path
from experiments.nan_fit07.io import read,sha,write


def main():
    root=Path(__file__).parent;target=root/'SOURCE_MANIFEST.json'
    if target.exists():raise FileExistsError('Never reseal an existing condition')
    previous=read('experiments/nan_gym05/SOURCE_MANIFEST.json');dependencies=dict(previous['model_dependencies_sha256'])
    for name in ('experiments/nan_gym05/common.py','experiments/nan_gym05/observe.py','experiments/nan_fit07/io.py',
                 'mechinterp/nanbeige_bpd_gym/resident_head.py'):
        dependencies[name]=sha(name)
    host=['sources.py','report.py','tests/test_sources.py','fixtures/host_labels.jsonl']
    files=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p!=target and p.relative_to(root).as_posix() not in host)
    tokens=read(root/'fixtures/tokenizer_receipt.json');assert not tokens['pretrained_weights_loaded']
    assert tokens['public_sha256']==sha(root/'fixtures/public.jsonl')
    engineering=read(root/'receipts/resident_placement_random_v2.json')
    assert engineering['status']=='passed' and not engineering['pretrained_weights_loaded']
    assert engineering['resident_source_sha256']==sha('mechinterp/nanbeige_bpd_gym/resident_head.py')
    sae={}
    for backend in ('nf4','bf16'):
        sae[backend]={str(layer):sha(Path('runs/nan_gym05')/backend/f'sae_tower/layer{layer}.npz') for layer in (5,11,17,21)}
    write(target,{'experiment':'NAN-CTX-08','frozen_UTC':datetime.now(timezone.utc).isoformat(),
        'worker_files_sha256':{name:sha(root/name) for name in files},'host_only_files_sha256':{name:sha(root/name) for name in host},
        'model_dependencies_sha256':dependencies,'SAE_dictionary_sha256':sae,
        'optional_learned_asset_producer_manifest_sha256':sha('experiments/nan_fit07/SOURCE_MANIFEST.json'),
        'learned_asset_rule':'all completed final T1/T2 cells in the same backend; missing if gate/fit fails; never effect selection',
        'tokenizer_config_sha256':tokens['tokenizer_config_sha256'],'checkpoint_revision':previous['checkpoint_revision'],
        'pretrained_condition_inference_started_at_freeze':False,'formal_inferential_comparisons':0,
        'BPD_donor_transfer_enabled':False,'sealed_test_accessed':False})
    print({'manifest_sha256':sha(target),'worker_files':len(files),'model_dependencies':len(dependencies),'host_files':len(host)})


if __name__=='__main__':main()
