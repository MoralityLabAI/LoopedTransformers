"""Future-condition namespace; source verification fails closed until frozen."""
from pathlib import Path
from experiments.nan_ram28.common import read,write,append,sha,THREADS
from experiments.nan_mlp27.common import load_sources,intervention_rows,source_group,validate_packages,parse,rows

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_bf31')


def verify():
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    if not seal.get('pretrained_inference_authorized_after_predecessors_end',False):raise RuntimeError('BF31 is not frozen for inference')
    for name,digest in seal['critical_files_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in seal['additional_dependencies_sha256'].items():assert sha(name)==digest,name
    from experiments.nan_mlp27.common import verify as inherited
    inherited()
    assert sha('experiments/nan_mlp27/SOURCE_MANIFEST.json')==seal['method_source_manifest_sha256']
    return seal
