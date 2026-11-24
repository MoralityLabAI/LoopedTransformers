import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_ram28')
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
def verify():
    manifest=read(ROOT/'SOURCE_MANIFEST.json')
    for name,digest in manifest['critical_files_sha256'].items():assert sha(ROOT/name)==digest,name
    from experiments.nan_mlp27.common import verify as inherited
    inherited()
    assert sha(Path('experiments/nan_mlp27/SOURCE_MANIFEST.json'))==manifest['method_source_manifest_sha256']
    return manifest
