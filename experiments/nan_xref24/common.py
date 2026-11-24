import hashlib
import importlib
import json
from pathlib import Path

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_xref24')
SOURCES={'xref19':'experiments.nan_xref19','xref22':'experiments.nan_xref22'}


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(mode='runtime'):
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    for name,h in seal['critical_files_sha256'].items():assert sha(ROOT/name)==h,name
    for source,description in seal['upstream'].items():
        common=importlib.import_module(SOURCES[source]+'.common')
        assert sha(common.ROOT/'SOURCE_MANIFEST.json')==description['source_manifest_sha256']
        common.verify(mode)
        assert sha(common.ROOT/'audit.py')==description['audit_'+mode+'_sha256']
    return seal
