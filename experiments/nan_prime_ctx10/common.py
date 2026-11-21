"""Only public CTX08 data is available through the model-facing loader."""
from pathlib import Path
from experiments.nan_fit07.io import read,rows,sha,write

CTX_SEAL='7f729d76bd0b37eabbeaf4a2fd8dc0faa7621c01f8830554bbf424a63dd4d9ec'
SDK_COMMIT='ed70053b6a7a4079d6976e03bdcd7d420356d594'


def public_rows():
    root=Path('experiments/nan_ctx08');assert sha(root/'SOURCE_MANIFEST.json')==CTX_SEAL
    seal=read(root/'SOURCE_MANIFEST.json');file=root/'fixtures/public.jsonl'
    assert sha(file)==seal['worker_files_sha256']['fixtures/public.jsonl']
    public=rows(file);dev=[r for r in public if r['split']=='development']
    assert len(public)==291 and len(dev)==288 and len({r['id'] for r in dev})==288 and len({r['root'] for r in dev})==4
    return public,dev


def verify_source(root):
    seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['model_dependencies_sha256'].items():assert sha(name)==h,name
    return seal
