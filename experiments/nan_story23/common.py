"""Source/receipt helpers; live observations use authenticated health or journals."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_story23')
SDK_COMMIT='ed70053b6a7a4079d6976e03bdcd7d420356d594'


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(v) for v in Path(path).read_bytes().splitlines() if v]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def verify(mode='runtime'):
    seal=read(ROOT/'SOURCE_MANIFEST.json');assert mode in ('runtime','git')
    for name,expected in seal['critical_files_sha256'].items():
        assert sha(ROOT/name)==expected,name
    for name,expected in seal['dependencies'].items():
        assert sha(name)==expected[mode+'_sha256'],name
    return seal


def public_rows():
    public=rows(ROOT/'fixtures/public.jsonl')
    assert len(public)==144 and len({r['id'] for r in public})==144
    assert len({r['root'] for r in public})==12
    return public


def qualified(audit,passes):
    if audit is None:return False
    assert audit['source_manifest_sha256']==read(ROOT/'config.json')['FIT16_IO18_seal']
    cells=[c for c in audit['cells'] if c['passes']==passes]
    return bool(audit['status']=='passed' and len(cells)==1 and cells[0]['status']=='passed'
        and cells[0]['training']['optimizer_steps']==16)


def fit_asset(backend,passes):
    path=Path('runs/nan_fit16_io18/audits')/(backend+'.json')
    audit=read(path) if path.exists() else None
    if not qualified(audit,passes):return None
    folder=Path('runs/nan_fit16_io18')/backend/f't{passes}_layer21_down_proj_rank8'
    receipt=read(folder/'receipt.json');cell=next(c for c in audit['cells'] if c['passes']==passes)
    assert receipt['status']=='completed' and receipt['optimizer_steps_completed']==16
    assert sha(folder/'receipt.json')==cell['receipt_sha256']
    checkpoint=folder/receipt['final_checkpoint']['file']
    assert sha(checkpoint)==receipt['final_checkpoint']['sha256']==cell['checkpoint_sha256']
    diagnostic=folder/'diagnostic_local.json'
    assert sha(diagnostic)==receipt['diagnostic_artifact_sha256'][diagnostic.name]
    return dict(folder=str(folder),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        receipt_sha256=sha(folder/'receipt.json'),audit_sha256=sha(path),receipt=receipt,
        static_CAL_mask=read(diagnostic)['static_CAL_mask'])


def eligible_rows(backend,software=False):
    public=public_rows()
    admitted=True if software else fit_asset(backend,2) is not None
    return [r for r in public if r['mode']=='native' or admitted]
