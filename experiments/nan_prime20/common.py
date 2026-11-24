"""Public COMP14 source bindings and write-once model records; no oracle imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_prime20')
COMP14_SEAL = 'b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f'
SDK_COMMIT = 'ed70053b6a7a4079d6976e03bdcd7d420356d594'


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(v) for v in Path(path).read_bytes().splitlines() if v]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def public_rows():
    parent = Path('experiments/nan_comp14')
    assert sha(parent / 'SOURCE_MANIFEST.json') == COMP14_SEAL
    seal = read(parent / 'SOURCE_MANIFEST.json'); path = parent / 'fixtures/public.jsonl'
    assert sha(path) == seal['critical_files_sha256']['fixtures/public.jsonl']
    public = [dict(r, upstream_ordinal=i) for i, r in enumerate(rows(path)) if r['arm'] in ('focused', 'copy', 'focused_erased')]
    dev = [r for r in public if r['split'] == 'development']
    assert len(public) == 144 and len(dev) == 96 and len({r['id'] for r in dev}) == 96
    assert len({r['root'] for r in dev}) == 16
    return public, dev


def verify(mode='runtime'):
    seal = read(ROOT / 'SOURCE_MANIFEST.json'); assert mode in ('runtime', 'git')
    for name, expected in seal['critical_files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in seal['dependencies'].items():
        assert sha(name) == expected[mode + '_sha256'], name
    return seal
