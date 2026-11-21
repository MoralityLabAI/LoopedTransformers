import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_xref19')


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


def verify(mode='runtime'):
    seal = read(ROOT / 'SOURCE_MANIFEST.json'); assert mode in ('runtime', 'git')
    for name, expected in seal['critical_files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in seal['dependencies'].items():
        assert sha(name) == expected[mode + '_sha256'], name
    return seal


def qualified(audit, passes):
    assert audit['source_manifest_sha256'] == read(ROOT / 'config.json')['FIT16_seal']
    matches = [r for r in audit['cells'] if r['passes'] == passes]
    return bool(audit['status'] == 'passed' and len(matches) == 1 and matches[0]['status'] == 'passed'
                and matches[0]['training']['optimizer_steps'] == 16)
