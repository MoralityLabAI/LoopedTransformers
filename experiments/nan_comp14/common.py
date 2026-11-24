"""Static inputs and write-once artifacts; no live atomic progress files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_comp14')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(v) for v in Path(path).read_bytes().splitlines() if v]


def events(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    return [json.loads(v) for v in data[:data.rfind(b'\n') + 1].splitlines()]


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def verify():
    seal = read(ROOT / 'SOURCE_MANIFEST.json')
    for name, expected in seal['critical_files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in seal['dependencies_sha256'].items():
        assert sha(name) == expected, name
    return seal
