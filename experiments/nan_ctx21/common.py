"""Static sources, append-only journals and exclusive immutable receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_ctx21')


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(v) for v in Path(path).read_bytes().splitlines() if v]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify():
    seal = read(ROOT / 'SOURCE_MANIFEST.json')
    for key in ('critical_files_sha256', 'dependencies_sha256'):
        for name, expected in seal[key].items():
            path = ROOT / name if key == 'critical_files_sha256' else Path(name)
            assert sha(path) == expected, str(path)
    return seal
