"""Source locks distinguish exact runtime bytes from exact Git checkout bytes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_grad15')


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(v) for v in Path(path).read_bytes().splitlines() if v]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def verify(mode='runtime'):
    assert mode in ('runtime', 'git')
    seal = read(ROOT / 'SOURCE_MANIFEST.json')
    for name, expected in seal['critical_files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in seal['dependencies'].items():
        assert sha(name) == expected[mode + '_sha256'], name
    return seal


def eligible(audits):
    from experiments.nan_comp14.report import grade_groups, gates
    cfg = read('experiments/nan_comp14/config.json'); result = {}
    for backend in ('nf4', 'bf16'):
        a = audits[backend]
        assert a['status'] == 'passed' and a['source_manifest_sha256'] == read(ROOT / 'config.json')['upstream_comp14_seal']
        assert a['contexts_checked'] == 240 and a['missing_contexts'] == 0
        chosen = next(v for v in gates(a, grade_groups(a['records']), cfg)
                      if v['template'] == 'focused' and v['endpoint'] == 'first_token_loop2')
        result[backend] = chosen
    return dict(passed=all(v['passed'] for v in result.values()), screens=result,
                rule='Both backends must pass the original COMP14 focused first-token screen; no substitutions or changed thresholds.')
