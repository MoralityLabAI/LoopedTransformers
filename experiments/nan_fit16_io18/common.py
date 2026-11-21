"""Write-once receipts, exact runtime/Git byte views and upstream eligibility."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = Path('runs/nan_fit16_io18')


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(v) for v in Path(path).read_bytes().splitlines() if v]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_digest(loop):
    from experiments.nan_recomp09.common import target_digest as bounded_hash
    return bounded_hash(loop)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(mode='runtime'):
    assert mode in ('runtime', 'git')
    seal = read(ROOT / 'SOURCE_MANIFEST.json')
    for name, expected in seal['critical_files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in seal['dependencies'].items():
        assert sha(name) == expected[mode + '_sha256'], name
    return seal


def eligible(comp14_audits, grad15_audits):
    from experiments.nan_grad15.common import eligible as competence
    decision = competence(comp14_audits)
    checks = {}
    for b in ('nf4', 'bf16'):
        a = grad15_audits[b]
        assert a['source_manifest_sha256'] == read(ROOT / 'config.json')['upstream_grad15_seal']
        conditions = {v['passes']: v for v in a['conditions']}
        checks[b] = (a['status'] == 'passed' and a['worker_status'] == 'completed'
            and a['CAL_prefixes_verified'] == 4 and a['optimizer_steps'] == 0
            and set(conditions) == {1, 2} and all(v['condition_status'] == 'passed'
                and v['successful_gradient_cases'] == v['archived_cases'] == 4
                and v['missing_registered_cases'] == 0 for v in conditions.values()))
    return dict(passed=decision['passed'] and all(checks.values()), competence=decision,
        gradient_prerequisites=checks,
        rule='Both unchanged focused first-token screens and all eight GRAD15 cases per backend; no bounded/legacy substitution.')
