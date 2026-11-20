"""Repair scope collisions in the hash-bound host probe, with no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ORIGINAL = Path('experiments/nan_ranksite13/probe.py')
ORIGINAL_SHA = '4b3d2d24fbcf15066b0a85c5cd8d10916fe971ff14f1d071c671da0c872adcf2'
EXPERIMENT_SEAL = '27b3e5e636abcdf9217f88a5f19999c46b56d9723c76e6f5263c781e93239ddc'
ROOT = Path(__file__).parent
PATCHES = (
    ('from .common import read,sha,write,condition,key,verify',
     'from .common import read,sha,write,condition,key as condition_key,verify'),
    ("passes=cfg['passes'];name=key(cfg)", "passes=cfg['passes'];condition_name=condition_key(cfg)"),
    ("folder=runs/'conditions'/name;receipt=", "folder=runs/'conditions'/condition_name;receipt="),
    ("read(runs/'audit'/(name+'.json'))", "read(runs/'audit'/(condition_name+'.json'))"),
    ("write(output/(name+'_cached.json'),", "write(output/(condition_name+'_cached.json'),"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def amended_source():
    raw = ORIGINAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA
    source = raw.decode('utf-8')
    for before, after in PATCHES:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def verify_registration():
    registration = json.loads((ROOT / 'REGISTRATION.json').read_text(encoding='utf-8'))
    assert registration['original_experiment_seal_sha256'] == EXPERIMENT_SEAL
    assert registration['original_probe_sha256'] == ORIGINAL_SHA
    assert registration['amended_source_sha256'] == hashlib.sha256(amended_source().encode()).hexdigest()
    for name, expected in registration['host_files_sha256'].items():
        assert digest(ROOT / name) == expected, name
    assert digest('experiments/nan_ranksite13/SOURCE_MANIFEST.json') == EXPERIMENT_SEAL
    return registration


def run(runs, index):
    verify_registration()
    from experiments.nan_ranksite13.common import condition, key, verify, read, write
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify()
    cfg = condition(index); name = key(cfg)
    meta = runs / 'metadata' / name
    raw = (meta / 'events.jsonl').read_bytes()
    events = [json.loads(v) for v in raw[:raw.rfind(b'\n') + 1].splitlines()]
    assert any(v['event'] == 'worker_receipt_closed' for v in events)
    worker = read(meta / 'receipt.json')
    assert worker['status'] == 'completed' and worker['source_manifest_sha256'] == EXPERIMENT_SEAL
    destination = runs / 'probes' / (name + '_cached.json')
    provenance = dict(registration_sha256=digest(ROOT / 'REGISTRATION.json'),
                      original_probe_sha256=ORIGINAL_SHA, repair_sha256=digest(__file__),
                      amended_source_sha256=hashlib.sha256(amended_source().encode()).hexdigest(),
                      scope='host variable-scope repair only; all cached rows, formulas, controls and grouping unchanged')
    if destination.exists():
        prior = read(destination)
        assert prior['analysis_amendment'] == provenance and prior['status'] == 'passed'
        return {'condition': name, 'status': 'already_completed_amended_probe'}
    # A fresh private namespace preserves original relative imports and source
    # root. It does not modify the frozen module or any running worker's globals.
    namespace = {'__name__': 'experiments.nan_ranksite13.probe_host01',
                 '__package__': 'experiments.nan_ranksite13', '__file__': str(ORIGINAL.resolve())}
    exec(compile(amended_source(), str(ORIGINAL) + '[host01]', 'exec'), namespace)
    original_write = namespace['write']; buffered = []

    def capture(path, value):
        if Path(path) == destination:
            assert not buffered and value['status'] == 'passed'
            buffered.append(value)
        else:
            original_write(path, value)

    namespace['write'] = capture
    namespace['probe'](runs, index)
    assert len(buffered) == 1
    result = buffered[0]; result['analysis_amendment'] = provenance
    folder = runs / 'conditions' / name
    result['local_derivative_artifact_sha256'] = {
        file: digest(folder / file) for file in
        ('probe_local_codes.npz', 'probe_local_Haar.npz', 'probe_local_index.json')}
    assert result['checkpoint_sha256'] == read(folder / 'receipt.json')['final_checkpoint']['sha256']
    assert len(result['groups']) == 28 and sum(v['vectors'] for v in result['groups']) == 644
    assert len(result['geometry']) == (6 if cfg['module'] == 'down_proj' else 0)
    write(destination, result)
    return {'condition': name, 'status': 'completed_amended_probe', 'groups': 28, 'vectors': 644}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--condition-index', type=int, required=True)
    p.add_argument('--runs', default='runs/nan_ranksite13'); a = p.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    print(run(Path(a.runs), a.condition_index), flush=True)


if __name__ == '__main__':
    main()
