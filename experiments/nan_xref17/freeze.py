from datetime import datetime, timezone
import hashlib
import subprocess
from .common import ROOT, RUNS, read, sha, write


def main():
    assert not RUNS.exists()
    old = read('experiments/nan_fit16/SOURCE_MANIFEST.json')
    assert sha('experiments/nan_fit16/SOURCE_MANIFEST.json') == read(ROOT / 'config.json')['FIT16_seal']
    deps = set(old['dependencies']) | {
        'experiments/nan_fit16/SOURCE_MANIFEST.json', 'experiments/nan_fit16/config.json',
        'mechinterp/nanbeige_bpd_gym/probe_towers.py', 'mechinterp/nanbeige_bpd_gym/crossref.py',
        'mechinterp/sae_causal_transfer/adapter.py'}
    bindings = {}
    for name in sorted(deps):
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
        bindings[name] = dict(runtime_sha256=sha(name), git_sha256=hashlib.sha256(blob).hexdigest())
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.md') and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    tower_manifests = {}
    for backend in ('nf4', 'bf16'):
        from pathlib import Path
        base = Path('runs/nan_gym05') / backend / 'sae_tower'; manifest = read(base / 'manifest.json')
        assert len(manifest['receipts']) == 4
        for receipt in manifest['receipts']:
            layer = receipt['layer_zero_based']; assert receipt['backend'] == backend and receipt['history'][-1]['step'] == 1000
            assert receipt['DEV_used_for_fitting_or_selection'] is False
            for name, field in [(f'layer{layer}.npz', 'artifact_sha256'), (f'layer{layer}_random_initial.npz', 'random_initial_sha256'), (f'layer{layer}_fit_manifest.json', 'fit_manifest_sha256')]:
                assert sha(base / name) == receipt[field]
        tower_manifests[backend] = dict(path=str(base / 'manifest.json').replace('\\', '/'), sha256=sha(base / 'manifest.json'))
    write(ROOT / 'SOURCE_MANIFEST.json', dict(experiment='NAN-XREF-17', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, dependencies=bindings, SAE_tower_manifests=tower_manifests,
        exposure='COMP14/GRAD15 and earlier caches exposed; FIT16 still running during registration. No factor-outcome or feature selection.',
        new_model_calls=0, optimizer_steps=0, formal_inferential_comparisons=0, sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), critical_files=len(critical), dependencies=len(bindings)), flush=True)


if __name__ == '__main__':
    main()
