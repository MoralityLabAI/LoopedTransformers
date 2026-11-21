from datetime import datetime, timezone
import hashlib
import subprocess
from .common import ROOT, RUNS, read, sha, write


def main():
    assert not RUNS.exists()
    old = read('experiments/nan_xref17/SOURCE_MANIFEST.json')
    assert sha('experiments/nan_xref17/SOURCE_MANIFEST.json') == 'a901c6a1389b85508a742e3eb4db7e9e0d21ae4b64a44bd6b1340e1b742712fd'
    assert sha('experiments/nan_fit16_io18/SOURCE_MANIFEST.json') == read(ROOT / 'config.json')['FIT16_seal']
    dependencies = set(old['dependencies']) | {'experiments/nan_xref17/' + name for name in
        ('SOURCE_MANIFEST.json', 'worker.py', 'cache.py', 'audit.py', 'config.json', 'CACHE_INVENTORY.json')} | {
        'experiments/nan_fit16_io18/SOURCE_MANIFEST.json', 'experiments/nan_fit16_io18/config.json'}
    bindings = {}
    for name in sorted(dependencies):
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
        bindings[name] = dict(runtime_sha256=sha(name), git_sha256=hashlib.sha256(blob).hexdigest())
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.md') and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    write(ROOT / 'SOURCE_MANIFEST.json', dict(experiment='NAN-XREF-19', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, dependencies=bindings, SAE_tower_manifests=old['SAE_tower_manifests'],
        exposure='Original FIT16 T1/T2 writer failures after two updates exposed. IO18 queued, no replacement checkpoints analyzed. No numerical analysis changes.',
        parent_XREF17_seal='a901c6a1389b85508a742e3eb4db7e9e0d21ae4b64a44bd6b1340e1b742712fd',
        new_model_calls=0, optimizer_steps=0, formal_inferential_comparisons=0, sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), critical_files=len(critical), dependencies=len(bindings)), flush=True)


if __name__ == '__main__':
    main()
