from datetime import datetime, timezone
import hashlib
import subprocess
from .common import ROOT, RUNS, read, sha, write


def main():
    assert not RUNS.exists()
    old = read('experiments/nan_comp14/SOURCE_MANIFEST.json')
    assert sha('experiments/nan_comp14/SOURCE_MANIFEST.json') == read(ROOT / 'config.json')['upstream_comp14_seal']
    dependencies = set(old['dependencies_sha256']) | {
        'mechinterp/nanbeige_bpd_gym/native_components.py', 'mechinterp/nanbeige_bpd_gym/native_saved_tensors.py',
        'mechinterp/nanbeige_bpd_gym/recomputed_native.py', 'mechinterp/nanbeige_bpd_gym/BPD_FRAMEWORK_LOCK.json',
        'experiments/nan_comp14/common.py', 'experiments/nan_comp14/report.py',
        'experiments/nan_comp14/config.json', 'experiments/nan_comp14/SOURCE_MANIFEST.json'}
    bindings = {}
    for name in sorted(dependencies):
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
        bindings[name] = dict(runtime_sha256=sha(name), git_sha256=hashlib.sha256(blob).hexdigest())
    init = read('experiments/nan_fit07/SOURCE_MANIFEST.json')['initialization']
    for backend in ('nf4', 'bf16'):
        assert sha('runs/nan_gym05/' + backend + '/linear_baselines/layer21_down_proj.npz') == init[backend]['file_sha256']
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.md') and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    write(ROOT / 'SOURCE_MANIFEST.json', dict(experiment='NAN-GRAD-15', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, dependencies=bindings, initialization=init,
        adapter_commit='1ac36353b36581b6a634626bcd55542e53dae456', BPD_commit='8f658b789138d73f497b1a2a05f5783745147d40',
        exposure='COMP14 NF4 aggregates observed. Fixed focused recipe and first two CAL roots, both worlds, before any GRAD15 forward/backward. No successful-prefix selection.',
        pretrained_GRAD15_started=False, optimizer_steps=0, formal_inferential_comparisons=0, sealed_test_accessed=False))
    print({'source_manifest_sha256': sha(ROOT / 'SOURCE_MANIFEST.json'), 'critical_files': len(critical), 'dependencies': len(bindings)})


if __name__ == '__main__':
    main()
