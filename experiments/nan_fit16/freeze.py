"""Lock exact runtime and Git views before pretrained FIT16 execution."""
from datetime import datetime, timezone
import hashlib
import subprocess
from .common import ROOT, RUNS, read, sha, write


def main():
    assert not RUNS.exists()
    old = read('experiments/nan_grad15/SOURCE_MANIFEST.json'); cfg = read(ROOT / 'config.json')
    assert sha('experiments/nan_grad15/SOURCE_MANIFEST.json') == cfg['upstream_grad15_seal']
    assert sha('experiments/nan_comp14/SOURCE_MANIFEST.json') == cfg['upstream_comp14_seal']
    deps = set(old['dependencies']) | set(read('experiments/nan_recomp09/SOURCE_MANIFEST.json')['model_dependencies_sha256']) | {
        'experiments/nan_grad15/common.py', 'experiments/nan_grad15/config.json', 'experiments/nan_grad15/SOURCE_MANIFEST.json',
        'experiments/nan_fit07/worker.py', 'experiments/nan_fit07/pointwise.py', 'experiments/nan_fit07/io.py',
        'experiments/nan_recomp09/common.py', 'experiments/nan_recomp09/fit.py',
        'experiments/nan_comp14/audit.py', 'mechinterp/nanbeige_bpd_gym/audit_fit.py',
        'mechinterp/nanbeige_bpd_gym/audit_native.py', 'mechinterp/nanbeige_bpd_gym/audit_suite_sources.py'}
    bindings = {}
    for name in sorted(deps):
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
        bindings[name] = dict(runtime_sha256=sha(name), git_sha256=hashlib.sha256(blob).hexdigest())
    for backend in ('nf4', 'bf16'):
        assert sha('runs/nan_gym05/' + backend + '/linear_baselines/layer21_down_proj.npz') == old['initialization'][backend]['file_sha256']
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
                if p.is_file() and p.suffix in ('.py', '.json', '.jsonl', '.md') and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    write(ROOT / 'SOURCE_MANIFEST.json', dict(experiment='NAN-FIT-16', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, dependencies=bindings, initialization=old['initialization'],
        adapter_commit=old['adapter_commit'], BPD_commit=old['BPD_commit'],
        exposure='COMP14 and NF4 GRAD15 success exposed; all CAL worlds retained. Before all FIT16 pretrained execution; no successful-prefix selection.',
        pretrained_FIT16_started=False, formal_inferential_comparisons=0, sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), critical_files=len(critical), dependencies=len(bindings)), flush=True)


if __name__ == '__main__':
    main()
