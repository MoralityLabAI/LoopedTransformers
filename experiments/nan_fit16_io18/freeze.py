from datetime import datetime, timezone
import hashlib
import subprocess
from .common import ROOT, RUNS, read, sha, write


def main():
    assert not RUNS.exists()
    assert sha('experiments/nan_fit16/SOURCE_MANIFEST.json') == '35371bbb5323be88ae4eca4cb43677c2f583494b8a46a5f7b7aa89c2fc710139'
    parent = read('experiments/nan_fit16/SOURCE_MANIFEST.json')
    dependencies = set(parent['dependencies']) | {'experiments/nan_fit16/' + name for name in
        ('SOURCE_MANIFEST.json', 'fit.py', 'worker.py', 'common.py', 'config.json', 'audit.py', 'report.py')}
    bindings = {}
    for name in sorted(dependencies):
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
        bindings[name] = dict(runtime_sha256=sha(name), git_sha256=hashlib.sha256(blob).hexdigest())
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.jsonl', '.md') and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    write(ROOT / 'SOURCE_MANIFEST.json', dict(experiment='NAN-FIT16-IO18', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, dependencies=bindings, initialization=parent['initialization'], adapter_commit=parent['adapter_commit'], BPD_commit=parent['BPD_commit'],
        parent_source_seal='35371bbb5323be88ae4eca4cb43677c2f583494b8a46a5f7b7aa89c2fc710139',
        supervision_commit='9689206dc4c9c390e189cdd7ffd19bc735ceb9a2',
        supervision_action_sha256=sha('runs/nan_fit16_io18_supervision_action.json'),
        exposure='Original NF4 T1 partial failure exposed, T2 naturally running. Only repeated step writer repaired; no numerical/criterion changes or warm starts.',
        pretrained_IO18_started=False, formal_inferential_comparisons=0, sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), critical_files=len(critical), dependencies=len(bindings)), flush=True)


if __name__ == '__main__':
    main()
