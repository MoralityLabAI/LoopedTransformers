"""Exact source/tokenizer/native-evidence lock before any PRIME20 model call."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from .common import ROOT, RUNS, public_rows, read, sha, write, COMP14_SEAL


def main():
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', required=True); a = p.parse_args()
    assert not any((RUNS / backend).exists() for backend in ('nf4', 'bf16'))
    assert not (RUNS / 'operations').exists()
    assert sha('experiments/nan_comp14/SOURCE_MANIFEST.json') == COMP14_SEAL
    old = read('experiments/nan_comp14/SOURCE_MANIFEST.json'); _, public = public_rows()
    deps = set(old['dependencies_sha256']) | {
        'experiments/nan_comp14/SOURCE_MANIFEST.json', 'experiments/nan_comp14/common.py', 'experiments/nan_comp14/config.json',
        'experiments/nan_comp14/parser.py', 'experiments/nan_comp14/audit.py', 'experiments/nan_comp14/report.py',
        'experiments/nan_comp14/fixtures/public.jsonl', 'experiments/nan_comp14/fixtures/host_labels.jsonl', 'experiments/nan_comp14/fixtures/tokenizer_receipt.json',
        'experiments/nan_prime06/wire.py', 'experiments/nan_grad15/common.py', 'experiments/nan_grad15/config.json',
        'mechinterp/nanbeige_bpd_gym/prime_evidence.py', 'experiments/nan_recomp09/common.py', 'experiments/nan_fit07/io.py'}
    bindings = {}
    for name in sorted(deps):
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
        bindings[name] = dict(runtime_sha256=sha(name), git_sha256=hashlib.sha256(blob).hexdigest())
    bindings['nanbeige_competence_suite.py'] = dict(runtime_sha256=sha('nanbeige_competence_suite.py'),
        git_sha256=hashlib.sha256(Path('nanbeige_competence_suite.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    evidence = {}; audits = {}
    for backend in ('nf4', 'bf16'):
        auditfile = Path('runs/nan_comp14/audits') / (backend + '.json'); audit = read(auditfile)
        assert audit['status'] == 'passed' and audit['contexts_checked'] == 240 and audit['missing_contexts'] == 0 and audit['worker_status'] == 'completed'
        audits[backend] = sha(auditfile); evidence[backend] = {}
        for row in public:
            record = Path('runs/nan_comp14') / backend / f"case_{row['upstream_ordinal']:04d}.json"; value = read(record)
            assert value['id'] == row['id']; state = record.parent / value['state_file']; assert sha(state) == value['state_file_sha256']
            evidence[backend][row['id']] = dict(record_sha256=sha(record), state_sha256=sha(state))
    mock = read(ROOT / 'receipts/mock_transport_v1.json'); discovery = read(ROOT / 'receipts/linux_discovery_v1.json')
    assert mock['status'] == discovery['status'] == 'passed' and mock['official_mock_episodes'] == 6 and discovery['contexts'] == 96
    for name, expected in mock['source_sha256'].items():
        assert sha(ROOT / name) == expected, 'Mock-tested source changed: ' + name
    assert sha('nanbeige_competence_suite.py') == mock['entry_point_sha256']
    critical = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.md', '.sh') and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    critical['.gitattributes'] = sha(ROOT / '.gitattributes')
    write(ROOT / 'SOURCE_MANIFEST.json', dict(experiment='NAN-PRIME-20', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical, dependencies=bindings, native_evidence=evidence, upstream_comp14_audits=audits,
        tokenizer_config_sha256=sha(Path(a.snapshot) / 'tokenizer_config.json'), tokenizer_json_sha256=sha(Path(a.snapshot) / 'tokenizer.json'),
        exposure='COMP14 and earlier native/SDK/fit results exposed; IO18 fitting running, XREF19 queued; no source-outcome filtering.',
        adapter_commit='1ac36353b36581b6a634626bcd55542e53dae456',
        paid_API_calls=0, PRIME20_pretrained_started=False, formal_inferential_comparisons=0, sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), critical_files=len(critical), dependencies=len(bindings)), flush=True)


if __name__ == '__main__':
    main()
