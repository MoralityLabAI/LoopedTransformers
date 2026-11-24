"""One-time source/dependency/parser/resource lock before CTX21 model calls."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from .common import ROOT, RUNS, read, rows, sha, write
from .source_audit import validate, old_graphs


def main():
    assert not RUNS.exists(), 'Source freeze must precede the registered run'
    cfg = read(ROOT/'config.json'); folder = ROOT/'fixtures'
    validate(rows(folder/'public.jsonl'),rows(folder/'host_labels.jsonl'),read(folder/'tokenizer_receipt.json'),cfg,old_graphs())
    assert read(folder/'independent_source_audit.json')['status']=='passed'
    names = set(read('experiments/nan_comp14/SOURCE_MANIFEST.json')['dependencies_sha256'])
    names.update(('experiments/nan_comp14/parser.py','experiments/nan_ctx08/receipts/resident_placement_random_v2.json',
        'mechinterp/nanbeige_bpd_gym/validate_resident_placement.py'))
    engineering = read('experiments/nan_ctx08/receipts/resident_placement_random_v2.json')
    assert engineering['status']=='passed' and engineering['physical_state_vectors_bit_exact']==40
    assert engineering['resident_source_sha256']==sha('mechinterp/nanbeige_bpd_gym/resident_head.py')
    assert engineering['validation_source_sha256']==sha('mechinterp/nanbeige_bpd_gym/validate_resident_placement.py')
    orchestration=read(ROOT/'receipts/orchestration_mock_v1.json')
    assert orchestration['status']=='passed' and not orchestration['CUDA_used'] and not orchestration['pretrained_model_loaded']
    for field,name in [('producer_sha256','engineering.py'),('worker_sha256','worker.py'),('common_sha256','common.py')]:
        assert orchestration[field]==sha(ROOT/name)
    git_bytes = {}
    for name in sorted(names):
        raw = Path(name).read_bytes(); blob = subprocess.check_output(['git','show','HEAD:'+name])
        assert raw.replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n'), 'Uncommitted dependency content: '+name
        git_bytes[name] = hashlib.sha256(blob).hexdigest()
    host = ['fixtures/host_labels.jsonl']
    critical = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl') and p.name!='SOURCE_MANIFEST.json'
        and '__pycache__' not in p.parts and str(p.relative_to(ROOT)).replace('\\','/') not in host}
    critical['.gitattributes']=sha(ROOT/'.gitattributes')
    seal = dict(experiment=cfg['experiment'],frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical,host_files_sha256={name:sha(ROOT/name) for name in host},
        validation_files_sha256={name:sha(name) for name in ('.github/scripts/verify_ctx21_checkout.py','.github/workflows/nan-ctx21-cpu.yml')},
        dependencies_sha256={name:sha(name) for name in sorted(names)},dependencies_git_blob_sha256=git_bytes,
        CI_byte_policy='Each dependency has exact observed runtime bytes and exact committed Git bytes, equal after whole-file CRLF-to-LF conversion only. Runtime always requires original exact bytes; CI accepts only those two frozen hashes.',
        checkpoint_revision=cfg['checkpoint_revision'],adapter_commit='1ac36353b36581b6a634626bcd55542e53dae456',
        prefreeze_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        exposure='Prior exposed COMP14/CTX08/fitting outcomes known. CTX21 sources and all criteria frozen before its first pretrained inference.',
        engineering_scope='Existing random22-layer width64,19-token placement/40-vector/full-head parity plus MOCK_ONLY actual-worker I/O, injected OOM and budget stops. No pretrained8K capacity proof or real CTX21 result.',
        pretrained_inference_started=False,neural_retries=0,formal_inferential_comparisons=0,changes_old_failed_gates=False,sealed_test_accessed=False)
    write(ROOT/'SOURCE_MANIFEST.json',seal)
    print(dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),critical_files=len(critical),dependencies=len(names)))


if __name__=='__main__':
    main()
