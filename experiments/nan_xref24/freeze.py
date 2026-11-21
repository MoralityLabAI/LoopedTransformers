"""Freeze the separate audit repair; existing neural/worker artifacts untouched."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import subprocess
from .common import ROOT,RUNS,SOURCES,read,sha,write
from .repair import repaired_tree


def main():
    assert not (RUNS/'operations').exists() and not (ROOT/'SOURCE_MANIFEST.json').exists()
    upstream={}
    for key,module in SOURCES.items():
        folder=Path(module.replace('.','/'));file=folder/'audit.py'
        _,changes=repaired_tree(file.read_text(encoding='utf-8'))
        upstream[key]=dict(source_manifest_sha256=sha(folder/'SOURCE_MANIFEST.json'),
            audit_runtime_sha256=sha(file),audit_git_sha256=hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+file.as_posix()])).hexdigest(),
            changed_expressions=changes)
    inputs={};source=Path('runs/nan_xref19');events=[__import__('json').loads(v) for v in (source/'queue_events.jsonl').read_bytes().splitlines() if v]
    for backend in ('nf4','bf16'):
        for name in ('worker_terminal','audit_terminal'):
            matches=[e for e in events if e.get('backend')==backend and e['event']==name];assert len(matches)==1
        receipt=source/backend/'receipt.json';assert read(receipt)['status']=='completed'
        logfile=source/'queue_logs'/(backend+'_audit.log')
        inputs[backend]=dict(receipt_sha256=sha(receipt),original_audit_log_sha256=sha(logfile),
            original_audit_exit_code=next(e['exit_code'] for e in events if e.get('backend')==backend and e['event']=='audit_terminal'))
    critical={str(f.relative_to(ROOT)).replace('\\','/'):sha(f) for f in sorted(ROOT.rglob('*'))
        if f.is_file() and f.suffix in ('.py','.md','.json') and '__pycache__' not in f.parts}
    critical['.gitattributes']=sha(ROOT/'.gitattributes')
    write(ROOT/'SOURCE_MANIFEST.json',dict(experiment='NAN-XREF-24',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        critical_files_sha256=critical,upstream=upstream,closed_xref19_inputs=inputs,
        exposure='Original XREF19 workers and failed decoder-orientation audits exposed; preserve all original failures. XREF22 worker remains unstarted at registration preparation.',
        prospective_scope='Exactly three decoder .T expressions in unchanged independent auditors, plus distinct output receipt namespace; no worker/model/grade replay.',
        planned_formal_comparisons=0,new_model_calls=0,sealed_test_accessed=False))
    print(dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),critical_files=len(critical)))


if __name__=='__main__':main()
