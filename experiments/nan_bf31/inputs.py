"""Closed RAM28 provenance and independent audit locations; no raw response copy."""
import json
from pathlib import Path
from .common import ROOT,RUNS,read,sha


def locations(artifact):
    own=Path(artifact);binding=read(own/'matched_inputs.json')
    expected=read(ROOT/'SOURCE_MANIFEST.json')['RAM28_source_manifest_sha256']
    assert binding['NF4_source_manifest_sha256']==expected
    nf4=Path(binding['NF4_artifact_root']);manifest=nf4/'nf4/manifest.json'
    assert sha(manifest)==binding['NF4_runtime_manifest_sha256']
    assert read(manifest)['source_manifest_sha256']==expected
    return dict(nf4=nf4/'nf4',bf16=own/'bf16'),dict(nf4=nf4/'host',bf16=own/'host')


def audit_records(artifact):
    bases,hosts=locations(artifact)
    for backend,host in hosts.items():
        journal=host/'source_groups.jsonl'
        if not journal.exists():continue
        seen=set()
        with journal.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):continue
                entry=json.loads(line);assert entry['file'] not in seen;seen.add(entry['file'])
                meta=host/entry['file'];assert sha(meta)==entry['sha256'];m=read(meta)
                file=host/m['audit_file'];assert sha(file)==m['audit_sha256'];r=read(file)
                assert r['backend']==backend
                r['source_group']=m['source_group'];yield r
