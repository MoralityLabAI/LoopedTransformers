"""Refuse direct loader calls without the authenticated one-model queue lane."""
import os
from .common import ROOT,RUNS,read,sha


def verify_worker():
    closed=read(RUNS/'predecessor_lane_closed.json')
    assert closed['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    assert closed['all_authenticated_predecessors_ended'] and closed['known_model_host_audit_inventory_clear']
    runtime=read(RUNS/'bf16_runtime.json')
    assert runtime['pid']==os.getpid() and runtime['backend']=='bf16' and runtime['condition']=='NAN-BF-31'
    published=read(RUNS/'source_publication.json')
    assert published['source_manifest_sha256']==closed['source_manifest_sha256'] and published['remote_commit']
