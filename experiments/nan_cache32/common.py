"""Explicit source/model/host separation; no numerical imports on this path."""
from pathlib import Path
from experiments.nan_ram28.common import read,write,append,sha,THREADS
from experiments.nan_full26.common import load_sources,intervention_rows,parse,rows
from experiments.nan_mlp27.common import source_group

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_cache32')


def verify(kind='model'):
    if kind not in ('model','host'):raise ValueError('Explicit verification lane required')
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    if seal.get('complete_source_freeze_before_inference') is not True:raise RuntimeError('CACHE32 is not frozen for inference')
    for name,digest in seal['critical_files_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in seal['model_dependencies_sha256'].items():
        assert 'host_labels' not in name and not name.endswith('/oracle.py'),'Host oracle in model lane'
        assert sha(name)==digest,name
    if kind=='host':
        for name,digest in seal['host_dependencies_sha256'].items():assert sha(name)==digest,name
    return seal


def verify_worker(backend):
    import os
    seal=sha(ROOT/'SOURCE_MANIFEST.json');closed=read(RUNS/'predecessor_lane_closed.json')
    assert closed['source_manifest_sha256']==seal and closed['all_authenticated_predecessors_ended']
    assert closed['known_model_host_audit_inventory_clear']
    publication=read(RUNS/'source_publication.json')
    assert publication['source_manifest_sha256']==seal and publication['remote_commit']
    runtime=read(RUNS/backend/'runtime.json')
    assert runtime['pid']==os.getpid() and runtime['backend']==backend and runtime['condition']=='NAN-CACHE-32'
    assert runtime['source_manifest_sha256']==seal
    permit=read(RUNS/(backend+'_worker_permit.json'))
    assert permit['source_manifest_sha256']==seal and permit['backend']==backend
    assert permit['all_preceding_actual_workers_ended'] is True


def assert_no_torch():
    import sys
    assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
