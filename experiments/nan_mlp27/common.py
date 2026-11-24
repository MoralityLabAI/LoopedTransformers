from pathlib import Path
from experiments.nan_full26.common import THREADS,read,rows,write,append,sha,load_sources,intervention_rows,parse

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_mlp27')

def verify():
    from experiments.nan_full26.common import verify as original
    original();seal=read(ROOT/'SOURCE_MANIFEST.json')
    for name,h in seal['files'].items():assert sha(name)==h,name
    return seal

def validate_packages(kind):
    from importlib.metadata import version
    for name,expected in read(ROOT/'ENVIRONMENT.json')[kind].items():
        assert version(name)==expected,'Installed numerical package changed: '+name

def source_group(row,label=None,AB_condition=None):
    label=label or {}
    arm=row.get('arm') or ('copy' if row.get('copy_control',label.get('copy_control',row['source_id'].endswith(':copy'))) else 'main')
    result=dict(arm=arm,suite=row.get('suite'),hops=label.get('hops'),context_budget=row.get('context_budget'),
        position=row.get('position'),task=row.get('task'),query_type=row.get('query_type'),event_count=row.get('event_count'),AB_condition=AB_condition)
    return {k:v for k,v in result.items() if v is not None}

def qualify(receipts):
    """Resource-only admission; never inspect correctness or generated text."""
    causes=[]
    for backend in ('nf4','bf16'):
        r=receipts.get(backend)
        if r is None:return dict(status='unrun_missing_closed_FULL26_receipt',causes=[],run_both_backends=False)
        f=r.get('failure') or {};kind=f.get('type','');message=f.get('message','')
        oom=kind=='OutOfMemoryError' and 'CUDA' in message
        admission=kind=='AssertionError' and message in (
            'Unadmitted input capacity; preserve and stop','A/B capacity unavailable; no serial replacement')
        if oom or admission:causes.append(dict(backend=backend,type=kind,message=message))
    return dict(status='qualified_capacity_failure' if causes else 'unrun_no_FULL26_capacity_failure',
        causes=causes,run_both_backends=bool(causes),task_outcomes_used=False)
