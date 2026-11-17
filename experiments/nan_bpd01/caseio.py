"""Nonmutating view of immutable initial tapes and explicit repaired controls."""
import json
from pathlib import Path
from .head import prefix_logprob_spread

def load_case(runs,backend,case_id):
    runs=Path(runs);name=case_id.replace(':','_')+'.json'
    record=json.loads((runs/backend/name).read_text())
    repaired=runs/'prefix_repair'/backend/name
    if repaired.exists():
        status=json.loads((repaired.parent/'receipt.json').read_text())
        if status['status']!='completed':raise RuntimeError('Prefix repair is incomplete')
        extra=json.loads(repaired.read_text())
        if extra['case_id']!=case_id:raise ValueError('Repair case binding mismatch')
        record['interventions'].update(extra['interventions'])
    for name,arm in record['interventions'].items():
        spread=prefix_logprob_spread(arm['delayed'])
        arm['prefix_logprob_spread']=spread;arm['canonical_prefix_consistent']=spread==0
        arm['source_stage']='post_readout_prefix_repair' if name.startswith('random_shared_dark_') else 'initial'
    return record
