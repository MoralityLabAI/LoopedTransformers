"""Full fixed schedule; no answer, model output, gold, device or score input."""
from collections import Counter
import hashlib
import json

ROUTES=('component_only','static_CAL_mean','reached_state')
STAGES=dict(nan_story23=72,nan_gym05=322,nan_comp14=240,nan_ctx08=291,nan_ctx21=624)


def cap(row):return 64 if row['stage']=='nan_story23' else 256 if row['schema']['type']=='grid' else 128


def build(public,bindings):
    assert len(public)==1549 and len({r['id'] for r in public})==1549
    assert Counter(r['stage'] for r in public)==STAGES
    assert all(r['stage']=='nan_story23' for r in public[:72])
    assert all(type(bindings[r['id']]['tokens']) is int and 0<bindings[r['id']]['tokens']+256<=9216 for r in public)
    eligible=[r for r in public if (r['stage']=='nan_story23' and r['arm']=='main') or (r['stage']=='nan_comp14' and r['arm']=='focused')]
    assert len(eligible)==72 and Counter(r['stage'] for r in eligible)==dict(nan_story23=24,nan_comp14=48)
    assert all(bindings[r['id']]['tokens']+256<=4608 for r in eligible)
    lookup={r['id']:i for i,r in enumerate(eligible)};jobs=[];index=0
    def add(lane,ordinal,indices,modes):
        members=[dict(source_id=public[i]['id'],source_index=i,mode=mode,initial_tokens=bindings[public[i]['id']]['tokens'],
                      initial_ids_sha256=bindings[public[i]['id']]['ids_sha256']) for i,mode in zip(indices,modes)]
        jobs.append(dict(execution_index=len(jobs),lane=lane,ordinal=ordinal,members=members))
    while index<len(public):
        indices=[index]
        if index+1<len(public) and max(bindings[public[i]['id']]['tokens'] for i in (index,index+1))+256<=4608:indices.append(index+1)
        add('baseline',index,indices,['native']*len(indices));index+=len(indices)
        for i in indices:
            if public[i]['id'] not in lookup:continue
            for route_index,route in enumerate(ROUTES):add('AB',72*route_index+lookup[public[i]['id']],[i,i],['native',route])
    baseline=[j for j in jobs if j['lane']=='baseline'];ab=[j for j in jobs if j['lane']=='AB']
    assert sum(len(j['members']) for j in baseline)==1549 and len(ab)==216
    assert {j['ordinal'] for j in ab}==set(range(216))
    slots=[f"{j['lane']}_{j['ordinal']:04d}_r{i}" for j in jobs for i in range(len(j['members']))]
    assert len(slots)==len(set(slots))==1981
    return dict(condition='NAN-CACHE-32',scope_baselines=1549,scope_AB_pairs=216,trajectory_slots_per_backend=1981,
        baseline_initial_batches=len(baseline),baseline_batch_sizes=dict(Counter(len(j['members']) for j in baseline)),
        full_cap_row_prefixes=sum(cap(public[m['source_index']]) for j in jobs for m in j['members']),
        sources_selected_by_answers=False,jobs=jobs)


def digest(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
