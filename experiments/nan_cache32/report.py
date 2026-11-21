"""All 3962 scheduled rows, strict verification and root-aware missing bounds."""
import json
from collections import defaultdict
from pathlib import Path
from .common import ROOT,read,write,sha,load_sources,source_group
from .plan import build,digest

OUTPUTS=('reports/nanbeige_bpd_gym/NAN_CACHE32_PROGRESS.md','reports/nanbeige_bpd_gym/nan_cache32_progress.json')


def score(slot):
    if not slot.get('primary_verified'):return None
    g=slot['grade'];return int(g['joint_correct'] if slot['stage']=='nan_story23' else g['final_correct'])


def collect(plan,source,gold,items):
    byid={r['id']:r for r in source};groups=defaultdict(list);effects=defaultdict(list);pairs=defaultdict(list)
    for job in plan['jobs']:
        slots={}
        for backend in ('nf4','bf16'):
            batch=items.get((backend,job['lane'],job['ordinal']));values=[]
            for row,m in enumerate(job['members']):
                r=byid[m['source_id']];condition=('component_only','static_CAL_mean','reached_state')[job['ordinal']//72] if job['lane']=='AB' else None
                group=source_group(r,gold[r['id']],condition);slot=batch['slots'][row] if batch else dict(primary_verified=False)
                if batch:
                    assert slot['source_id']==r['id'] and slot['original_row']==row
                    assert slot['source_group']==group and slot['backend']==backend
                value=score(slot);values.append(value)
                groups[backend,r['stage'],r['split'],job['lane'],m['mode'],json.dumps(group,sort_keys=True)].append((r['root'],value,slot.get('grade') if value is not None else None))
            slots[backend]=values
            if job['lane']=='AB':
                r=byid[job['members'][0]['source_id']];route=job['members'][1]['mode'];a,b=values
                group=source_group(r,gold[r['id']],route)
                effects[backend,r['stage'],r['split'],route,json.dumps(group,sort_keys=True)].append((r['root'],a,b))
        for row,m in enumerate(job['members']):
            r=byid[m['source_id']];group=source_group(r,gold[r['id']],('component_only','static_CAL_mean','reached_state')[job['ordinal']//72] if job['lane']=='AB' else None)
            pairs[r['stage'],r['split'],job['lane'],m['mode'],json.dumps(group,sort_keys=True)].append((r['root'],slots['bf16'][row],slots['nf4'][row]))
    tables=[]
    for key,values in sorted(groups.items()):
        roots=defaultdict(list);endpoints=defaultdict(lambda:dict(correct=0,verified=0))
        for root,value,grade in values:
            roots[root].append(value)
            if grade:
                for field,result in grade.items():endpoints[field]['correct']+=int(result);endpoints[field]['verified']+=1
        known=[v for _,v,_ in values if v is not None];correct=sum(known);N=len(values);unknown=N-len(known)
        fields=['format_valid','final_correct']+(['checkpoints_correct','path_correct','joint_correct'] if key[1]=='nan_story23' else [])
        for field in fields:
            item=endpoints[field];item.update(registered=N,unknown=N-item['verified'],
                accuracy_bounds=[item['correct']/N,(item['correct']+N-item['verified'])/N])
        bounds=[(sum(x for x in vv if x is not None)/len(vv),(sum(x for x in vv if x is not None)+sum(x is None for x in vv))/len(vv)) for vv in roots.values()]
        tables.append(dict(backend=key[0],stage=key[1],split=key[2],lane=key[3],mode=key[4],source_group=json.loads(key[5]),registered=N,
            verified=len(known),correct=correct,unknown=unknown,world_accuracy_bounds=[correct/N,(correct+unknown)/N],independent_roots=len(roots),
            root_macro_accuracy_bounds=[sum(b[0] for b in bounds)/len(bounds),sum(b[1] for b in bounds)/len(bounds)],endpoint_counts=dict(endpoints)))
    def differences(mapping,AB):
        result=[]
        for key,values in sorted(mapping.items()):
            roots=defaultdict(list);known=[]
            for root,a,b in values:
                roots[root].append(((b if b is not None else 0)-(a if a is not None else 1),(b if b is not None else 1)-(a if a is not None else 0)))
                if a is not None and b is not None:known.append((a,b))
            bounds=[(sum(v[0] for v in vv)/len(vv),sum(v[1] for v in vv)/len(vv)) for vv in roots.values()]
            entry=dict(registered_pairs=len(values),verified_pairs=len(known),unknown_pairs=len(values)-len(known),independent_roots=len(roots),
                A_correct=sum(a for a,b in known),B_correct=sum(b for a,b in known),gains=sum(b>a for a,b in known),losses=sum(b<a for a,b in known),
                root_macro_difference_bounds=[sum(v[0] for v in bounds)/len(bounds),sum(v[1] for v in bounds)/len(bounds)])
            if AB:entry.update(backend=key[0],stage=key[1],split=key[2],route=key[3],source_group=json.loads(key[4]),difference='B_edit_minus_A_native')
            else:entry.update(stage=key[0],split=key[1],lane=key[2],mode=key[3],source_group=json.loads(key[4]),difference='NF4_minus_BF16')
            result.append(entry)
        return result
    assert sum(r['registered'] for r in tables)==3962
    return dict(accuracy=tables,paired_precision_accuracy=differences(pairs,False),AB_behavior=differences(effects,True))


def run(artifact):
    from experiments.nan_full26.host import labels
    from .paired import batches
    artifact=Path(artifact);source,_=load_sources();plan=build(source,read('experiments/nan_full26/tokenizer_receipt.json')['contexts'])
    items=batches(artifact);data=collect(plan,source,labels(),items)
    data.update(experiment='NAN-CACHE-32',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),plan_sha256=digest(plan),
        registered_rows=3962,baseline_registered_per_backend=1549,AB_pairs_registered_per_backend=216,
        full_scope_retained=True,geometry=read(artifact/'host/paired_geometry.json') if (artifact/'host/paired_geometry.json').exists() else dict(status='unrun_or_incomplete_host'),
        model_terminal=read(artifact/'model_lane_terminal.json') if (artifact/'model_lane_terminal.json').exists() else None,
        host_terminal=read(artifact/'host_lane_terminal.json') if (artifact/'host_lane_terminal.json').exists() else None,
        evidence_class='exposed_development_single_checkpoint',verdict='REDESIGN_OR_PILOT_ONLY',formal_comparisons=0,sealed_test_accessed=False)
    Path(OUTPUTS[1]).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# NAN-CACHE-32: full matched cached development gym','',
        'All 1,549 baselines and 216 genuine B2 A/B pairs per precision remain. Fixed source order interleaves each eligible source with its three fresh A/B jobs. Two native phase caches, chunk128 prefill, streamed full-vocabulary BF16 head, exact cache ancestry and independent byte/numerical audits. This is a distinct execution/schedule condition; legacy frozen outcomes are unchanged.','',
        'Only normal target/stage/source integrity plus complete token/state/numerical and batch-ancestry verification qualify a primary grade. Missing/failed/unverified slots stay unknown. No orphan rescue, reply retry, refitting or donor transfer. Story joint/final/checkpoint/path/format endpoints remain separate. Roots are evidence units; tokens, worlds, loops, routes and features repeat. Formal comparisons0, REDESIGN_OR_PILOT_ONLY; meaningful causal/sham floors and power uncalibrated. Original168 test sealed; continuous goal active.','',
        '|Backend|Stage|Split|Lane|Mode|Stratum|Registered|Verified|Correct|Unknown|Roots|Accuracy bounds|',
        '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in data['accuracy']:
        lines.append('|'+ '|'.join(str(r[k]) for k in ('backend','stage','split','lane','mode'))+'|'+json.dumps(r['source_group'],sort_keys=True)+'|'+'|'.join(str(r[k]) for k in ('registered','verified','correct','unknown','independent_roots'))+'|'+str(r['world_accuracy_bounds'])+'|')
    lines+=['','All paired precision and A/B difference bounds, all five story endpoints and root-level geometry are in the complete aggregate. Generated traces do not establish faithful reasoning, mediation, necessity or numerical equivalence.','',
        '[Complete aggregate](nan_cache32_progress.json) · [Protocol](../../experiments/nan_cache32/PROTOCOL.md)']
    Path(OUTPUTS[0]).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
