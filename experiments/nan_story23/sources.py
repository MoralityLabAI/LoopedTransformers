"""Materialize one outcome-blind draw and all four main online routing modes."""
import json
from .common import ROOT,read,write
from .worlds import PEOPLE,ENTITIES,ROOMS,draw,render,canonical


def materialize(cfg):
    cases=[];host=[]
    for split in ('calibration','development'):
        n=cfg['CAL_roots_per_query_type'] if split=='calibration' else cfg['DEV_roots_per_query_type']
        for qi,kind in enumerate(cfg['query_types']):
            for index in range(n):
                source=draw(cfg,split,kind,index,qi)
                for world in ('A','B'):
                    data=source['worlds'][world];count=source['event_count'];checks=data['checkpoints']
                    key=source['root']+':'+world
                    schema=dict(checkpoints=checks,nodes=list(ENTITIES+PEOPLE+ROOMS))
                    gold=dict(source_case_id=key,root=source['root'],world=world,split=split,query_type=kind,event_count=count,
                        intermediate_rooms=[data['states'][t]['room'] for t in checks],path=data['states'][count]['path'],
                        final_room=data['states'][count]['room'],canonical_trace=canonical(data['states'],checks,count))
                    cases.append(dict(source_case_id=key,root=source['root'],world=world,split=split,query_type=kind,event_count=count,
                        schema=schema,prompts={arm:render(source,world,arm) for arm in cfg['native_arms']}))
                    host.append(gold)
    public=[];labels=[]
    for mode in ('native',*cfg['additional_main_routes']):
        for case,gold in zip(cases,host):
            for arm in cfg['native_arms'] if mode=='native' else ('main',):
                row={k:v for k,v in case.items() if k!='prompts'}
                row.update(id=case['source_case_id']+':'+arm+':'+mode,arm=arm,mode=mode,prompt=case['prompts'][arm])
                public.append(row);labels.append(gold|dict(id=row['id'],arm=arm,mode=mode))
    assert len(public)==144 and len({r['root'] for r in public})==12
    assert all(a['prompt']!=b['prompt'] for a,b in zip(public,public[1:])), 'Separate legitimate identical prompt slots from immediate retry fence'
    for case in cases:
        if case['world']=='A':
            pair=next(c for c in cases if c['source_case_id']==case['source_case_id'][:-1]+'B')
            assert case['prompts']['erased']==pair['prompts']['erased']
    return public,labels


def main():
    cfg=read(ROOT/'config.json');public,host=materialize(cfg);folder=ROOT/'fixtures';folder.mkdir(exist_ok=False)
    for name,values in [('public.jsonl',public),('host_labels.jsonl',host)]:
        with (folder/name).open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in values))
    write(folder/'source_receipt.json',dict(roots=12,CAL_roots=3,DEV_roots=9,source_worlds=24,registered_trajectories=144,
        query_types=cfg['query_types'],DEV_event_counts=cfg['DEV_event_counts'],CAL_event_count=cfg['CAL_event_count'],
        source_only=True,model_outcomes_used=False,source_regeneration_after_outcomes=False,sealed_test_accessed_or_used_as_exclusion=False))
    print(dict(roots=12,source_worlds=24,trajectories=144,pretrained_model_calls=0))


if __name__=='__main__':
    main()
