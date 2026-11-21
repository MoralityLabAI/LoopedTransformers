"""Source-only finite story simulator: containment, feeds and recorded beliefs."""
import copy
import hashlib
import random

PEOPLE=tuple('P'+str(i) for i in range(4))
ENTITIES=tuple('E'+str(i).zfill(2) for i in range(12))
ROOMS=tuple('R'+str(i) for i in range(8))


def path(graph,entity):
    result=[entity]
    while result[-1] not in ROOMS:
        assert result[-1] in graph and len(result)<=len(graph),'Invalid/cyclic recorded world'
        result.append(graph[result[-1]])
        assert result[-1] not in result[:-1],'Cyclic recorded world'
    return result


def simulate(initial,events,query,checkpoints):
    actual=dict(initial);belief={p:dict(initial) for p in PEOPLE}
    nested={a:{b:dict(initial) for b in PEOPLE} for a in PEOPLE};states={}
    for event in events:
        item,destination,viewers=event['item'],event['destination'],event['viewers']
        assert event['actor'] in viewers and set(viewers)<=set(PEOPLE)
        actual[item]=destination
        for a in viewers:
            belief[a][item]=destination
            for b in viewers:
                nested[a][b][item]=destination
        for graph in [actual,*belief.values(),*(g for by in nested.values() for g in by.values())]:
            for node in graph:
                path(graph,node)
        if event['time'] in checkpoints or event['time']==len(events):
            graph=actual if query['type']=='actual' else belief[query['observer']] if query['type']=='belief' else nested[query['observer']][query['subject']]
            states[event['time']]=dict(room=path(graph,query['target'])[-1],path=path(graph,query['target']))
    return states


def draw(cfg,split,query_type,index,query_index):
    seed=int.from_bytes(hashlib.sha256(f"{cfg['source_seed']}:{split}:{query_type}:{index}".encode()).digest()[:8],'big')
    rng=random.Random(seed);people=list(PEOPLE);rng.shuffle(people);entities=list(ENTITIES);rng.shuffle(entities)
    chain=entities[:4];noise=entities[4:];holder=people[0];observer,subject=people[1:3]
    initial={p:rng.choice(ROOMS) for p in PEOPLE}
    initial.update({x:rng.choice(ROOMS+PEOPLE) for x in ENTITIES})
    for a,b in zip(chain,chain[1:]):
        initial[a]=b
    initial[chain[-1]]=holder
    count=cfg['CAL_event_count'] if split=='calibration' else cfg['DEV_event_counts'][index]
    checks=[count//4,count//2,3*count//4];handoff={t-1:people[j] for t,j in zip(checks,(1,2,0))}
    first=(2*(query_index*3+index+(5 if split=='calibration' else 0)))%8
    a,b=ROOMS[first],ROOMS[(first+1)%8];others=[r for r in ROOMS if r not in (a,b)];rng.shuffle(others);c,d=others[:2]
    events=[]
    for time in range(1,count+1):
        if time in handoff:
            event=dict(time=time,actor=holder,item=chain[-1],destination=handoff[time],viewers=list(PEOPLE),kind='handoff')
            holder=handoff[time]
        elif time==count-5:
            viewers={holder}
            if query_type in ('belief','nested_belief'):
                viewers.add(observer)
            if query_type=='nested_belief':
                viewers.add(subject)
            event=dict(time=time,actor=holder,item=holder,destination=a,viewers=sorted(viewers),kind='move')
        elif time in (count-3,count-1) and query_type!='actual':
            viewers={holder}
            if time==count-3 and query_type=='nested_belief':
                viewers.add(subject)
            event=dict(time=time,actor=holder,item=holder,destination=c if time==count-3 else d,viewers=sorted(viewers),kind='move')
        elif time in checks:
            viewers={holder,*rng.sample(list(PEOPLE),2)}
            event=dict(time=time,actor=holder,item=holder,destination=rng.choice(ROOMS),viewers=sorted(viewers),kind='move')
        else:
            movers=[p for p in PEOPLE if time<count-5 or p!=people[0]]
            item=rng.choice(movers+noise);actor=item if item in PEOPLE else rng.choice(PEOPLE)
            destination=rng.choice(ROOMS) if item in PEOPLE else rng.choice(ROOMS+PEOPLE)
            viewers={actor,*[p for p in PEOPLE if rng.random()<.45]}
            event=dict(time=time,actor=actor,item=item,destination=destination,viewers=sorted(viewers),kind='move' if item in PEOPLE else 'place')
        events.append(event)
    other=copy.deepcopy(events);other[count-6]['destination']=b
    query=dict(type=query_type,target=chain[0],observer=observer,subject=subject)
    worlds={w:dict(initial=initial,events=e,query=query,checkpoints=checks) for w,e in [('A',events),('B',other)]}
    for world,expected in [('A',a),('B',b)]:
        states=simulate(**worlds[world]);assert states[count]['room']==expected
        worlds[world]['states']=states
    return dict(root=f'story23:{split}:{query_type}:{index:02d}',split=split,query_type=query_type,event_count=count,worlds=worlds)


def canonical(states,checkpoints,count):
    fields=[f'T{t}='+states[t]['room'] for t in checkpoints]
    fields+=['PATH='+'>'.join(states[count]['path']),'FINAL='+states[count]['room'],'END']
    return ';'.join(fields)


def render(source,world,arm):
    data=source['worlds'][world];query=data['query'];count=source['event_count']
    names=dict(zip(PEOPLE,('Mira','Theo','Inez','Jonah')))
    room_names=('boathouse','observatory','archive','workshop','greenhouse','signal tower','storehouse','cabin')
    catalog='; '.join(f'{p}={names[p]}' for p in PEOPLE)+'. Rooms: '+'; '.join(f'{r}={name}' for r,name in zip(ROOMS,room_names))+'.'
    header=('During an expedition, couriers move nested cases between rooms and people. Each entity has exactly one parent: a case, a person or a room. '
        'An entity is in the room reached by repeatedly following its parent links. Moving a parent moves everything inside it. '
        'The initial parent table is common knowledge. Only the listed feed viewers see an event and who else sees it. '
        'The acting person is always a viewer. A viewer updates its own recorded parent map for that event. '
        "A's record of B's belief updates that parent link only when BOTH A and B are viewers. "
        'Unseen events leave records unchanged. Answer using these exact recorded-map rules.\n'+catalog+'\n')
    if arm=='erased':
        body='Initial parents: [not provided]\nEvent history: [not provided]\n'
    else:
        body='Initial parents (left entity is inside/carried by right entity):\n'+'\n'.join(f'{item} -> {parent}' for item,parent in sorted(data['initial'].items()))+'\nEvent history:\n'
        for event in data['events']:
            actor,item,destination=event['actor'],event['item'],event['destination']
            action=(f'{names[actor]} [{actor}] moved to [{destination}]' if event['kind']=='move' else
                f'{names[actor]} [{actor}] handed [{item}] to [{destination}]' if event['kind']=='handoff' else
                f'{names[actor]} [{actor}] placed [{item}] in [{destination}]')
            body+=f"T{event['time']:03d}. {action}. Feed viewers: {','.join(event['viewers'])}.\n"
    perspective=(f"the actual room of [{query['target']}]" if query['type']=='actual' else
        f"the room of [{query['target']}] in {names[query['observer']]}'s [{query['observer']}] own recorded belief" if query['type']=='belief' else
        f"the room of [{query['target']}] in {names[query['observer']]}'s [{query['observer']}] record of what {names[query['subject']]} [{query['subject']}] believes")
    checks=data['checkpoints'];request=('\nQuestion: report '+perspective+f' after events {checks[0]}, {checks[1]}, {checks[2]}, and {count}. '
        'Give the final parent-link path from the queried entity to its room in THAT same map.\n'
        f"Output exactly one line: T{checks[0]}=R?;T{checks[1]}=R?;T{checks[2]}=R?;PATH=entity>...>room;FINAL=R?;END\n"
        'Replace each R? with one room ID R0 through R7. Use entity/person IDs in PATH. No prose or Markdown.\n')
    if arm=='copy':
        request+='Copy control: ignore the question and copy this verified trace exactly:\n'+canonical(data['states'],checks,count)+'\n'
    return header+body+request
