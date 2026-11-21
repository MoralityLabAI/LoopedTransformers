"""Independent textual world replay and independent whole-trace grammar."""
import re

ROOMS={f'R{i}' for i in range(8)}
PEOPLE={f'P{i}' for i in range(4)}


def trace_packet(text,schema):
    if not isinstance(text,str) or len(text)>512:
        return None
    parts=[v.strip() for v in text.strip().split(';')]
    if len(parts)!=6 or parts[-1]!='END':
        return None
    states=[]
    for raw,time in zip(parts[:3],schema['checkpoints']):
        pair=raw.split('=')
        if len(pair)!=2:
            return None
        label,value=[v.strip() for v in pair]
        if not re.fullmatch('T[0-9]+',label) or int(label[1:])!=time or value not in ROOMS:
            return None
        states.append(value)
    tail=[]
    for raw,label in zip(parts[3:5],('PATH','FINAL')):
        pair=raw.split('=')
        if len(pair)!=2 or pair[0].strip()!=label:
            return None
        tail.append(pair[1].strip())
    path=[v.strip() for v in tail[0].split('>')]
    if not 2<=len(path)<=16 or any(v not in schema['nodes'] for v in path) or tail[1] not in ROOMS:
        return None
    return dict(intermediate_rooms=states,path=path,final_room=tail[1])


def grade(text,schema,gold):
    packet=trace_packet(text,schema)
    if packet is None:
        return dict(format_valid=0,final_correct=0,intermediate_correct=[0,0,0],path_correct=0,trace_exact=0)
    intermediate=[int(a==b) for a,b in zip(packet['intermediate_rooms'],gold['intermediate_rooms'])]
    final=int(packet['final_room']==gold['final_room']);path=int(packet['path']==gold['path'])
    return dict(format_valid=1,final_correct=final,intermediate_correct=intermediate,path_correct=path,
        trace_exact=int(bool(final and path and all(intermediate))))


def replay(prompt,query_type):
    assert 'Initial parents: [not provided]' not in prompt
    initial=dict(re.findall(r'^(E[0-9]{2}|P[0-3]) -> (E[0-9]{2}|P[0-3]|R[0-7])$',prompt,re.MULTILINE))
    assert len(initial)==16 and set(initial)==PEOPLE|{f'E{i:02d}' for i in range(12)}
    question=prompt.split('\nQuestion: report ',1)[1].split('\nOutput exactly',1)[0]
    target=re.search(r'room of \[(E[0-9]{2})\]',question).group(1)
    observers=re.findall(r'\[(P[0-3])\]',question)
    assert len(observers)=={'actual':0,'belief':1,'nested_belief':2}[query_type]
    times=[int(v) for v in re.search(r'after events ([0-9]+), ([0-9]+), ([0-9]+), and ([0-9]+)\.',question).groups()]
    parsed=[]
    for time,actor,action,viewers in re.findall(r'^T([0-9]+)\. [^\[]+\[(P[0-3])\] (.+)\. Feed viewers: ([P0-3,]+)\.$',prompt,re.MULTILINE):
        targets=re.findall(r'\[(E[0-9]{2}|P[0-3]|R[0-7])\]',action)
        if action.startswith('moved to '):
            assert len(targets)==1;item=actor;destination=targets[0]
        else:
            assert (action.startswith('handed ') or action.startswith('placed ')) and len(targets)==2
            item,destination=targets
        watching=set(viewers.split(','));assert actor in watching and watching<=PEOPLE
        parsed.append(dict(time=int(time),actor=actor,item=item,destination=destination,viewers=tuple(sorted(watching))))
    assert [e['time'] for e in parsed]==list(range(1,times[-1]+1))
    # Independently indexed recorded views: empty tuple=actual, (A,)=A's view,
    # (A,B)=A's view of B's view. No simulator/generator import.
    views={():dict(initial)}
    views.update({(p,):dict(initial) for p in sorted(PEOPLE)})
    views.update({(a,b):dict(initial) for a in sorted(PEOPLE) for b in sorted(PEOPLE)})
    query=() if query_type=='actual' else tuple(observers)
    checkpoints={}

    def follow(graph):
        route=[target];seen={target}
        while route[-1] not in ROOMS:
            assert route[-1] in graph
            node=graph[route[-1]];assert node not in seen;seen.add(node);route.append(node)
            assert len(route)<=17
        return route

    for event in parsed:
        for key,view in views.items():
            if all(p in event['viewers'] for p in key):
                view[event['item']]=event['destination']
        if event['time'] in times:
            checkpoints[event['time']]=follow(views[query])
    return dict(intermediate_rooms=[checkpoints[t][-1] for t in times[:3]],path=checkpoints[times[-1]],
        final_room=checkpoints[times[-1]][-1],initial=initial,events=parsed,query_people=observers,target=target,checkpoints=times[:3])
