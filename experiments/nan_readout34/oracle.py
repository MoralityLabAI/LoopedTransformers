"""Independent rendered-text simulator and split-field grader. No generator import."""
import re


def replay_text(prompt):
    initial=dict(re.findall(r'^(E[0-9]{2}|P[0-3]) -> (E[0-9]{2}|P[0-3]|R[0-7])$',prompt,re.M))
    assert set(initial)=={f'E{i:02d}' for i in range(12)}|{f'P{i}' for i in range(4)}
    query=re.search(r'^Query: type=(actual|belief|nested_belief);target=(E[0-9]{2});observer=(P[0-3]);subject=(P[0-3]);times=([0-9,]+)$',prompt,re.M)
    assert query
    kind,target,a,b,raw_times=query.groups();times=list(map(int,raw_times.split(',')))
    assert len(times)==4 and len(set(times))==4 and times==sorted(times)
    people=sorted(n for n in initial if n.startswith('P'))
    maps={():dict(initial)}
    maps.update({(p,):dict(initial) for p in people})
    maps.update({(p,q):dict(initial) for p in people for q in people})
    selected=() if kind=='actual' else (a,) if kind=='belief' else (a,b)
    parsed=re.findall(r'^T([0-9]+): actor=(P[0-3]);parent\((E[0-9]{2}|P[0-3])\)=(E[0-9]{2}|P[0-3]|R[0-7]);viewers=([P0-3,]+)$',prompt,re.M)
    assert [int(row[0]) for row in parsed]==list(range(1,times[-1]+1))
    states={}
    for raw_time,actor,item,destination,raw_viewers in parsed:
        viewers=set(raw_viewers.split(','));assert actor in viewers and viewers<=set(people)
        for view,graph in maps.items():
            if set(view)<=viewers:graph[item]=destination
            # Validate every view, not only the selected path.
            for node in graph:
                seen=set()
                while not node.startswith('R'):
                    assert node in graph and node not in seen
                    seen.add(node);node=graph[node]
        time=int(raw_time)
        if time in times:
            route=[target]
            while not route[-1].startswith('R'):route.append(maps[selected][route[-1]])
            states[time]=route
    return dict(checkpoints=times[:3],checkpoint_rooms=[states[t][-1] for t in times[:3]],
        final_room=states[times[-1]][-1],path=states[times[-1]],single_time=times[1])


def packet(text,schema):
    if type(text) is not str or len(text)>512 or any(ord(c)>127 for c in text):return None
    fields=text.strip(' \r\n\t').split(';')
    if not fields or fields.pop()!='END':return None
    parsed=[]
    for field in fields:
        pair=field.split('=')
        if len(pair)!=2:return None
        parsed.append(pair)
    keys=[p[0] for p in parsed];kind=schema['readout'];rooms={f'R{i}' for i in range(8)}
    if kind=='final':
        if keys!=['FINAL'] or parsed[0][1] not in rooms:return None
        return dict(final_room=parsed[0][1])
    if kind=='single_checkpoint':
        if keys!=['AT'] or parsed[0][1] not in rooms:return None
        return dict(single_room=parsed[0][1])
    if kind=='checkpoint_array':
        if keys!=['CHECKS']:return None
        values=parsed[0][1].split(',')
        if len(values)!=3 or any(v not in rooms for v in values):return None
        return dict(checkpoint_rooms=values)
    if kind!='packet':raise ValueError('Unknown readout')
    if keys!=[*(f'T{t}' for t in schema['checkpoints']),'PATH','FINAL']:return None
    values=[p[1] for p in parsed];route=values[3].split('>')
    if any(v not in rooms for v in values[:3]+values[4:]):return None
    if not 2<=len(route)<=16 or any(n not in schema['nodes'] for n in route):return None
    return dict(checkpoint_rooms=values[:3],path=route,final_room=values[4])


def grade(text,schema,label):
    answer=packet(text,schema);kind=schema['readout']
    wanted=({'final_room':label['final_room']} if kind=='final' else {'single_room':label['checkpoint_rooms'][1]}
        if kind=='single_checkpoint' else {'checkpoint_rooms':label['checkpoint_rooms']} if kind=='checkpoint_array'
        else {k:label[k] for k in ('checkpoint_rooms','path','final_room')})
    fields={k:int(answer is not None and answer.get(k)==v) for k,v in wanted.items()}
    return dict(format_valid=int(answer is not None),fields_correct=fields,joint_exact=int(all(fields.values())),
        checkpoint_items_correct=[int(answer is not None and answer['checkpoint_rooms'][i]==v) for i,v in enumerate(label['checkpoint_rooms'])]
        if 'checkpoint_rooms' in wanted else None)
