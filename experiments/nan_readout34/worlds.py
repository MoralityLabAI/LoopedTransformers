"""Outcome-blind independent roots with two context lengths and one-event twins."""
import copy
import hashlib
import random

PEOPLE=tuple(f'P{i}' for i in range(4))
ENTITIES=tuple(f'E{i:02d}' for i in range(12))
ROOMS=tuple(f'R{i}' for i in range(8))
NODES=ENTITIES+PEOPLE+ROOMS


def follow(graph,target):
    result=[target]
    while result[-1] not in ROOMS:
        assert result[-1] in graph and len(result)<=16
        result.append(graph[result[-1]])
        assert result[-1] not in result[:-1]
    return result


def replay(initial,events,query,checks):
    # Generator oracle uses only the queried map. The independent auditor
    # reconstructs all 21 views from rendered text, without this module.
    graph=dict(initial);states={}
    required=() if query['type']=='actual' else (query['observer'],) if query['type']=='belief' else (query['observer'],query['subject'])
    for event in events:
        assert event['actor'] in event['viewers']
        if all(p in event['viewers'] for p in required):
            graph[event['item']]=event['destination']
        if event['time'] in (*checks,len(events)):
            route=follow(graph,query['target'])
            states[event['time']]=dict(room=route[-1],path=route)
    return states


def draw(cfg,split,kind,index,count):
    # Context length is deliberately excluded from the root seed. Initial
    # layout, labels and target role are repeated measures of the same root.
    key=f"{cfg['source_seed']}:{split}:{kind}:{index}"
    rng=random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],'big'))
    people=list(PEOPLE);entities=list(ENTITIES);rooms=list(ROOMS)
    rng.shuffle(people);rng.shuffle(entities);rng.shuffle(rooms)
    holder,observer,subject=people[:3];chain=entities[:4];noise=entities[4:]
    initial={p:rng.choice(ROOMS) for p in PEOPLE}
    initial.update({e:rng.choice(ROOMS+PEOPLE) for e in ENTITIES})
    for left,right in zip(chain,chain[1:]):initial[left]=right
    initial[chain[-1]]=holder
    a,b,c,d=rooms[:4];events=[]
    for time in range(1,count+1):
        special=time-(count-5)
        if special in (0,2,4):
            if special==0:
                destination=a if kind=='nested_belief' else c;viewers=list(PEOPLE)
            elif special==2:
                destination=a if kind=='belief' else c if kind=='nested_belief' else d
                viewers=[holder,observer] if kind=='belief' else [holder,subject]
            else:
                destination=a if kind=='actual' else d
                viewers=list(PEOPLE) if kind=='actual' else [holder,observer] if kind=='nested_belief' else [holder]
            actor=item=holder
        else:
            candidates=list(PEOPLE if time<count-5 else tuple(p for p in PEOPLE if p!=holder))+noise
            item=rng.choice(candidates);actor=item if item in PEOPLE else rng.choice(PEOPLE)
            destination=rng.choice(ROOMS) if item in PEOPLE else rng.choice(ROOMS+PEOPLE)
            viewers=sorted({actor,*[p for p in PEOPLE if rng.random()<.45]})
        events.append(dict(time=time,actor=actor,item=item,destination=destination,viewers=sorted(viewers)))
    twin=copy.deepcopy(events)
    changed=count-1 if kind=='actual' else count-3 if kind=='belief' else count-5
    twin[changed-1]['destination']=b
    query=dict(type=kind,target=chain[0],observer=observer,subject=subject)
    checks=[count//4,count//2,3*count//4];worlds={}
    for name,evs,expected in [('A',events,a),('B',twin,b)]:
        states=replay(initial,evs,query,checks)
        assert states[count]['room']==expected
        worlds[name]=dict(initial=initial,events=evs,query=query,checks=checks,states=states)
    assert sum(x!=y for x,y in zip(events,twin))==1
    return dict(root=f'readout34:{split}:{kind}:{index:02d}',split=split,query_type=kind,event_count=count,worlds=worlds)


def gold(data,count):
    checks=data['checks'];states=data['states']
    return dict(checkpoints=checks,checkpoint_rooms=[states[t]['room'] for t in checks],
        final_room=states[count]['room'],path=states[count]['path'],single_time=checks[1])


def canonical(label,readout):
    if readout=='final':return f"FINAL={label['final_room']};END"
    if readout=='single_checkpoint':return f"AT={label['checkpoint_rooms'][1]};END"
    if readout=='checkpoint_array':return 'CHECKS='+','.join(label['checkpoint_rooms'])+';END'
    assert readout=='packet'
    return ';'.join([*(f'T{t}={room}' for t,room in zip(label['checkpoints'],label['checkpoint_rooms'])),
        'PATH='+'>'.join(label['path']),'FINAL='+label['final_room'],'END'])


def render(source,world,readout,arm):
    data=source['worlds'][world];q=data['query'];count=source['event_count'];checks=data['checks']
    text=('Finite courier record task. Each person or entity has one parent. Follow parent links until a room to find its room. '
        'Moving a parent moves its contents. Initial parents are common knowledge. '
        'Only listed viewers see an event and who else sees it; its actor is always a viewer. '
        'A viewer updates that parent link in its own recorded map. '
        "A's record of B's map updates that link only when BOTH A and B are viewers. Unseen events leave records unchanged. "
        'Use these recorded-map rules, not assumptions about human beliefs. '
        'Room IDs are R0 through R7; person IDs P0 through P3; entity IDs E00 through E11.\n')
    # Examples are fixed, source independent and demonstrate each answer form.
    examples={
        'final':'If the requested final room is R3, answer FINAL=R3;END.',
        'single_checkpoint':'If the requested checkpoint room is R4, answer AT=R4;END.',
        'checkpoint_array':'If the three requested checkpoint rooms in order are R1, R2, R4, answer CHECKS=R1,R2,R4;END.',
        'packet':f'Example of the form only: T{checks[0]}=R1;T{checks[1]}=R2;T{checks[2]}=R4;PATH=E00>P0>R3;FINAL=R3;END. Replace every example value with this case\'s value.'}
    text+='Answer-format example: '+examples[readout]+'\n'
    if arm=='erased':text+='Initial parents: [not provided]\nEvents: [not provided]\n'
    else:
        text+='Initial parents:\n'+''.join(f'{a} -> {b}\n' for a,b in sorted(data['initial'].items()))+'Events:\n'
        for e in data['events']:
            text+=f"T{e['time']:03d}: actor={e['actor']};parent({e['item']})={e['destination']};viewers={','.join(e['viewers'])}\n"
    perspective=(f"actual" if q['type']=='actual' else f"{q['observer']}'s own recorded" if q['type']=='belief' else f"{q['observer']}'s recorded view of {q['subject']}'s")
    text+=f"Query: type={q['type']};target={q['target']};observer={q['observer']};subject={q['subject']};times={','.join(map(str,checks+[count]))}\n"
    text+=f"Use the {perspective} map for {q['target']}. "
    requests={
        'final':f'Give only the room after event {count}. Format FINAL=R?;END.',
        'single_checkpoint':f'Give only the room after event {checks[1]}. Format AT=R?;END.',
        'checkpoint_array':f"Give only the rooms after events {checks[0]}, {checks[1]}, {checks[2]}, in that order. Format CHECKS=R?,R?,R?;END.",
        'packet':f"Give rooms after events {checks[0]}, {checks[1]}, {checks[2]}, then the final parent path and room after event {count}. Format T{checks[0]}=R?;T{checks[1]}=R?;T{checks[2]}=R?;PATH=entity>...>room;FINAL=R?;END."}
    text+=requests[readout]+' No prose, Markdown or extra fields. Output exactly one line.\n'
    if arm=='copy':text+='Copy control: the complete verified answer is supplied below. Copy it exactly; do not recompute:\n'+canonical(gold(data,count),readout)+'\n'
    return text
