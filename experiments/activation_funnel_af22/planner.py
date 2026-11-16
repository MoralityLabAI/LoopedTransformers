"""Host-transition Bellman backup with a neural boundary value function.

At each visited host state, enumerate exactly two legal Horn rule-fire layers.
V_0(s) is supplied by a shared, trained neural distance/reachability head.
V_1(s) = min(C_stop(s), min_{legal a} (1+V_0(T(s,a)))).
The greedy two-step policy uses Q_2(s,a) = 1+V_1(T(s,a)).
The one-step control uses Q_1(s,a) = 1+V_0(T(s,a)); the direct
control uses raw rule logits. STOP is an actual policy action in all modes.

Terminal GOAL states have exact public boundary V=0 and stop cost=0.
Unexpanded interior states receive only neural boundary values, not BFS truth.
"""
from __future__ import annotations
from collections import deque
import math
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym as base
from . import gym

MODES=('direct','vi1','vi2')
MAX_DEPTH=2
CAP=16.


def state_graph(c,s):
    root=frozenset(s);depth={root:0}; todo=deque([root]);edges={}
    while todo:
        state=todo.popleft();level=depth[state]
        edges[state]={}
        if base.GOAL in state or level==MAX_DEPTH:continue
        for a in gym.enabled(c,state):
            next_state,proof=gym.sound_fire(c,state,a)
            if proof is None:raise AssertionError('non-STOP transition must have host proof')
            edges[state][a]=next_state
            if next_state not in depth:
                depth[next_state]=level+1;todo.append(next_state)
    if any(depth[u]>MAX_DEPTH for u in depth):raise AssertionError('planner overflow')
    return tuple(depth.keys()),edges


def neural_values(dist,reach,has_goal):
    """Public terminal indicator and neural values; source BFS distances absent."""
    p=torch.sigmoid(reach)
    finite=torch.clamp(F.softplus(dist),min=0.,max=CAP)
    v0=p*finite+(1-p)*CAP
    stopcost=p*CAP
    if has_goal.dtype!=torch.bool:raise TypeError('goal boundary must be boolean')
    z=torch.zeros_like(v0)
    return torch.where(has_goal,z,v0),torch.where(has_goal,z,stopcost)


def backup(values,stop_cost,edges):
    """One synchronous min-plus Bellman update using host-verified transitions."""
    return {s:min(stop_cost[s],min((1.+values[t] for t in nexts.values()),default=float('inf')))
            for s,nexts in edges.items()}


def score_from_values(c,root,edges,v0,stop_cost,raw):
    """Calculate all three greedy action vectors using the same neural states."""
    a=edges[root];score={mode:[-1e4]*(gym.STOP+1) for mode in MODES}
    for i in gym.enabled(c,root):
        if i not in a:raise AssertionError('host legal-rule edge missing')
    for i in a:
        score['direct'][i]=float(raw[i])
        score['vi1'][i]=-(1.+v0[a[i]])
    b1=backup(v0,stop_cost,edges)
    for i in a:score['vi2'][i]=-(1.+b1[a[i]])
    score['direct'][gym.STOP]=float(raw[gym.STOP])
    score['vi1'][gym.STOP]=-stop_cost[root]
    score['vi2'][gym.STOP]=-stop_cost[root]
    chosen={m:max(range(len(v)),key=lambda i:v[i]) for m,v in score.items()}
    if not a:chosen={m:gym.STOP for m in MODES}
    trace={'nodes':[{'state':sorted(s),'value0':v0[s], 'stop_cost':stop_cost[s],
              'edges':{str(i):sorted(t) for i,t in edges[s].items()}}
             for s in edges], 'root_raw':list(map(float,raw))}
    return score,chosen,trace


def decisions(model,cases,states,batch_size=128):
    """Batch all neighborhood nodes; share identical computed values across modes.

    Every entry contains only model outputs and host-defined successor edges.
    The original source BFS oracle is never invoked on this path.
    """
    if len(cases)!=len(states) or not cases:raise ValueError('nonempty equal input rosters required')
    all_cases=[];all_states=[];records=[]
    for c,s in zip(cases,states):
        nodes,edges=state_graph(c,s);begin=len(all_cases)
        all_cases.extend([c]*len(nodes));all_states.extend(nodes)
        records.append((nodes,edges,begin))
    logits=[];distances=[];reaches=[]
    model.eval()
    with torch.no_grad():
        for off in range(0,len(all_cases),batch_size):
            x,known,_=gym.observations(all_cases[off:off+batch_size],all_states[off:off+batch_size])
            raw,d,reach=model.full(x,known)
            logits.extend(raw.tolist());distances.extend(d.tolist());reaches.extend(reach.tolist())
    result=[]
    for c,s,(nodes,edges,begin) in zip(cases,states,records):
        root=frozenset(s);v0={};stop={}
        for i,node in enumerate(nodes):
            idx=begin+i
            d=torch.tensor([distances[idx]],dtype=torch.float32)
            r=torch.tensor([reaches[idx]],dtype=torch.float32)
            b=torch.tensor([base.GOAL in node],dtype=torch.bool)
            value,price=neural_values(d,r,b)
            v0[node]=float(value.item());stop[node]=float(price.item())
        scores,selected,trace=score_from_values(c,root,edges,v0,stop,logits[begin])
        for i,row in enumerate(trace['nodes']):
            row['distance_logit']=float(distances[begin+i])
            row['reach_logit']=float(reaches[begin+i])
        trace['nodes_forward_count']=len(nodes)
        trace['terminal_boundary']='host-goal-membership-only'
        result.append({'scores':scores,'selected':selected,'trace':trace})
    return result


def check_trace_math(c,s,trace,scores):
    """Independent arithmetic reconstruction from persisted public graph and V0."""
    nodes,edges=state_graph(c,s)
    saved=trace['nodes']
    if len(nodes)!=len(saved) or len(nodes)!=trace['nodes_forward_count']:
        raise ValueError('Bellman node roster altered')
    v0={};stop={}
    for state,row in zip(nodes,saved):
        if row['state']!=sorted(state) or row['edges']!={str(i):sorted(t) for i,t in edges[state].items()}:
            raise ValueError('Bellman host successor trace altered')
        if not all(math.isfinite(float(row[k])) and 0<=row[k]<=CAP+1e-4 for k in ('value0','stop_cost')):
            raise ValueError('invalid Bellman cost')
        if base.GOAL in state and (row['value0'] or row['stop_cost']):
            raise ValueError('terminal goal boundary altered')
        dist=torch.tensor([row['distance_logit']],dtype=torch.float32)
        reach=torch.tensor([row['reach_logit']],dtype=torch.float32)
        expected_value,expected_stop=neural_values(dist,reach,torch.tensor([base.GOAL in state]))
        if abs(float(expected_value.item())-row['value0'])>1e-5 or \
           abs(float(expected_stop.item())-row['stop_cost'])>1e-5:
            raise ValueError('Bellman neural boundary arithmetic altered')
        v0[state]=float(row['value0']);stop[state]=float(row['stop_cost'])
    expected,selected,_=score_from_values(c,frozenset(s),edges,v0,stop,trace['root_raw'])
    for mode in scores:
        if mode not in MODES:raise ValueError('unknown scored Bellman mode')
        if len(scores[mode])!=gym.STOP+1 or max(abs(a-b) for a,b in zip(expected[mode],scores[mode]))>1e-5:
            raise ValueError('Bellman action score mismatch')
    return selected
