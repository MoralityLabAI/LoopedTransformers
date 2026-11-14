"""FC-08 preregistered non-neural, compute-bounded Horn proof search policies.

All procedures receive ONLY graph facts, rules and goal. No labels, reference
proofs, traces or minimum proof lengths are consulted. The fixed max of 24
popped states and 5 rule actions was preregistered before test construction.
"""
from __future__ import annotations
from collections import deque
import heapq
import time
from typing import Any

BUDGET=5
MAX_POPPED=24


def _relaxed_distance(task:dict)->dict[str,int]:
    """Reverse hyperedges with conjunction relaxed to single prerequisites.

    Every rule head -> each of its premises is a directed edge in the
    relaxed reverse graph. BFS yields a lower-bound, not a proof oracle.
    """
    backwards={a:[] for a in task["atoms"]}
    for rule in task["rules"]:
        for premise in rule["premises"]:
            backwards[rule["head"]].append(premise)
    dist={task["goal"]:0}
    queue=deque([task["goal"]])
    while queue:
        node=queue.popleft()
        for incoming in backwards[node]:
            if incoming not in dist:
                dist[incoming]=dist[node]+1
                queue.append(incoming)
    return dist


def _legal(task:dict,known:frozenset[str])->list[int]:
    return [i for i,rule in enumerate(task["rules"])
            if rule["head"] not in known and
            set(rule["premises"]).issubset(known)]


def bounded_best_first(task:dict,*,budget:int=BUDGET,
                       max_popped:int=MAX_POPPED)->dict[str,Any]:
    """Best-first legal proof search with an explicit state-expansion cap.

    The priority is (g + min relaxed distance from any proved atom,
    g, insertion order). Every popped source state is counted. Search
    never invokes source gold, a reference proof or an exact solver.
    An exhausted budget leaves the last explored trace as a failed
    proposal, so the independent checker can record a fault reason.
    """
    if not 1<=budget<=5 or max_popped<1:
        raise ValueError("invalid preregistered budget")
    relaxed=_relaxed_distance(task)
    all_atoms=set(task["atoms"])
    known0=frozenset(task["facts"])
    if not known0<=all_atoms or task["goal"] not in all_atoms:
        raise ValueError("source syntax contains foreign atom")
    serial=0
    last_trace=[]
    best_trace=[]
    best_heur=float("inf")
    def h(known:frozenset[str])->int:
        return min((relaxed.get(a,len(task["atoms"])+1) for a in known),
                   default=len(task["atoms"])+1)
    heap=[(h(known0),0,serial,known0,())]
    seen={known0}
    popped=0
    started=time.perf_counter()
    while heap and popped<max_popped:
        _,g,_,known,trace=heapq.heappop(heap)
        popped+=1
        last_trace=list(trace)
        heur=h(known)
        if heur<best_heur or (heur==best_heur and len(trace)<len(best_trace)):
            best_heur=heur
            best_trace=list(trace)
        if task["goal"] in known:
            return {"trace":list(trace),"states_popped":popped,
                    "found_goal":True,
                    "symbolic_search_ms":round(1000*(time.perf_counter()-started),6)}
        if g>=budget:
            continue
        for rule_id in _legal(task,known):
            rule=task["rules"][rule_id]
            next_known=known|frozenset([rule["head"]])
            if next_known in seen:
                continue
            seen.add(next_known)
            serial+=1
            next_trace=trace+(rule_id,)
            g2=g+1
            heapq.heappush(heap,(g2+h(next_known),g2,serial,next_known,next_trace))
    return {"trace":best_trace if best_trace else last_trace,
            "states_popped":popped,"found_goal":False,
            "symbolic_search_ms":round(1000*(time.perf_counter()-started),6)}


def symbolic_trace(task:dict,method:str,backward_fn,unary_fn)->dict:
    """Execute preregistered static source-only control, with actual time."""
    start=time.perf_counter()
    if method=="bounded_best_first":
        return bounded_best_first(task)
    if method=="backward_slice":
        trace=backward_fn(task,budget=BUDGET)
    elif method=="unary_bfs":
        trace=unary_fn(task,budget=BUDGET)
    else:
        raise ValueError("unsupported symbolic policy")
    return {"trace":trace,
            "states_popped":0,
            "found_goal":None,
            "symbolic_search_ms":round(1000*(time.perf_counter()-start),6)}
