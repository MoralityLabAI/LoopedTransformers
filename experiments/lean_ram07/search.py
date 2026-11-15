"""RAM-05 matched native/recurrent proof search with full Lean axiom-gated release."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time
import ram07_core as c
from runtime import Session,ProbeTimeout,certify,compile_source

@dataclass
class Node:
    actions:list
    state:dict
    ordered:list|None=None
    index:int=0

def search(lean,original,theorem,ranker,out,guard,budget,*,clock=time.perf_counter,session_type=Session,certificate=certify,replayer=compile_source):
    started=clock();end=started+budget['seconds'];active_end=end-.5
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    calls=[];stack=[];visited=set();processes=0;restarts=0;server=None
    accepted=False;reason='no_progress';chosen=[];final=None;replay=None;stats=Counter();root_signature=None
    def open_server():
        nonlocal processes
        processes+=1
        return session_type(lean,original,theorem,out/f'server{processes:02d}',guard,min(10.,max(.01,active_end-clock())))
    try:
        server=open_server();root_signature=c.signature(server.initial['goals'])
        stack=[Node([],server.initial)];visited.add(root_signature)
        while stack:
            guard.check()
            if clock()>=active_end:reason='deadline';break
            if len(calls)>=budget['evaluations']:reason='evaluation_budget';break
            node=stack[-1]
            if len(node.actions)>=budget['transitions']:
                stats['transition_budget']+=1;reason='transition_budget';stack.pop();continue
            if node.ordered is None:
                pool=c.candidates(node.state)
                node.ordered=ranker.rank(node.state,pool) if ranker is not None else pool
                if Counter(node.ordered)!=Counter(pool):raise AssertionError('ranker changed registered grammar')
            if node.index>=len(node.ordered):stack.pop();continue
            family,script=node.ordered[node.index];node.index+=1;actions=node.actions+[script]
            before=len(node.state['goals'])
            try:
                result=server.query(actions,min(budget['request_seconds'],max(.01,active_end-clock())))
            except ProbeTimeout:
                calls.append(dict(family=family,script=script,actions=actions,accepted=False,observation_unknown=True,depth=len(actions)))
                stats['request_timeouts']+=1
                if restarts>=2 or clock()>=active_end:reason='request_timeout';break
                restarts+=1;server=open_server()
                if c.signature(server.initial['goals'])!=root_signature:raise AssertionError('server restart changed original root')
                continue
            result.update(family=family,script=script,actions=actions,depth=len(actions),
              before_signature=c.signature(node.state['goals']),before_goals=before)
            calls.append(result)
            if clock()>=active_end:reason='deadline';break
            outcome=c.label(node.state,result);result['outcome']=outcome
            if outcome==0:stats['failed_or_unchanged']+=1;continue
            goals=result['goals'];sig=c.signature(goals)
            result.update(after_signature=sig,after_goals=len(goals),creates_branches=len(goals)>before,
                          closes_goals=len(goals)<before)
            if not goals:
                chosen=actions;server.close()
                if hasattr(server,'ensure_clean'):server.ensure_clean()
                server=None
                if clock()>=active_end:reason='verification_budget';break
                processes+=1
                final=certificate(lean,original,theorem,actions,out/'certificate',guard,min(10.,active_end-clock()))
                if not final['accepted'] or clock()>=active_end:reason='final_rejected_or_budget';break
                processes+=1
                replay=replayer(lean,Path(final['source_file']).read_text(encoding='utf8'),out/'independent_replay',guard,min(10.,active_end-clock()))
                accepted=bool(replay['accepted'] and replay['source_sha256']==final['source_sha256'] and clock()<=end)
                reason='independently_certified' if accepted else 'replay_rejected_or_budget'
                break
            if len(goals)>budget['max_goals']:
                stats['goal_cap']+=1;continue
            if sig in visited:stats['deduplicated']+=1;continue
            visited.add(sig)
            stats['transitions']+=1
            if len(goals)>1:stats['multi_goal_transitions']+=1
            stack.append(Node(actions,result))
    except ProbeTimeout:
        reason='startup_or_request_timeout'
    finally:
        if server is not None:
            was_closed=getattr(server,'closed',False)
            server.close()
            if not was_closed and hasattr(server,'ensure_clean'):server.ensure_clean()
    elapsed=clock()-started
    if elapsed>budget['seconds']:accepted=False;reason='deadline_shutdown_overrun'
    return dict(accepted=accepted,reason=reason,actions=chosen,logical_evaluations=len(calls),
      actual_lean_processes=processes,server_restarts=restarts,elapsed_seconds=elapsed,budget=budget,
      calls=calls,diagnostics=dict(stats),states_visited=len(visited),max_path_transitions=max((r['depth'] for r in calls),default=0),
      certificate=final,replay=replay,original_source_sha256=c.sha(original),initial_signature=root_signature)
