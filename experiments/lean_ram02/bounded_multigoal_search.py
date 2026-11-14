"""Bounded complete-goal-list search with real Lean certificates and fresh replay."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
import model_data as m
from resources import guarded_process

ALLOWED_AXIOMS={'propext','Classical.choice','Quot.sound'}

def canonical_goals(goals):
    """A syntactic signature, not a semantic equivalence test. Preserve multiplicity."""
    mapping={}
    def metavariable(match):
        token=match.group(0)
        return mapping.setdefault(token,f'?m{len(mapping)}')
    result=[]
    for goal in goals:
        if goal.count('⊢')!=1: raise ValueError('expected one printed goal per branch')
        lines=[line.strip() for line in goal.splitlines() if not re.match(r'^\s*case\s+',line)]
        text='\n'.join(' '.join(line.split()) for line in lines if line.strip())
        result.append(re.sub(r'\?m(?:\.[A-Za-z_]+)*\.\d+',metavariable,text))
    return result

def signature(goals):
    canonical=canonical_goals(goals)
    goal_hashes=[hashlib.sha256(g.encode()).hexdigest() for g in canonical]
    state_hash=hashlib.sha256(json.dumps(canonical,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    return state_hash,goal_hashes

def axiom_audit(log):
    if log.count('RAM02_AXIOMS_BEGIN')!=1 or log.count('RAM02_AXIOMS_END')!=1:
        return False,[]
    section=log.split('RAM02_AXIOMS_BEGIN')[1].split('RAM02_AXIOMS_END')[0]
    if 'sorryAx' in section: return False,['sorryAx']
    records=re.findall(r"'([^']+)'\s+(does not depend on any axioms|depends on axioms:\s*\[[^\]]*\])",section,re.S)
    if len(records)!=1: return False,[]
    record=records[0][1]
    if record=='does not depend on any axioms': return True,[]
    names=[x.strip() for x in record.split('[',1)[1].split(']',1)[0].split(',') if x.strip()]
    return set(names)<=ALLOWED_AXIOMS,names

def extract_goals(log):
    counts=re.findall(r'RAM02_GOAL_COUNT=(\d+)',log)
    goals=re.findall(r'RAM02_GOAL_BEGIN\s*\n(.*?)\nRAM02_GOAL_END',log,re.S)
    if len(counts)!=1 or len(goals)!=int(counts[0]):
        raise ValueError('missing or ambiguous Lean goal observation')
    goals=[g.strip() for g in goals]
    canonical_goals(goals)
    return goals

class LeanCompiler:
    def __init__(self, executable, guard, *, lean_path=None):
        self.executable,self.guard=executable,guard

    def source(self, original, theorem, actions, probe):
        from research.refutation_bench.fc12 import scratch
        # Original source preparation and tactic validation stay pinned to FC-12.
        source=scratch.make_source(original,theorem,actions,probe=False)
        lines=source.splitlines()
        endings=[i for i,line in enumerate(lines) if re.fullmatch(r'end\s+[\w.]+',line.strip())]
        at=endings[-1] if endings else len(lines)
        if probe:
            report=[
                '  run_tac',
                '    let goals ← Lean.Elab.Tactic.getGoals',
                '    Lean.logInfo m!"RAM02_GOAL_COUNT={goals.length}"',
                '    for goal in goals do',
                '      Lean.logInfo m!"RAM02_GOAL_BEGIN\\n{← Lean.Meta.ppGoal goal}\\nRAM02_GOAL_END"',
                '  all_goals sorry',
            ]
        else:
            report=['#eval IO.println "RAM02_AXIOMS_BEGIN"','#print axioms '+theorem,
                    '#eval IO.println "RAM02_AXIOMS_END"']
        lines[at:at]=report
        return '\n'.join(lines)+'\n'

    def __call__(self, original, theorem, actions, probe, timeout, prefix):
        source=self.source(original,theorem,actions,probe)
        return self.compile_source(source,probe,timeout,prefix)

    def compile_source(self, source, probe, timeout, prefix):
        prefix.parent.mkdir(parents=True,exist_ok=True)
        path=prefix.with_suffix('.lean')
        path.write_text(source,encoding='utf8',newline='\n')
        result=guarded_process([self.executable,str(path.resolve())],prefix.with_suffix('.log'),self.guard,timeout)
        log=result.pop('log')
        gate,axioms=(True,[]) if probe else axiom_audit(log)
        accepted=result['returncode']==0 and not result['timed_out'] and 'error:' not in log and gate
        goals=[]
        if accepted and probe:
            try: goals=extract_goals(log)
            except ValueError: accepted=False
        if not probe:
            accepted=accepted and "declaration uses 'sorry'" not in log and 'sorryAx' not in log
        return dict(**result,accepted=accepted,probe=probe,remaining_goals=goals,goal_count=len(goals),
                    axiom_gate_passed=gate if not probe else None,axioms=axioms,
                    source_sha256=m.digest(path),source_file=str(path),log_file=str(prefix.with_suffix('.log')),
                    log_sha256=m.digest(prefix.with_suffix('.log')))

@dataclass
class Node:
    actions:list
    goals:list
    transitions:int
    candidates:list|None=None
    next_candidate:int=0

def solve(original,theorem,initial_goals,ranker,evidence,compiler,*,generator=None,
          clock=time.perf_counter,seconds=20.,max_calls=12,max_transitions=3,per_call_seconds=8.):
    if generator is None:
        from research.refutation_bench.fc12 import core
        generator=core.generate
    started=clock(); end=started+seconds
    root_hash,_=signature(initial_goals)
    visited={root_hash}
    stack=[Node([],initial_goals,0)]
    calls=[]; chosen=[]; certified=False; reason='no_progress'; terminal=None
    stats=Counter()
    def budget():
        if clock()>=end: return 'hard_deadline'
        if len(calls)>=max_calls: return 'subprocess_budget'
        return None
    while stack:
        expired=budget()
        if expired: reason=expired; break
        node=stack[-1]
        if node.transitions>=max_transitions:
            stats['transition_budget']+=1; reason='transition_budget'; stack.pop(); continue
        if node.candidates is None:
            pool=generator(node.goals[0])
            candidates=[(f,s) for f in m.FAMILIES for s in pool.get(f,[])]
            node.candidates=ranker.rank(node.goals[0],candidates) if ranker else candidates
            if Counter(node.candidates)!=Counter(candidates): raise AssertionError('policy changed candidate grammar')
            if clock()>=end: reason='hard_deadline'; break
        if node.next_candidate>=len(node.candidates):
            stats['exhausted_states']+=1; stack.pop(); continue
        family,script=node.candidates[node.next_candidate]
        node.next_candidate+=1
        actions=node.actions+[script]
        before_hash,before_goals=signature(node.goals)
        result=compiler(original,theorem,actions,True,min(per_call_seconds,end-clock()),evidence/f'call{len(calls):02d}')
        result.update(family=family,script=script,depth=node.transitions+1,
                      state_before=before_hash,goal_hashes_before=before_goals,goals_before=len(node.goals))
        calls.append(result)
        if clock()>=end: reason='hard_deadline'; break
        if not result['accepted']:
            stats['compiler_rejections']+=1
            continue
        goals=result['remaining_goals']
        after_hash,after_goals=signature(goals)
        result.update(state_after=after_hash,goal_hashes_after=after_goals,
                      creates_branches=len(goals)>len(node.goals),
                      reduces_goal_count=len(goals)<len(node.goals),
                      all_goals_closed=len(goals)==0)
        if not goals:
            expired=budget()
            if expired: reason=expired; break
            final=compiler(original,theorem,actions,False,min(per_call_seconds,end-clock()),evidence/f'call{len(calls):02d}')
            final.update(family=family,script=script,depth=node.transitions+1)
            calls.append(final)
            certified=bool(final['accepted'] and not final['timed_out'] and clock()<=end)
            chosen=actions
            terminal=final if certified else None
            reason='kernel_certified' if certified else ('hard_deadline' if clock()>end else 'final_rejected')
            if certified or reason=='hard_deadline': break
            continue
        if after_hash in visited:
            stats['deduplicated_states']+=1
            continue
        visited.add(after_hash)
        stats['successful_transitions']+=1
        if len(goals)>1: stats['multigoal_transitions']+=1
        stack.append(Node(actions,goals,node.transitions+1))
    elapsed=clock()-started
    if elapsed>seconds: certified=False;reason='hard_deadline';terminal=None
    return dict(accepted=certified,reason=reason,chosen_actions=chosen,lean_subprocesses=len(calls),
        elapsed_seconds=elapsed,deadline_seconds=seconds,max_calls=max_calls,max_transitions=max_transitions,
        max_path_transitions=max((c.get('depth',0) for c in calls),default=0),
        states_visited=len(visited),diagnostics=dict(stats),calls=calls,final_certificate=terminal,
        initial_state_sha256=root_hash,original_source_sha256=hashlib.sha256(original.encode()).hexdigest())
