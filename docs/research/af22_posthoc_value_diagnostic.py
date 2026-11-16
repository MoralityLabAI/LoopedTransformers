"""Retrospective independent source-graph proof-cost/error certification for AF-22.

Uses the original scored *hosted* AF-22 JSONL outputs. This is diagnostic only,
never used for model selection, inference, or preregistered thresholds.
"""
import functools,json,math,os
from pathlib import Path
from experiments.activation_funnel_af22 import gym,planner,verify
from experiments.activation_funnel_af05 import gym as base
ROOT=Path(os.environ.get('AF22_HOSTED_ROOT','/mnt/data/af22_official'))
C={c.id:c for sp in ('held_layout','held_depth4','held_depth5') for c in gym.cases_for(sp)}

@functools.lru_cache(maxsize=None)
def graph(case_id):
 return verify.public_graph(C[case_id])

@functools.lru_cache(maxsize=100000)
def cost(case_id, atoms):
 c=C[case_id];rules,_,goal=graph(case_id);amap,_=gym.physical_maps(c.permutation)
 return verify.shortest_bfs(rules,frozenset(amap[i] for i in atoms),goal)

R={}
for seed in gym.SEEDS:
 rows=[json.loads(line) for line in (ROOT/f's{seed}'/'evaluation'/'episodes.jsonl').open()]
 by={}
 for sp in ('held_depth4','held_depth5'):
  measures={'first_positive':0,'selected_optimal':0,'premature_stop':0,
   'certified':0,'certified_correct':0,'positive_action_count':0,
   'value_error_total':0.,'stop_error_total':0.,'node_count':0,
   'max_error_total':0.,'min_oracle_gap_total':0.,'min_gap_count':0}
  for row in rows:
   if row['mode']!='vi2' or row['split']!=sp or not row['goal_truth'] or not row['actions']:
    continue
   c=C[row['case_id']];first=row['actions'][0];root=frozenset(first['before'])
   d=cost(c.id,tuple(sorted(root)))
   if d is None or d==0:continue
   measures['first_positive']+=1
   trace=first['plan_trace'];edges=planner.state_graph(c,root)[1]
   v0={};st={};epsilon=0.
   for node in trace['nodes']:
    s=frozenset(node['state']);true=cost(c.id,tuple(sorted(s)))
    vt=float(16 if true is None else true)
    ct=float(16 if true is not None and true>0 else 0)
    err=abs(node['value0']-vt);stoperr=abs(node['stop_cost']-ct)
    epsilon=max(epsilon,err,stoperr)
    measures['value_error_total']+=err
    measures['stop_error_total']+=stoperr
    measures['node_count']+=1
    v0[s]=vt;st[s]=ct
   measures['max_error_total']+=epsilon
   oracle_scores,_,_=planner.score_from_values(c,root,edges,v0,st,trace['root_raw'])
   oq={a:-oracle_scores['vi2'][a] for a in [*gym.enabled(c,root),gym.STOP]}
   optimal=min(oq.values());good={a for a,q in oq.items() if abs(q-optimal)<1e-5}
   bad=[q for a,q in oq.items() if a not in good]
   gap=min(bad)-optimal if bad else float('inf')
   selected=first['action'];correct=(selected in good)
   measures['selected_optimal']+=int(correct)
   measures['premature_stop']+=int(selected==gym.STOP)
   if math.isfinite(gap):
    measures['min_oracle_gap_total']+=gap;measures['min_gap_count']+=1
   certified=gap>2*epsilon+1e-6
   measures['certified']+=int(certified)
   measures['certified_correct']+=int(certified and correct)
   if certified and not correct:raise ValueError('empirical margin bound contradicted')
  n=measures['first_positive'];nodes=measures['node_count']
  measures['initial_optimal_action_rate']=measures['selected_optimal']/n if n else None
  measures['initial_stop_rate']=measures['premature_stop']/n if n else None
  measures['empirical_margin_certificate_rate']=measures['certified']/n if n else None
  measures['mean_leaf_value_abs_error']=measures['value_error_total']/nodes if nodes else None
  measures['mean_stop_cost_abs_error']=measures['stop_error_total']/nodes if nodes else None
  measures['mean_max_neighborhood_error']=measures['max_error_total']/n if n else None
  by[sp]=measures
 R[str(seed)]=by
out={'scope':'posthoc_hosted_scored_state_diagnostic_only',
 'source_commit':'5c16b342323977b3045f34c9be4eb24113dc691e',
 'ci_run':43788715688,'seeds':R}
(ROOT/'posthoc_value_diagnostic.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('seed depth init-optimal stop-rate certificate mean-value-error mean-stop-error')
for s,part in R.items():
 for sp,r in part.items():
  print(s,sp,r['first_positive'],f"{r['initial_optimal_action_rate']:.3f}",f"{r['initial_stop_rate']:.3f}",f"{r['empirical_margin_certificate_rate']:.3f}",f"{r['mean_leaf_value_abs_error']:.3f}",f"{r['mean_stop_cost_abs_error']:.3f}")
