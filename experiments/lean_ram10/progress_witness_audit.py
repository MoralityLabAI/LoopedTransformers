"""RAM-10 diagnostic on *old* RAM-08, independently replayed Lean effects.

Reads no RAM-10 MathCan TRAIN/DEV/TEST source. Retrospective labels only.
An accepted tactic that branches is not proof completion. We report an
observed closure-path witness and explicitly censor every unknown frontier.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
GRAPH=HERE.parent/'lean_ram09'
if str(GRAPH) not in sys.path:sys.path.insert(0,str(GRAPH))
from offline_graph import load_archive,source_splits,ObservedClosureGraph,goal_signature,key,ARCHIVE_SHA,TRANSITIONS_SHA

SOURCE_FAMILIES=('exact_typed','exact_local','rw_local','rcases_local','induction_list',
                 'constructor','intro_named','apply_local','decide','simp_all','omega','rfl')


def characterize(rows: list[dict]):
    """Count retained closure routes, with censored frontiers kept open-world.

    An original compiler-accepted transition is 'witnessed' only if its exact
    recorded successor state reaches an original compiler-accepted terminal
    within the finite recorded graph. This is a sufficient certificate for
    an *observed tactic route*, never for a whole source theorem proof.
    """
    graph=ObservedClosureGraph(rows)
    metrics=defaultdict(lambda:dict(tested=0,positive=0,closed_immediately=0,
                                    witnessed_continuation=0,unobserved_successor=0,
                                    observed_successor_without_witness=0,
                                    failed_or_unchanged=0,positive_roots=set(),witnessed_roots=set()))
    for row in rows:
        f=row['family'];m=metrics[f];m['tested']+=1
        if row['label']==0:
            m['failed_or_unchanged']+=1
            continue
        m['positive']+=1
        root=(row['source_file'],row['theorem'])
        m['positive_roots'].add(root)
        if row['label']==3:
            m['closed_immediately']+=1
            m['witnessed_roots'].add(root)
            continue
        goals=row['after']['goals']
        child=(*root,goal_signature(goals))
        if child not in graph.by_state:
            m['unobserved_successor']+=1
        elif child in graph.distance:
            m['witnessed_continuation']+=1
            m['witnessed_roots'].add(root)
        else:
            m['observed_successor_without_witness']+=1
    rendered={}
    for f in sorted(metrics):
        x=metrics[f]
        witnessed=x['closed_immediately']+x['witnessed_continuation']
        if sum(x[k] for k in ('closed_immediately','witnessed_continuation',
               'unobserved_successor','observed_successor_without_witness'))!=x['positive']:
            raise AssertionError('a compiler-positive action was lost or mislabeled')
        rendered[f]={k:v for k,v in x.items() if not isinstance(v,set)}
        rendered[f].update(witnessed_actions=witnessed,
          distinct_positive_roots=len(x['positive_roots']),
          distinct_witnessed_roots=len(x['witnessed_roots']),
          observed_witness_fraction_given_positive=(witnessed/x['positive'] if x['positive'] else None),
          possible_eventual_success_fraction_range=[witnessed/x['positive'],1.] if x['positive'] else None)
    return graph,rendered


def audit(artifact,output):
    if output.exists():raise FileExistsError('preserve first diagnostic receipt')
    original,audit_receipt=load_archive(artifact)
    train,dev=source_splits(original)
    result=dict(campaign='RAM10_historical_progress_vs_observed_witness_audit',
        epistemic_status='posthoc on RAM08 old source; never RAM10 test',
        archived_data_sha256=TRANSITIONS_SHA, original_training_zip_sha256=ARCHIVE_SHA,
        compiler_effects_independently_replayed=audit_receipt['checked'],
        ram10_source_checkout=False,ram10_models_trained=0,ram10_test_theorems_scored=0,
        registered_ram10_proof_admission_gate_unchanged=True,
        splits={})
    for name,subset in [('old_train',train),('old_dev_selected',dev)]:
        graph,groups=characterize(subset)
        result['splits'][name]=dict(compiler_observations=len(subset),
           state_count=len(graph.by_state),observed_closure_states=len(graph.distance),
           family_metrics=groups)
    result['interpretation']=(
        'Finite-graph route is sufficient evidence for compiler-observed action completion; '
        'unwitnessed accepted actions are censored/unknown, not failures. '
        'An observed tactic path alone is not a complete independently kernel-certified original source theorem.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original_ram08_zip',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    x=audit(a.original_ram08_zip,a.out)
    for name,part in x['splits'].items():
        print(name,'states',part['state_count'],'observed_reachable',part['observed_closure_states'])
        for f in SOURCE_FAMILIES:
            v=part['family_metrics'].get(f)
            if v:
                print(f,'accepted',v['positive'],'witnessed',v['witnessed_actions'],
                      'unobserved',v['unobserved_successor'],
                      'observed_without_route',v['observed_successor_without_witness'])

if __name__=='__main__':main()
