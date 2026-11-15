"""RAM-11F old-TRAIN only: prioritize unexecuted prefix alias checks for Lean.

The worklist NEVER labels a concatenation as successful. Each proposal is an
original recorded tactic p+a followed by original compiler-observed suffix at
an equivalent goal state q; only fresh Lean4.35 complete script replay may
upgrade the candidate to a valid original full-source theorem proof.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prefix_audit import PrefixGraph,goal_signature,load_archive,source_splits,prefix_key

PRIORITY_FAMILIES={
    'exact_typed':0,'exact_local':1,'rw_local':2,'rcases_local':3,'induction_list':4,
    'constructor':5,'intro_named':6,'simpa_local':7,
}
MAX_PER_ROOT=4


def worklist(train:list[dict],max_per_root:int=MAX_PER_ROOT)->tuple[list[dict],dict]:
    graph=PrefixGraph(train)
    options=[];considered=0;match_full_state=0
    for row in train:
        if row['label'] not in (1,2):continue
        target=(row['source_file'],row['theorem'],tuple(row['prefix']+[row['script']]))
        if target in graph.nodes:continue
        canonical=(row['source_file'],row['theorem'],goal_signature(row['after']['goals']))
        aliases=graph.goal_hash_nodes.get(canonical,[])
        if not aliases:continue
        considered+=1
        for alias in aliases:
            if alias not in graph.distance:continue
            before=graph.nodes[alias][0]['before']
            if row['after']['goals']!=before['goals'] or sorted(row['after']['definitions'])!=sorted(before['definitions']):continue
            match_full_state+=1
            suffix=graph.witness(alias)
            if not suffix:continue
            actions=[*row['prefix'],row['script'],*suffix]
            if len(actions)>8:continue
            root=(row['source_file'],row['theorem'])
            options.append(dict(source_file=root[0],theorem=root[1],
                original_source_sha256=row['source_sha256'],
                recorded_prefix=row['prefix'],candidate_family=row['family'],
                candidate_script=row['script'],candidate_goal_hash=canonical[2],
                alias_recorded_prefix=list(alias[2]),alias_prefix_witness=suffix,
                proposed_new_actions=actions,
                candidate_rank_evidence='compiler-verified individual effects, full concatenation not yet checked',
                historical_proof_score=0))
    # Minimize number of scripts executed per theorem; prioritize rare families,
    # shorter exact-prefix witnesses and stable lexical tie breaks.
    options.sort(key=lambda r:(r['source_file'],r['theorem'],
                PRIORITY_FAMILIES.get(r['candidate_family'],9),
                len(r['proposed_new_actions']),r['candidate_script'],
                tuple(r['recorded_prefix']),tuple(r['alias_recorded_prefix'])))
    selected=[];used=Counter();seen=set()
    for option in options:
        root=(option['source_file'],option['theorem'])
        actions=tuple(option['proposed_new_actions'])
        if actions in seen or used[root]>=max_per_root:continue
        selected.append(option);seen.add(actions);used[root]+=1
    audit=dict(original_train_candidates=len(train),
               candidate_actions_with_some_original_goal_alias=considered,
               matching_aliases_with_exact_printed_state_and_known_prefix_suffix=match_full_state,
               candidate_complete_scripts_selected=len(selected),
               source_theorem_roots_selected=len(used),
               selected_by_candidate_family=dict(sorted(Counter(x['candidate_family'] for x in selected).items())),
               max_per_source_theorem=max_per_root,
               target_source='original MathCan TRAIN c84e2025334e9a967c1297425a490f98d5b3d189',
               new_compiler_queries=0,full_source_proofs_certified=0,
               PEF_or_new_mathcan_test_source_read=False,
               data_limit='This is a deterministic PLAN; no concat has been compiled in fresh Lean.')
    return selected,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('output directory exists: preserve original worklist')
    rows,a=load_archive(args.archive)
    train,dev=source_splits(rows)
    selected,summary=worklist(train)
    args.out.mkdir(parents=True)
    content=''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in selected)
    (args.out/'candidate_complete_source_scripts.jsonl').write_text(content,encoding='utf8')
    summary.update(dataset_sha256=a['dataset_sha256'],worklist_sha256=hashlib.sha256(content.encode()).hexdigest())
    (args.out/'worklist_receipt.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print('RAM11F_SOURCE_ONLY_PLAN',json.dumps(summary,sort_keys=True))

if __name__=='__main__':main()
