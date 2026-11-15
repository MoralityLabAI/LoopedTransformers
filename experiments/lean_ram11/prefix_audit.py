"""RAM-11 retrospective audit of exact-prefix versus goal-quotient proof witnesses.

This checks the original RAM-08 independently replayed corpus, not new Lean
proofs. Exact-prefix witnesses follow recorded concrete tactic prefixes only.
Goal-quotient witnesses may splice unexecuted composite tactic scripts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lean_ram09'))
from offline_graph import load_archive, source_splits, goal_signature, ObservedClosureGraph, key

Prefix = tuple[str, str, tuple[str, ...]]


def prefix_key(row: dict) -> Prefix:
    return row['source_file'], row['theorem'], tuple(row['prefix'])


class PrefixGraph:
    """Genuine recorded tactic-prefix trie with complete-printed-state edge audit."""
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.nodes: dict[Prefix, list[dict]] = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.forward = defaultdict(list)
        self.distance: dict[Prefix, int] = {}
        self.chosen = {}
        self.root_sha: dict[tuple[str, str], str] = {}
        self.matched = 0
        self.censored = 0
        self.terminals = 0
        self.rejected = 0
        self.quotient_splices = 0
        self.definition_mismatches_on_goal_aliases = 0
        self.visited_goal_sha_to_prefix = {}
        self._build()

    def _build(self) -> None:
        # Full original source, theorem, prefix and printed goal-state must agree
        # across every candidate observation from the same recorded state.
        for row in self.rows:
            node = prefix_key(row)
            root = node[:2]
            sha = row['source_sha256']
            if root in self.root_sha and self.root_sha[root] != sha:
                raise ValueError('original theorem source SHA varies across observations')
            self.root_sha[root] = sha
            if goal_signature(row['before']['goals']) != row['state_sha256']:
                raise ValueError('original RAM08 state signature mismatch')
            if row['label'] not in (0, 1, 2, 3):
                raise ValueError('unknown compiler label')
            if self.nodes[node]:
                precedent = self.nodes[node][0]['before']
                if (precedent['goals'] != row['before']['goals'] or
                    sorted(precedent.get('definitions', [])) != sorted(row['before'].get('definitions', []))):
                    raise ValueError('original prefix changed its visible Lean state')
            self.nodes[node].append(row)
        self.goal_hash_nodes = defaultdict(list)
        for node, rows in self.nodes.items():
            h = rows[0]['state_sha256']
            self.goal_hash_nodes[(node[0],node[1],h)].append(node)

        for node, rows in self.nodes.items():
            for row in rows:
                label = row['label']
                if label == 0:
                    self.rejected += 1
                    continue
                after = row['after']
                if not after['accepted']:
                    raise ValueError('positive labelled outcome rejected by Lean')
                if label == 3:
                    if after['goals']:
                        raise ValueError('Lean goal closing action left goals')
                    self.terminals += 1
                    self.forward[node].append((None,row['script']))
                    continue
                if not after['goals']:
                    raise ValueError('positive nonterminal action closed all goals')
                # Strict child: the exact actual scripted prefix p followed by a.
                target = (node[0],node[1],node[2]+(row['script'],))
                same_visible = self.goal_hash_nodes.get((node[0],node[1],goal_signature(after['goals'])),[])
                if target not in self.nodes:
                    self.censored += 1
                    if same_visible:
                        self.quotient_splices += 1
                        if all(sorted(after.get('definitions',[])) !=
                            sorted(self.nodes[s][0]['before'].get('definitions',[])) for s in same_visible):
                            self.definition_mismatches_on_goal_aliases += 1
                    continue
                child_before = self.nodes[target][0]['before']
                if (after['goals'] != child_before['goals'] or
                    sorted(after.get('definitions',[])) != sorted(child_before.get('definitions',[]))):
                    raise ValueError('exact-prefix edge has altered printed Lean state')
                self.matched += 1
                self.forward[node].append((target,row['script']))
                self.predecessors[target].append((node,row['script']))
        queue = deque()
        for node, edges in self.forward.items():
            for nxt, action in edges:
                if nxt is None and node not in self.distance:
                    self.distance[node] = 1
                    self.chosen[node] = (None,action)
                    queue.append(node)
        while queue:
            nxt = queue.popleft()
            for source, action in self.predecessors.get(nxt,[]):
                if source not in self.distance:
                    self.distance[source] = 1 + self.distance[nxt]
                    self.chosen[source] = (nxt, action)
                    queue.append(source)

    def action_distance(self, row: dict) -> int|None:
        if row['label'] == 0:
            return None
        if row['label'] == 3:
            return 1
        node = prefix_key(row)
        next_node = (node[0], node[1], node[2]+(row['script'],))
        if next_node in self.nodes and next_node in self.distance:
            return 1+self.distance[next_node]
        return None

    def witness(self, node: Prefix) -> list[str]|None:
        if node not in self.distance:
            return None
        route=[];current=node
        while current is not None:
            nxt,a=self.chosen[current];route.append(a);current=nxt
        if len(route)!=self.distance[node]:
            raise ValueError('invalid shortest prefix witness')
        return route

    def summary(self)->dict:
        lengths=dict(sorted(Counter(self.distance.values()).items()))
        root_nodes=[n for n in self.nodes if not n[2]]
        return dict(original_theorem_roots=len(self.root_sha),states=len(self.nodes),
            observations=len(self.rows),terminal_effects=self.terminals,
            strict_matched_nonterminal_edges=self.matched,
            strict_censored_nonterminal_edges=self.censored,
            rejected_or_unchanged_effects=self.rejected,
            quotient_possible_unexecuted_prefix_splices=self.quotient_splices,
            quotient_goal_aliases_with_different_definitions=self.definition_mismatches_on_goal_aliases,
            states_with_exact_prefix_witness=len(self.distance),
            shortest_exact_prefix_witness_lengths=lengths,
            root_states_with_exact_prefix_witness=sum(r in self.distance for r in root_nodes),
            theorem_roots_with_some_exact_prefix_witness=len({n[:2] for n in self.distance}),
            nonterminal_labels_censored_when_prefix_absent=True,
            full_original_theorem_kernel_certificates=0)


def actions_by_family(rows: list[dict], prefix: PrefixGraph, quotient: ObservedClosureGraph)->dict:
    report=defaultdict(lambda: Counter())
    for row in rows:
        family=row['family']
        report[family]['attempts']+=1
        if row['label'] > 0:
            report[family]['compiler_accepted_progress']+=1
            if prefix.action_distance(row) is not None:
                report[family]['strict_prefix_witness']+=1
            if quotient.action_distance(row) is not None:
                report[family]['goal_quotient_witness']+=1
    return {n:dict(stats) for n,stats in sorted(report.items())}


def compare_graphs(rows: list[dict])->tuple[dict, PrefixGraph, ObservedClosureGraph]:
    p = PrefixGraph(rows)
    q = ObservedClosureGraph(rows)
    exact={(n[0],n[1],p.nodes[n][0]['state_sha256']) for n in p.distance}
    abstract=set(q.distance)
    if not exact <= abstract:
        raise AssertionError('exact-prefix routes must induce quotient routes')
    if len(q.by_state)!=len(p.nodes):
        raise AssertionError('unique source/original-prefix state count differs from goal quotient')
    result=dict(prefix=p.summary(),goal_quotient=q.summary(),
         quotient_only_state_count=len(abstract-exact),
         quotient_only_state_examples=[dict(file=n[0],theorem=n[1],goal_sha256=n[2]) for n in sorted(abstract-exact)[:12]],
         prefix_subset_quotient_verified=True,
         actions_by_family=actions_by_family(rows,p,q))
    return result,p,q


def score_frozen_models(archive:Path, original_train:list[dict], original_dev:list[dict])->dict:
    """Optional read-only old-source replay with SHA-pinned original models."""
    import torch
    sys.path.insert(0, str(HERE.parent / 'lean_ram09'))
    from offline_fit import originals,score,encode,split_internal
    torch.set_num_threads(2)
    fit, hold = split_internal(original_train)
    cohorts={'inner_holdback':hold,'old_ram08_development':original_dev}
    result={}
    with torch.no_grad():
        for name,model,sha in originals(archive):
            model.eval();split_data={}
            for split, split_rows in cohorts.items():
                p=PrefixGraph(split_rows);q=ObservedClosureGraph(split_rows)
                scores=[]
                for i in range(0,len(split_rows),64):
                    scores.extend(score(model,encode(split_rows[i:i+64])).tolist())
                grouped=defaultdict(list)
                for r,v in zip(split_rows,scores):grouped[prefix_key(r)].append((r,v))
                stats=Counter()
                for node,obs in grouped.items():
                    chosen=max(obs,key=lambda tup:tup[1])[0]
                    stats['states']+=1
                    stats['selected_progress']+=int(chosen['label']>0)
                    stats['selected_closure']+=int(chosen['label']==3)
                    if node in p.distance:
                        stats['available_prefix_witness_states']+=1
                        stats['selected_strict_prefix_witness']+=int(p.action_distance(chosen) is not None)
                    qnode=key(chosen)
                    if qnode in q.distance:
                        stats['available_quotient_witness_states']+=1
                        stats['selected_quotient_witness']+=int(q.action_distance(chosen) is not None)
                split_data[split]=dict(stats)
            result[name]=dict(original_checkpoint_sha256=sha,cohorts=split_data)
    return result


def audit(path:Path, with_models:bool=False)->dict:
    rows, original_audit=load_archive(path)
    train,dev=source_splits(rows)
    left,_,_=compare_graphs(train)
    right,_,_=compare_graphs(dev)
    result=dict(campaign='RAM11-retrospective-prefix-fiber-witness-integrity',
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_dataset_sha256=original_audit['dataset_sha256'],
        original_compiler_effects=len(rows),
        original_lean_sessions_for_independent_effect_replay=original_audit['independent_fresh_lean_sessions'],
        train=left,development=right,
        total_exact_prefix_witness_states=left['prefix']['states_with_exact_prefix_witness']+right['prefix']['states_with_exact_prefix_witness'],
        total_abstract_goal_quotient_states=left['goal_quotient']['states_with_observed_closure_path']+right['goal_quotient']['states_with_observed_closure_path'],
        total_quotient_only_states=left['quotient_only_state_count']+right['quotient_only_state_count'],
        newly_compiled_lean_theorems=0,
        PEF_theorems_scored=0,
        RAM10_fresh_compiler_observations=0,
        limitation='Posthoc exact-prefix recorded compiler paths. Neither graph proves a full theorem without new complete Lean source compilation.')
    if with_models:
        result['frozen_original_model_ranking']=score_frozen_models(path,train,dev)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--with-models',action='store_true',help='Replay three immutable old-source DEV rankings')
    args=parser.parse_args()
    if args.out.exists():parser.error('Audit output exists; preserve original report')
    result=audit(args.archive,args.with_models)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('RAM11_PREFIX_FIBER_AUDIT',json.dumps({k:result[k] for k in ['original_compiler_effects','total_exact_prefix_witness_states','total_abstract_goal_quotient_states','total_quotient_only_states','newly_compiled_lean_theorems']},sort_keys=True))

if __name__=='__main__':main()
