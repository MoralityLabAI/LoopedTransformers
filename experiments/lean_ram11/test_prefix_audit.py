"""RAM-11 exact-prefix graph contracts, including counterfactual-join regression."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import unittest
from prefix_audit import PrefixGraph, compare_graphs, goal_signature, prefix_key


def row(prefix,goal,script,after,label,kind='original',source_sha='fixture',definitions=()):
    before=[f'⊢ {goal}']
    return dict(source_file='MathCan/FiberFixture.lean',theorem=kind,
        development=False,prefix=list(prefix),before=dict(accepted=True,goals=before,definitions=list(definitions)),
        after=dict(accepted=label>0,goals=[f'⊢ {g}' for g in after] if label>0 else [],definitions=list(definitions)),
        state_sha256=goal_signature(before),source_sha256=source_sha,
        family='omega',script=script,label=label,delta=int(len(after)<len(before)))

class PrefixAuditTest(unittest.TestCase):
    def test_concrete_counterfactual_splice_is_censored(self):
        data=[row([], 'A', 'x', ['B'],2),
              row([], 'A', 'y', ['C'],2),
              row(['x'], 'B', 'done', [],3),
              row(['y'], 'C', 'q', ['B'],2)]
        result,graph,abstract=compare_graphs(data)
        self.assertEqual(result['prefix']['states_with_exact_prefix_witness'],2)
        self.assertEqual(result['goal_quotient']['states_with_observed_closure_path'],3)
        self.assertEqual(result['quotient_only_state_count'],1)
        self.assertEqual(graph.witness(prefix_key(data[0])),['x','done'])
        self.assertIsNone(graph.witness(prefix_key(data[3])))
        self.assertIsNone(graph.action_distance(data[3]))
        self.assertIsNotNone(abstract.action_distance(data[3]))
        self.assertEqual(graph.quotient_splices,1)

    def test_strict_edge_needs_entire_state_and_exact_script(self):
        data=[row([],'A','x',['B'],2),row(['x'],'B','done',[],3)]
        result,strict,_=compare_graphs(data)
        self.assertEqual(strict.matched,1)
        self.assertEqual(strict.summary()['states_with_exact_prefix_witness'],2)
        data[0]['after']['definitions']=['extraUnseenDefinition']
        with self.assertRaisesRegex(ValueError,'altered printed Lean state'):
            PrefixGraph(data)

    def test_prefix_state_cannot_change_under_same_prefix(self):
        data=[row([],'A','a',[],3),row([],'B','b',[],3)]
        with self.assertRaises(ValueError):PrefixGraph(data)

    def test_theorem_and_source_hash_isolation(self):
        data=[row([],'A','a',[],3),row(['x'],'B','z',[],3,source_sha='mutated')]
        with self.assertRaisesRegex(ValueError,'source SHA varies'):
            PrefixGraph(data)
        other=row([],'A','b',[],3,kind='other')
        result,strict,_=compare_graphs([data[0],other])
        self.assertEqual(result['prefix']['original_theorem_roots'],2)

    def test_positive_not_automatically_witness(self):
        data=[row([],'A','progress',['C'],2),row([],'A','failed',[],0)]
        graph=PrefixGraph(data)
        self.assertIsNone(graph.action_distance(data[0]))
        self.assertEqual(graph.censored,1)
        self.assertEqual(len(graph.distance),0)

    @unittest.skipUnless(os.environ.get('RAM11_ORIGINAL_ARCHIVE'), 'requires pinned RAM08 original artifact')
    def test_original_20930_case_counts(self):
        from prefix_audit import audit
        result=audit(Path(os.environ['RAM11_ORIGINAL_ARCHIVE']))
        self.assertEqual(result['original_compiler_effects'],20930)
        self.assertEqual(result['total_exact_prefix_witness_states'],203)
        self.assertEqual(result['total_abstract_goal_quotient_states'],248)
        self.assertEqual(result['total_quotient_only_states'],45)
        self.assertEqual(result['train']['prefix']['strict_matched_nonterminal_edges'],583)
        self.assertEqual(result['development']['prefix']['strict_matched_nonterminal_edges'],69)

if __name__=='__main__':unittest.main()
