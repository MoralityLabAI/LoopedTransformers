"""Deterministic unit checks for retrospective observed-closure graph."""
import unittest
from offline_graph import ObservedClosureGraph,goal_signature,key

def case(goal, script, label, after, prefix=(), name='synthetic'):
    before=[f'⊢ {goal}']
    return dict(source_file='MathCan/Synthetic.lean',theorem=name,
                development=False, prefix=list(prefix),
                before=dict(accepted=True,goals=before,definitions=[]),
                after=dict(accepted=label>0,goals=[f'⊢ {g}' for g in after] if label>0 else [],definitions=[]),
                state_sha256=goal_signature(before),source_sha256='fixture',
                family='omega',script=script,label=label,delta=int(len(after)<len(before)))

class GraphTest(unittest.TestCase):
    def test_shortest_witness_and_unobserved_censoring(self):
        rows=[case('A','x',2,['B']),case('A','bad',0,[]),
              case('B','step',2,['C']),case('B','blind',2,['UNOBSERVED']),
              case('C','done',3,[])]
        graph=ObservedClosureGraph(rows)
        self.assertEqual(graph.witness(key(rows[0])),['x','step','done'])
        self.assertEqual(graph.action_distance(rows[1]),None)
        self.assertEqual(graph.action_distance(rows[3]),None)
        self.assertEqual(graph.action_distance(rows[2]),2)
        self.assertEqual(graph.summary()['unobserved_nonterminal_successors'],1)
        self.assertEqual(graph.summary()['states_with_observed_closure_path'],3)
    def test_shortest_branch_and_theorem_isolation(self):
        rows=[case('A','slow',2,['B']),case('A','fast',3,[]),
              case('B','finish',3,[]),case('A','elsewhere',2,['B'],name='other')]
        g=ObservedClosureGraph(rows)
        self.assertEqual(g.witness(key(rows[0])),['fast'])
        self.assertIsNone(g.witness(key(rows[-1])))
        self.assertEqual(g.action_distance(rows[-1]),None)
    def test_bad_hash_is_rejected(self):
        row=case('X','finish',3,[]);row['state_sha256']='changed'
        with self.assertRaises(ValueError):ObservedClosureGraph([row])

if __name__ == '__main__':unittest.main()
