"""Pure graph-soundness tests with synthetic fixtures, never TRAIN/TEST data."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'lean_ram09'))
from offline_graph import goal_signature
from progress_witness_audit import characterize


def rec(before,script,label,after=None,accepted=None,root='T',family='rw_local'):
    goals=[f'⊢ {before}']
    if after is None:out=[]
    else:out=[f'⊢ {after}']
    if accepted is None:accepted=bool(label)
    return dict(source_file='MathCan/Synthetic.lean',theorem=root,
      before={'goals':goals,'accepted':True,'definitions':[]},
      after={'goals':out if label else goals,'accepted':accepted,'definitions':[]},
      script=script,family=family,label=label,state_sha256=goal_signature(goals),
      development=False,delta=1 if label==3 else 0)

class ProgressWitnessTest(unittest.TestCase):
    def test_unknown_finite_graph_successors_are_censored(self):
        rows=[rec('A','rw [h]',2,'B'),rec('A','rw [u]',2,'UNSEEN'),
              rec('B','exact hb',3),rec('C','induction xs',2,'UNSEEN',family='induction_list')]
        graph,t=characterize(rows)
        self.assertEqual(t['rw_local']['positive'],3)
        self.assertEqual(t['rw_local']['closed_immediately'],1)
        self.assertEqual(t['rw_local']['witnessed_continuation'],1)
        self.assertEqual(t['rw_local']['unobserved_successor'],1)
        self.assertEqual(t['induction_list']['positive'],1)
        self.assertEqual(t['induction_list']['witnessed_actions'],0)
        self.assertEqual(t['induction_list']['unobserved_successor'],1)
        self.assertEqual(len(graph.distance),2)
    def test_observed_successor_without_route_is_not_negative(self):
        rows=[rec('A','rewrite',2,'B'),rec('B','fails',0,accepted=False)]
        _,t=characterize(rows)
        self.assertEqual(t['rw_local']['observed_successor_without_witness'],1)
        self.assertEqual(t['rw_local']['possible_eventual_success_fraction_range'],[0.,1.])
    def test_immediate_terminal_always_has_observed_witness(self):
        _,t=characterize([rec('A','exact h',3,family='exact_local')])
        self.assertEqual(t['exact_local']['witnessed_actions'],1)
        self.assertEqual(t['exact_local']['possible_eventual_success_fraction_range'],[1.,1.])

if __name__=='__main__':unittest.main()
