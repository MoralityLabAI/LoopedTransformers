from __future__ import annotations
import unittest
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
RAM09=HERE if (HERE/"offline_graph.py").exists() else HERE.parent/"lean_ram09"
sys.path.insert(0,str(RAM09))
import torch
from offline_graph import ObservedClosureGraph
from test_offline_graph import case
from ram09b_bounds import witness_feasibility

class BoundTests(unittest.TestCase):
    def test_unflippable_wrong_winner(self):
        rows=[case('A','finish',3,[]),case('A','rejected',0,[])]
        graph=ObservedClosureGraph(rows)
        report=witness_feasibility(graph,torch.tensor([0.,1.]),.35)['counts']
        self.assertEqual(report['necessarily_wrong_under_any_bounded_delta'],1)
        self.assertEqual(report['maximum_witness_states_under_independent_bounded_corrections'],0)
    def test_possible_wrong_winner(self):
        rows=[case('A','finish',3,[]),case('A','rejected',0,[])]
        graph=ObservedClosureGraph(rows)
        report=witness_feasibility(graph,torch.tensor([0.,.3]),.35)['counts']
        self.assertEqual(report['nonwitness_winner_and_legal_flip'],1)
        self.assertEqual(report['maximum_witness_states_under_independent_bounded_corrections'],1)
    def test_guarded_witness_wins(self):
        rows=[case('A','finish',3,[]),case('A','rejected',0,[])]
        graph=ObservedClosureGraph(rows)
        report=witness_feasibility(graph,torch.tensor([1.,0.]),.35)['counts']
        self.assertEqual(report['guaranteed_witness_under_every_bounded_delta'],1)
        self.assertEqual(report['original_winner_is_witness'],1)

if __name__=='__main__':unittest.main()