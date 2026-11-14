from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import model as m

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_future_outcomes_and_metadata_never_enter_features(self):
        row=dict(before=dict(goals=['P : Prop\nh : P\n⊢ P','⊢ Q']),family='exact_local',script='exact h')
        poisoned=dict(row,theorem='SECRET',source_file='SECRET',prefix=['hidden'],after=dict(goals=[]),label=3,delta=2)
        for key,value in m.encode([row]).items():self.assertTrue(torch.equal(value,m.encode([poisoned])[key]))
    def test_pending_goals_change_features(self):
        a=dict(before=dict(goals=['⊢ P','⊢ Q']),family='rfl',script='rfl')
        b=dict(a,before=dict(goals=['⊢ P','⊢ R']))
        self.assertFalse(torch.equal(m.encode([a])['pending'],m.encode([b])['pending']))
    def test_both_heads_and_recurrence_receive_gradients(self):
        rows=[dict(before=dict(goals=['P : Prop\nh : P\n⊢ P','⊢ Q']),family='exact_local',script='exact h')]
        for width,loops in [(32,4),(64,4),(64,0)]:
            model=m.Policy(m.Config('test',width,loops));logits,delta=model(m.encode(rows))
            self.assertEqual(tuple(logits.shape),(1,4));self.assertEqual(tuple(delta.shape),(1,))
            (logits.square().sum()+delta.square().sum()).backward()
            self.assertIsNotNone(model.embedding.weight.grad)
            self.assertIsNotNone(model.delta.weight.grad)
            if loops:self.assertIsNotNone(model.cell[0].weight.grad)
    def test_ranker_preserves_registered_candidate_pool(self):
        model=m.Policy(m.Config('loop32',32,4)).eval()
        pool=[('omega','omega'),('exact_local','exact h')]
        self.assertCountEqual(model.rank(dict(goals=['P : Prop\nh : P\n⊢ P']),pool),pool)
if __name__=='__main__':unittest.main()
