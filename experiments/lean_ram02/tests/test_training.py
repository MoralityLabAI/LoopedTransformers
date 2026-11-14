from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import model_data as m
import train_local as t
from resources import ResourceGuard, ResourceLimit

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_sizes_and_metadata_isolation(self):
        row=dict(goal_text='P : Prop\nh : P\n⊢ P',family='exact_local',script='exact h')
        poison=dict(row,theorem='future',source_file='heldout',source_command='sorry',label=3,future_actions=['rfl'])
        for k,v in m.encode([row]).items(): self.assertTrue(torch.equal(v,m.encode([poison])[k]))
        for name,width,loops,n in [('loop32',32,4,82692),('loop64',64,4,198148),('flat64',64,0,173252)]:
            model=m.CandidatePolicy(m.Config(name,width,loops))
            self.assertEqual(sum(p.numel() for p in model.parameters()),n)
            self.assertEqual(tuple(model(m.encode([row])).shape),(1,4))

    def test_accumulation_preserves_weighted_effective_batch(self):
        rows=[dict(goal_text='P : Prop\nh : P\n⊢ P',family='exact_local',script='exact h',label=i%4) for i in range(8)]
        x,y,w=t.tensors(rows)
        class Guard:
            def check(self): pass
        t.seed_all(17)
        a=m.CandidatePolicy(m.Config('loop32',32,4))
        b=m.CandidatePolicy(m.Config('loop32',32,4)); b.load_state_dict(a.state_dict())
        oa=torch.optim.AdamW(a.parameters(),lr=.002)
        ob=torch.optim.AdamW(b.parameters(),lr=.002)
        t.update(a,oa,x,y,w,torch.Generator().manual_seed(19),dict(microbatch=2,accumulation=4,gradient_clip=1.),Guard())
        t.update(b,ob,x,y,w,torch.Generator().manual_seed(19),dict(microbatch=8,accumulation=1,gradient_clip=1.),Guard())
        for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa,pb,atol=2e-6,rtol=2e-5)

    def test_resource_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            g=ResourceGuard(Path(td)/'rss.jsonl',max_rss_gib=0)
            g.sample()
            with self.assertRaises(ResourceLimit): g.check()

    def test_checkpoint_checks_data_fingerprints(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.pt'
            torch.save(dict(state_sha256='bad',effect_sha256=m.EFFECT_SHA),p)
            with self.assertRaises(ValueError): t.load_checkpoint(p)

if __name__=='__main__': unittest.main()
