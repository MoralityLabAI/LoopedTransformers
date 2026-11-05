import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gym import (make_problem,teacher_states,applicable,features,TwoClockHRM,
                 set_loss,lean_source,save_checkpoint,evaluate)

class GymTests(unittest.TestCase):
    def test_proofs_closed(self):
        for family in ('chain','fork','mix'):
            for seed in range(20):
                p=make_problem(seed,4,family)
                states=teacher_states(p)
                known=set(p.facts)
                for old,actions in states:
                    self.assertEqual(known,old)
                    self.assertTrue(actions)
                    chosen=next(a for a in p.teacher if a in actions)
                    self.assertIn(chosen,applicable(p,known))
                    known.add(p.rules[chosen][1])
                self.assertIn(p.goal,known)
                self.assertNotIn('sorry',lean_source([p]))
    def test_masks_and_gradients(self):
        p=make_problem(12,4,'mix')
        known,correct=teacher_states(p)[0]
        x,m,l=features(p,known,48)
        y=torch.zeros(48,dtype=torch.bool);y[correct]=True
        net=TwoClockHRM(width=32,fast_steps=2,slow_steps=2)
        logits=net(x[None],m[None],l[None]); loss=set_loss(logits,y[None]);loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(net.fast.qkv.weight.grad)
        self.assertIsNotNone(net.slow.qkv.weight.grad)
        self.assertTrue((logits[0][~l]<-1000).all())
    def test_rule_permutation_equivariance(self):
        p=make_problem(12,3)
        known=teacher_states(p)[0][0]
        order=list(reversed(range(len(p.rules))))
        from gym import Problem
        q=Problem(p.facts,[p.rules[i] for i in order],p.goal,[],p.depth,p.family)
        net=TwoClockHRM(width=32)
        net.eval()
        with torch.no_grad():
            x,m,l=features(p,known,48);xx,mm,ll=features(q,known,48)
            a=net(x[None],m[None],l[None])[0];b=net(xx[None],mm[None],ll[None])[0]
        self.assertLess((a[order]-b[:len(order)]).abs().max().item(),1e-4)
    def test_checkpoint_and_eval(self):
        model=TwoClockHRM(width=32)
        opt=torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest.pt'
            save_checkpoint(p,model,opt,{'width':32},1,.4)
            ck=torch.load(p,weights_only=False)
            self.assertEqual(ck['step'],1)
            res=evaluate(model,seed=3,n=3,max_rules=48,depth_min=2,depth_max=3,device='cpu')
            self.assertLessEqual(res['solved'],3)
            self.assertEqual(res['invalid_actions'],0)
            self.assertFalse(res['kernel_checked'])

if __name__=='__main__':unittest.main()
