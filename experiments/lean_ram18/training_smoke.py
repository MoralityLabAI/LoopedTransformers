"""RAM18 synthetic torch CPU smoke, only for manual training workflow."""
from __future__ import annotations
from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
import torch
from balanced import VerifiedCurriculum, draw_batch

class TrainingSmoke(unittest.TestCase):
    def test_architecture_budget_and_pair_initialization(self):
        torch.set_num_threads(2)
        pairs={}
        for arch in ("loop64","flat71"):
            m=train.make_model(arch,17)
            pairs[arch]=sum(p.numel() for p in m.parameters())
            self.assertEqual(train.state_fingerprint(m.state_dict()),
                             train.state_fingerprint(train.make_model(arch,17).state_dict()))
        self.assertLess(abs(pairs["flat71"]-pairs["loop64"])/pairs["loop64"],.015)
        self.assertEqual(train.FAMILIES23[:20],train.old.FAMILIES)
    def test_synthetic_genuine_lean_goal_encoder_and_gradients(self):
        model=train.make_model("loop64",43)
        rows=[dict(before={"goals":["n : Nat\n⊢ n = n"],"definitions":[]},
                   script="induction n <;> simp",family="nat_induction",label=3,delta=1),
              dict(before={"goals":["n : Nat\n⊢ n=n"],"definitions":[]},
                   script="rfl",family="rfl",label=3,delta=1)]
        opt=torch.optim.AdamW(model.parameters(),lr=.002)
        opt.zero_grad();loss=train.single_loss(model,rows)
        self.assertTrue(bool(torch.isfinite(loss).item()))
        loss.backward();opt.step()

if __name__=="__main__":unittest.main(verbosity=2)
