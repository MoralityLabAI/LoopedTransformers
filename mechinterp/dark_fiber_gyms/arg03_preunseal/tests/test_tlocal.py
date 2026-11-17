"""Model-free CPU tests for the prospective time-local Nanbeige H2a control."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mechinterp.dark_fiber_gyms.arg03_preunseal.run_tlocal_dev import (
    check_dev_path, load_development, evaluate_tlocal_case, main,
)

class ToyLoop:
    n=1
    def prepare(self, ids):
        t=ids.float()
        emb=torch.stack([t/20, t.square()/200, t.sin(), t.cos(), torch.ones_like(t)],dim=-1)
        return emb, {}
    def run(self,h,lo,hi,ctx):
        return h+0.1*h.roll(1,dims=1)
    def readout(self,h,ctx):
        w=torch.tensor([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.],
                        [0.,0.,1.,0.,0.],[0.,0.,0.,1.,0.]])
        return torch.log_softmax(h[:,-3:,:]@w.T,dim=-1)

class ToyEngine:
    op_ids=dict(zip("INSR",range(4)))
    loop=ToyLoop()
    def rows(self,row):
        ids=torch.stack([
            torch.tensor(list(range(17))+[i,j],dtype=torch.long)
            for i in range(4) for j in range(4)])
        return ids,17

class TestTimeLocal(unittest.TestCase):
    def test_readout_and_norm_match_at_each_depth(self):
        row={"case_id":"arg03:development:synthetic:1","public_sha256":"2"*64}
        r=evaluate_tlocal_case(ToyEngine(),row,sigma=0.15,width=6,seed=99)
        self.assertEqual(set(r["depths"]),{"2","4","8"})
        self.assertTrue(r["historical_h2a_unchanged"])
        for v in r["depths"].values():
            self.assertEqual(v["pre_head_logprob_error"]["structured"],0.0)
            self.assertEqual(v["pre_head_logprob_error"]["random_matched"],0.0)
            self.assertLess(v["matched_l2_dose"]["relative_error"],1e-5)
            self.assertGreater(v["next_full_vocab_l2"]["structured"],0.)
            self.assertEqual(len(v["next_64_program_scores"]["structured"]),64)
            self.assertEqual(len(v["next_64_program_scores"]["random_matched"]),64)
            self.assertIn("structured",v["ranking_change"])
            self.assertIn("random_matched",v["ranking_change"])

    def test_reject_old_test_and_non_dev_paths(self):
        for path in (Path("runs/test.jsonl"),Path("runs/heldout/development.jsonl"),
                     Path("runs/final/test_full168/development.jsonl")):
            with self.assertRaises(ValueError):
                check_dev_path(path)
        check_dev_path(Path("runs/development.jsonl"))
        check_dev_path(Path("runs/development_finalR_valid_public.jsonl"))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"development.jsonl"
            row={"case_id":"arg03:test:signed:1","public_sha256":"4"*64,
                 "topology":"fork","observed_degree":12,"query":{},
                 "demos":[{},{}]}
            p.write_text(json.dumps(row)+"\n")
            with self.assertRaises(ValueError):
                load_development(p)
            row["case_id"]="arg03:development_finalR_valid:signed:1"
            p.write_text(json.dumps(row)+"\n")
            self.assertEqual(len(load_development(p)),1)
            self.assertEqual(main(["--public-jsonl",str(p),
                                    "--compactification-root",tmp,
                                    "--out-dir",str(Path(tmp)/"results"),
                                    "--dry-run"]),0)

if __name__=="__main__":unittest.main()
