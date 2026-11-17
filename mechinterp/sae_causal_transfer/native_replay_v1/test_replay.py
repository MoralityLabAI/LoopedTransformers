"""LS-SAE10 CPU unit tests; no pretrained network or heldout in these tests."""
from __future__ import annotations
import unittest

import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_hard_task_v1 import run_native as NATIVE8
from mechinterp.sae_causal_transfer.native_replay_v1 import run_native as RUN
from mechinterp.sae_causal_transfer.native_replay_v1 import audit as AUD


class TinyNative:
    puzzle_emb_len=16
    def act_step(self,zh,zl,x):
        new_zl = .83*zl + .02*zh
        new_zh = .95*zh + .015*new_zl + .004*x
        return new_zh,new_zl
    def digit_logits(self,zh):
        return zh[:,16:,:9].float()


class ReplayForensicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def synthetic(self):
        anchor={"zh":torch.ones((1,97,512),dtype=torch.float32)*0.2,
                "zl":torch.ones((1,97,512),dtype=torch.float32)*0.1,
                "x_emb":torch.ones((1,97,512),dtype=torch.float32)*0.3}
        basis=np.eye(512,dtype=np.float64)
        directions={"state_power":basis[0], "logit_power":basis[1],
                    "sae":None, "random":basis[3]}
        return TinyNative(),anchor,directions

    def test_native_original_and_instrumented_same_grid(self):
        model,anchor,directions=self.synthetic()
        old=NATIVE8.native_all_outputs(model,anchor,5,directions)
        shadow,states=RUN.instrumented_forward(model,anchor,5,directions)
        self.assertEqual(set(states),set(RUN.HORIZONS))
        self.assertEqual(set(old),set(shadow))
        self.assertEqual(old,shadow)
        flags={k:v is not None for k,v in directions.items()}
        self.assertNotIn("sae",old["ACT6"])
        for horizon,name,value in RUN.each_variant(old,flags):
            self.assertEqual(np.asarray(value).shape,(81,9))
        self.assertEqual(RUN.comparison(old,shadow,flags,np.asarray([5]))["ACT12"]["clean"]["max_absolute_drift"],0)

    def test_drift_stat_recomputes_all81_argmax_and_center(self):
        a=np.zeros((81,9),dtype=np.float64); a[:,0]=2
        b=a.copy();b[2,1]=3
        expected=RUN.drift_stats(a,b,np.asarray([2,3]))
        independent=AUD.independent_stats(a.tolist(),b.tolist(),[2,3])
        self.assertEqual(expected["all81_argmax_changes"],1)
        self.assertEqual(expected["blank_argmax_changes"],1)
        self.assertEqual(expected["all81_argmax_changes"],independent["all81_argmax_changes"])
        self.assertEqual(expected["blank_argmax_changes"],independent["blank_argmax_changes"])
        for key in ("max_absolute_drift","centered_drift_rms"):
            self.assertAlmostEqual(expected[key],independent[key],places=12)

    def test_unit_directions_exact_order_and_ineligible_sae(self):
        _,anchor,directions=self.synthetic()
        order=RUN.ordered_edits(directions)
        self.assertEqual(len(order),1+3*3*2)
        self.assertEqual(order[0][0],"clean")
        self.assertEqual(order[1][:3],("state_power",0.2,"plus"))
        self.assertEqual(order[-1][:3],("random",8.0,"minus"))
        self.assertEqual(len(set((a,b,c) for a,b,c,_ in order)),len(order))

    def test_auditor_checks_original_digit_coverage(self):
        z=[[0.]*9 for _ in range(81)]
        with self.assertRaises(ValueError):
            AUD.rows_ok(z[:-1])
        with self.assertRaises(ValueError):
            AUD.rows_ok([[0.]*8]+z[1:])
        good={"ACT3":{"clean":z},"ACT6":{"clean":z},"ACT12":{"clean":z}}
        emitted=list(AUD.all_variants(good,{x:False for x in AUD.ARMS}))
        self.assertEqual(len(emitted),3)
        bad=dict(good);bad["ACT6"]={"clean":z,"random":{"8.00":{"plus":z,"minus":z}}}
        with self.assertRaises(ValueError):
            list(AUD.all_variants(bad,{x:False for x in AUD.ARMS}))

    def test_original_baseline_fail_closed(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"faux.json";p.write_text("{}")
            with self.assertRaises(ValueError):
                RUN.run(p,Path(td)/"out.json")


if __name__=="__main__":
    unittest.main()
