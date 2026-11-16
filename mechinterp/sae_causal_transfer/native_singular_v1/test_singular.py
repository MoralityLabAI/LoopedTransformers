"""LS-SAE05 exact native JVP/VJP power geometry and source-lock tests."""
import unittest

import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_singular_v1 import run_native as R
from mechinterp.sae_causal_transfer.native_singular_v1 import families as F
from mechinterp.sae_causal_transfer.native_singular_v1 import audit as A


class LinearNativeToy:
    """A genuine 97×512 PyTorch tied recurrence with one amplifying mode."""
    puzzle_emb_len=16
    def act_step(self,zh,zl,x_emb):
        scale=torch.full((512,),.9,dtype=zh.dtype,device=zh.device)
        scale[0]=2.
        return zh*scale,zl


class FrozenSingularTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def anchor(self):
        return {"zh":torch.zeros((1,97,512),dtype=torch.float32),
                "zl":torch.zeros((1,97,512),dtype=torch.float32),
                "x_emb":torch.zeros((1,97,512),dtype=torch.float32)}

    def test_untouched_sudoku_split_count_and_visibility(self):
        fingerprints=F.manifest()
        self.assertEqual(set(fingerprints),set(F.SEEDS))
        all_rows=[r for name in F.SEEDS for r in F.roster(name)]
        self.assertEqual(len(all_rows),20)
        self.assertEqual(len({tuple(x["target"]) for x in all_rows}),20)
        for name in F.SEEDS:
            for row in F.roster(name):
                self.assertEqual(row["x"].count(0),F.HOLES[name])
                self.assertEqual(F.median_blank(row["x"]),
                                 [i for i,d in enumerate(row["x"]) if not d][F.HOLES[name]//2])
        with self.assertRaises(ValueError):
            F.roster("unregistered")
        with self.assertRaises(ValueError):
            F.median_blank([0]*81)

    def test_exact_native_JtJ_four_step_power_finds_narrow_amplifier(self):
        model=LinearNativeToy();anchor=self.anchor();token=24
        power,progress=R.power_direction(model,anchor,token,"test_30",202625030000)
        self.assertEqual(len(progress),4)
        self.assertIsNotNone(power)
        self.assertAlmostEqual(float(np.linalg.norm(power)),1,places=8)
        self.assertGreater(abs(float(power[0])),.99)
        goal=anchor["zh"][0,token].clone()
        _,jvp=R.one_jvp(model,anchor,token,goal,power)
        self.assertGreater(float(jvp.norm()),3.99)
        for k,step in enumerate(progress):
            self.assertEqual(step["step"],k)
            self.assertEqual(step["status"],"eligible")
            self.assertAlmostEqual(step["jvp_dot_vjp"],step["squared_gain"],places=4)
            nxt=progress[k+1]["input_direction_512"] if k<3 else power
            v=np.asarray(step["reverse_gradient_512"])
            np.testing.assert_allclose(v/np.linalg.norm(v),nxt,atol=1e-7)

    def test_native_forward_reverse_dot_equality(self):
        m=LinearNativeToy();a=self.anchor();token=21
        site=a["zh"][0,token].clone()
        v=np.zeros(512,dtype=np.float64);v[0]=1
        _,j=R.one_jvp(m,a,token,site,v)
        w=R.one_vjp(m,a,token,site,j)
        self.assertAlmostEqual(float(j.norm()),4,places=6)
        self.assertAlmostEqual(float(w[0]),16,places=6)
        self.assertAlmostEqual(float(v@w),float(j.norm()**2),places=6)

    def test_scientific_gates_are_puzzle_clustered(self):
        good=[{"power_full_jvp_gain":2.0,"best_random_full_jvp_gain":.3,
               "gain_difference":1.7,"event":True} for _ in range(8)]
        r=R.family_score(good,"test_30")
        self.assertTrue(r["registered_success"])
        self.assertEqual(r["power_gain_events"],8)
        self.assertEqual(r["puzzle_bootstrap95"],[1.7,1.7])
        for x in good:x.update({"event":False,"power_full_jvp_gain":.1,
                                  "best_random_full_jvp_gain":.3,"gain_difference":-.2})
        self.assertFalse(R.family_score(good,"test_30")["registered_success"])

    def test_independent_source_hash_rejects_wrong_receipt(self):
        with self.assertRaises((KeyError,ValueError)):
            A.audit({"schema":R.SCHEMA,"protocol_commit":R.PROTOCOL_COMMIT,
                     "source_checkpoint_sha256":"altered"})


if __name__=="__main__":
    unittest.main()
