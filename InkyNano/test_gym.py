import json
import unittest
try:
    import numpy as np
    from gym_data import build
    from gym_train import admit, Mesh
    from gym_export import pack4, unpack4
    import torch
    from routing import parse_rows
except ImportError:
    np=None


@unittest.skipIf(np is None,"numpy/torch/safetensors unavailable")
class GymTests(unittest.TestCase):
    def trace(self):
        roster=[{"root":"r","turn":"0","family":"f","split":"train","prompt_len":1,"token_ids":[1,2,3]}]
        rows=[]
        for t in range(3):
            for l in range(24):
                for p in range(3 if 6<=l<18 else 1):
                    rows.append(json.dumps({"root":"r","turn":"0","token":t,"layer":l,"pass":p,
                                            "experts":list(range(t*8,t*8+8)),"weights":[1.]*8}))
        return roster,parse_rows(rows,True)

    def test_no_future_in_features(self):
        roster,rows=self.trace()
        d=build(roster,rows,1)
        self.assertEqual(d["x"].shape,(2,24,256))
        self.assertEqual(float(d["x"][0,0,8:16].sum()),0.)
        self.assertEqual(float(d["y"][0,0,8:16].sum()),1.)
        self.assertAlmostEqual(float(d["x"][1,0,128+8:128+16].sum()),.5)
        self.assertEqual(float(d["x"][1,0,128+16:128+24].sum()),0.)

    def test_admissions(self):
        old=frozenset(range(8))
        score=np.arange(128,dtype=float)
        self.assertEqual(admit(score,old,8,0),old)
        new=admit(score,old,8,2)
        self.assertEqual(len(new-old),2)
        self.assertEqual(new-old,{126,127})
        self.assertEqual(len(new),8)

    def test_mesh(self):
        m=Mesh()
        self.assertEqual(tuple(m(torch.zeros(2,24,256)).shape),(2,24,128))

    def test_pack(self):
        w=torch.linspace(-2,2,67).reshape(1,67)
        q,s,shape=pack4(w)
        reconstructed=unpack4(q,s,shape)
        self.assertEqual(tuple(reconstructed.shape),tuple(w.shape))
        self.assertLess(float((w-reconstructed).abs().max()),.16)


if __name__=="__main__":
    unittest.main()
