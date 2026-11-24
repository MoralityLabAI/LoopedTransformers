"""Protocol controls for 06G; native certification runs in the separate campaign."""
import unittest
import torch

from ppt_gym.olr06e_multigoal import Root
from ppt_gym.olr06g_portfolio import (
    ARMS,MENU,RESTARTS,CAP,Reader,packet,schedule,calibrate,digest_weights,
    classical_fit,classical_predict,utility,setup,
)

RULES=(("aa","ab"),("ab","bb"),("bc","ca"),("cc","dd"),("ad","dc"),("ba","cb"))
ROOT=Root("dev",RULES,"aaaaaaa",("bbbbbaa","abbaaab","aaaaabb")).validate()


class PortfolioTests(unittest.TestCase):
    def test_every_schedule_and_restart_charge(self):
        for arm in MENU:
            r=schedule(ROOT,arm)
            self.assertLessEqual(r["total_work"],CAP)
            self.assertEqual(r["total_work"],r["expansions"]+r["abstract_expansions"])
            self.assertEqual(r["total_work"],sum(p["work"] for p in r["phases"]))
            if len(r["phases"])==2:
                a,b=r["phases"]
                self.assertEqual(b["cap"],CAP-a["work"])
                self.assertFalse(set(a["resolved"])&set(b["targets"]))
                self.assertEqual(set(a["targets"])-set(a["resolved"]),set(b["targets"]))

    def test_matched_initialization_and_packet(self):
        setup();x,i,scans=packet(ROOT)
        self.assertEqual(x.shape,(11,156));self.assertEqual(i.shape,(11,11))
        self.assertEqual(scans,24)
        self.assertEqual(float(x[:,0].sum()),9.)
        hashes=[]
        for arm in ARMS:
            torch.manual_seed(17);m=Reader(arm)
            if arm in ("bi","bi1","one_way"):hashes.append(digest_weights(m))
            z=m(x[None],i[None])
            self.assertEqual(z.shape,(1,len(MENU)));self.assertTrue(torch.isfinite(z).all())
            z.sum().backward()
        self.assertEqual(len(set(hashes)),1)

    def test_calibration_and_simple_predictors(self):
        setup();x,i,_=packet(ROOT)
        x=x.repeat(6,1,1);i=i.repeat(6,1,1)
        y=torch.arange(len(MENU),dtype=torch.float32).repeat(6,1)/len(MENU)
        state=classical_fit(x,i,y)
        for arm in ("ridge","knn"):
            pred=classical_predict(state,x,i,arm)
            self.assertTrue(torch.allclose(pred,y,atol=1e-6))
            alpha,grid=calibrate(pred,y,y)
            self.assertEqual(alpha,0.)
        self.assertRaises(ValueError,utility,{"total_work":1,"found":{0:0,1:1,2:2}},3)


if __name__=="__main__":unittest.main()
