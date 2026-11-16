"""AF-27 scoring and rule-gap preflight without neural training."""
import unittest
from experiments.activation_funnel_af27 import gap
from experiments.activation_funnel_af26 import factor


def source_values():
    direct=[-1e4]*15
    vi2=[-1e4]*15
    direct[0],direct[1],direct[14]=3.,2.,0.
    vi2[0],vi2[1],vi2[14]=-3.,-1.,-2.
    return {"direct":direct,"vi2":vi2}


class GapContracts(unittest.TestCase):
    def test_native_margins_and_stop_gates(self):
        v=source_values()
        r=gap.gap_receipt(v,(0,1))
        self.assertEqual(r["direct"]["top_rule"],0)
        self.assertEqual(r["vi2"]["top_rule"],1)
        self.assertEqual(r["direct"]["margin"],1.)
        self.assertEqual(r["vi2"]["margin"],2.)
        self.assertTrue(r["same_top_legal_rule"] is False)
        self.assertTrue(r["direct_continues"])
        self.assertTrue(r["vi2_continues"])

    def test_ties_track_and_lowest_slot(self):
        v=source_values()
        v["vi2"][0]=v["vi2"][1]=-1.
        r=gap.legal_margin(v["vi2"],(0,1))
        self.assertEqual(r["top_rule"],0)
        self.assertTrue(r["near_tie_1e5"])
        self.assertEqual(r["margin"],0)

    def test_single_legal_rule_has_no_margin(self):
        v=source_values()
        r=gap.legal_margin(v["direct"],(1,))
        self.assertIsNone(r["margin"])
        self.assertIsNone(r["runner_up_rule"])

    def test_never_cross_native_logit_scale(self):
        v=source_values()
        # Make native VI2 choose STOP yet Direct choose legal 0.
        v["vi2"][14]=0.
        self.assertEqual(factor.choose(v,(0,1),"vd")["chosen"],14)
        self.assertEqual(factor.choose(v,(0,1),"dd")["chosen"],0)

    def test_legal_mask_rejects_bad_ids(self):
        v=source_values()
        for legal in ((2,1),(0,0),(0,14),()):
            with self.subTest(legal=legal):
                if legal==():
                    with self.assertRaises(ValueError):
                        gap.gap_receipt(v,legal)
                else:
                    with self.assertRaises(ValueError):
                        gap.gap_receipt(v,legal)

    def test_interior_min_plus_path_decomposition(self):
        root=[0,1,2,3]
        first=[0,1,2,3,4]
        terminal=[0,1,2,3,4,14]
        trace={
            "root_raw":[0]*15,
            "nodes":[
                {"state":root,"edges":{"0":first},
                 "value0":10.,"stop_cost":8.,"reach_logit":0.,"distance_logit":0.},
                {"state":first,"edges":{"3":terminal},
                 "value0":6.,"stop_cost":1.,"reach_logit":0.,"distance_logit":0.},
                {"state":terminal,"edges":{},
                 "value0":0.,"stop_cost":0.,"reach_logit":0.,"distance_logit":0.},
            ]
        }
        rec=gap.value_path_details(trace,(0,))
        self.assertTrue(rec["0"]["interior_stop_binds"])
        self.assertEqual(rec["0"]["two_step_rule_score"],-2.)
        self.assertEqual(rec["0"]["min_continuation_cost"],1.)


if __name__=="__main__":
    unittest.main()
