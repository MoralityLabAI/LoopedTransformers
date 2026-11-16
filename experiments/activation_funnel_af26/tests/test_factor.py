"""AF-26 fixed, source-only factorial contracts; run before checkpoint scores."""
import unittest

from experiments.activation_funnel_af26 import factor


def score_pair():
    """Two original scorers with independently varying gate and rule rank."""
    direct = [-1e4] * 15
    vi2 = [-1e4] * 15
    direct[0],direct[1],direct[14] = 3.,2.,1.
    vi2[0],vi2[1],vi2[14] = -3.,-1.,0.
    return {"direct":direct,"vi2":vi2}


class FactorContracts(unittest.TestCase):
    def test_original_dd_and_vv_reproduced_exactly(self):
        p=score_pair()
        a=factor.choose(p,(0,1),"dd",{"direct":0,"vi2":14})
        b=factor.choose(p,(0,1),"vv",{"direct":0,"vi2":14})
        self.assertEqual(a["chosen"],0)
        self.assertEqual(b["chosen"],14)
        self.assertEqual((a["gate_source"],a["rule_source"]),("direct","direct"))

    def test_native_gate_and_crossed_rank(self):
        p=score_pair()
        self.assertEqual(factor.choose(p,(0,1),"dv")["chosen"],1)
        self.assertEqual(factor.choose(p,(0,1),"vd")["chosen"],14)
        self.assertEqual(factor.choose(p,(0,1),"nd")["chosen"],0)
        self.assertEqual(factor.choose(p,(0,1),"nv")["chosen"],1)
        self.assertFalse(factor.choose(p,(0,1),"nd")["gate_selected_stop"])
        self.assertTrue(factor.choose(p,(0,1),"vd")["gate_selected_stop"])

    def test_rule_head_selection_only_legal(self):
        p=score_pair()
        self.assertEqual(factor.choose(p,(1,),"dd")["chosen"],1)
        self.assertEqual(factor.choose(p,(1,),"dv")["chosen"],1)

    def test_ties_follow_original_physical_slot_order(self):
        p=score_pair()
        p["direct"][0]=p["direct"][1]=p["direct"][14]=2.
        p["vi2"][0]=p["vi2"][1]=p["vi2"][14]=0.
        for arm in factor.ARMS:
            self.assertEqual(factor.choose(p,(0,1),arm)["chosen"],0)

    def test_no_legal_rules_stop_in_every_arm(self):
        p=score_pair()
        p["direct"]=[-1e4]*14 + [4.]
        p["vi2"]=[-1e4]*14 + [2.]
        for arm in factor.ARMS:
            self.assertEqual(factor.choose(p,(),arm)["chosen"],14)

    def test_disallow_cross_scale_score_mixing(self):
        p=score_pair()
        self.assertEqual(factor.choose(p,(0,1),"dv")["chosen"],1)
        p["direct"][14]=-1000.
        p["direct"][0]=-999.
        p["direct"][1]=-999.5
        # Large affine shift to Direct does not affect native STOP gate.
        self.assertEqual(factor.choose(p,(0,1),"dv")["chosen"],1)

    def test_source_action_and_mask_tampering_rejected(self):
        p=score_pair()
        with self.assertRaises(ValueError):
            factor.choose(p,(0,1),"dd",{"direct":1,"vi2":14})
        with self.assertRaises(ValueError):
            factor.choose(p,(1,0),"dd")
        with self.assertRaises(ValueError):
            factor.choose(p,(0,0),"dd")
        with self.assertRaises(ValueError):
            factor.choose(p,(0,15),"dd")
        with self.assertRaises(ValueError):
            factor.choose({"direct":p["direct"]},(0,1),"dd")
        with self.assertRaises(ValueError):
            factor.choose(p,(0,1),"oracle")

    def test_registered_factorial_outcome(self):
        base={"dd":240,"dv":131+55,"vd":131+10,"vv":131}
        self.assertEqual(factor.outcome(base),"STOP_ONLY")
        base["dv"]=131+10
        base["vd"]=131+55
        self.assertEqual(factor.outcome(base),"RULE_ONLY")
        base["dv"]=131+55
        self.assertEqual(factor.outcome(base),"BOTH")
        base["dv"]=131+54
        base["vd"]=131+54
        self.assertEqual(factor.outcome(base),"NEITHER")
        with self.assertRaises(ValueError):
            factor.outcome({"dd":230,"dv":190,"vd":140,"vv":131})

    def test_factorial_interaction(self):
        counts={"dd":240,"dv":185,"vd":180,"vv":131}
        c=factor.factor_contrast(counts)
        self.assertEqual(c["direct_minus_vi2"],109)
        self.assertEqual(c["interaction"],6)


if __name__=="__main__":
    unittest.main()
