"""Frozen carry and baseline protocol tests; no fake proof claims."""
import unittest
import torch
from ppt_gym.olr_metta04_carry import (
    perturb,signed_majority,pattern,calibrate,likelihood,paired_contrast,
)


class CarryAuditTests(unittest.TestCase):
    def test_signed_vote_reference(self):
        self.assertEqual(signed_majority([[-1.,1.,1.,1.,1.,1.]]),0)
        self.assertEqual(signed_majority([[1.,1.,1.,1.,1.,1.]]),1)
        self.assertEqual(signed_majority([[1.,1.,1.],[-1.,1.,1.]]),0)

    def test_zero_and_sham_have_equal_input_local_dose(self):
        c=torch.arange(1.,25.).reshape(1,24)
        z,dz=perturb(c,"zero",root_id="r",seed=17,decision=2)
        s,ds=perturb(c,"sham",root_id="r",seed=17,decision=2)
        self.assertTrue(torch.equal(z,torch.zeros_like(c)))
        self.assertAlmostEqual(dz,ds,places=4)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(s-c)),ds,places=4)
        self.assertFalse(torch.equal(s,z))
        t,_=perturb(c,"sham",root_id="r",seed=17,decision=2)
        self.assertTrue(torch.equal(t,s))
        self.assertEqual(perturb(None,"sham",root_id="r",seed=17,decision=0),(None,0.))

    def test_calibration_reads_only_supplied_train_labels(self):
        rows=[]; labels=[]
        for i in range(48):
            winner=i%2
            # Source one perfectly reliable claim, source two always wrong.
            f=[[float(2*winner-1),1.,1.,1.,1.,1.],
               [float(1-2*winner),1/3,1.,1.,1.,1.]]
            f.extend([[0.,0.,0.,0.,0.,0.]]*4)
            rows.append({"root_id":str(i),"model_features":f})
            labels.append({"root_id":str(i),"source_first_winner":winner})
        counts=calibrate(rows,labels)
        for i,row in enumerate(rows):
            self.assertEqual(likelihood(row["model_features"],counts),i%2)
        with self.assertRaises(ValueError):
            calibrate(rows,labels[:-1])

    def test_pair_bootstrap_keeps_seeds_together(self):
        rows=[]
        for pair in ("a","b"):
            for world in (0,1):
                for seed in (17,29,43):
                    for policy,score in (("x",True),("y",False)):
                        rows.append({"split":"dev","pair_id":pair,"world":world,
                                     "seed":seed,"policy":policy,"kernel_closed":score})
        c=paired_contrast(rows,"dev","x","y","kernel_closed",repetitions=100)
        self.assertEqual(c["effect"],1.)
        self.assertEqual(c["ci95"],[1.,1.])
        self.assertEqual(c["pair_clusters"],2)


if __name__=="__main__":
    unittest.main()
