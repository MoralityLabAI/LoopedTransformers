"""LGA-10F pre-source controls. Only exposed model seed 61064 is used."""
import copy
import unittest

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope13b.source_feasibility import QuadraticObserver
from mechinterp.loopscope13c.audit_receipt import audit_native
from mechinterp.loopscope13f.audit61069 import check_levelset_donors
from mechinterp.loopscope13f.manifold import (
    SEED, S_VALUES, N_DIRECTIONS, curvature, levelset_candidates, prepare_levelset,
)
from mechinterp.loopscope13f.source61069 import evaluate_case, locked_source


def fixture():
    R=torch.eye(96,dtype=torch.float64)[:,:10]
    V=torch.eye(96,dtype=torch.float64)[:,11:75]
    z=torch.zeros(96,dtype=torch.float64)
    z[0]=0.1
    z[10]=0.5
    H=torch.eye(96,dtype=torch.float64)
    H[11,10]=H[10,11]=0.3
    H[12,12]=0.25
    H[13,13]=1.5
    return z,R,V,H


class LevelsetPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_finite_levelset_roots_keep_geometry(self):
        z,R,V,H=fixture()
        rows=list(levelset_candidates(z,R,V,H,base=0,goal=10,seed=61064))
        self.assertEqual(len(rows),1+N_DIRECTIONS*len(S_VALUES))
        self.assertTrue(torch.equal(rows[0][1],z))
        self.assertTrue(all(idx==i for i,(idx,_,__) in enumerate(rows)))
        valid=[(idx,donor,meta) for idx,donor,meta in rows[1:]
               if donor is not None]
        self.assertGreaterEqual(len(valid),10)
        for idx,donor,meta in valid:
            self.assertEqual(meta["root_reason"],None)
            self.assertLessEqual(abs(curvature(donor,H)-curvature(z,H)),1e-11)
            self.assertAlmostEqual(float(donor.norm()),float(z.norm()),delta=1e-12)
            self.assertLessEqual(float(torch.linalg.vector_norm(R.T@(donor-z))),1e-12)
            self.assertLessEqual(float(torch.linalg.vector_norm(V.T@(donor-z))),0.2)
            self.assertLessEqual(float(torch.linalg.vector_norm(donor-z)),0.2*float(z.norm()))
            self.assertGreater(abs(meta["s"]),0)

    def test_deterministic_frozen_direction_stream(self):
        z,R,V,H=fixture()
        a=list(levelset_candidates(z,R,V,H,base=0,goal=10,seed=61064))
        b=list(levelset_candidates(z,R,V,H,base=0,goal=10,seed=61064))
        self.assertEqual(len(a),len(b))
        for (i,va,ma),(j,vb,mb) in zip(a,b):
            self.assertEqual(i,j)
            self.assertEqual(ma,mb)
            if va is None:
                self.assertIsNone(vb)
            else:
                self.assertTrue(torch.equal(va,vb))

    def test_zero_curvature_gradient_has_explicit_failures(self):
        z,R,V,_=fixture()
        H=torch.eye(96,dtype=torch.float64)
        rows=list(levelset_candidates(z,R,V,H,base=0,goal=10,seed=61064))
        self.assertEqual(len(rows),33)
        self.assertTrue(torch.equal(rows[0][1],z))
        self.assertTrue(all(value is None and entry["root_reason"]==
            "DEGENERATE_CURVATURE_TANGENT" for _,value,entry in rows[1:]))

    def test_curvature_bracket_can_be_infeasible(self):
        z,R,V,H=fixture()
        H[10,11]=H[11,10]=1e-9
        rows=list(levelset_candidates(z,R,V,H,base=0,goal=10,seed=61064))
        self.assertEqual(len(rows),33)
        self.assertTrue(any(v is None and d["root_reason"]=="CURVATURE_ROOT_UNBRACKETED"
                            for _,v,d in rows[1:]))

    def test_source_geometry_failures_are_explicit(self):
        z,R,V,H=fixture()
        bad=H.clone();bad[10,11]=.9;bad[11,10]=0.
        with self.assertRaisesRegex(ValueError,"nonsymmetric"):
            prepare_levelset(z,R,V,bad)
        with self.assertRaisesRegex(ValueError,"overlaps"):
            prepare_levelset(z,R,torch.eye(96,dtype=torch.float64)[:,10:74],H)
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            z[0]=float("nan")
            prepare_levelset(z,R,V,H)

    def test_rehashed_donor_candidate_corruption_rejected(self):
        z,R,V,H=fixture()
        prefix="case_00_00__"
        h32=np.zeros((1,81,48),dtype=np.float32)
        original=z.numpy().copy()
        original[0]=0.125  # Exactly representable in FP32 for the corruption control
        Q=float(original@H.numpy()@original)
        # Deliberately fail physical-dense curvature so the HVP receipt is
        # absent, as in a normal early donor-rejection trace.
        target=Q+1e-4
        cap={prefix+"repair__attempt_00__intended":original.copy(),
             prefix+"repair__attempt_00__actual":original.astype(np.float32).astype(np.float64),
             prefix+"repair__attempt_00__probe32":original.astype(np.float32)}
        entry={"index":0,"kind":"natural","s":0.,"t":0.,
            "root_reason":None,"passed":False,"reasons":["DONOR_PHYSICAL_DENSE_CURVATURE"],
            "relative_distance":0.,"dose_error":0.,"k_error":0.,
            "signed_j_error":0.,"intended_levelset_error":0.,
            "physical_dense_curvature_error":1e-4}
        case={"goal":0,"donor_attempts":[entry]}
        K=np.eye(96)
        J=np.eye(96)[:10]
        errors=[]
        check=lambda cond,msg: errors.append(msg) if not cond else None
        # Signed J source error uses 0 for natural candidate.
        check_levelset_donors(case,cap,prefix,original,target,h32,K,J,
                              R.numpy(),H.numpy(),check)
        self.assertEqual(errors,[])
        cap[prefix+"repair__attempt_00__intended"]=original*1.2
        check_levelset_donors(case,cap,prefix,original,target,h32,K,J,
                              R.numpy(),H.numpy(),check)
        self.assertGreater(len(errors),0)

    def test_first_generation_guard_fails_closed(self):
        with self.assertRaises(ValueError):
            locked_source("61069")
        with self.assertRaises(ValueError):
            locked_source("0"*40)

    def test_exposed_61064_source_geometry_and_numpy_endpoint(self):
        torch.manual_seed(61064)
        model32=TinyRecursiveModel().float().eval().requires_grad_(False)
        model64=copy.deepcopy(model32).double()
        x,_=fixed_data(1,61064)
        with torch.no_grad():
            h32=model32.reason_states(x)[2]
            h64=model64.reason_states(x)[2]
        hp=full_head_kernel(model64.head.weight).projector
        arrays={}
        case=evaluate_case(model32,model64,h32,h64,x,hp,
            QuadraticObserver(),torch.zeros(48,dtype=torch.float64),
            torch.ones(48,dtype=torch.float64),0,0,arrays,teacher_seed=61064)
        self.assertTrue(case["source_eligible"],case.get("exception"))
        self.assertTrue(case.get("geometry_eligible"),case.get("geometry_reason"))
        self.assertEqual(set(case["arms"]),{
            "frozen_baseline","fixed_donor_oracle_filter","levelset_repair"})
        self.assertIn("case_00_00__common__natural_hessian",arrays)
        self.assertIn("case_00_00__common__natural_complement",arrays)
        self.assertGreater(len(case["donor_attempts"]),0)
        self.assertLessEqual(len(case["donor_attempts"]),33)
        self.assertTrue(all(entry["index"]==i for i,entry in
                            enumerate(case["donor_attempts"])))
        for name,arm in case["arms"].items():
            if not arm["complete"]:
                self.assertFalse(arm["native_fp32_ready"])
                continue
            self.assertEqual(arm["n_shams"],8)
            self.assertEqual(len(arm["native"]["arms"]),9)
            p="case_00_00__"+name+"__"
            intended=np.concatenate([arrays[p+"selected"].reshape(1,96),
                                     arrays[p+"shams"]],axis=0)
            failures=[]
            def check(x,msg):
                if not x: failures.append(msg)
            result=audit_native(
                arm["native"],lambda key: arrays[p+"native__"+key],
                check,target=case["target_curvature_original"],
                intended=intended,h32=h32.numpy(),h64=h64.numpy(),
                reference_j=arrays["case_00_00__common__reference_jacobian64"],
                head_projector=hp.numpy(),head=model64.head.weight.numpy())
            self.assertEqual(failures,[])
            self.assertEqual(result["native_ready"],arm["native_fp32_ready"])

if __name__ == "__main__":
    unittest.main()
