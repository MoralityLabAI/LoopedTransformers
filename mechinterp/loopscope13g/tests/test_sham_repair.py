"""Pre-source LGA-10G sham-level-set tests; only exposed model seed 61064."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope13b.source_feasibility import QuadraticObserver
from mechinterp.loopscope13c.audit_receipt import audit_native
from mechinterp.loopscope13g.source61070 import evaluate_case, locked_source
import mechinterp.loopscope13g.sham_repair as sham_module
from mechinterp.loopscope13b.spectral_match import make_spectral_shams
from mechinterp.loopscope13g.sham_repair import (
    FALLBACK_PARENTS, LOCAL_DIRECTIONS, OFFSET_GRID,
    correction_seed, sham_corrections, _rank_original_gate,
)
from mechinterp.loopscope13g.audit61070 import audit_sham_repairs


def synthetic_fiber():
    R=torch.eye(96,dtype=torch.float64)[:,:10]
    V=torch.eye(96,dtype=torch.float64)[:,11:75]
    z0=torch.zeros(96,dtype=torch.float64);z0[0]=0.1
    q0=torch.zeros(96,dtype=torch.float64);q0[10]=1.
    H=torch.eye(96,dtype=torch.float64)
    H[11,12]=H[12,11]=0.2
    reference=SimpleNamespace(
        raw={"z0":z0,"q0":q0,"complement_basis":V,"row_basis":R},
        receipt={"null_radius":1.,"selected_curvature":float((z0+q0)@(H@(z0+q0)))},
    )
    c=0.35
    parent=z0+c*q0+math.sqrt(1.-c*c)*V[:,0]
    return reference,parent,H


class ShamLevelsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytic_sham_correction_stays_on_same_curvature_and_cosine(self):
        ref,parent,H=synthetic_fiber()
        candidates=list(sham_corrections(
            parent,ref,H,teacher_seed=61064,base_index=0,
            goal=10,sham_index=0,parent_candidate_index=12))
        self.assertEqual(len(candidates),LOCAL_DIRECTIONS*len(OFFSET_GRID))
        admitted=0
        for cand,receipt in candidates:
            self.assertIn(receipt["s"],OFFSET_GRID)
            self.assertIn(receipt["j"],(0,1))
            if cand is None:
                continue
            admitted+=1
            self.assertIsNone(receipt["reason"])
            w=receipt["_axis"]
            self.assertEqual(tuple(w.shape),(64,))
            self.assertAlmostEqual(float(w.norm()),1.,delta=1e-10)
            self.assertAlmostEqual(float(cand.norm()),float(parent.norm()),delta=1e-10)
            r=float(ref.receipt["null_radius"])
            q0=ref.raw["q0"]
            c=float((cand-ref.raw["z0"])@q0)/(r*r)
            self.assertAlmostEqual(c,0.35,delta=1e-10)
            Q=float(cand@(H@cand))
            self.assertLessEqual(abs(Q-ref.receipt["selected_curvature"]),1e-10)
        self.assertGreaterEqual(admitted,6)

    def test_unbracketed_curvature_roots_are_preserved(self):
        ref,parent,H=synthetic_fiber()
        ref.receipt["selected_curvature"]+=10.0
        items=list(sham_corrections(parent,ref,H,
             teacher_seed=61064,base_index=1,goal=20,
             sham_index=0,parent_candidate_index=3))
        self.assertEqual(len(items),12)
        self.assertTrue(all(c is None and receipt["reason"] in (
            "LOCAL_ROOT_UNBRACKETED","LOCAL_ROOT_RESIDUAL")
            for c,receipt in items))

    def test_zero_tangent_gradient_fails_without_candidate(self):
        ref,parent,H=synthetic_fiber()
        H=torch.eye(96,dtype=torch.float64)
        items=list(sham_corrections(parent,ref,H,teacher_seed=61064,
            base_index=0,goal=0,sham_index=0,parent_candidate_index=0))
        self.assertEqual(len(items),12)
        self.assertTrue(all(c is None and r["reason"]==
            "DEGENERATE_LOCAL_CURVATURE_GRADIENT" for c,r in items))

    def test_score_is_signed_and_catches_scalar_probe_failure(self):
        gate=SimpleNamespace(donor_q32=0.,donor_q64=0.,tau=1e-8)
        receipt={"trial_qnative":2.5e-8,"trial_q64_actual":0.,
                 "trial_q32_probe":2.5e-8,"trial_q64_probe":0.}
        self.assertAlmostEqual(_rank_original_gate(receipt,gate),2.5,delta=1e-10)
        self.assertIsNone(_rank_original_gate({},gate))
        self.assertEqual(FALLBACK_PARENTS*LOCAL_DIRECTIONS*len(OFFSET_GRID),36)

    def test_proposal_seed_is_stable_and_source_specific(self):
        self.assertEqual(correction_seed(61064,1,10,0,4,1),
                         correction_seed(61064,1,10,0,4,1))
        self.assertNotEqual(correction_seed(61064,1,10,0,4,1),
                            correction_seed(61064,1,10,0,4,0))

    def test_bounded_fallback_is_exercised_and_uses_corrected_sham(self):
        """Analytic full selector: frozen proposals fail, one corrected root passes."""
        R=torch.eye(96,dtype=torch.float64)[:,:10]
        P=torch.zeros((96,96),dtype=torch.float64)
        P[torch.arange(75),torch.arange(75)]=1.
        z=torch.zeros(96,dtype=torch.float64)
        z[0]=.1;z[10]=1.
        hdiag=torch.ones(96,dtype=torch.float64)
        hdiag[11:75]=torch.linspace(.6,1.4,64,dtype=torch.float64)
        H=torch.diag(hdiag)
        def project(value):
            return (P@value.reshape(96)).reshape(2,48)
        reference=make_spectral_shams(
            z,R,project,lambda v:H@v.reshape(96),
            teacher_seed=61064,base_index=0,goal=0,
            expected_complement_dim=64)
        self.assertTrue(reference.eligible,reference.reason)

        class RejectThenAccept:
            def __init__(self):
                self.physical=SimpleNamespace(hessian=H)
                self.hessian=H
                self.donor_q32=0.
                self.donor_q64=0.
                self.tau=1e-8
                self.allow=False
                self.calls=0
            def __call__(self,candidate,accepted):
                self.calls+=1
                v=candidate.detach().double().reshape(96)
                record={
                    "passed":self.allow,
                    "failed_checks":[] if self.allow else ["FP32_NATIVE_PARITY"],
                    "trial_qnative":5e-8 if not self.allow else 0.,
                    "trial_q64_actual":0.,
                    "trial_q32_probe":5e-8 if not self.allow else 0.,
                    "trial_q64_probe":0.,
                }
                actual=v.float().double()
                return record,{"intended":v.clone(),"actual32":actual,
                               "dense_hvp":H@actual}
        gate=RejectThenAccept()
        original=sham_module.sham_corrections
        def enable_fallback(*args,**kwargs):
            gate.allow=True
            return original(*args,**kwargs)
        with patch.object(sham_module,"sham_corrections",
                          side_effect=enable_fallback):
            plan=sham_module.make_lattice_shams(
                reference,gate,teacher_seed=61064,
                base_index=0,goal=0,n_shams=8,max_attempts=32)
        self.assertTrue(plan.eligible,plan.reason)
        self.assertEqual(plan.receipt["n_rescued_shams"],1)
        self.assertEqual(len(plan.shams),8)
        rescues=[a for a in plan.receipt["rescue_attempts"] if a["accepted"]]
        self.assertEqual(len(rescues),1)
        ix=rescues[0]["candidate_index"]
        self.assertTrue(torch.equal(plan.shams[0].reshape(96),
                         plan.raw["rescue_candidate_intended"][ix]))
        self.assertEqual(plan.raw["rescue_root_vectors"].shape[1],96)
        self.assertEqual(plan.raw["rescue_root_axes"].shape[1],64)
        self.assertLessEqual(len(plan.receipt["rescue_attempts"]),36)

        # Independent NumPy auditor must reconstruct the corrected root,
        # rounded physical edit and its donor-sham parent from raw arrays.
        prefix="case_00_00__"
        arm_prefix=prefix+"levelset_sham_repair__"
        source={
            prefix+"common__natural_hessian":H.numpy(),
            arm_prefix+"selected":z.numpy(),
            arm_prefix+"shams":plan.raw["shams"].numpy(),
        }
        for key in ("z0","q0","complement_basis","row_basis"):
            source[arm_prefix+"repair_geometry__"+key]=reference.raw[key].numpy()
        for key in ("candidate_intended","rescue_root_vectors",
                    "rescue_root_axes","rescue_candidate_intended",
                    "rescue_candidate_actual","rescue_candidate_hvp_dense"):
            source[arm_prefix+"selection__"+key]=plan.raw[key].numpy()
        class Cap:
            def __contains__(self,k):return k in source
            def __getitem__(self,k):return source[k]
        case={
            "base":0,"goal":0,"source_seed":61064,
            "arms":{"levelset_sham_repair":{
                "selection_receipt":plan.receipt,
                "n_rescued_shams":1,
                "n_rescue_trials":len(plan.receipt["rescue_attempts"]),
                "filter_donor_scalars":{"qnative":0.,"qpromoted":0.,"q_tolerance":1e-8},
            }},
        }
        errors=[]
        checked=audit_sham_repairs(case,Cap(),prefix,
            np.zeros((1,81,48),dtype=np.float32),
            np.zeros((10,48),dtype=np.float64),
            lambda v,msg: errors.append(msg) if not v else None)
        self.assertEqual(errors,[])
        self.assertEqual(checked,1)


    def test_auditor_source_does_not_import_model_or_torch(self):
        path=Path(__file__).resolve().parents[1]/"audit61070.py"
        ast_tree=ast.parse(path.read_text())
        names=[]
        for node in ast.walk(ast_tree):
            if isinstance(node,ast.Import):
                names.extend(t.name for t in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:
                names.append(node.module)
        self.assertFalse(any(x=="torch" or x.startswith("torch.")
                             or x.endswith("source61070")
                             or x.endswith("sham_repair") for x in names))

    def test_freeze_gate_fails_closed(self):
        with self.assertRaises(ValueError):
            locked_source("61070")
        with self.assertRaises(ValueError):
            locked_source("0"*40)

    def test_exposed_61064_four_arm_selection_and_native_numpy_gate(self):
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
            "frozen_baseline","fixed_donor_oracle_filter",
            "levelset_repair","levelset_sham_repair"})
        fixed=case["arms"]["levelset_repair"]
        repaired=case["arms"]["levelset_sham_repair"]
        if fixed["complete"]:
            self.assertTrue(case.get("sham_repair_reused_complete_original"))
            self.assertTrue(repaired["complete"])
            self.assertEqual(repaired["n_rescued_shams"],0)
        if "case_00_00__levelset_sham_repair__selected" in arrays:
            class Cap:
                def __contains__(self,k):return k in arrays
                def __getitem__(self,k):return arrays[k]
            errors=[]
            audit_sham_repairs(case,Cap(),"case_00_00__",h32.numpy(),
                               model64.head.weight.detach().numpy(),
                               lambda v,msg: errors.append(msg) if not v else None)
            self.assertEqual(errors,[])
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
            result=audit_native(
                arm["native"],lambda key: arrays[p+"native__"+key],
                lambda v,msg: failures.append(msg) if not v else None,
                target=case["target_curvature_original"],
                intended=intended,h32=h32.numpy(),h64=h64.numpy(),
                reference_j=arrays["case_00_00__common__reference_jacobian64"],
                head_projector=hp.numpy(),head=model64.head.weight.numpy())
            self.assertEqual(failures,[])
            self.assertEqual(result["native_ready"],arm["native_fp32_ready"])

if __name__=="__main__":
    unittest.main()
