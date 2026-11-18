"""Source-free paired receipt, corruption, and pre-generation guard checks.

The toy is linear with an exactly zero Hessian. Proposal/source seed 61064
is already exposed. These tests never initialize or generate seed 61066.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import fixed_data, tensor_int64_sha
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope13b.spectral_match import make_spectral_shams
from mechinterp.loopscope13c import source_feasibility as pilot
from mechinterp.loopscope13c.audit_receipt import audit, audit_candidates, audit_geometry
from mechinterp.loopscope13c.precision_oracle import dense_margin_hessian
from mechinterp.loopscope13c.roundtrip_match import PhysicalGate, make_roundtrip_shams


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.zeros((10, 48), dtype=torch.float32)
        weight[:, 38:] = torch.eye(10, dtype=torch.float32)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, state):
        return state @ self.weight.T


class _Shift(torch.nn.Module):
    def forward(self, state):
        return torch.roll(state, shifts=7, dims=-1)


class _LinearToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = _Head()
        self.block = _Shift()


def _toy_case(base=0, goal=0, *, dense_valid=True, fail_first_native=False, fail_first_primary=False):
    dtype = torch.float64
    model32 = _LinearToy().eval()
    model64 = copy.deepcopy(model32).double()
    h32 = torch.zeros((1, 81, 48), dtype=torch.float32)
    h32[:, :, 0] = 1.
    h64 = h32.double()
    hp = torch.eye(48, dtype=dtype)
    hp[38:, 38:] = 0.
    pair_p = torch.block_diag(hp, hp)
    g = h64[0, [(goal - 1) % 81, goal]].reshape(96)
    K = pair_p - torch.outer(g, g) / (g @ g)
    R = torch.eye(96, dtype=dtype)[:, 72:82]
    selected = torch.zeros(96, dtype=dtype)
    selected[1], selected[72] = 1., .2
    data, labels = fixed_data(8, 61064)
    donor, metadata = legal_predecessor_donor(data[base], goal)
    prefix = f"case_{base:02d}_{goal:02d}__"
    arrays = {"source_inputs": data.numpy(), "source_labels": labels.numpy(),
              "head_weight": model64.head.weight.detach().numpy(),
              "head_weight_fp32": model32.head.weight.detach().numpy(), "head_projector": hp.numpy(),
              f"base_{base:02d}__hrec_fp64": h64.numpy(), f"base_{base:02d}__hrec_fp32": h32.numpy(),
              prefix + "donor_input": donor.numpy()}
    counts = {key: 0 for key in ("reference_construction", "dense_primary", "baseline_final_fp64",
                                "baseline_final_fp32", "roundtrip_final_fp64", "roundtrip_final_fp32")}

    def hvp(vector, category="reference_construction"):
        counts[category] += 1
        if fail_first_primary and category == "baseline_final_fp64" and counts[category] == 1:
            raise ArithmeticError("synthetic primary oracle failure")
        return torch.zeros_like(vector)

    reference = make_spectral_shams(selected, R, lambda v: K @ v, hvp,
        teacher_seed=61064, base_index=base, goal=goal)
    if not reference.eligible:
        raise AssertionError(reference.reason)
    dense, dense_receipt, dense_raw = dense_margin_hessian(lambda v: hvp(v, "dense_primary"))
    shared = {"selected": selected, "hpair": g, "hessian": dense, "dense_passed": dense_valid,
              "k": K, "jacobian": R.T, "head": model64.head.weight.detach(), "egrad": g,
              "row_basis": R, "q0": reference.raw["q0"], "target": 0.}
    gate = PhysicalGate(selected=selected, hpair=g, hessian=dense, k_projector=K, jacobian=R.T,
                        head=shared["head"], egrad=g, row_basis=R, q0=shared["q0"], target_curvature=0.)
    new = make_roundtrip_shams(reference, gate, teacher_seed=61064, base_index=base, goal=goal)
    case = {"base": base, "goal": goal, "raw_prefix": prefix, "case_reason": None,
            "donor_metadata": metadata, "k_dimension": 75, "rank_future_j": 10,
            "n_full_j_dark": 65, "reference": reference.receipt, "dense_primary": dense_receipt,
            "arms": {arm: pilot._initial_arm() for arm in pilot.ARMS}}
    pilot._capture_raw(arrays, prefix + "reference__", reference.raw)
    pilot._capture_raw(arrays, prefix, {"future_jacobian": R.T, "restricted_jacobian": R.T,
                        "energy_gradient": g, "natural_pair": selected, "hrec_pair": g,
                        "dense_hessian": dense, "dense_hessian_raw": dense_raw})
    real_native = pilot.native_precision_check
    call_count = 0

    def native(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if fail_first_native and call_count == 1:
            raise ArithmeticError("synthetic baseline native failure")
        return real_native(*args, **kwargs)

    with mock.patch.object(pilot, "native_precision_check", side_effect=native):
        pilot._validate_captured_arms({"baseline": reference, "roundtrip": new}, shared=shared,
            case=case, arrays=arrays, counted_hvp=hvp, model32=model32, model64=model64,
            h32=h32, h64=h64, metadata=metadata, head_projector=hp)
    case["hvp_counts"] = counts
    return case, arrays


def _write_fixture(folder, cases, arrays):
    raw, receipt = Path(folder) / "raw.npz", Path(folder) / "receipt.json"
    np.savez_compressed(raw, **arrays)
    summaries, paired = pilot._summaries(cases)
    x, y = arrays["source_inputs"], arrays["source_labels"]
    document = {
        "excluded_source_seed": 61064, "excluded_model_seed": 61064,
        "n_excluded_bases": 8, "n_goals_per_base": 8, "n_cases": 64,
        "n_untrained_model_initializations": 1, "n_new_trained_model_roots": 0,
        "registered_min_admitted": 61, "new_test_generated": False,
        "trained_weights_loaded": False, "edited_task_outcomes_scored": False,
        "native_gate_used_in_proposal_search": False, "paired_proposal_seed_namespace": "LGA10B",
        "source_inputs_sha256": hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest(),
        "source_labels_sha256": hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest(),
        "arms": summaries, "paired": paired, "cases": cases,
        "hvp_counts": {key: sum(c["hvp_counts"][key] for c in cases) for key in cases[0]["hvp_counts"]},
        "native_hvp_counts": {key: sum(c["arms"][arm].get("native", {}).get("hvp_counts", {}).get(key, 0)
                                      for c in cases for arm in pilot.ARMS)
                              for key in ("native32", "promoted_exact", "promoted_probe")},
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }
    receipt.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
    receipt.with_suffix(".json.sha256").write_text(hashlib.sha256(receipt.read_bytes()).hexdigest() + "  receipt.json\n")
    return receipt, raw


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_paired_receipt_and_rehashed_tampering(self):
        cases, arrays = [], {}
        for base in range(8):
            for goal in pilot.GOALS:
                case, values = _toy_case(base, goal)
                cases.append(case)
                arrays.update(values)
        with tempfile.TemporaryDirectory(prefix="lga10c-synthetic-") as folder:
            receipt, raw = _write_fixture(folder, cases, arrays)
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit(receipt, raw, expected_seed=61064)
            self.assertEqual(result["audit_status"], "PASS", result["failures"][:20])
            self.assertEqual(result["n_cases_audit_passed"], 64)
            key = "case_00_00__roundtrip__candidate_actual32"
            arrays[key] = arrays[key].copy()
            arrays[key][0, 9] += .001
            key = "case_00_10__roundtrip__native__hvp32"
            arrays[key] = arrays[key].copy()
            arrays[key][1, 1] += .5
            receipt, raw = _write_fixture(folder, cases, arrays)
            with contextlib.redirect_stdout(io.StringIO()):
                damaged = audit(receipt, raw, expected_seed=61064)
            self.assertEqual(damaged["audit_status"], "FAIL")
            self.assertTrue(any("candidate FP32 addition differs" in e for e in damaged["failures"]))
            self.assertTrue(any("native aggregate" in e for e in damaged["failures"]))

    def test_native_exception_preserves_both_selections_and_other_arm(self):
        case, arrays = _toy_case(fail_first_native=True)
        baseline, new = case["arms"]["baseline"], case["arms"]["roundtrip"]
        self.assertTrue(baseline["primary_eligible"])
        self.assertFalse(baseline["native_ready"])
        self.assertEqual(baseline["native_reason"], "EXPLICIT_NATIVE_VALIDATION_EXCEPTION")
        self.assertTrue(new["primary_eligible"] and new["native_ready"] and new["overall_ready"])
        self.assertEqual(len(arrays["case_00_00__roundtrip__candidate_intended"]), 8)

    def test_dense_invalidity_blocks_both_primary_arms(self):
        case, _ = _toy_case(dense_valid=False)
        for arm in pilot.ARMS:
            self.assertFalse(case["arms"][arm]["primary_eligible"])
            self.assertEqual(case["arms"][arm]["primary_reason"], "DENSE_HESSIAN_SYMMETRY_FAILURE")
            self.assertTrue(case["arms"][arm]["native_ready"])

    def test_primary_exception_still_measures_same_arm_native(self):
        case, arrays = _toy_case(fail_first_primary=True)
        baseline, new = case["arms"]["baseline"], case["arms"]["roundtrip"]
        self.assertFalse(baseline["primary_eligible"])
        self.assertTrue(baseline["native_ready"])
        self.assertEqual(baseline["primary_reason"], "EXPLICIT_PRIMARY_VALIDATION_EXCEPTION")
        self.assertIn("case_00_00__baseline__native__hvp32", arrays)
        self.assertTrue(new["overall_ready"])

    def test_truncated_candidate_log_cannot_claim_budget_exhaustion(self):
        case, arrays = _toy_case()
        prefix = case["raw_prefix"]
        problems = []
        check = lambda okay, message: problems.append(message) if not okay else None
        geometry = audit_geometry(case, lambda key: arrays[prefix + key], check,
                                  head_projector=arrays["head_projector"], head=arrays["head_weight"])
        selection = case["arms"]["roundtrip"]["selection"]
        selection["attempts"] = selection["attempts"][:1]
        selection["n_shams"] = 1
        selection["candidate_evaluations"] = 1
        selection["eligible"] = False
        selection["reason"] = "ROUNDED_CANDIDATE_BUDGET_EXHAUSTED"
        for key in ("attempt_start", "candidate_start", "candidate_intended", "candidate_actual32", "candidate_hvp_dense", "shams"):
            name = prefix + "roundtrip__" + key
            arrays[name] = arrays[name][:1]
        def get(key):
            return arrays[prefix + ("baseline__shams" if key == "baseline_shams" else "roundtrip__" + key)]
        audit_candidates(case, get, check, geometry=geometry, hpair=arrays[prefix + "hrec_pair"],
                         head=arrays["head_weight"], dense=arrays[prefix + "dense_hessian"], expected_seed=61064)
        self.assertTrue(any("budget exhaustion ended before" in message for message in problems))
        selection.update({"attempts": [], "n_shams": 0, "candidate_evaluations": 0,
                          "reason": "GRID_NO_ATTAINABLE_TARGET"})
        for key in ("attempt_start", "candidate_start", "candidate_intended", "candidate_actual32", "candidate_hvp_dense", "shams"):
            name = prefix + "roundtrip__" + key
            arrays[name] = arrays[name][:0]
        problems.clear()
        audit_candidates(case, get, check, geometry=geometry, hpair=arrays[prefix + "hrec_pair"],
                         head=arrays["head_weight"], dense=arrays[prefix + "dense_hessian"], expected_seed=61064)
        self.assertTrue(any("unverified shared termination" in message for message in problems))

    def test_reported_first_order_failure_is_audited_as_failure(self):
        case, arrays = _toy_case()
        prefix = case["raw_prefix"]
        K = arrays[prefix + "reference__k_projector"].copy()
        K[0, 0] += .1
        arrays[prefix + "reference__k_projector"] = K
        case["reference"]["reason"] = "INVALID_FIRST_ORDER_GEOMETRY"
        case["reference"]["eligible"] = False
        problems = []
        check = lambda okay, message: problems.append(message) if not okay else None
        result = audit_geometry(case, lambda key: arrays[prefix + key], check,
                                head_projector=arrays["head_projector"], head=arrays["head_weight"])
        self.assertEqual(problems, [])
        self.assertEqual(result["shared_geometry_failure_verified"], "INVALID_FIRST_ORDER_GEOMETRY")

    def test_uncommitted_code_blocks_before_rng(self):
        with tempfile.TemporaryDirectory(prefix="lga10c-provenance-") as folder:
            root = Path(folder)
            for relative in pilot.REQUIRED_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic committed guard fixture\n")

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT, text=True).strip()

            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic guard fixture")
            head = git("rev-parse", "HEAD")
            fake_file = root / "mechinterp/loopscope13c/source_feasibility.py"
            with mock.patch.object(pilot, "__file__", str(fake_file)):
                self.assertEqual(pilot._provenance(head)["freeze_commit"], head)
                extra = root / "mechinterp/loopscope13c/uncommitted.py"
                extra.write_text("# uncommitted dependency\n")
                for staged in (False, True):
                    if staged:
                        git("add", str(extra.relative_to(root)))
                    with self.subTest(staged=staged), mock.patch.object(pilot.torch, "manual_seed") as seed, \
                         mock.patch.object(pilot, "fixed_data") as data:
                        with self.assertRaises(subprocess.CalledProcessError):
                            pilot.run(root / "output.json", root / "raw.npz", freeze_commit=head)
                        seed.assert_not_called()
                        data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
