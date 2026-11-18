"""Source-free paired capture, semantic corruption and frozen-source guards.

The linear toy has an exact zero Hessian and uses exposed source 61064.
No test initializes or generates the prospective 61067 source.
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

from mechinterp.loopscope04.gradient_flow import fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope11.pair_fiber import project_into_k
from mechinterp.loopscope13c.tests.test_receipt import _LinearToy
from mechinterp.loopscope13d import source_feasibility as pilot
from mechinterp.loopscope13d.audit_receipt import audit, audit_witness, _audit_case
from mechinterp.loopscope13d.capture import write_capture


def _toy_case(base=0, goal=0, *, fail_selection=False, fail_promoted=False, fail_native=False,
              invalidate_native_dense=False, real_model=False):
    data, labels = fixed_data(8, 61064)
    donor, metadata = legal_predecessor_donor(data[base], goal)
    model32 = _LinearToy().eval()
    model64 = copy.deepcopy(model32).double()
    h32 = torch.zeros((1, 81, 48), dtype=torch.float32)
    h32[:, :, 0] = 1.
    h64 = h32.double()
    h64[:, :, 2] += 2. ** -20
    hdon32 = h32.clone()
    hdon32[0, (goal - 1) % 81, 1] += 1.
    hdon32[0, goal, 24] += .2
    hp = torch.eye(48, dtype=torch.float64)
    hp[38:, 38:] = 0.
    if real_model:
        with torch.random.fork_rng():
            torch.manual_seed(61064)
            model32 = pilot.TinyRecursiveModel().float().eval().requires_grad_(False)
        model64 = copy.deepcopy(model32).double()
        with torch.no_grad():
            h32 = model32.reason_states(data[base:base + 1])[2]
            h64 = model64.reason_states(data[base:base + 1])[2]
            hdon32 = model32.reason_states(donor[None])[2]
        hp = pilot.full_head_kernel(model64.head.weight).projector
    parent = pilot.build_common_native_parent(model64, pilot.QuadraticObserver(), h32, hdon32,
        torch.zeros(48, dtype=torch.float64), torch.ones(48, dtype=torch.float64),
        goal=goal, original_class=metadata["original_goal_label"], donor_class=metadata["donor_goal_label"],
        teacher_seed=61064, base_index=base, head_projector=hp)
    if not parent["eligible"]:
        raise AssertionError(parent["reason"])
    selected = parent["directions"]["pair_energy_tangent"].reshape(96)
    target_hvp = pilot.native_margin_hvp(model64, h32.double(), goal,
        metadata["donor_goal_label"], metadata["original_goal_label"])(selected)
    target = float(selected @ target_hvp)
    prefix = f"case_{base:02d}_{goal:02d}__"
    case = {"base": base, "goal": goal, "raw_prefix": prefix, "case_reason": None,
        "donor_metadata": metadata, "k_dimension": 75, "radius": parent["radius"],
        "target_curvature_native": target, "common_target_hvp_count": 1, "common_target_hvp_attempt_count": 1,
        "common_parent": {"eligible": True, "reason": None, "native_source": parent["native_source"],
            "clean_future_margin": parent["clean_future_margin"],
            "selected_signed_derivative": parent["selected_signed_derivative"],
            "pair_positions": parent["pair_positions"], "natural_radius": parent["radius"]},
        "arms": {arm: pilot._initial_arm() for arm in pilot.ARMS}}
    arrays = {"source_inputs": data.numpy(), "source_labels": labels.numpy(),
        "head_weight": model64.head.weight.detach().numpy(), "head_weight_fp32": model32.head.weight.detach().numpy(),
        "head_projector": hp.numpy(), f"base_{base:02d}__hrec_fp64": h64.numpy(),
        f"base_{base:02d}__hrec_fp32": h32.numpy(), prefix + "donor_input": donor.numpy(),
        prefix + "hdon_fp32": hdon32.numpy()}
    K = torch.stack([project_into_k(axis.reshape(2, 48), parent["fiber"], hp).reshape(96)
                     for axis in torch.eye(96, dtype=torch.float64)], dim=1)
    pilot._capture(arrays, prefix + "common__", {"selected": selected,
        "natural_pair": parent["fiber"].donor.reshape(96), "energy_gradient": parent["fiber"].egrad.reshape(96),
        "k_projector": K, "application_pair64": h32[0, [(goal - 1) % 81, goal]].double().reshape(96),
        "future_gradient": parent["future_gradient"].reshape(96), "future_k_gradient": parent["future_k_gradient"].reshape(96),
        "target_hvp_native": target_hvp})
    pilot._capture(arrays, prefix + "common__parent_direction__", parent["directions"])
    geometries = {arm: pilot.build_geometry(model64, state, h32, parent, hp, teacher_seed=61064,
        base_index=base, goal=goal, original_class=metadata["original_goal_label"], donor_class=metadata["donor_goal_label"])
        for arm, state in ((pilot.ARMS[0], h64), (pilot.ARMS[1], h32.double()))}
    if invalidate_native_dense:
        geometries[pilot.ARMS[1]].dense_receipt["passed"] = False
    counts = {"selection": 0, "promoted": 0, "native": 0}
    originals = {"selection": pilot._selection_validation, "promoted": pilot._promoted_validation,
                 "native": pilot.native_precision_check}
    def endpoint(name, should_fail, *args, **kwargs):
        counts[name] += 1
        # Every endpoint sees both selected captures, even if it raises first.
        for arm in pilot.ARMS:
            if prefix + arm + "__selection__shams" not in arrays:
                raise AssertionError("endpoint ran before both selections were archived")
        if should_fail and counts[name] == 1:
            raise ArithmeticError("synthetic " + name + " failure")
        return originals[name](*args, **kwargs)
    with mock.patch.object(pilot, "_selection_validation", side_effect=lambda *a, **kw: endpoint("selection", fail_selection, *a, **kw)), \
         mock.patch.object(pilot, "_promoted_validation", side_effect=lambda *a, **kw: endpoint("promoted", fail_promoted, *a, **kw)), \
         mock.patch.object(pilot, "native_precision_check", side_effect=lambda *a, **kw: endpoint("native", fail_native, *a, **kw)):
        pilot._evaluate_selected_arms(geometries, case=case, arrays=arrays, model32=model32, model64=model64,
            h32=h32, h64=h64, selected=selected, target=target, head_projector=hp, metadata=metadata)
    return case, arrays


def _write_fixture(folder, cases, arrays):
    raw, receipt = Path(folder) / "raw.npz", Path(folder) / "receipt.json"
    if raw.exists():
        raw.unlink()
    capture = write_capture(raw, arrays)
    summaries, paired = pilot._summaries(cases)
    document = {"excluded_source_seed": 61064, "excluded_model_seed": 61064,
        "n_excluded_bases": 8, "n_goals_per_base": 8, "n_cases": 64,
        "n_untrained_model_initializations": 1, "n_new_trained_model_roots": 0,
        "registered_min_admitted": 61, "common_native_parent": True, "new_test_generated": False,
        "trained_weights_loaded": False, "edited_task_outcomes_scored": False, "trained_release_authorized": False,
        "precision_endpoints_used_in_proposal_search": False, "paired_proposal_seed_namespace": "LGA10B",
        "source_inputs_sha256": hashlib.sha256(np.asarray(arrays["source_inputs"], dtype="<i8").tobytes()).hexdigest(),
        "source_labels_sha256": hashlib.sha256(np.asarray(arrays["source_labels"], dtype="<i8").tobytes()).hexdigest(),
        "arms": summaries, "paired": paired, "cases": cases, **pilot._measurement_counts(cases),
        "donor_precision_ceiling": pilot._donor_precision_ceiling(cases),
        "capture": capture, "raw_sha256": capture["raw_sha256"]}
    receipt.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
    receipt.with_suffix(".json.sha256").write_text(hashlib.sha256(receipt.read_bytes()).hexdigest() + "  receipt.json\n")
    return receipt, raw


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_capture_and_rehashed_semantic_corruption(self):
        cases, arrays = [], {}
        for base in range(8):
            for goal in pilot.GOALS:
                case, values = _toy_case(base, goal)
                cases.append(case)
                arrays.update(values)
        self.assertTrue(all(case["arms"][arm]["promoted_ready"] and case["arms"][arm]["native_fp32_ready"]
                            for case in cases for arm in pilot.ARMS))
        self.assertEqual(sum(pilot._measurement_counts(cases)["hvp_counts"].values()), 27712)
        self.assertEqual(pilot._measurement_counts(cases)["independent_first_gradient_count"], 6912)
        with tempfile.TemporaryDirectory(prefix="lga10d-toy-") as folder:
            receipt, raw = _write_fixture(folder, cases, arrays)
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit(receipt, raw, expected_seed=61064)
            self.assertEqual(result["audit_status"], "PASS", result["failures"][:20])
            self.assertEqual(result["n_cases_audit_passed"], 64)
            mutations = {"case_00_00__common__natural_pair": (1, .01),
                "case_00_10__reference_derivatives__selection__candidate_actual32": ((0, 9), .01),
                "case_00_20__native_derivatives__promoted__pair_gradients": ((1, 2, 0, 1), .01),
                "case_00_30__reference_derivatives__native__hvp32": ((1, 1), .5)}
            for key, (index, value) in mutations.items():
                arrays[key] = arrays[key].copy()
                arrays[key][index] += value
            key = "case_00_40__native_derivatives__promoted__pair_gradients"
            arrays[key] = arrays[key].astype(np.float32)
            row = cases[5]["arms"][pilot.ARMS[1]]
            row["geometry"]["hvp_counts"]["reference_construction"] = 65
            row["reference"]["construction_hvp_count"] = 65
            receipt, raw = _write_fixture(folder, cases, arrays)
            with contextlib.redirect_stdout(io.StringIO()):
                damaged = audit(receipt, raw, expected_seed=61064)
            self.assertEqual(damaged["audit_status"], "FAIL")
            for diagnostic in ("native source subtraction", "candidate FP32 addition", "Richardson reconstruction", "native aggregate",
                               "promoted witness dtype", "spectral action capture"):
                self.assertTrue(any(diagnostic in failure for failure in damaged["failures"]), diagnostic)

    def test_each_endpoint_exception_keeps_other_measurements(self):
        for endpoint in ("selection", "promoted", "native"):
            with self.subTest(endpoint=endpoint):
                case, arrays = _toy_case(**{"fail_" + endpoint: True})
                first, second = [case["arms"][arm] for arm in pilot.ARMS]
                field = {"selection": "selection_validated", "promoted": "promoted_ready", "native": "native_fp32_ready"}[endpoint]
                self.assertFalse(first[field])
                for other in set(("selection_validated", "promoted_ready", "native_fp32_ready")) - {field}:
                    self.assertTrue(first[other], other)
                self.assertTrue(all(second[key] for key in ("selection_validated", "promoted_ready", "native_fp32_ready")))
                self.assertIn("case_00_00__native_derivatives__selection__shams", arrays)
                if endpoint != "selection":
                    counts = pilot._measurement_counts([case])
                    self.assertFalse(counts["call_counts_complete"])
                    self.assertEqual(counts["incomplete_call_count_endpoints"][0]["endpoint"], endpoint)

    def test_native_dense_failure_keeps_native_fp32_endpoint_independent(self):
        case, _ = _toy_case(invalidate_native_dense=True)
        for arm in pilot.ARMS:
            self.assertFalse(case["arms"][arm]["promoted_ready"])
            self.assertTrue(case["arms"][arm]["native_fp32_ready"])

    def test_gradient_witness_detects_rehashed_cached_oracle_bias(self):
        case, arrays = _toy_case()
        prefix = "case_00_00__native_derivatives__promoted__"
        witness = copy.deepcopy(case["arms"][pilot.ARMS[1]]["promoted"]["witness"])
        actual = arrays[prefix + "actual_edits64"]
        # Bias a cache-derived HVP while preserving all witness gradients.
        arrays[prefix + "fresh_hvps"] = 1e-5 * actual
        failures = []
        passed, _, _ = audit_witness(witness, lambda key: arrays[prefix + key],
            lambda okay, message: failures.append(message) if not okay else None,
            intended=arrays[prefix + "intended_edits"], h32=arrays["base_00__hrec_fp32"], goal=0, target=0.)
        self.assertFalse(passed)
        self.assertTrue(any("promoted witness gate" in failure for failure in failures))

    def test_exposed_real_model_case_has_coherent_capture(self):
        case, arrays = _toy_case(real_model=True)
        failures = []
        _audit_case(case, arrays, lambda okay, message: failures.append(message) if not okay else None,
            x=arrays["source_inputs"], hp=arrays["head_projector"], head=arrays["head_weight"], expected_seed=61064)
        self.assertEqual(failures, [])
        self.assertTrue(all("selection" in case["arms"][arm] for arm in pilot.ARMS))

    def test_auditor_import_has_no_model_matcher_or_torch(self):
        root = Path(pilot.__file__).resolve().parents[2]
        program = ("import sys; import mechinterp.loopscope13d.audit_receipt; "
                   "assert 'torch' not in sys.modules; "
                   "assert 'mechinterp.loopscope13d.geometry' not in sys.modules; "
                   "assert 'mechinterp.loopscope13c.precision_oracle' not in sys.modules")
        subprocess.run([__import__("sys").executable, "-c", program], cwd=root, check=True, capture_output=True)

    def test_common_target_exception_preserves_an_auditable_paired_failure(self):
        case, arrays = _toy_case()
        case.update({"common_target_hvp_count": 0, "case_reason": "EXPLICIT_COMMON_CASE_EXCEPTION",
                     "exception": {"type": "ArithmeticError", "message": "synthetic target failure"}})
        case.pop("target_curvature_native")
        case["arms"] = {arm: pilot._initial_arm() for arm in pilot.ARMS}
        for arm in pilot.ARMS:
            case["arms"][arm]["selection_reason"] = case["case_reason"]
        arrays = {key: value for key, value in arrays.items()
                  if not any("__" + arm + "__" in key for arm in pilot.ARMS)
                  and key != "case_00_00__common__target_hvp_native"}
        failures = []
        _audit_case(case, arrays, lambda okay, message: failures.append(message) if not okay else None,
            x=arrays["source_inputs"], hp=arrays["head_projector"], head=arrays["head_weight"], expected_seed=61064)
        self.assertEqual(failures, [])

    def test_generation_guard_rejects_untracked_and_staged_source_before_rng(self):
        with tempfile.TemporaryDirectory(prefix="lga10d-guard-") as folder:
            root = Path(folder)
            for name in pilot.REQUIRED_PATHS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic committed dependency\n")
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT, text=True).strip()
            git("init", "-q")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                "commit", "-qm", "synthetic implementation lock")
            head = git("rev-parse", "HEAD")
            with mock.patch.object(pilot, "__file__", str(root / "mechinterp/loopscope13d/source_feasibility.py")):
                self.assertEqual(pilot._provenance(head)["freeze_commit"], head)
                extra = root / "mechinterp/loopscope13d/uncommitted.py"
                extra.write_text("# uncommitted imported dependency\n")
                for staged in (False, True):
                    if staged:
                        git("add", str(extra.relative_to(root)))
                    with self.subTest(staged=staged), mock.patch.object(pilot.torch, "manual_seed") as seed, \
                         mock.patch.object(pilot, "fixed_data") as data:
                        with self.assertRaises(subprocess.CalledProcessError):
                            pilot.run(root / "out.json", root / "raw.npz", freeze_commit=head)
                        seed.assert_not_called()
                        data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
