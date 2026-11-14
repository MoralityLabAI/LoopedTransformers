"""Independent certificate boundary tests, including optional actual Lean replay."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.activation_funnel import certify as C


def fixture() -> dict:
    data = {
        "case_id": "certificate_fixture",
        "atoms": ["start_a", "start_b", "goal"],
        "facts": [0],
        "rules": [{"premises": [0], "head": 2}, {"premises": [1], "head": 2}],
        "goal": 2,
        "candidate_traces": [[0], [1]],
        "statuses": ["valid", "tainted"],
    }
    payload = {"schema": "af01-target-v1", "atoms": data["atoms"],
               "facts": [0], "rules": data["rules"], "goal": 2,
               "allowed_axioms": []}
    data["target_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode()
    ).hexdigest()
    return data


def row(data: dict, branch):
    return {"case_id": data["case_id"], "branch": branch,
            "target_digest": data["target_digest"], "arm": "fixture", "seed": 17}


class CertificateBoundaryTests(unittest.TestCase):
    def test_original_context_accepts_only_supported_candidate(self):
        host = C.HostTheorem.read(fixture())
        self.assertTrue(C.check_branch(host, 0).accepted)
        self.assertFalse(C.check_branch(host, 1).accepted)
        self.assertIn("missing_premise", C.check_branch(host, 1).reason)

    def test_preview_claim_cannot_add_a_fact(self):
        data = fixture()
        data["statuses"] = ["dead", "valid"]
        host = C.HostTheorem.read(data)
        self.assertTrue(C.check_branch(host, 0).accepted)
        self.assertFalse(C.check_branch(host, 1).accepted)

    def test_host_target_tampering_is_rejected(self):
        for field, value in [("goal", 0), ("facts", [0, 1]),
                             ("rules", [{"premises": [0], "head": 1},
                                        {"premises": [1], "head": 2}])]:
            with self.subTest(field=field):
                data = fixture()
                data[field] = value
                with self.assertRaisesRegex(C.CertificateError, "digest mismatch"):
                    C.HostTheorem.read(data)

    def test_candidate_must_derive_original_goal(self):
        data = fixture()
        data["candidate_traces"] = [[1], [0]]
        host = C.HostTheorem.read(data)
        self.assertFalse(C.check_branch(host, 0).accepted)
        self.assertTrue(C.check_branch(host, 1).accepted)

    def test_structured_action_rejects_source_injection_and_bool(self):
        host = C.HostTheorem.read(fixture())
        for action in [True, False, "0", "0\naxiom evil : False", -1, 2, 0.0]:
            with self.subTest(action=action):
                with self.assertRaises(C.CertificateError):
                    C.check_branch(host, action)

    def test_atom_names_reject_source_syntax(self):
        for malicious in ["P\naxiom evil : False", "P : Prop", "P)", "_foo", "P\n"]:
            data = fixture()
            data["atoms"][0] = malicious
            with self.assertRaisesRegex(C.CertificateError, "identifiers"):
                C.HostTheorem.read(data)

    def test_case_id_is_never_interpolated_into_lean(self):
        data = fixture()
        data["case_id"] = "../../bad\n#print axioms False"
        _, source = C.render_selected(C.HostTheorem.read(data), 0)
        self.assertNotIn(data["case_id"], source)
        self.assertNotIn("../", source)

    def test_native_baseline_keeps_missing_root_undefined(self):
        host = C.HostTheorem.read(fixture())
        _, source = C.render_native_search(host)
        self.assertIn("(f0 : P0)", source)
        self.assertNotIn("(f1 : P1)", source)
        self.assertIn("af01_unavailable_1", source)
        self.assertIn("  first\n  | exact r0", source)
        self.assertNotIn("\naxiom ", source)
        self.assertNotIn("sorry", source)

    def test_wrong_axiom_declaration_and_duplicates_fail(self):
        names = ["af01_proof_0123456789abcdef01234567"]
        good = f"'{names[0]}' does not depend on any axioms\n"
        C.check_axiom_output(good, names)
        for bad in ["", good + good, "'wrong' does not depend on any axioms\n",
                    f"'{names[0]}' depends on axioms: [sorryAx]\n",
                    good + "'extra' does not depend on any axioms\n"]:
            with self.subTest(output=bad):
                with self.assertRaises(C.KernelAuditError):
                    C.check_axiom_output(bad, names)

    def test_missing_lean_fails_even_for_all_stops(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(C.shutil, "which", return_value=None):
            with self.assertRaisesRegex(C.KernelAuditError, "unavailable"):
                C.certify_selected([data], [row(data, None)], tmp)
            receipt = json.loads((Path(tmp) / "kernel_audit.json").read_text())
            self.assertFalse(receipt["kernel_complete"])
            self.assertEqual(receipt["stop_rows"], 1)

    def test_wrong_lean_version_fails(self):
        completed = subprocess.CompletedProcess(["lean", "--version"], 0,
                                                "Lean (version 4.35.0)\n", "")
        with mock.patch.object(C.shutil, "which", return_value="/tmp/lean"), \
                mock.patch.object(C.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(C.KernelAuditError, "4.19.0"):
                C.LeanCompiler.discover()

    def test_exit_zero_without_receipt_cannot_certify(self):
        compiler = C.LeanCompiler("/tmp/lean", "Lean (version 4.19.0)", "a" * 64)
        completed = subprocess.CompletedProcess(["lean"], 0, "unrelated output\n", "")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(C.subprocess, "run", return_value=completed):
            path = Path(tmp) / "Example.lean"
            path.write_text("theorem example0 (P : Prop) (h : P) : P := h\n")
            result = compiler.run(path, ["example0"])
            self.assertFalse(result["kernel_success"])
            self.assertFalse(result["axiom_audit_passed"])

    def test_row_cannot_rebind_target(self):
        data = fixture()
        bad_row = row(data, 0)
        bad_row["target_digest"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(C.CertificateError, "outcome target digest"):
                C.certify_selected([data], [bad_row], tmp)

    def test_unknown_and_duplicate_host_ids_fail(self):
        data = fixture()
        unknown = row(data, 0)
        unknown["case_id"] = "not-the-host"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(C.CertificateError, "unknown host"):
                C.certify_selected([data], [unknown], tmp)
            with self.assertRaisesRegex(C.CertificateError, "duplicate episode"):
                C.certify_selected([data, copy.deepcopy(data)], [row(data, 0)], tmp)


class ActualLeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lean = os.environ.get("AF01_LEAN_EXE") or shutil.which("lean")
        if not cls.lean:
            raise unittest.SkipTest("actual Lean unavailable; strict audit CLI still fails closed")
        C.LeanCompiler.discover(cls.lean)

    def test_selected_native_and_negative_kernel_checks(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = C.certify_selected([data], [row(data, 0), row(data, 1), row(data, None)],
                                         root / "selected", self.lean)
            self.assertTrue(receipt["kernel_complete"])
            self.assertEqual(receipt["kernel_certified_rows"], 1)
            self.assertEqual(receipt["rejected_submission_rows"], 1)
            self.assertEqual(receipt["stop_rows"], 1)
            baseline = C.native_baseline([data], root / "native", self.lean)
            self.assertEqual(baseline["kernel_successes"], 1)
            guards = C.negative_guards(root / "guards", self.lean)
            self.assertTrue(guards["passed"])


if __name__ == "__main__":
    unittest.main()
