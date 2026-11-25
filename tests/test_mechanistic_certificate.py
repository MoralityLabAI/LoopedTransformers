"""MC-01 validation. Synthetic capture fixtures exercise the file interface only."""
import json
from pathlib import Path
import tempfile
import struct
import unittest

import numpy as np

from research_gym.analysis.mechanistic_certificate import (
    BN254, M31, Statement, audit_finite, check, encode, oracle, relation,
)
from research_gym.scripts.mechanistic_certificate import audit_capture, controls, file_hash


class ExactRelationTests(unittest.TestCase):
    def setUp(self):
        self.labels = (0, 2, 1, 2)
        self.statement = Statement((2, 2, 1), 0, 3, 2)
        self.circuit = relation(3, 3)
        self.witness = encode(self.statement, self.labels)

    def test_exact_trace_both_fields(self):
        for field in (BN254, M31):
            self.assertEqual(self.circuit.failures(self.witness, field), [])

    def test_public_and_private_wire_counts(self):
        self.assertEqual(self.circuit.public_count, 15)
        self.assertEqual(len(self.witness), 55)
        self.assertEqual(len(self.circuit.constraints), 70)
        large = relation(16, 16)
        self.assertEqual(len(large.constraints), 4945)
        self.assertEqual(len(large.names) - 1 - large.public_count, 4368)

    def test_wrong_start(self):
        self.assertFalse(check(self.statement, (1, 2, 1, 2))["abstract_valid"])

    def test_wrong_end(self):
        self.assertFalse(check(self.statement, (0, 2, 1, 1))["abstract_valid"])

    def test_correct_endpoint_wrong_middle(self):
        self.assertFalse(check(self.statement, (0, 1, 1, 2))["abstract_valid"])

    def test_each_nonconstant_wire_is_constrained(self):
        for field in (BN254, M31):
            for i in range(1, len(self.witness)):
                wrong = self.witness.copy()
                wrong[i] = (wrong[i] + 1) % field
                self.assertTrue(self.circuit.failures(wrong, field), (field, i))

    def test_no_implicit_source_trust(self):
        result = check(self.statement, self.labels)
        self.assertTrue(result["abstract_valid"])
        self.assertFalse(result["trusted_capture_accept"])

    def test_source_linkage_accepts_exact_capture(self):
        result = check(self.statement, self.labels, expected=self.statement,
                       trusted_observed=self.labels)
        self.assertTrue(result["trusted_capture_accept"])
        self.assertFalse(result["cryptographic_source_binding"])
        self.assertEqual(result["causal_fidelity"], "unmeasured")

    def test_counterfeit_oracle_witness(self):
        result = check(self.statement, self.labels, expected=self.statement,
                       trusted_observed=(0, 1, 1, 2))
        self.assertTrue(result["abstract_valid"])
        self.assertFalse(result["observations_bound"])
        self.assertFalse(result["trusted_capture_accept"])

    def test_wrong_expected_program(self):
        other = Statement((1, 2, 1), 0, 3, 2)
        self.assertFalse(check(self.statement, self.labels, expected=other,
                               trusted_observed=self.labels)["trusted_capture_accept"])

    def test_wrong_expected_horizon(self):
        other = Statement(self.statement.graph, 0, 2, 1)
        self.assertFalse(check(self.statement, self.labels, expected=other,
                               trusted_observed=self.labels)["trusted_capture_accept"])

    def test_reject_noncanonical_and_boolean_wires(self):
        for value in (-1, BN254, True, 0.0):
            wrong = self.witness.copy()
            wrong[1] = value
            with self.assertRaises(ValueError):
                self.circuit.failures(wrong)

    def test_constant_wire_cannot_be_changed(self):
        wrong = self.witness.copy()
        wrong[0] = 0
        with self.assertRaises(ValueError):
            self.circuit.failures(wrong)

    def test_malformed_layout_and_field(self):
        with self.assertRaises(ValueError):
            self.circuit.failures(self.witness[:-1])
        with self.assertRaises(ValueError):
            self.circuit.failures(self.witness, 15)

    def test_malformed_statement(self):
        cases = [((0,), 0, 1, 0), ((0, 2), 0, 1, 0), ((0, 1), -1, 1, 0),
                 ((0, 1), 0, 0, 0), ((0, 1), 0, 65, 0), ((True, 1), 0, 1, 0),
                 ([0, 1], 0, 1, 0)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Statement(*args)

    def test_malformed_labels(self):
        for labels in ((0, 2), (0, 2, 1, -1), (0, 2, 1, True)):
            with self.assertRaises(ValueError):
                encode(self.statement, labels)

    def test_circom_has_every_explicit_constraint(self):
        text = self.circuit.circom()
        self.assertEqual(text.count(" === "), len(self.circuit.constraints))
        self.assertIn("main {public [public_values]}", text)
        self.assertNotIn("<--", text)
        self.assertNotIn("assert(", text)

    def test_exhaustive_control_suite(self):
        with tempfile.TemporaryDirectory() as d:
            result = controls(Path(d))
            self.assertEqual(result["valid_field_checks"], 712)
            self.assertEqual(result["candidate_traces_checked"], 2187)
            self.assertEqual(result["candidate_traces_accepted"], 81)
            self.assertEqual(result["single_wire_mutations_rejected"], 108)
            self.assertFalse(result["example_export"]["circom_compiled"])


class FiniteAbstractionTests(unittest.TestCase):
    def test_positive_with_interventions(self):
        result = audit_finite([(3, 2, 1, 0), (0, 0, 2, 2)], (0, 0, 1, 1),
                              [(1, 0), (0, 1)], (0, 0, 1, 1), (0, 1))
        self.assertTrue(result["exact_on_declared_domain"])

    def test_passive_shadow_is_not_causal_evidence(self):
        args = ([(3, 2, 1, 0)], (0, 0, 1, 1), [(1, 0)], (0, 1, 0, 1), (0, 1))
        self.assertTrue(audit_finite(*args, domain=(0, 3))["exact_on_declared_domain"])
        self.assertFalse(audit_finite(*args)["exact_on_declared_domain"])

    def test_domain_escape_is_a_failure(self):
        result = audit_finite([(3, 2, 1, 0), (0, 1, 0, 1)], (0, 0, 1, 1),
                              [(1, 0), (0, 0)], (0, 1, 0, 1), (0, 1), domain=(0, 3))
        self.assertIn("domain_escape", [d["kind"] for d in result["defects"]])

    def test_explicit_commutation_defect(self):
        result = audit_finite([(1, 0)], (0, 1), [(0, 1)], (0, 1), (0, 1))
        self.assertEqual(len(result["defects"]), 2)
        self.assertTrue(all(d["kind"] == "commutation" for d in result["defects"]))

    def test_malformed_and_empty_audit(self):
        for transitions in ([], [(0,)], [(False, 1)], [(0, 2)]):
            with self.assertRaises(ValueError):
                audit_finite(transitions, (0, 1), [(0, 1)], (0, 1), (0, 1))
        with self.assertRaises(ValueError):
            audit_finite([(0, 1)], (0, 1), [(0, 1)], (0, 1), (0, 1), domain=())


class CaptureAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trace = self.root / "trace.npz"
        self.metrics = self.root / "metrics.json"
        self.graph = (1, 2, 0)
        self.labels = oracle(self.graph, 0, 4)
        self.arrays = {
            "full_states": np.zeros((5, 1, 4, 2)),
            "input_embedding": np.zeros((1, 4, 2)),
            "tokens": np.array([[0, 1, 1, 2, 2, 0, 0]]),
            "permutations": np.array([self.graph]), "starts": np.array([0]),
            "predictions": np.array(self.labels[1:]).reshape(4, 1),
            "own_head_logits": np.eye(3)[np.array(self.labels[1:])].reshape(4, 1, 3),
            "oracle": np.array(self.labels[1:]).reshape(4, 1)}
        # Deliberately constructed file fixture, not an executed neural experiment.
        self.receipt = {"schema_version": "semantic_echo.looped_chase.v1",
                        "status": "UNTRAINED_SMOKE", "checkpoint_sha256": None,
                        "fixed_depth_compatible": True, "items": 1, "max_loops": 4,
                        "config": {"n_nodes": 3, "fixed_depth": True},
                        "native_parity": {"passed": True, "logit_parity": {"passed": True}}}
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        np.savez_compressed(self.trace, **self.arrays)
        self.receipt["trace_sha256"] = file_hash(self.trace)
        self.metrics.write_text(json.dumps({"receipt": self.receipt}))
        self.pins = {"expected_trace": file_hash(self.trace), "expected_metrics": file_hash(self.metrics),
                     "expected_checkpoint": None, "allow_untrained": True}

    def run_audit(self):
        return audit_capture(self.trace, self.metrics, **self.pins)

    def test_constructed_capture(self):
        result = self.run_audit()
        self.assertEqual(result["trusted_capture_accept_count"], 1)
        self.assertFalse(result["cryptographic_source_binding"])
        self.assertEqual(result["causal_fidelity"], "unmeasured")

    def test_capture_damage_fails_pinned_identity(self):
        self.trace.write_bytes(self.trace.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_receipt_damage_fails_pinned_identity(self):
        self.metrics.write_text(self.metrics.read_text() + " ")
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_smoke_requires_opt_in(self):
        self.pins["allow_untrained"] = False
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_checkpoint_requires_independent_pin(self):
        self.receipt.update(status="MEASURED_CHECKPOINT", checkpoint_sha256="a" * 64)
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()
        self.pins["expected_checkpoint"] = "a" * 64
        self.assertEqual(self.run_audit()["trusted_capture_accept_count"], 1)

    def test_readout_labels_must_match_logits(self):
        self.arrays["predictions"][0, 0] = 0
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_graph_must_match_tokens(self):
        self.arrays["tokens"][0, 1] = 2
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_oracle_is_recomputed(self):
        self.arrays["oracle"][0, 0] = 0
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_native_parity_is_required(self):
        self.receipt["native_parity"]["logit_parity"]["passed"] = False
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_nonfinite_state_rejected(self):
        self.arrays["full_states"][1, 0, 0, 0] = float("nan")
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_wrong_middle_is_counted_despite_correct_endpoint(self):
        self.arrays["predictions"][0, 0] = 0
        self.arrays["own_head_logits"][0, 0] = [1, 0, 0]
        self.save()
        result = self.run_audit()
        self.assertEqual(result["final_answer_correct_count"], 1)
        self.assertEqual(result["trusted_capture_accept_count"], 0)
        self.assertEqual(result["rows"][0]["first_wrong_visit"], 1)

    def test_integer_labels_required(self):
        self.arrays["predictions"] = self.arrays["predictions"].astype(float)
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_missing_source_array_rejected(self):
        del self.arrays["full_states"]
        self.save()
        with self.assertRaises(ValueError):
            self.run_audit()


class CompilerFormatTests(unittest.TestCase):
    def test_wtns_roundtrip_and_truncation(self):
        from research_gym.scripts.mc01_compiler_parity import read_wtns
        header = struct.pack("<I", 32) + BN254.to_bytes(32, "little") + struct.pack("<I", 3)
        payload = b"".join(i.to_bytes(32, "little") for i in (1, 0, 2))
        raw = b"wtns" + struct.pack("<II", 2, 2)
        raw += struct.pack("<IQ", 1, len(header)) + header
        raw += struct.pack("<IQ", 2, len(payload)) + payload
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.wtns"
            path.write_bytes(raw)
            self.assertEqual(read_wtns(path), (BN254, [1, 0, 2]))
            for wrong in (raw[:-1], raw + b"x", b"bad!" + raw[4:]):
                path.write_bytes(wrong)
                with self.assertRaises(ValueError):
                    read_wtns(path)

    def test_wtns_writer_roundtrip_and_validation(self):
        from research_gym.scripts.mc01_compiler_parity import read_wtns, write_wtns
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mutation.wtns"
            write_wtns(path, [1, 2, BN254 - 1])
            self.assertEqual(read_wtns(path), (BN254, [1, 2, BN254 - 1]))
            for values in ([], [0, 1], [1, -1], [1, BN254], [True, 1], [1, 2.0]):
                with self.assertRaises(ValueError):
                    write_wtns(path, values)

    def test_exported_constraint_evaluator(self):
        from research_gym.scripts.mc01_compiler_parity import satisfies
        rows = [[{"1": "1"}, {"2": "1"}, {"3": "1"}]]
        self.assertTrue(satisfies(rows, [1, 2, 3, 6], BN254))
        self.assertFalse(satisfies(rows, [1, 2, 3, 7], BN254))


if __name__ == "__main__":
    unittest.main()
