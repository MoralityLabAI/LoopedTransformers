"""Synthetic scoring contract tests; no native traces or reserved fixtures."""
import copy
import unittest

from native_trace import score


def fixture():
    row = dict(
        phase="development", root_id="synthetic-root", case_id="synthetic-case",
        checkpoint="fixture", source_id="fixture-source", basis_id="fixture-basis",
        placement_sha256="fixture-placement", policy_sha256="fixture-policy",
        head_parity=True,
    )
    row["arms"] = {
        name: dict(correct=name in {"clean", "geometry"}, nll=0.5 if name == "geometry" else 1.0,
                   state_radius=1.0, displacement_l2=2.0)
        for name in ("clean", "corrupt", "baseline", "geometry", "sham")
    }
    return row


class NativeTraceContract(unittest.TestCase):
    def test_synthetic_recovery(self):
        answer = score([fixture()], 0.01)
        self.assertEqual((answer["n_units"], answer["n_cases"]), (1, 1))
        row = answer["pilot"][0]
        self.assertEqual((row["paired_effect"], row["sham_effects"]), ("1/1", ["0/1"]))
        self.assertEqual((row["recovery"], row["corruption"], row["mean_nll_gain"]), (1, 0, 0.5))

    def test_duplicate_and_phase_rejected(self):
        row = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score([row, copy.deepcopy(row)], 0.01)
        row["phase"] = "non-development-fixture"
        with self.assertRaisesRegex(ValueError, "development only"):
            score([row], 0.01)

    def test_parity_dose_and_nan_rejected(self):
        row = fixture()
        row["head_parity"] = False
        with self.assertRaisesRegex(ValueError, "parity"):
            score([row], 0.01)
        row = fixture()
        row["arms"]["sham"]["displacement_l2"] = 1.0
        with self.assertRaisesRegex(ValueError, "dose mismatch"):
            score([row], 0.01)
        row = fixture()
        row["arms"]["geometry"]["nll"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid endpoint"):
            score([row], 0.01)
