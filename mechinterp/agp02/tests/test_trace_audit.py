"""AGP-02 source-only paired native GRU state and covariance regression tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mechinterp.agp02.trace_audit import (
    ARMS, DEPTHS, SEEDS, WIDTH, analyze, canonical, as_rows,
)


def public_fixture() -> list[dict]:
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            digest = hashlib.sha256(f"{seed}/{arm}".encode()).hexdigest()
            for rid in ("root-a", "root-b"):
                for cid in ("case-0", "case-1"):
                    magnitude = 1.0 + 0.2 * int(cid[-1]) + 0.3 * int(rid[-1] == "b")
                    for hint in (0, 1):
                        for depth in DEPTHS:
                            state = [0.0] * WIDTH
                            state[0] = float(depth) / 16.0
                            if depth > 6 and hint == 1:
                                state[1] = float(magnitude)
                            rows.append({
                                "init_seed": seed,
                                "arm": arm,
                                "root_id": rid,
                                "case_id": cid,
                                "grader_hint": hint,
                                "depth": depth,
                                "state": state,
                                "checkpoint_sha256": digest,
                            })
    return rows


class NativeFiberAuditTests(unittest.TestCase):
    def test_full_paired_shape_and_mean_displacement(self):
        result = analyze(public_fixture())
        self.assertEqual(result["verdict"],
                         "PASS_RETROSPECTIVE_PAIRED_NATIVE_STATE_AUDIT")
        self.assertEqual(result["native_state_rows"], 480)
        self.assertEqual(result["unique_host_roots"], 2)
        self.assertEqual(result["models"]["103:host_trait"]["depths"]["6"][
            "mean_l2"], 0.0)
        late = result["models"]["103:host_trait"]["depths"]["12"]
        self.assertAlmostEqual(late["mean_l2"], 1.25)
        self.assertAlmostEqual(late["top1_uncentered_energy_fraction"], 1.0)
        self.assertAlmostEqual(late["top1_centered_energy_fraction"], 1.0)
        self.assertGreater(late["shared_mean_energy_fraction"], 0.9)
        self.assertAlmostEqual(
            result["averages_over_fixed_initializers"]["trait_cf"]["8"][
                "mean_l2"], 1.25)

    def test_all_registered_models_and_pairs_required(self):
        source = public_fixture()
        for mutate in (
            lambda s: s.pop(),
            lambda s: s.append(copy.deepcopy(s[0])),
            lambda s: s[0].__setitem__("state", [0.0] * 15),
            lambda s: s[0]["state"].__setitem__(0, float("nan")),
            lambda s: s[0].__setitem__("checkpoint_sha256", "f" * 64),
            lambda s: s[0].__setitem__("arm", "other-arm"),
            lambda s: s[0].__setitem__("depth", 10),
        ):
            with self.subTest(mutate=mutate):
                rows = copy.deepcopy(source)
                mutate(rows)
                with self.assertRaises(ValueError):
                    analyze(rows)

    def test_pre_hint_differences_rejected(self):
        rows = public_fixture()
        target = next(r for r in rows if r["depth"] == 6 and r["grader_hint"] == 1)
        target["state"][1] += 0.03
        with self.assertRaisesRegex(ValueError, "pre-injection"):
            analyze(rows)

    def test_jsonl_gzip_reader_and_canonical_receipt(self):
        import gzip
        rows = public_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(canonical(row).decode())
            reconstructed = list(as_rows(path))
            self.assertEqual(rows, reconstructed)
            result = analyze(reconstructed)
            self.assertTrue(canonical(result).endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
