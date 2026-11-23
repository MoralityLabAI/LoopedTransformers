"""Read-only CPU diagnosis invariants; synthetic fixtures carry no research claims."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from ppt_gym.cpu_diagnostics import (
    _verify_artifacts, choice_geometry, shared_decisions,
)
from ppt_gym.data import sha256_file


class CloudDiagnosticsTests(unittest.TestCase):
    def test_choice_geometry_counts_only_real_discriminating_forks(self):
        rows = [
            {"lineage": "root_a", "candidates": ["h0"], "labels": [1]},
            {"lineage": "root_a", "candidates": ["r0", "r1"], "labels": [1, 0]},
            {"lineage": "root_b", "candidates": ["r0", "r1"], "labels": [1, 1]},
            {"lineage": "root_b", "candidates": ["r0", "r1", "r2"], "labels": [0, 0, 1]},
        ]
        result = choice_geometry(rows)
        self.assertEqual(result["independent_lineages"], 2)
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["menu_singletons"], 1)
        self.assertEqual(result["menu_forks"], 3)
        self.assertEqual(result["discriminating_forks"], 2)
        self.assertEqual(result["ambiguous_optimal_forks"], 1)
        self.assertAlmostEqual(result["mean_discriminating_forks_per_root"], 1)
        with self.assertRaisesRegex(ValueError, "independent"):
            choice_geometry([])

    def test_compares_only_exactly_same_packet_and_menu(self):
        a = {
            "events": [
                {"public_packet_sha256": "A", "menu_sha256": "1", "tactic": "apply r0"},
                {"public_packet_sha256": "B", "menu_sha256": "2", "tactic": "apply h0"},
                {"public_packet_sha256": "C", "menu_sha256": "3", "tactic": "apply r3"},
            ],
        }
        b = {
            "events": [
                {"public_packet_sha256": "A", "menu_sha256": "1", "tactic": "apply r1"},
                {"public_packet_sha256": "B", "menu_sha256": "9", "tactic": "apply h0"},
                {"public_packet_sha256": "C", "menu_sha256": "3", "tactic": "apply r3"},
            ],
        }
        self.assertEqual(shared_decisions(a, b), (2, 1))
        self.assertEqual(shared_decisions(b, a), (2, 1))

    def test_refuses_edited_or_moved_native_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "paired_roots.jsonl"
            record.write_text('{"root":"a"}\n', encoding="utf-8")
            trusted = sha256_file(record)
            _verify_artifacts(root, {record.name: trusted})
            record.write_text('{"root":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _verify_artifacts(root, {record.name: trusted})
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _verify_artifacts(root, {"../bad": trusted})


if __name__ == "__main__":
    unittest.main()
