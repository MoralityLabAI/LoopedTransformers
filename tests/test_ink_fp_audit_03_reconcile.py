import unittest

from research_gym.ink_fp_audit_03 import reconcile as rc


class ReconcileTests(unittest.TestCase):
    def test_reports_match_receipts(self):
        for path in rc.REPORTS:
            if path.exists():
                self.assertEqual(rc.check(path, require=rc.REQUIRED.get(path, ())), [], path.name)

    def test_audit02_verified_numbers(self):
        s = rc.effort_stats(rc.records(rc.A02 / "receipts/inky_base_dev_20261128/records.jsonl"))
        self.assertEqual([s["efforts"][e]["correct"] for e in rc.EFFORTS], [89, 87, 84])
        self.assertEqual([s["efforts"][e]["completion_tokens"] for e in rc.EFFORTS], [45831, 58330, 70048])
        self.assertEqual((s["truncated_total"], s["failures"], s["failures_truncated"]), (28, 28, 28))
        self.assertEqual(len(s["saturated_families"]), 9)
        self.assertEqual((s["efforts"]["medium"]["rescues_vs_low"], s["efforts"]["high"]["rescues_vs_low"]), (0, 0))

    def test_hand_edit_is_detected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.md"
            p.write_text("<!-- reconcile:a02_base_dev:start -->\n| Low | 90/96 |\n<!-- reconcile:a02_base_dev:end -->\n",
                         encoding="utf-8")
            self.assertEqual(rc.check(p), ["a02_base_dev"])
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))  # CRLF copy still detected
            self.assertEqual(rc.check(p), ["a02_base_dev"])
            self.assertEqual(rc.check(p, require=("a02_frontier8192",)), ["missing:a02_frontier8192", "a02_base_dev"])


if __name__ == "__main__":
    unittest.main()
