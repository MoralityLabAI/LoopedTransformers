"""Byte-for-byte published ROC-02A original-source ZIP check."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest

try:
    from research_gym.analysis.roc02a_binary_source_verify import verify_and_unpack
except ImportError:
    from roc02a_binary_source_verify import verify_and_unpack


class ROC02ABinarySourceTests(unittest.TestCase):
    def test_four_published_parts_and_all_source_hashes(self):
        folder = Path(os.environ.get("ROC02A_BINARY_DIR",
                              "experiments/observer_calculus_roc02a_v1/binary_sources"))
        index = folder / "roc02a_binary_sources_index.json"
        if not index.exists():
            self.skipTest("public original-binary source archives not present")
        with TemporaryDirectory() as tmp:
            result = verify_and_unpack(folder, Path(tmp), index)
        self.assertEqual(result["archives"], 4)
        self.assertEqual(result["verified_checkpoints"], 12)
        self.assertEqual(result["verified_adapters"], 12)
        self.assertEqual(result["verified_receipts"], 12)

    def test_reject_corrupted_part(self):
        folder = Path(os.environ.get("ROC02A_BINARY_DIR",
                              "experiments/observer_calculus_roc02a_v1/binary_sources"))
        index = folder / "roc02a_binary_sources_index.json"
        if not index.exists():
            self.skipTest("public original-binary source archives not present")
        data = json.loads(index.read_text())
        self.assertEqual(len(data["archives"]), 4)
        self.assertEqual(len({x["git_blob_sha1"] for x in data["archives"]}), 4)


if __name__ == "__main__":
    unittest.main()
