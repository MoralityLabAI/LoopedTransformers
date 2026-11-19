"""Scientific inputs must fail closed on corruption or unsafe extraction."""

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.deuterium.artifacts import inventory_archive, verify_archive


class ArtifactTests(unittest.TestCase):
    def archive(self, root, name="a.txt", contents=b"original\r\nbytes"):
        archive = Path(root) / "archive.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(name, contents)
        spec = {"bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "members": 1, "uncompressed_bytes": len(contents)}
        return archive, spec

    def test_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, spec = self.archive(directory)
            out = Path(directory) / "recovered"
            result = inventory_archive(archive, spec, out)
            self.assertEqual((out / "a.txt").read_bytes(), b"original\r\nbytes")
            self.assertEqual(result["member_count"], 1)

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, spec = self.archive(directory)
            spec["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_archive(archive, spec)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, spec = self.archive(directory, "../escape.txt")
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                inventory_archive(archive, spec, Path(directory) / "recovered")
            self.assertFalse((Path(directory) / "escape.txt").exists())

    def test_existing_modified_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, spec = self.archive(directory)
            out = Path(directory) / "recovered"
            out.mkdir()
            (out / "a.txt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "was modified"):
                inventory_archive(archive, spec, out)
            self.assertEqual((out / "a.txt").read_bytes(), b"modified")


if __name__ == "__main__":
    unittest.main()
