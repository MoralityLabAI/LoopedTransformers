"""Relocation must reject source mutations before scoring reserved seeds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.deuterium import relocated_runner as runner


class PortTests(unittest.TestCase):
    def test_original_source_hashes_are_still_frozen(self):
        registration = json.loads(runner.REGISTRATION.read_text())
        for path, expected in registration["source_sha256"].items():
            self.assertEqual(runner.sha256(runner.PORT / path), expected, path)

    def test_scoring_rejects_missing_registration(self):
        with self.assertRaisesRegex(ValueError, "registration-commit"):
            runner.frozen_source_check(None, runner.HERE / "protocol.json")

    def test_scoring_rejects_unregistered_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "protocol.json"
            other.write_bytes((runner.HERE / "protocol.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "alternative protocol"):
                runner.frozen_source_check(runner.PINNED_REGISTRATION, other)

    def test_scoring_requires_pinned_upstream(self):
        with patch.dict("os.environ", {"DEUTERIUM_UPSTREAM": ""}):
            with self.assertRaisesRegex(ValueError, "Missing DEUTERIUM_UPSTREAM"):
                runner.frozen_source_check(runner.PINNED_REGISTRATION, runner.HERE / "protocol.json")


if __name__ == "__main__":
    unittest.main()
