import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.deuterium import relocated_runner, relocated_verifier


class RelocatedVerifierTests(unittest.TestCase):
    def test_disallows_arbitrary_git_requests(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized verifier"):
            relocated_verifier.historical_git_show(["git", "status"])

    def test_disallows_unregistered_commit(self):
        argv = ["git", "-C", str(relocated_runner.PORT), "show", "0" * 40 + ":experiments/deuterium/protocol.json"]
        with self.assertRaisesRegex(ValueError, "Unregistered Git source"):
            relocated_verifier.historical_git_show(argv)

    def test_original_bytes_are_used_when_upstream_is_verified(self):
        for source, location in relocated_verifier.ALLOWED.items():
            argv = ["git", "-C", str(relocated_runner.PORT), "show", relocated_runner.PINNED_REGISTRATION + ":" + source]
            with patch.object(relocated_runner, "frozen_source_check", return_value={}):
                self.assertEqual(relocated_verifier.historical_git_show(argv), Path(location).read_bytes())


if __name__ == "__main__":
    unittest.main()
