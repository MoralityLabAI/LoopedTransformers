"""Archived ROC-02 sample hash gate and sealed-label test."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from research_gym.analysis.roc02_dataset_archive import pack, verify_archive
    from research_gym.analysis.roc02_delayed_rule_source import write_dataset
except ImportError:
    from roc02_dataset_archive import pack, verify_archive
    from roc02_delayed_rule_source import write_dataset


class ROC02DatasetBundleTests(unittest.TestCase):
    def test_deterministic_archive(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"cases"
            manifest=write_dataset(source)
            first=root/"a.tar.gz"
            second=root/"b.tar.gz"
            a=pack(source,first)
            b=pack(source,second)
            self.assertEqual(first.read_bytes(),second.read_bytes())
            self.assertEqual(a["archive_sha256"],b["archive_sha256"])
            self.assertEqual(a["valid_cases"],1664)
            self.assertEqual(a["invalid_fixtures"],24)
            self.assertEqual(verify_archive(first,manifest)["member_sha256"],a["member_sha256"])

    def test_published_bundle(self):
        root=Path(__file__).resolve().parents[1]
        manifest=root/"experiments/roc02_delayed_rule_v1/source_manifest.json"
        bundle=root/"experiments/roc02_delayed_rule_v1/roc02_source_cases.tar.gz"
        if not (manifest.is_file() and bundle.is_file()):
            self.skipTest("published data artifact unavailable in local source-only mode")
        result=verify_archive(bundle,json.loads(manifest.read_text()))
        self.assertEqual(result["valid_cases"],1664)
        self.assertEqual(result["invalid_fixtures"],24)


if __name__=="__main__":
    unittest.main()
