import json
from pathlib import Path
import unittest

from research_gym.interp.aux_gym_v2 import (
    ABSTAIN, MODULAR, POINTER, build, metrics, rows, split_ops, symbolic,
)


class AuxV2Tests(unittest.TestCase):
    def test_hole_is_absorbing(self):
        self.assertEqual(symbolic("masked_pointer_chase",
            {"start": 0, "hops": 3, "mapping": [1, -1, 0]}),
            [POINTER, POINTER + 1, ABSTAIN, ABSTAIN])

    def test_affine_and_identity(self):
        self.assertEqual(symbolic("modular_recurrence", {"start": 3, "operations": []}), [835])
        self.assertEqual(symbolic("modular_recurrence",
            {"start": 3, "operations": [[2, 4], [3, 1]]}), [835, 842, 846])

    def test_abstain_shortcut(self):
        examples = [{"target_token": t, "difficulty": 2, "semantic": {"mapping": [], "hops": 2}}
                    for t in (ABSTAIN, ABSTAIN, POINTER, POINTER + 1)]
        m = metrics(examples, [ABSTAIN] * 4, ABSTAIN)
        self.assertEqual(m["exact"], 0.5)
        self.assertEqual(m["masked"]["node_exact"], 0)
        self.assertEqual(m["masked"]["abstain_recall"], 1)
        self.assertAlmostEqual(m["macro_recall_observed_classes"], 1 / 3)

    def test_split(self):
        from research_gym.interp.bundle import BundleConfig
        config = BundleConfig.from_mapping(json.loads(Path("configs/rmp_task_bundle_v0.json").read_text()))
        self.assertEqual({k: len(v) for k, v in split_ops(config).items()},
                         {"train": 144, "monitor_train": 73, "locked": 20,
                          "validation": 16, "monitor_calibration": 14, "ood_stress": 5})

    def test_balanced_small_build(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data"
            manifest = build(out, Path("configs/rmp_task_bundle_v0.json"), 2026112602, 1, 1, 100000)
            owners = {}
            seen = set()
            for name in manifest["files"]:
                split = name.split("/")[0]
                data = rows(out / name)
                for r in data:
                    self.assertNotIn(r["fingerprint"], seen)
                    seen.add(r["fingerprint"])
                    self.assertEqual(owners.setdefault(r["group_id"], split), split)
                    self.assertEqual(r["target_token"], r["symbolic_trace_tokens"][-1])
                if split != "train":
                    self.assertEqual(set(manifest["files"][name]["targets"].values()),
                                     {4 if split.startswith("locked") else 8})
            zeros = [r for r in rows(out / "train/modular_recurrence.jsonl") if r["difficulty"] == 0]
            self.assertEqual({r["target_token"] for r in zeros}, set(range(MODULAR, MODULAR + 17)))


if __name__ == "__main__":
    unittest.main()
