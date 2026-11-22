import unittest

import numpy as np
import torch

from research_gym.inkling_routing_01.core import (
    ARMS, WIDTH, TARGET_COORD, patch_vector, sham_coordinate,
)
from research_gym.inkling_routing_01.ndif import apply_native_patch, plan, reader_call


class NativeHookContractTests(unittest.TestCase):
    def test_reader_call_index(self):
        self.assertEqual(reader_call(2), 6)
        self.assertEqual(reader_call(3), 18)
        with self.assertRaises(ValueError):
            reader_call(1)

    def test_plan_manifest_and_no_secret(self):
        manifest = plan(2, 2)
        self.assertEqual(len(manifest["root_prompts"]), 2)
        self.assertEqual(manifest["restore_call"], 7)
        self.assertEqual(len(manifest["manifest_sha256"]), 64)
        self.assertNotIn("API_KEY", str(manifest))
        self.assertEqual(manifest, plan(2, 2))

    def test_torch_patch_matches_numpy_and_is_restorable(self):
        rng = np.random.default_rng(13)
        value = rng.normal(size=WIDTH).astype(np.float32)
        value[TARGET_COORD] = 32.4
        sham = sham_coordinate("capital")
        for arm in ARMS:
            with self.subTest(arm=arm):
                bank = torch.from_numpy(np.tile(value, (2, 3, 1)).copy())
                expected = patch_vector(value, arm, sham)
                original = apply_native_patch(bank, arm, sham)
                self.assertTrue(np.allclose(bank[:, -1].numpy(), expected, rtol=1e-5, atol=1e-5))
                bank[:, -1] = original
                self.assertTrue(np.array_equal(bank[0, -1].numpy(), value))
                self.assertTrue(np.array_equal(bank[:, 0].numpy(), np.tile(value, (2, 1))))


if __name__ == "__main__":
    unittest.main()
