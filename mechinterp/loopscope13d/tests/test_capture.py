"""Lossless alias capture and corruption checks using numeric fixtures only."""
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from mechinterp.loopscope13d.capture import open_capture, write_capture


class CaptureTests(unittest.TestCase):
    def test_aliases_preserve_dtype_shape_and_exact_value_bytes(self):
        vector = np.arange(4, dtype=np.float32)
        arrays = {"one": vector, "same": vector.copy(), "different_dtype": vector.astype(np.float64),
                  "different_shape": vector.reshape(2, 2), "scalar": np.array(-0., dtype=np.float64)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.npz"
            metadata = write_capture(path, arrays)
            self.assertEqual(metadata["logical_array_count"], 5)
            self.assertEqual(metadata["unique_array_count"], 4)
            self.assertEqual(metadata["logical_to_physical"]["one"], metadata["logical_to_physical"]["same"])
            with open_capture(path, metadata) as raw:
                for key, value in arrays.items():
                    self.assertEqual(raw[key].shape, value.shape)
                    self.assertEqual(raw[key].dtype, value.dtype)
                    self.assertEqual(raw[key].tobytes(), value.tobytes())
                    self.assertFalse(raw[key].flags.writeable)
                self.assertEqual(raw["scalar"].shape, ())
            with self.assertRaises(FileExistsError):
                write_capture(path, arrays)

    def test_rehashed_container_does_not_hide_changed_numeric_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.npz"
            metadata = write_capture(path, {"probe": np.arange(6, dtype=np.float64)})
            key = metadata["logical_to_physical"]["probe"]
            buffer = io.BytesIO()
            np.save(buffer, np.arange(6, dtype=np.float64) + 1., allow_pickle=False)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(key + ".npy", buffer.getvalue())
            metadata["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata["raw_file_bytes"] = path.stat().st_size
            with self.assertRaisesRegex(ValueError, "content digest"):
                open_capture(path, metadata)

    def test_alias_redirect_is_detected_and_objects_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.npz"
            metadata = write_capture(path, {"a": np.ones(3), "b": np.zeros(3)})
            bad = copy.deepcopy(metadata)
            bad["logical_to_physical"]["a"] = bad["logical_to_physical"]["b"]
            with self.assertRaisesRegex(ValueError, "alias map"):
                open_capture(path, bad)
            with self.assertRaisesRegex(ValueError, "without objects"):
                write_capture(Path(folder) / "bad.npz", {"bad": np.array([object()])})


if __name__ == "__main__":
    unittest.main()
