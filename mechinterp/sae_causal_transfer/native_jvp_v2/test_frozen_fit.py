"""LS-SAE04 preheldout byte-sealed calibration and split-isolation tests."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT
from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as NEW
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as PACK
from mechinterp.sae_causal_transfer.native_jvp_v2 import audit as AUD


class FrozenFitSourceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_unseen_pilot_and_test_rosters(self):
        manifests=NEW.verify_disjoint()
        self.assertEqual(set(manifests),set(NEW.BASES))
        for key in NEW.BASES:
            rosters=NEW.roster(key)
            self.assertEqual(len(rosters),NEW.SIZES[key])
            self.assertEqual(OLD.fingerprint(rosters),manifests[key])
            for row in rosters:
                self.assertEqual(row["seed"]//100000000,2026)
                self.assertEqual(len(row["x"]),81)
                self.assertEqual(row["x"].count(0),NEW.HOLES[key])
                self.assertGreater(row["rectangles"],0)
                self.assertEqual(OLD.count_solutions(row["x"]),1)
        with self.assertRaises(ValueError):
            NEW.roster("unregistered")

    def test_exact_npz_bytes_canonical_fingerprints_and_tamper_detection(self):
        rng=np.random.default_rng(431)
        fitted=FIT.fit_only(rng.normal(size=(4,81,512)).astype(np.float32))
        registered=OLD.fingerprint(NEW.roster("fit_30"))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            npz=base/"fit.npz";manifest=base/"fit-manifest.json"
            original=PACK.make_pack(fitted,registered,PACK.MODEL_SHA,npz,manifest)
            restored,other,manifest_sha=PACK.load_pack(npz,manifest)
            self.assertEqual(original,other)
            self.assertEqual(original["fit_sha256"],fitted.fingerprint)
            self.assertEqual(len(manifest_sha),64)
            np.testing.assert_array_equal(restored.basis,fitted.basis)
            np.testing.assert_array_equal(restored.physical_decoder,fitted.physical_decoder)
            with np.load(npz,allow_pickle=False) as loaded:
                bad={key:loaded[key].copy() for key in loaded.files}
            bad["encoder_bias"][0]+=.01
            np.savez_compressed(npz,**bad)
            with self.assertRaises(ValueError):
                PACK.load_pack(npz,manifest)

    def test_registered_source_manifest_rejects_fake_pilot_or_checkpoint(self):
        fake={"schema":"LS-SAE04/preheldout-source-sealed-native-512D-JVP-v1",
              "preregistration_commit":PACK.PROTOCOL_COMMIT,
              "source_checkpoint_sha256":"0"*64}
        with self.assertRaises((ValueError,KeyError)):
            AUD.audit(fake)

    def test_source_only_pack_rejects_bad_checkpoint(self):
        rng=np.random.default_rng(40)
        fit=FIT.fit_only(rng.normal(size=(4,81,512)).astype(np.float32))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                PACK.make_pack(fit,"anything","0"*64,
                     Path(directory)/"frozen.npz",Path(directory)/"manifest.json")


if __name__=="__main__":
    unittest.main()
