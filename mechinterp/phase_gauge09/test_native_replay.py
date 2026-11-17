"""COE-09 native source receipt guards, using a synthetic fixture only."""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
try:
    from .native_label_replay import load_native,replay,REPS,HEADS,SOURCE_SCHEMA,SOURCE_MODEL_SHA,FIT_STRATA,TEST_STRATA
    from .label_alignment import projected_words
except ImportError:
    from native_label_replay import load_native,replay,REPS,HEADS,SOURCE_SCHEMA,SOURCE_MODEL_SHA,FIT_STRATA,TEST_STRATA
    from label_alignment import projected_words

class NativeReplayTests(unittest.TestCase):
    def fake(self,folder:Path):
        arrays={r:np.zeros((4,9,8,11 if r=='visible' else 512),dtype=np.float32) for r in REPS}
        for rep in REPS:
            for s in range(4):
                for d in range(9):
                    for t in range(8):
                        arrays[rep][s,d,t,0] = float((d+t+s)%3)
        path=folder/'native.npz'
        np.savez_compressed(path,**arrays)
        receipt={'schema':SOURCE_SCHEMA,'source_checkpoint_sha256':SOURCE_MODEL_SHA,
            'source_benchmark_sha256':'fake-manifest-unit-only',
            'fit_strata':list(FIT_STRATA),'heldout_strata':list(TEST_STRATA),
            'visits':8,'num_digits':9,'pairs_per_stratum':36,
            'metadata':{'activation_sha256':{}},'representations':{}}
        for rep,arr in arrays.items():
            receipt['metadata']['activation_sha256'][rep]=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            v=[1.]+[0.]*(arr.shape[-1]-1)
            code=projected_words(arr,np.array(v),0.5)
            receipt['representations'][rep]={head:{'direction':v,'threshold':0.5,
                                        'heldout_codes':code[2:].tolist()}
                                 for head in HEADS}
        return path,receipt

    def test_exact_fixture_replay(self):
        with tempfile.TemporaryDirectory() as td:
            path,rcpt=self.fake(Path(td))
            src=load_native(path,rcpt)
            res=replay(src,rcpt,permutations=6)
            self.assertEqual(res['denominator_per_head'],18)
            self.assertEqual(set(res['results']),set(REPS))
            self.assertEqual(res['results']['visible']['cyclic']['modes']['ordered']['total'],18)

    def test_tamper_digest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path,rcpt=self.fake(Path(td))
            rcpt['metadata']['activation_sha256']['dark']='0'*64
            with self.assertRaises(ValueError):load_native(path,rcpt)

    def test_tamper_heldout_code_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path,rcpt=self.fake(Path(td))
            src=load_native(path,rcpt)
            rcpt['representations']['dark']['ordered']['heldout_codes'][0][0]^=1
            with self.assertRaises(ValueError):replay(src,rcpt,permutations=2)

    def test_wrong_strata_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path,rcpt=self.fake(Path(td))
            rcpt['heldout_strata'].reverse()
            with self.assertRaises(ValueError):load_native(path,rcpt)

if __name__=='__main__':unittest.main()
