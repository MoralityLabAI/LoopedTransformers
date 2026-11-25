import tempfile
import unittest
from pathlib import Path
import numpy as np
from activation_io import read_activation
from synthetic_fixture import arrays, write
from analyze import measure, report


class GeometryTests(unittest.TestCase):
    def test_projection(self):
        r=measure(arrays())
        self.assertTrue(r["identity_query_exact"])
        self.assertEqual(r["arms"]["effect"]["effect_donor_projection"],1)
        self.assertEqual(r["arms"]["sham"]["effect_donor_projection"],0)
        self.assertAlmostEqual(r["effect_clean_prefix_l2"],2)
        self.assertIsNone(r["arms"]["identity"]["effect_donor_cosine"])

    def test_bf16_read_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"case.safetensors"
            a=arrays()
            sha=write(p,a)
            got=read_activation(p,sha,3)
            self.assertTrue(all(np.array_equal(a[k],got[k]) for k in a))
            with self.assertRaises(ValueError):
                read_activation(p,"0"*64,3)
            with self.assertRaises(ValueError):
                read_activation(p,sha,4)

    def test_root_denominator(self):
        c={"case_id":"x","root_id":13}
        m={"cases":[c,{"case_id":"y","root_id":13}]}
        r=report(m,[(c,{},arrays())])
        self.assertEqual(r["n_complete_identity_checked_roots"],0)
        self.assertEqual(r["n_saved_complete_cases"],1)
