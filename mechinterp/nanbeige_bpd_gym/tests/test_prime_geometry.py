import unittest
import numpy as np
from mechinterp.nanbeige_bpd_gym.audit_prime import vector_parity,parity_groups


class PrefixGeometryTests(unittest.TestCase):
    def test_byte_closure_distinguishes_signed_zero_and_undefined_scale(self):
        zero=np.zeros(3,dtype=np.float32);negative=zero.copy();negative[0]=-0.
        self.assertTrue(vector_parity(zero,zero)['bit_exact'])
        r=vector_parity(zero,negative)
        self.assertFalse(r['bit_exact']);self.assertIsNone(r['relative_l2']);self.assertEqual(r['max_abs'],0.)

    def test_repeated_variants_keep_equal_root_weight(self):
        key=(21,'residual_after_layer',1,'pointer',False)
        sample=lambda v:dict(bit_exact=False,max_abs=v,relative_l2=v)
        result=parity_groups({key:[('a',sample(0.))]*4+[('b',sample(2.))]})[0]
        self.assertEqual(result['source_roots'],2);self.assertEqual(result['vectors'],5)
        self.assertEqual(result['root_macro_relative_l2'],1.)


if __name__=='__main__':unittest.main()
