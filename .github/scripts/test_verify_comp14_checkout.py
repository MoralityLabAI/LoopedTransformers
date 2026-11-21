import hashlib
import unittest
from verify_comp14_checkout import registered_mode


class CheckoutBytes(unittest.TestCase):
    def test_only_two_named_legacy_dependencies_allow_whole_file_crlf(self):
        raw = b'first\nsecond\n'; expected = hashlib.sha256(b'first\r\nsecond\r\n').hexdigest()
        named = 'mechinterp/dark_fiber_gyms/arg03_nanbeige.py'
        self.assertIsNotNone(registered_mode(named, raw, expected, dependency=True))
        self.assertIsNone(registered_mode('other.py', raw, expected, dependency=True))
        self.assertIsNone(registered_mode(named, raw, expected, dependency=False))
        self.assertIsNone(registered_mode(named, b'changed\nsecond\n', expected, dependency=True))

    def test_exact_bytes_need_no_conversion(self):
        raw = b'fixed\r\n'; expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(registered_mode('any.py', raw, expected), 'exact')


if __name__ == '__main__':
    unittest.main()
