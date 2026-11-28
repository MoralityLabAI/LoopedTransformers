import unittest
try:
    import torch
    from spectral import nested_svd, apply_factors, relative_error
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class SpectralTests(unittest.TestCase):
    def test_nested_prefixes(self):
        w = torch.diag(torch.tensor([4., 3., 2., 1.]))
        l, r, s = nested_svd(w, 4)
        errors = [relative_error(w, l, r, k) for k in range(1,5)]
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertLess(errors[-1], 1e-6)
        x = torch.randn(6,4, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(apply_factors(x,l,r,4), x @ w.T)
        self.assertLess(relative_error(w,l,r,4,x), 1e-6)

    def test_reject_integer_storage(self):
        with self.assertRaises(ValueError):
            nested_svd(torch.ones(2,2,dtype=torch.int8), 1)


if __name__ == "__main__":
    unittest.main()
