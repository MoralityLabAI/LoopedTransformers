import base64
import random
import unittest
import zlib

from chantless_bridge import MODEL, REVISION, decode, selection, plan, route


def bitmap(ids, compressed=False):
    raw = bytes(sum(128 >> (i % 8) for i in ids if i // 8 == b) for b in range(16))
    return {"n": 128, "count": len(ids), "enc": "packbits+zlib" if compressed else "packbits",
            "b64": base64.b64encode(zlib.compress(raw) if compressed else raw).decode()}


def mask(support):
    return {"schema": "chantless.mask/v3", "source": {"repo": MODEL, "revision": REVISION}, "experts": support}


class ChantlessBridgeTests(unittest.TestCase):
    def test_bits(self):
        ids = (0, 1, 7, 8, 63, 64, 126, 127)
        for compress in (False, True):
            self.assertEqual(decode(bitmap(ids, compress)), ids)
        self.assertEqual(decode(list(reversed(ids))), ids)

    def test_released_layer_zero(self):
        for b64, count in [("P/fxc3h9f+//uvvvF7uf/A==", 96), ("N4e1QrA8GaCiMgMEEgidWA==", 48)]:
            self.assertEqual(len(decode({"n": 128, "count": count, "b64": b64})), count)

    def test_invalid(self):
        for spec in ([1, 1], [128], [True], {**bitmap([0]), "count": 2}, {**bitmap([0]), "n": 127}):
            with self.assertRaises(ValueError):
                decode(spec)
        with self.assertRaises(ValueError):
            selection(mask({"0": list(range(8))}))
        with self.assertRaises(ValueError):
            selection(mask({str(i): list(range(7)) for i in range(24)}))
        with self.assertRaises(ValueError):
            selection({**mask({}), "source": {"repo": MODEL, "revision": "bad"}})

    def test_plan(self):
        manifest = {"source_model": MODEL, "source_revision": REVISION, "num_layers": 24,
                    "total_experts": 3072, "bundle_format": "gguf_v3_standalone_tensor_bundles",
                    "backbone": {"bytes": 100}, "experts": {str(l): {str(e): {"span_bytes": 10+l} for e in range(128)} for l in range(24)}}
        rows = {str(l): list(range(48)) for l in range(24)}
        result = plan(mask(rows), manifest, "a", "b")
        self.assertEqual(result["expert_span_bytes"], 48*sum(range(10, 34)))
        self.assertEqual(result["capacity_per_layer"], 48)
        self.assertEqual(len(selection(mask({}))[0]), 128)
        with self.assertRaises(ValueError):
            plan(mask(rows), {**manifest, "source_revision": "wrong"}, "a", "b")

    def test_compact_parity(self):
        rng = random.Random(17)
        for k in (8, 12, 16, 32, 48, 96, 128):
            for _ in range(200):
                ids = sorted(rng.sample(range(128), k))
                s = [rng.random() for _ in range(128)]
                b = [rng.uniform(-0.3, 0.3) for _ in range(128)]
                ref = route(s, b, ids, resident=ids[:4])
                compact = sorted(range(k), key=lambda j: (-(s[ids[j]]+b[ids[j]]), j))[:8]
                self.assertEqual(ref["local_ids"], tuple(compact))
                self.assertEqual(ref["ids"], tuple(ids[j] for j in compact))
                self.assertAlmostEqual(sum(ref["weights"]), 8)
                self.assertEqual(ref["miss_ids"], tuple(e for e in ref["ids"] if e not in ids[:4]))

    def test_ties_and_bias(self):
        ids = list(range(16))
        self.assertEqual(route([0.5]*128, [0]*128, ids)["ids"], tuple(range(8)))
        s = [0.5]*128
        s[15] = 0.25
        b = [0]*128
        b[15] = 1
        r = route(s, b, ids)
        self.assertEqual(r["ids"][0], 15)
        self.assertAlmostEqual(r["weights"][0]*2, r["weights"][1])


if __name__ == "__main__":
    unittest.main()
