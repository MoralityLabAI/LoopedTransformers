import json
import unittest
from routing import parse_rows, simulate, train_prior


def row(token, ids, p=0, layer=6, root="r", turn="a"):
    return json.dumps({"root":root, "turn":turn, "token":token, "layer":layer,
                       "pass":p, "experts":list(ids), "weights":[1.0]*8})


class RoutingTests(unittest.TestCase):
    def test_turn_vs_token_oracle(self):
        rows = parse_rows([row(0, range(8)), row(1, range(8,16))])
        whole = simulate(rows, 8, 0, 10)
        token = simulate(rows, 8, 1, 10)
        self.assertEqual(whole["root_mean_retained_mass"], .5)
        self.assertEqual(token["root_mean_retained_mass"], 1.)
        self.assertEqual(whole["total_transfer_bytes"], 80)
        self.assertEqual(token["total_transfer_bytes"], 160)

    def test_shared_cache_across_passes(self):
        rows = parse_rows([row(0, range(8), p=p) for p in range(3)])
        result = simulate(rows, 8, 1, 10)
        self.assertEqual(result["total_transfer_bytes"], 80)
        self.assertEqual(result["roots"]["r"]["visits"], 3)

    def test_previous_cannot_read_current(self):
        rows = parse_rows([row(0, range(8,16)), row(1, range(8,16))])
        result = simulate(rows, 8, 1, 10, "previous")
        self.assertEqual(result["root_mean_retained_mass"], .5)
        self.assertEqual(result["total_transfer_bytes"], 160)

    def test_prior(self):
        train = parse_rows([row(0, range(8,16), root="train")])
        test = parse_rows([row(0, range(8,16))])
        self.assertEqual(simulate(test, 8, 0, 10, "global", train_prior(train))["root_mean_retained_mass"], 1.)

    def test_duplicate_and_incomplete(self):
        r = row(0, range(8))
        with self.assertRaises(ValueError):
            parse_rows([r,r])
        with self.assertRaises(ValueError):
            parse_rows([r], strict=True)

    def test_equal_weight_roots(self):
        rows = parse_rows([row(0, range(8), root="a")] +
                          [row(t, range(8,16), root="b") for t in range(3)])
        result = simulate(rows, 8, 0, 10, "global")
        self.assertEqual(result["root_mean_retained_mass"], .5)


if __name__ == "__main__":
    unittest.main()
