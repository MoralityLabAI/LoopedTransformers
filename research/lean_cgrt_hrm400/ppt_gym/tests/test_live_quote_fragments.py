from unittest import TestCase

from ppt_gym.bound_quotes import outcomes
from ppt_gym.data import digest
from ppt_gym.live_quote_fragments import NativeFragments, NativeQuoteBackup
from ppt_gym.private_quotes import QUOTES


def fixture(*, unobserved=False):
    public = [{"goals": [index]} for index in range(4)]
    root, inner, left, right = keys = [digest(value) for value in public]
    menus = [("constructor",), ("apply r0", "apply r1"), ("apply h0",), ("apply h1",)]
    if unobserved:
        menus[1] += ("apply r2",)
    states = {key: {"public": state, "menu": menu} for key, state, menu in zip(keys, public, menus, strict=True)}

    def edge(before, action, after):
        return {"before": before, "action": action, "after": after, "status": "progress" if after else "closed"}

    fragments = [{"start": root, "steps": [edge(root, "constructor", inner), edge(inner, "apply r0", left)]},
                 {"start": inner, "steps": [edge(inner, "apply r1", right), edge(right, "apply h1", None)]},
                 {"start": left, "steps": [edge(left, "apply h0", None)]}]
    billing = [{"bits": dict.fromkeys(QUOTES, world),
                "prices": {"constructor": 1, "apply r0": 1 + 2 * world, "apply r1": 3 - 2 * world,
                           "apply h0": 1, "apply h1": 1}} for world in range(2)]
    return {"root": root, "states": states, "fragments": fragments}, billing


class LiveQuoteFragmentTests(TestCase):
    def test_stitched_proof_credits_a_paid_quote_inside_a_native_proof(self):
        log, billing = fixture()
        native = NativeFragments(log)
        model = NativeQuoteBackup(native, dict.fromkeys(QUOTES, 1), billing)
        no_quotes = [model.solve(world, 4, 0) for world in range(2)]
        with_quotes = [model.solve(world, 4, 1) for world in range(2)]
        self.assertEqual(sum(row["proof"] for row in no_quotes), 1)
        self.assertTrue(all(row["proof"] and row["quote_effort"] == 1 for row in with_quotes))
        for row in with_quotes:
            self.assertEqual(row["script"][0], "constructor")
            self.assertIn("quote", row["trace"][1])
        self.assertFalse(any(fragment["start"] == native.root and fragment["steps"][-1]["status"] == "closed"
                             for fragment in log["fragments"]))

    def test_unobserved_alternative_stays_unknown(self):
        log, billing = fixture(unobserved=True)
        model = NativeQuoteBackup(NativeFragments(log), dict.fromkeys(QUOTES, 1), billing)
        known, possible = model.value(log["root"], 4, 8, model.initial_belief, 0, 0)
        self.assertEqual((known, possible), (.5, 1.))

    def test_missing_anchor_is_rejected(self):
        log, _ = fixture()
        log["fragments"] = log["fragments"][1:] + log["fragments"][:1]
        with self.assertRaisesRegex(ValueError, "restarted outside"):
            NativeFragments(log)

    def test_shared_late_detour_uses_opposite_native_tariff(self):
        case = {
            "plans": (("apply r0", "apply r2"), ("apply r1", "apply r2")),
            "graph": {"rules": (([1], 0), ([2], 0), ([3], 2))},
            "channels": {"ledger_u": {"rule": "apply r2"}, "ledger_v": {"rule": "apply h0"},
                         "reconcile": {"rule": "apply r0", "counterpart": "apply r1"}},
            "late_link": {"quote": "ledger_u", "counterpart": "apply r3"},
        }
        for tariffs, bits in outcomes(case):
            bit = bits["ledger_u"]
            self.assertEqual(tariffs["apply r2"], 1 + 3 * bit)
            self.assertEqual(tariffs["apply r3"], 1 + 3 * (1 - bit))
