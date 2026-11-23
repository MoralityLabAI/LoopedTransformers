from unittest import TestCase

import torch

from ppt_gym.live_quote_meter import fit_meter, predict_billing, public_graph
from ppt_gym.live_quote_policy import FEATURES, LiveQuoteTRM, PublicQuoteFeatures, TwoSlotQuoteActor


class LiveQuotePolicyTests(TestCase):
    def setUp(self):
        self.case = {"graph": {"tid": "test", "family": "alt-dead", "depth": 2, "seed": 1,
                               "n_atoms": 5, "facts": [0, 1], "rules": [([0], 2), ([1], 2), ([0], 3), ([1], 3)],
                               "goal": [2, 3]},
                     "channels": {"ledger_u": {"rule": "apply r2"}, "ledger_v": {"rule": "apply h0"},
                                  "reconcile": {"rule": "apply r0", "counterpart": "apply r1"}},
                     "late_link": {"quote": "ledger_u", "counterpart": "apply r3"}}
        observed = {"words": ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
                    "costs": {"late": (1, 4), "flag": (1, 2), "band_1": (1, 6)}}
        self.case["billing"] = predict_billing(self.case, observed)
        self.estimated = fit_meter([self.case])

    def test_meter_recovers_observed_prices_and_binds_both_directions(self):
        self.assertEqual(self.estimated["costs"]["late"], (1, 4))
        self.assertEqual(predict_billing(self.case, self.estimated), self.case["billing"])
        features = PublicQuoteFeatures(self.case, self.estimated)
        for world, row in enumerate(self.case["billing"]):
            support = features.posterior((("ledger_v", row["bits"]["ledger_v"]),),
                                         (("apply r3", row["prices"]["apply r3"]),))
            self.assertEqual(support, [world])

    def test_public_grammar_has_both_late_derivations(self):
        graph = public_graph(self.case)
        root = graph.edges[graph.root]["constructor"]["after"]
        pending = graph.edges[root]["apply r0"]["after"]
        late = graph.edges[pending]["apply h0"]["after"]
        self.assertEqual(set(graph.states[late]["menu"]), {"apply r2", "apply r3"})

    def test_both_actors_use_same_masked_public_tensor_with_finite_gradients(self):
        public = {"goals": [{"target": {"a": "P3"}}]}
        features = PublicQuoteFeatures(self.case, self.estimated)
        view = {"fees": {"ledger_u": 1, "ledger_v": 3, "reconcile": 4}, "budget": 8}
        menu, tensor = features.encode(public, ("apply r2", "apply r3"), view, 2, 2, (), (), {})
        self.assertEqual(menu, ("apply r2", "apply r3", "ledger_u"))
        self.assertEqual(tuple(tensor.shape), (3, 3, FEATURES))
        for model in (TwoSlotQuoteActor(), LiveQuoteTRM()):
            model(tensor).square().mean().backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
