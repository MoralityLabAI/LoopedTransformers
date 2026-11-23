from unittest import TestCase

import torch

from configs.run_ppt_mathcan_sparse import Transition, observed_support, play
from configs.run_ppt_native_feedback import typed_batch
from configs.run_ppt_mathcan_progress import factor_targets
from ppt_gym.mathcan_deadline import deadline_utility, supported_distances
from ppt_gym.mathcan_progress import progress_curve, split_completion_curve
from ppt_gym.native_feedback import PrefixGraph, SuccessorFeedbackRoute, closure_curve, horizon_labels
from ppt_gym.premise_budget import BudgetRecurrence, budget_state, shift_budget
from ppt_gym.priced_fragments import FragmentGraph
from ppt_gym.data import digest
from ppt_gym.two_slot import TwoSlotPolicy
from ppt_gym.mathcan_sparse import Frame, RecordedSearch


class Archive:
    key = ("MathCan/Example.lean", "example", ())
    states = {key: {name: {"state_sha256": "state", "source_sha256": "source"}
                    for name in ("reject", "unknown", "close")}}

    def step(self, key, action):
        return {"reject": ("rejected", key),
                "unknown": ("censored", None),
                "close": ("archived_closure", None)}[action]


class SparseSearchTests(TestCase):
    def test_censor_has_no_return_and_rejection_preserves_the_prefix(self):
        search = RecordedSearch(Archive())
        initial = Frame(Archive.key, 3)
        after = search.step(initial, "reject")["after"]
        self.assertEqual(after.key, initial.key)
        self.assertEqual(after.left, 2)
        self.assertNotIn("reject", search.menu(after))
        censored = play(search, initial.key, 3, lambda _: "unknown")
        self.assertTrue(censored["censored"])
        self.assertIsNone(censored["observed_return"])
        failed = play(search, initial.key, 1, lambda _: "reject")
        self.assertFalse(failed["censored"])
        self.assertEqual(failed["observed_return"], -.05)

    def test_support_transfers_across_histories_only_at_same_prefix(self):
        search = RecordedSearch(Archive())
        initial = Frame(Archive.key, 3)
        after = search.step(initial, "reject")["after"]
        tape = [Transition(initial, "reject", after, "rejected", False, None, False),
                Transition(initial, "close", None, "archived_closure", True, 1, True),
                Transition(Frame(("another", "example", ()), 3), "unknown", None,
                           "archived_closure", True, 1, True)]
        self.assertEqual(observed_support(search, tape)[after], {"close"})

    def test_deadline_backup_joins_sampled_continuations_and_charges_retries(self):
        initial = Frame(Archive.key, 8)
        next_key = (*Archive.key[:2], ("intro",))
        next_state = Frame(next_key, 8)
        tape = [(initial, "reject", "rejected", initial.key, None),
                (initial, "intro", "accepted_nonterminal", next_key, None),
                (next_state, "close", "archived_closure", None, None),
                (initial, "unknown", "censored", None, None)]
        self.assertEqual(supported_distances(tape), [3, 2, 1])
        # The slow tactic can have stronger eventual success and still miss the deadline.
        curves = torch.tensor([[[.7] * 8, [0., 0.] + [1.] * 6]])
        self.assertEqual(int(deadline_utility(curves, torch.tensor([2])).argmax()), 0)
        self.assertEqual(int(deadline_utility(curves, torch.tensor([8])).argmax()), 1)

    def test_disposition_targets_distinguish_censor_and_rejection(self):
        frame = Frame(Archive.key, 8)
        rows = factor_targets([(frame, "reject", "rejected", frame.key, (None, 8)),
                               (frame, "unknown", "censored", None, (None, 8)),
                               (frame, "close", "archived_closure", None, (1, 8))])
        self.assertEqual([row.accepted for row in rows], [False, True, True])
        self.assertEqual([row.backed_up for row in rows], [(9, 0), (9, 0), (1, 8)])

    def test_retry_consumes_a_check_and_excludes_the_rejected_action(self):
        acceptance = torch.tensor([[0.25, 0.75]])
        conditional = torch.tensor([[[1., 1., 1.], [0.5, 0.5, 1.]]])
        curves = progress_curve(acceptance, conditional)
        expected = torch.tensor([[[.25, .53125, .53125], [.375, .4375, .8125]]])
        torch.testing.assert_close(curves, expected)
        torch.testing.assert_close(progress_curve(acceptance[:, :1], conditional[:, :1]), conditional[:, :1] * .25)

    def test_split_completion_keeps_one_check_mass_and_conditional_tail(self):
        forecast = split_completion_curve(torch.tensor([[.25, .3, .4]]), torch.tensor([[.5, .75, 1.]]))
        torch.testing.assert_close(forecast, torch.tensor([[.25, .625, 1.]]))

    def test_native_frontier_is_unknown_until_a_checked_closure_exists(self):
        graph = PrefixGraph({"nodes": [{"prefix": [], "choices": {
            "intro": {"status": "progress"}, "reject": {"status": "rejected"}}}]})
        self.assertEqual(horizon_labels(*graph.backed_up(())), (0, -1, -1, -1, -1, -1, -1, -1))
        self.assertEqual(horizon_labels(3, 2), (0, -1, 1, 1, 1, 1, 1, 1))

    def test_native_progress_and_retry_spend_the_current_check(self):
        disposition = torch.tensor([[[.1, .6, .2, .1], [.5, 0., 0., .5]]])
        delayed = torch.tensor([[[.5, .5, .5], [0., 0., 0.]]])
        torch.testing.assert_close(closure_curve(disposition, delayed),
                                   torch.tensor([[[.1, .55, .55], [.5, .55, .7]]]))

    def test_successor_critic_reads_both_proof_slots(self):
        model = SuccessorFeedbackRoute(TwoSlotPolicy(bindings=True)).eval()
        with torch.no_grad():
            model.delta.weight.zero_()
            model.delta.bias.zero_()
            goals = torch.randn(1, 2, 32)
            tokens = torch.randn(1, 3, 2, 32)
            logits, successors = model.successor(tokens, goals)
            torch.testing.assert_close(successors, goals[:, None].expand(-1, 3, -1, -1))
            self.assertEqual(logits.shape, (1, 3, 4))
            curve = model.continuation(successors)
            self.assertTrue(bool((curve[..., 1:] >= curve[..., :-1]).all()))

    def test_native_tactic_prompt_keeps_structured_hypothesis_type(self):
        class ObserveMenu:
            def batch(self, _visible, menu, _history, _remaining):
                return menu

        public = {"goals": [{"hypotheses": [{"name": "r0", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}}]}]}
        encoded = typed_batch(ObserveMenu(), public, ("apply r0", "rfl"), (), 2)
        self.assertEqual(encoded, ["apply r0\n{\"imp\":[{\"a\":\"P0\"},{\"a\":\"P1\"}]}", "rfl"])

    def test_premise_costs_sum_and_repeat_under_one_budget(self):
        model = BudgetRecurrence()
        mass = torch.zeros(3, 9)
        mass[0, 0], mass[1, 1], mass[2, 2] = 1., 1., 1.
        choices = torch.tensor([[1, 2], [2, 2], [1, 0]])
        actual = model.joint(mass, choices)
        self.assertEqual(actual.argmax(-1).tolist(), [3, 4, 1])
        public = {"goals": [
            {"target": {"a": "P1"}, "hypotheses": [
                {"name": "r0", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}}]},
            {"target": {"a": "P0"}, "hypotheses": []},
        ]}
        state = budget_state(public, ("apply r0",), torch.zeros(1, 8))
        self.assertEqual(state.projected, ((1, 1),))

    def test_priced_backup_can_join_checked_fragments_without_an_episode_return(self):
        root, child = {"goals": [{"target": {"a": "P1"}}]}, {"goals": [{"target": {"a": "P0"}}]}
        start, end = digest(root), digest(child)
        graph = FragmentGraph({
            "prices": {"apply r0": 1, "apply h0": 2},
            "public_states": {start: {"public": root, "menu": ["apply r0"]},
                              end: {"public": child, "menu": ["apply h0"]}},
            "fragments": [
                {"start": start, "allowance": 1, "steps": [
                    {"before": start, "after": end, "action": "apply r0", "cost": 1, "status": "progress"}]},
                {"start": end, "allowance": 2, "steps": [
                    {"before": end, "after": None, "action": "apply h0", "cost": 2, "status": "closed"}]},
            ],
        })
        backed_up, empirical = graph.labels(start, "apply r0")
        self.assertEqual(backed_up, (0, 0, 1, 1, 1, 1, 1, 1))
        self.assertEqual(empirical, (0, -1, -1, -1, -1, -1, -1, -1))
        self.assertEqual(graph.certificate(start), (3, ["apply r0", "apply h0"]))
        impulse = torch.ones(1, 9)
        cost = shift_budget(impulse, torch.tensor([3]), start=1)
        torch.testing.assert_close(cost, torch.tensor([[0., 0., 1., 1., 1., 1., 1., 1.]]))
