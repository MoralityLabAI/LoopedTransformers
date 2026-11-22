from dataclasses import asdict
from math import inf
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from configs.check_ppt_dependent_witness import action_costs
from configs.prepare_ppt_mathcan_costs import cost_roots
from configs.pretrain_ppt_slot_mixed import mathcan_cost_loss
from configs.probe_ppt_witness_workspace import binder_intervention, observe_fork, prefix_carry, relabel_witness_prefix
from configs.run_ppt_slot_continuations import (
    branch_return, decisions, learn_pair, offline, paired_loss, paired_margins, probe_continuation, replay_paired,
    restore_policy, training_return,
)
from configs.run_ppt_two_slot import SCHEMA, TRM_SCHEMA, calibrate, evaluate, fork_features
from ppt_gym.data import digest, sha256_file
from ppt_gym.dependent_witness import FORK, PREFIX, action_projection, proof_view
from ppt_gym.diagnostic_policy import DiagnosticPolicy
from ppt_gym.slot_bindings import action_bindings
from ppt_gym.slot_path_bank import SlotPathBank
from ppt_gym.slot_witness_bank import SlotWitnessBank, WitnessWorkspace
from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM, candidate_state, text_chunks
from ppt_gym.two_slot_fixture import BUNDLES, CHECKS, FAMILY, GUARD, PROBES, ROUTE, continuation_fixtures, fixtures, menu, registered_tactic


def observation(finish):
    hypotheses = [{"name": name, "type": {"raw": "Nat"}} for name in "abcuvw"]
    equalities = ("a = b", "a = c", "b = c", "b = u", "c = u", f"{finish} = v")
    hypotheses.extend({"name": f"h{i}", "type": {"raw": equality}} for i, equality in enumerate(equalities))
    focus, pending = ({"target": {"raw": f"a = {target}"}, "hypotheses": hypotheses} for target in "uv")
    initial = {"target": {"and": [focus["target"], pending["target"]]}, "hypotheses": hypotheses}
    return [initial], [focus, pending]


class TwoSlotTests(unittest.TestCase):
    def test_opening_weight_changes_only_opening_gradients(self):
        record = {"trials": [{"fork_index": 0, "reward": reward, "success": True} for reward in (.75, .725)]}

        def frames(model, trace):
            yield 0, 0, model[0], None
            yield 1, 0, model[1], None

        gradients = []
        with patch("configs.run_ppt_slot_continuations.decisions", side_effect=frames):
            for weight in (1., 30.):
                model = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2)),
                                                torch.nn.Parameter(torch.zeros(2))])
                optimizer = torch.optim.SGD(model.parameters(), lr=0.)
                learn_pair(model, optimizer, [record, record], .7, opening_weight=weight)
                gradients.append([p.grad.clone() for p in model])
        torch.testing.assert_close(gradients[1][0], 30 * gradients[0][0])
        torch.testing.assert_close(gradients[1][1], gradients[0][1])

    def test_paired_regression_and_rl_use_the_same_targets_and_batch_count(self):
        suffix = ("apply h2", "apply h3")

        def trace(opening, action, reward):
            return {"fork_index": 0, "success": reward > 0, "reward": reward, "events": [
                {"action": opening, "menu": FORK}, {"action": action, "menu": suffix},
            ]}

        records = [{"trials": [trace(opening, suffix[0], reward) for opening, reward in zip(FORK, returns)],
                    "probes": [{"index": 1, "alternatives": [trace(FORK[0], suffix[1], 0.)]}, None]}
                   for returns in ((.75, .725), (.725, .75))]

        def frames(policy, episode):
            scores = policy.weight.flatten()
            for i, row in enumerate(episode["events"]):
                yield i, row["menu"].index(row["action"]), scores, scores[:, None]

        targets = []
        with patch("configs.run_ppt_slot_continuations.decisions", side_effect=frames):
            for regression in (False, True):
                policy = torch.nn.Linear(2, 1, bias=False)
                with patch("configs.run_ppt_slot_continuations.paired_loss", wraps=paired_loss) as loss:
                    self.assertEqual(replay_paired(policy, records, certificate_cost=True, regress_returns=regression), 1)
                targets.append([(torch.as_tensor(call.args[1]).tolist(), call.kwargs.get("selected"))
                                for call in loss.call_args_list])
            self.assertEqual(offline(policy, records, imitate=False, certificate_cost=True), 4)
        self.assertEqual(len(targets[0]), 6)
        self.assertEqual(*targets)

    def test_mathcan_cost_contrasts_use_complete_paths_and_skip_unknown_actions(self):
        start = ("MathCan/Train.lean", "example", ())
        tail = (*start[:2], ("split",))
        choices = {
            start: {"direct": ("archived_closure", None), "split": ("accepted_nonterminal", tail),
                    "unknown": ("censored", None)},
            tail: {"close": ("archived_closure", None)},
        }
        replay = SimpleNamespace(starts=[start], observations={key: {"goals": ["⊢ True"]} for key in choices},
                                 states={key: {action: {"source_sha256": "source"} for action in menu}
                                         for key, menu in choices.items()},
                                 step=lambda key, action: choices[key][action])
        root, = cost_roots(replay)
        state, = root["states"]
        self.assertEqual(state["completed"], {"direct": ("direct",), "split": ("split", "close")})
        self.assertEqual(state["unscored"], {"unknown": "censored"})
        scores = torch.zeros(2, requires_grad=True)

        def policy(_goals, actions, _budget):
            self.assertEqual(actions, ["direct", "split"])
            return scores, None

        mathcan_cost_loss(policy, root).backward()
        self.assertLess(scores.grad[0], scores.grad[1])

    def test_certificate_only_closure_supplies_reward_without_a_policy_decision(self):
        budgets = []

        def policy(_goals, menu, remaining, _carry):
            budgets.append(remaining)
            return torch.zeros(len(menu)), torch.zeros(len(menu), 1, 2)

        policy.advance = lambda _chosen, _next_goals: None
        trace = {"budget": 1, "success": True, "reward": .725, "events": [
            {"goals": [], "menu": ("exact h0", "exact h1"), "action": "exact h0"},
            {"goals": [], "menu": ("exact h2",), "action": "exact h2", "certificate_only": True},
        ]}
        self.assertEqual(len(list(decisions(policy, trace))), 1)
        self.assertEqual(budgets, [1])
        self.assertEqual(training_return(trace, certificate_cost=True), .725)

    def test_lookahead_trm_restores_recorded_depth_without_carry(self):
        trm = TwoSlotTRM(bindings=True, witnesses=True, passes=2)
        fit = {"split": "deep-witness", "witness_workspace": True, "passes": {"trm_lookahead": 2}}
        restored = restore_policy(fit, "trm_lookahead", trm.state_dict())
        self.assertIsInstance(restored, TwoSlotTRM)
        self.assertEqual(restored.passes, 2)
        self.assertFalse(restored.witness_workspace)
        self.assertEqual(restored.macs(2), trm.macs(2))
        self.assertIsNone(restored.advance(None, []))

    def test_exploratory_margin_uses_complete_layouts_and_keeps_paired_losses(self):
        def result(*complete):
            return {"pairs": {str(i): {"complete": ok} for i, ok in enumerate(complete)}}

        report = paired_margins({"live_rl": result(True, False, True, False),
                                 "trm_lookahead": result(False, True, True, False)})
        self.assertEqual(report["layouts"], 4)
        self.assertEqual(report["controls"]["trm_lookahead"],
                         {"live_rl_only": 1, "control_only": 1, "ties": 2, "net_complete_pairs": 0})

    def test_residual_carry_does_not_add_a_duplicate_observation(self):
        initial, _ = observation("b")
        model = TwoSlotPolicy(width=16, residual_carry=True)
        observed = model.encode(initial)
        torch.testing.assert_close(model.advance(observed, initial), torch.zeros_like(observed))

    def test_local_probe_replays_prefix_with_matched_random_draws(self):
        prefix = {"action": "constructor"}
        opening = {"action": BUNDLES[0]}
        event = {"goals": ["public state"], "menu": ["rw [h2]", "rw [h3]"], "action": "rw [h2]"}
        original = {"events": [prefix, opening, event], "success": False, "reward": -.075}
        better = {"events": [prefix, opening, {**event, "action": "rw [h3]"}], "success": True, "reward": .825}
        with patch("configs.run_ppt_slot_continuations.trajectory", side_effect=[original, better]) as rollout:
            trace, probe = probe_continuation(None, None, None, None, BUNDLES[0], random.Random(831))
        a, b = rollout.call_args_list
        self.assertEqual(a.kwargs["draws"], b.kwargs["draws"])
        self.assertEqual(b.kwargs["forced_prefix"], ("constructor", BUNDLES[0], "rw [h3]"))
        self.assertEqual(probe["index"], 2)
        self.assertEqual(branch_return({"trials": [trace], "probes": [probe]}, 0, True), .825)

    def test_long_context_is_chunked_without_losing_unicode_and_charged(self):
        text = "h : α → β ∧ γ\n" * 65
        parts = text_chunks(text, 384)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(part.encode("utf-8")) <= 384 for part in parts))
        model = TwoSlotPolicy(width=16, passes=1)
        goals = [{"target": {"pp": text}, "hypotheses": []}]
        self.assertEqual(model.encode(goals).shape, (1, 16))
        self.assertGreater(model.extra_context_macs(goals), 0)

    def test_directed_slot_bank_distinguishes_two_hops_from_three(self):
        hypotheses = [{"name": f"h{i}", "type": {"raw": edge}} for i, edge in enumerate(
            ("a = b", "a = c", "b = d", "d = u", "c = e", "e = f", "f = u", "u = r"))]
        goals = [{"target": {"raw": "a = u"}, "hypotheses": hypotheses}]
        bank = SlotPathBank(8)
        view, seeds, gates = bank.start(goals, ("rw [h0]", "rw [h1]"))
        first, _ = bank.advance(view, seeds, gates)
        second, _ = bank.advance(view, first, gates)
        torch.testing.assert_close(first[view.read_indices], torch.tensor([.64, 0.]))
        torch.testing.assert_close(second[view.read_indices], torch.tensor([.64, .512]))
        second[view.read_indices].sum().backward()
        self.assertGreater(bank.edge_gate.bias.grad.item(), 0.)

    def test_shared_witness_bank_requires_both_conjuncts_after_binding(self):
        def atom(text):
            return {"raw": text}

        hypotheses = [{"name": f"h{i}", "type": atom(text)} for i, text in enumerate(
            ("P 0", "P 1", "R 0", "S 0", "R 1"))]
        hypotheses.extend({"name": f"h{5 + n}",
                           "type": {"imp": [{"and": [atom(f"R {n}"), atom(f"S {n}")]}, atom(f"Q {n}")]}}
                          for n in (0, 1))
        native = [{"target": atom(text), "hypotheses": hypotheses} for text in ("P ?w", "Q ?w", "Nat")]
        visible = proof_view(native)
        self.assertEqual(action_projection(visible, FORK[0]), [None, atom("Q 0")])
        self.assertEqual(action_projection(visible, FORK[1]), [None, atom("Q 1")])
        self.assertEqual(native[1]["target"], atom("Q ?w"))
        self.assertEqual(action_costs(visible, FORK, 2, hops=1), [inf, inf])
        self.assertEqual(action_costs(visible, FORK, 2, hops=2), [7, inf])
        self.assertEqual(action_costs(visible, FORK, 2), [7, inf])
        bank = SlotWitnessBank(8)
        view, seeds, gates = bank.start(visible, FORK)
        evidence, _ = bank.advance(view, seeds, gates)
        torch.testing.assert_close(evidence[view.read_indices].reshape(2, 2),
                                   torch.tensor([[1., .4096], [1., 0.]]))
        evidence[view.read_indices].sum().backward()
        self.assertGreater(bank.edge_gate.bias.grad.item(), 0.)
        for model in (TwoSlotPolicy(witnesses=True), TwoSlotTRM(witnesses=True)):
            scores, _ = model(visible, FORK, 6)
            (scores[0] - scores[1]).backward()
            self.assertGreater(model.path_bank.read.weight.grad.abs().sum().item(), 0.)

    def test_diagnostic_memory_preserves_the_earlier_write_when_the_clock_changes(self):
        context = torch.tensor([[1., .75, 0., 1.], [1., .69, 0., 0.]])
        events = torch.tensor([[[0, 1], [-1, 0]], [[0, 1], [1, 0]]])
        for warm, is_trm in ((TwoSlotPolicy(), False), (TwoSlotTRM(), True)):
            model = DiagnosticPolicy(warm, context.shape[-1], trm=is_trm)
            memory = model.memory(context, events)
            torch.testing.assert_close(memory[0, 0], memory[1, 0], atol=1e-6, rtol=0.)
            self.assertFalse(torch.equal(memory[0, 1], memory[1, 1]))
            if not is_trm:
                reset = model.memory(context, events, reset=True)
                no_left = model.memory(context[1:], events.new_tensor([[[-1, 0], [1, 0]]]))
                torch.testing.assert_close(reset[1], no_left[0], atol=1e-6, rtol=0.)

    def test_witness_workspace_reuses_only_the_same_public_context(self):
        hypotheses = [{"name": "h0", "type": {"raw": "P 0"}}, {"name": "h1", "type": {"raw": "P 1"}}]
        for n, depth in ((0, 5), (1, 6)):
            head = {"raw": f"Q {n}"}
            for name in "ABCDEF"[:depth]:
                premise = {"raw": f"{name} {n}"}
                hypotheses.append({"name": f"h{len(hypotheses)}", "type": {"imp": [premise, head]}})
                head = premise
            hypotheses.append({"name": f"h{len(hypotheses)}", "type": head})
        prefix = [{"target": {"raw": "∃ n, P n ∧ Q n"}, "hypotheses": hypotheses}]
        fork = [{"target": {"raw": f"{predicate} ?w"}, "hypotheses": hypotheses} for predicate in ("P", "Q")]
        bank = SlotWitnessBank(8)

        def unroll(goals, actions, remembered=None):
            view, values, gates, keys = bank.resume(goals, actions, remembered)
            for _ in range(2):
                values, _ = bank.advance(view, values, gates)
            read = values[view.read_indices].reshape(view.shape)
            return read, WitnessWorkspace(keys, values)

        _, remembered = unroll(prefix, ("apply Exists.intro",))
        warm, _ = unroll(fork, FORK, remembered)
        fresh, _ = unroll(fork, FORK)
        torch.testing.assert_close(warm[:, 1], torch.tensor([.8 ** 6, .8 ** 7]))
        torch.testing.assert_close(fresh[:, 1], torch.zeros(2))
        changed = [{**goal, "hypotheses": hypotheses[:-1]} for goal in fork]
        isolated, _ = unroll(changed, FORK, remembered)
        torch.testing.assert_close(isolated[:, 1], torch.zeros(2))
        warm[:, 1].sum().backward()
        self.assertGreater(bank.edge_gate.bias.grad.item(), 0.)

        models = [policy(witnesses=True, witness_workspace=True, anonymous_witness=anonymous)
                  for policy in (TwoSlotPolicy, TwoSlotTRM) for anonymous in (False, True)]
        for model in models:
            _, predicted = model(prefix, ("apply Exists.intro",), 10)
            carried = model.advance(candidate_state(predicted, 0), fork)
            self.assertIs(carried.workspace, predicted.workspace)
            _, continued = model(fork, FORK, 8, carried)
            self.assertNotEqual(continued.workspace.keys, ())

            event = {"menu": list(FORK), "goals": fork}
            observed = []
            with torch.no_grad(), model.path_bank.read.register_forward_hook(
                    lambda _module, args, _output: observed.append(args[0].clone())):
                natural, channels = observe_fork(model, event, 10, carried)
                identity, _ = observe_fork(model, event, 10, carried, channels)
                donor = tuple(channel.flip(0) for channel in channels)
                observe_fork(model, event, 10, carried, donor)
                recovered, _ = observe_fork(model, event, 10, carried)
            torch.testing.assert_close(identity, natural, rtol=0, atol=0)
            torch.testing.assert_close(recovered, natural, rtol=0, atol=0)
            for i in range(2):
                torch.testing.assert_close(observed[i + 4][::2], observed[i][::2], rtol=0, atol=0)
                torch.testing.assert_close(observed[i + 4][1::2], donor[i], rtol=0, atol=0)

            targets = [goal["target"] for goal in fork]
            conjunction = [{"target": {"and": targets}, "hypotheses": hypotheses}]
            events = [{"goals": goals, "menu": [action], "action": action}
                      for goals, action in zip((prefix, conjunction), PREFIX, strict=True)]
            events.append(event)
            renamed = relabel_witness_prefix(events)
            self.assertEqual(relabel_witness_prefix(renamed), events)
            self.assertEqual(renamed[-1]["goals"][0]["hypotheses"][0]["name"], "h1")
            with torch.no_grad():
                carry = prefix_carry(model, events, 10)
                logits, signals = observe_fork(model, event, 10, carry)
                result = binder_intervention(model, events, 10, carry, logits, signals)
            torch.testing.assert_close(torch.tensor(result["menu_reversed"]), logits)
            if model.anonymous_witness:
                torch.testing.assert_close(torch.tensor(result["renamed"]), logits, atol=1e-6, rtol=0.)
                cue = model.cues(fork, FORK, 10 - len(PREFIX))
                torch.testing.assert_close(cue[0], cue[1], atol=0., rtol=0.)
                self.assertFalse(signals[0][0].equal(signals[0][1]))

    def test_certificate_cost_retains_success_cost_and_ties_failures(self):
        self.assertEqual(training_return({"reward": -.075, "success": False}, True), 0.)
        self.assertEqual(training_return({"reward": -.175, "success": False}, True), 0.)
        self.assertAlmostEqual(training_return({"reward": .825, "success": True}, True), .825)
        self.assertAlmostEqual(training_return({"reward": -.075, "success": False}), -.075)

    def test_binding_swap_changes_pending_support_for_only_the_matching_candidate(self):
        _, left = observation("b")
        _, right = observation("c")
        for goal in (left, right):
            self.assertEqual(action_bindings(goal, BUNDLES[0])[0][4], action_bindings(goal, BUNDLES[1])[0][4])
        self.assertEqual(action_bindings(left, BUNDLES[0])[1][4], 1.)
        self.assertEqual(action_bindings(left, BUNDLES[1])[1][4], 0.)
        self.assertEqual(action_bindings(right, BUNDLES[0])[1][4], 0.)
        self.assertEqual(action_bindings(right, BUNDLES[1])[1][4], 1.)
        self.assertEqual(action_bindings(left, "rw [h0]")[1][4], 0.)
        for model in (TwoSlotPolicy(bindings=True), TwoSlotTRM(bindings=True)):
            scores, _ = model(left, BUNDLES, CHECKS - 1)
            scores.sum().backward()
            self.assertGreater(float(model.binding.weight.grad.abs().sum()), 0.)

    def test_prefix_carry_changes_fork_scores_and_receives_gradient(self):
        initial, goals = observation("b")
        model = TwoSlotPolicy(width=16, passes=1)
        _, predicted = model(initial, menu(initial, 0), CHECKS)
        carry = model.advance(predicted[0], goals)
        scores, _ = model(goals, menu(goals, 1), CHECKS - 1, carry)
        reset, _ = model(goals, menu(goals, 1), CHECKS - 1)
        self.assertEqual(carry.shape, (2, 16))
        self.assertFalse(torch.allclose(scores, reset))
        (scores[0] - scores[1]).backward()
        self.assertGreater(float(model.write.weight.grad.abs().sum()), 0.)

    def test_calibration_fits_only_the_actor(self):
        records = []
        for number in range(2):
            for finish in ("b", "c"):
                initial, goals = observation(finish)
                events = [{"goals": initial, "menu": ("constructor",)}, {"goals": goals, "menu": BUNDLES}]
                rewards = (.9, -.1) if finish == "b" else (-.1, .9)
                records.append({"case": {"cluster": str(number)}, "trials": [
                    {"events": events, "reward": reward} for reward in rewards]})
        model = TwoSlotPolicy(width=16, passes=1)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        calibrate(model, records)
        self.assertFalse(torch.equal(model.actor.weight, before["actor.weight"]))
        for name, value in model.state_dict().items():
            if not name.startswith("actor."):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(fork_features(model, records[0])).all())

    def test_registered_pairs_and_bundles(self):
        training = {case.root for case, _ in fixtures("train")}
        dev = list(fixtures("dev"))
        structures = {case.family for case, _ in dev}
        self.assertEqual(structures, set(PROBES) - {ROUTE})
        for family in structures:
            probe = PROBES[family]
            self.assertEqual(tuple(winner for case, winner in dev if case.family == family), probe.fork)
        self.assertTrue(all(case.family == FAMILY for case, _ in fixtures("train")))
        self.assertFalse(training.intersection(case.root for case, _ in dev))
        self.assertTrue(all(registered_tactic(action) for action in BUNDLES))
        self.assertFalse(registered_tactic("all_goals rw [h9]"))

    def test_continuation_layout_pairs_are_disjoint(self):
        train = list(continuation_fixtures("train"))
        dev = list(continuation_fixtures("dev"))
        confirm = list(continuation_fixtures("confirm"))
        counterfactual = list(continuation_fixtures("counterfactual"))
        self.assertEqual(len(confirm), 48)
        self.assertEqual(len(counterfactual), 48)
        self.assertEqual((len(train), len(dev)), (24, 48))
        self.assertFalse({case.cluster for case, _ in train} & {case.cluster for case, _ in dev})
        self.assertFalse({case.cluster for case, _ in train + dev} & {case.cluster for case, _ in confirm})
        self.assertFalse({case.cluster for case, _ in train + dev + confirm}
                         & {case.cluster for case, _ in counterfactual})
        for left, right in zip(dev[::2], dev[1::2], strict=True):
            self.assertEqual(left[0].cluster, right[0].cluster)
            self.assertEqual((left[1], right[1]), BUNDLES)
            self.assertNotEqual(left[0].statement, right[0].statement)

    def test_all_arms_score_the_same_structural_pairs(self):
        repl = SimpleNamespace(version="4.19.0")
        tree = SimpleNamespace(stats=dict)
        learned = TwoSlotPolicy().state_dict()
        calibrated = {**learned, "actor.bias": learned["actor.bias"] + 1}
        trm = TwoSlotTRM().state_dict()
        calls = []

        def score(_repl, _tree, case, *, policy, reset_slots):
            calls.append((case.root, float(policy.actor.bias.detach()), reset_slots))
            return {"success": not (case.family == GUARD and case.tid.endswith("_1"))}

        with TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "fit.pt"
            torch.save({"schema": SCHEMA, "lean": repl.version, "learned": learned, "calibrated": calibrated,
                        "training": digest([asdict(case) for case, _ in fixtures("train")])}, checkpoint)
            control = Path(scratch) / "trm.pt"
            torch.save({"schema": TRM_SCHEMA, "parent_sha256": sha256_file(checkpoint), "weights": trm}, control)
            with patch("configs.run_ppt_two_slot.trajectory", side_effect=score):
                report = evaluate(repl, tree, checkpoint, trm_checkpoint=control)

        dev = [case.root for case, _ in fixtures("dev")]
        for arm in report["results"].values():
            self.assertEqual((arm["complete_pairs"], arm["proofs"]), (2, 5))
            self.assertEqual(len(arm["pairs"]), 3)
        for weights, reset in ((learned, False), (calibrated, False), (learned, True), (trm, True)):
            self.assertEqual(calls[:len(dev)], [(root, float(weights["actor.bias"]), reset) for root in dev])
            del calls[:len(dev)]
        self.assertFalse(calls)
        arms = report["results"]
        self.assertAlmostEqual(arms["trm"]["parameters"] / arms["persistent"]["parameters"], 1., delta=.03)
