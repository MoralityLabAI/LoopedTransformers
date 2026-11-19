"""Forensic semantics that materially affect interpretation of old claims."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("deuterium_records", Path(__file__).parents[1] / "records.py")
records = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records)


def graft(**kwargs):
    d = {key: 0.0 for key in records.SIGNALS}
    d.update(accuracy_delta=0.01, false_commit_rate_delta=0.0, false_reject_rate_delta=0.0)
    return {"deltas": d, "scale": 1.0, "mean_abs_logit_delta": 0.1, **kwargs}


class HistoricalSemanticsTests(unittest.TestCase):
    def test_lexicographic_control_win_can_tie_every_task_metric(self):
        candidate = graft(mean_abs_logit_delta=0.01)
        random = graft(mean_abs_logit_delta=0.1)
        self.assertEqual(records.rank_key(candidate)[:4], records.rank_key(random)[:4])
        self.assertGreater(records.rank_key(candidate), records.rank_key(random))

    def test_historical_guardrails_allow_five_point_regressions(self):
        row = graft()
        row["deltas"].update(accuracy_delta=-0.05, false_reject_rate_delta=0.05,
                             repair_preservation_rate_delta=0.01)
        self.assertTrue(records.guardrail(row))
        row["deltas"]["false_reject_rate_delta"] = 0.050001
        self.assertFalse(records.guardrail(row))

    def test_score_file_holdout_can_repeat_training_examples(self):
        row = {"trajectory_id": "same", "target_action": "A", "scores": [{"action": "A", "logprob": 0}]}
        cards = [{"name": "train", "rows": [row]}, {"name": "heldout", "rows": [dict(row)]}]
        audit = records.overlap_analysis(cards, "arc")
        self.assertEqual(audit["unique_ids"], 1)
        self.assertEqual(audit["folds"][1]["unseen_in_other_cards"], 0)

    def test_wordle_episode_and_event_identity_are_distinct(self):
        row = {"trajectory_id": "w_808", "step": 0, "target_action": "[crane]",
               "scores": [{"action": "[crane]", "logprob": 0}]}
        next_step = {**row, "step": 6, "target_action": "[swept]"}
        cards = [{"name": "card", "rows": [row, next_step]}]
        events = records.overlap_analysis(cards, "wordle", include_step=True)
        episodes = records.overlap_analysis(cards, "wordle")
        self.assertEqual(events["unique_ids"], 2)
        self.assertEqual(episodes["unique_ids"], 1)
        self.assertEqual(events["target_conflict_ids"], [])

    def test_wordle_replay_uses_recorded_gate_not_inferred_step(self):
        row = {"trajectory_id": "w", "step": 4, "post_initial_gate": False,
               "target_action": "[crane]", "scores": [{"action": "[crane]", "logprob": 0},
                                                         {"action": "[house]", "logprob": -1}]}
        result = records.score_policy([row], {"penalized_action": "[crane]", "penalty": 8}, wordle=True)
        self.assertEqual(result["policy_hits"], 1)
        self.assertEqual(result["damages"], 0)

    def test_arc_condition_respects_margin_endpoint(self):
        row = {"trajectory_id": "arc", "target_action": "A", "scores": [
            {"action": "D", "logprob": 0}, {"action": "A", "logprob": -0.9}]}
        policy = {"top_action": "D", "runner_up_action": "A", "max_margin": 1, "penalty": 1}
        self.assertEqual(records.score_policy([row], policy)["rescues"], 1)
        row["scores"][1]["logprob"] = -1.01
        self.assertEqual(records.score_policy([row], policy)["rescues"], 0)


if __name__ == "__main__":
    unittest.main()
