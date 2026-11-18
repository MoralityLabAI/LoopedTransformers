"""Fixture-only checks; these never open training/development namespaces."""
from dataclasses import replace
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import run_lrm_rev02 as runner
from lrm_rev02_fixture import encode_prefix, encode_stream, initial_posterior, make_pilot_episode


FIXTURE_WEIGHT_SEED = 160224731


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner.configure_cpu()

    def model(self):
        torch.manual_seed(FIXTURE_WEIGHT_SEED)
        return runner.RevisionGRU()

    def test_causal_prefix_and_target_timing(self):
        episode = make_pilot_episode(0, 0, "fixture")
        changed = replace(episode, replacements=tuple(1 - x for x in episode.replacements))
        inputs = torch.tensor([encode_stream(episode, episode.correction_slots),
                               encode_stream(changed, changed.correction_slots)], dtype=torch.float32)
        model = self.model()
        logits, _ = model(inputs)
        self.assertTrue(torch.equal(logits[0, :12], logits[1, :12]))

        # A different full-history teacher cannot change the pre-correction
        # head's direct supervision. The full teacher enters only at 18/24.
        probe = torch.zeros((1, 24, 3), dtype=torch.float32, requires_grad=True)
        initial = torch.tensor([[0.2, 0.3, 0.5]])
        target_a, target_b = torch.tensor([[0.1, 0.2, 0.7]]), torch.tensor([[0.7, 0.2, 0.1]])
        weights = {"initial": .25, "corrected": .25, "reminded": .5}
        loss_a, _ = runner.training_loss(probe, initial, target_a, weights)
        gradient_a, = torch.autograd.grad(loss_a, probe, retain_graph=True)
        loss_b, _ = runner.training_loss(probe, initial, target_b, weights)
        gradient_b, = torch.autograd.grad(loss_b, probe)
        self.assertTrue(torch.equal(gradient_a[:, 11], gradient_b[:, 11]))
        self.assertFalse(torch.equal(gradient_a[:, 17], gradient_b[:, 17]))
        self.assertFalse(torch.equal(gradient_a[:, 23], gradient_b[:, 23]))
        unused = [i for i in range(24) if i not in (11, 17, 23)]
        self.assertEqual(float(gradient_a[:, unused].abs().max()), 0)

    def test_fixture_optimizer_wiring_and_cpu_contract(self):
        episode = make_pilot_episode(0, 1, "fixture")
        model = self.model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4995)
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32
                            for p in model.parameters()))
        before = runner.weights_tensor_sha256(model)
        inputs = torch.tensor([encode_stream(episode, episode.correction_slots)], dtype=torch.float32)
        initial = torch.tensor([list(map(float, initial_posterior(sum(episode.old))))])
        full = torch.tensor([list(map(float, episode.full_reference()))])
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0)
        logits, _ = model(inputs)
        loss, _ = runner.training_loss(logits, initial, full,
                                      {"initial": .25, "corrected": .25, "reminded": .5})
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer.step()
        self.assertNotEqual(before, runner.weights_tensor_sha256(model))

    def test_same_reached_state_and_private_label_exclusion(self):
        episode = make_pilot_episode(0, 2, "fixture")
        model = self.model()
        row = runner.score_episode(model, episode, 0, 2)
        changed = runner.score_episode(model, replace(episode, true_regime=(episode.true_regime + 1) % 3), 0, 2)
        self.assertEqual(row["arms"], changed["arms"])
        self.assertEqual(row["initial_probability"], changed["initial_probability"])
        self.assertEqual(row["paired_effect"], changed["paired_effect"])
        self.assertLessEqual(row["prefix_replay_logits_abs_max"], 1e-6)
        self.assertLessEqual(row["prefix_replay_hidden_abs_max"], 1e-6)
        self.assertEqual(row["prefix_replay_probability_abs_max"],
                         max(abs(a - b) for a, b in zip(row["prefix_replay_probability"],
                             row["arms"]["corrected"]["curve"][0]["probability"])))
        initial_curve = row["arms"]["corrected"]["curve"][0]
        for arm in row["arms"].values():
            self.assertEqual(arm["curve"][0], initial_curve)
            self.assertEqual([point["r"] for point in arm["curve"]], list(range(7)))
            with torch.no_grad():
                logits, _ = model(torch.tensor([encode_stream(episode, arm["slots"])], dtype=torch.float32))
            standalone = runner.posterior_from_logits(logits[0, -1])
            self.assertLessEqual(max(abs(a - b) for a, b in
                                     zip(standalone, arm["curve"][-1]["probability"])), 1e-6)

    def test_scoring_failure_identifies_arm_and_stage(self):
        episode = make_pilot_episode(0, 6, "fixture")
        original = runner.posterior_from_logits
        calls = 0

        def fail_at_second_reminder(logits):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise ArithmeticError("fixture nonfinite readout")
            return original(logits)

        with patch.object(runner, "posterior_from_logits", side_effect=fail_at_second_reminder):
            with self.assertRaises(runner.EpisodeScoreFailure) as failure:
                runner.score_episode(self.model(), episode, 0, 6)
        self.assertEqual(failure.exception.context,
                         {"phase": "reminder", "arm": "corrected", "stage": 2})

    def test_raw_scores_match_conditional_brier_and_root_mean(self):
        model = self.model()
        rows = [runner.score_episode(model, make_pilot_episode(0, i, "fixture"), 0, i)
                for i in (3, 4)]
        independent_effects = []
        for row in rows:
            target = row["target_full"]
            risks = {}
            for name, arm in row["arms"].items():
                for point in arm["curve"]:
                    probability = point["probability"]
                    risk = math.fsum(target[k] * math.fsum((p - int(i == k)) ** 2
                                                           for i, p in enumerate(probability))
                                     for k in range(3))
                    oracle_risk = math.fsum(target[k] * math.fsum((q - int(i == k)) ** 2
                                                                  for i, q in enumerate(target))
                                            for k in range(3))
                    self.assertAlmostEqual(point["expected_brier"], risk, places=12)
                    self.assertAlmostEqual(point["excess"], risk - oracle_risk, places=12)
                    self.assertAlmostEqual(math.fsum(probability), 1.0, places=12)
                risks[name] = arm["curve"][-1]["expected_brier"]
            independent_effects.append(math.fsum(risks[f"random_{b}"] for b in range(4)) / 4
                                       - risks["corrected"])
        means, curves, diagnostics = runner.summarize_rows(rows, runner.load_registration()["diagnostics"])
        self.assertAlmostEqual(means["paired_effect"], math.fsum(independent_effects) / 2, places=12)
        self.assertEqual(len(curves["corrected"]), 7)
        self.assertTrue(diagnostics["numeric_pass"])

    def test_checkpoint_roundtrip_preserves_fixture_forecasts(self):
        model = self.model()
        episode = make_pilot_episode(0, 5, "fixture")
        inputs = torch.tensor([encode_prefix(episode)], dtype=torch.float32)
        with torch.no_grad():
            expected, _ = model(inputs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.pt"
            torch.save(model.state_dict(), path)
            other = runner.RevisionGRU()
            other.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            with torch.no_grad():
                observed, _ = other(inputs)
            self.assertTrue(torch.equal(expected, observed))
            self.assertEqual(runner.weights_tensor_sha256(model), runner.weights_tensor_sha256(other))

    def test_source_freeze_rejects_mutation_and_moved_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True)
            source = repo / "source.py"
            source.write_text("value = 1\n")
            subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            manifest = runner.verify_source(commit, ["source.py"], repo)
            self.assertEqual(manifest["source_hashes"]["source.py"], hashlib.sha256(source.read_bytes()).hexdigest())
            source.write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "differs from commit"):
                runner.verify_source(commit, ["source.py"], repo)
            source.write_text("value = 1\n")
            subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "different commit"], cwd=repo, check=True)
            with self.assertRaisesRegex(ValueError, "HEAD differs"):
                runner.verify_source(commit, ["source.py"], repo)

    def test_budget_stop_preserves_empty_partial_without_opening_sources(self):
        config = runner.load_registration()
        source = {"schema_version": 1, "source_commit": "0" * 40,
                  "source_tree_sha": "1" * 40, "source_hashes": {"fixture": "2" * 64}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt"
            with patch.object(runner, "make_pilot_episode", side_effect=AssertionError("source opened")), \
                    patch.object(runner, "RevisionGRU", side_effect=AssertionError("model initialized")):
                summary = runner.run(config, source, output, runner.Budget(0, 2147483648))
            self.assertEqual(summary["run_status"], "BUDGET_STOP")
            self.assertEqual(summary["verdict"], "ENGINEERING_PILOT_ONLY")
            self.assertEqual(summary["preflight_status"], runner.PREFLIGHT_STATUS)
            self.assertEqual(summary["completed_roots"], [])
            self.assertEqual(summary["n_units_completed"], 0)
            self.assertEqual(summary["planned_roots"], [0, 1, 2, 3])
            self.assertEqual(summary["roots"][0]["development_count"], 0)
            self.assertEqual(summary["roots"][0]["failure_context"],
                             {"phase": "initialization", "root": 0})
            retained = json.loads((output / "root_0" / "root_summary.json").read_text())
            self.assertEqual(retained, summary["roots"][0])
            for relative, digest in summary["artifact_sha256"].items():
                self.assertEqual(runner.sha256_file(output / relative), digest)

    def test_failed_fixture_keeps_checkpoint_and_source_context(self):
        # Replace all source access and initialization with a separate fixture
        # namespace. This exercises failure retention without opening a pilot root.
        config = copy.deepcopy(runner.load_registration())
        config["training"].update(steps=1, batch_size=1, episodes_per_root=1)
        config["development"]["episodes_per_root"] = 1
        source = {"schema_version": 1, "source_commit": "0" * 40,
                  "source_tree_sha": "1" * 40, "source_hashes": {"fixture": "2" * 64}}
        fixtures = {"train": make_pilot_episode(0, 7, "fixture"),
                    "development": make_pilot_episode(0, 8, "fixture")}

        def fixture_source(root, index, phase):
            self.assertEqual((root, index), (0, 0))
            return fixtures[phase]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt"
            with patch.object(runner, "make_pilot_episode", side_effect=fixture_source), \
                    patch.object(runner, "model_seed", return_value=FIXTURE_WEIGHT_SEED), \
                    patch.object(runner, "sampler_seed", return_value=FIXTURE_WEIGHT_SEED + 1), \
                    patch.object(runner, "score_episode", side_effect=runner.EpisodeScoreFailure(
                        "fixture readout failure", {"phase": "reminder", "arm": "corrected", "stage": 2})):
                summary = runner.run(config, source, output, runner.Budget(60, 2147483648))
            self.assertEqual(summary["run_status"], "FAILED")
            self.assertEqual(summary["completed_roots"], [])
            root = summary["roots"][0]
            self.assertEqual(root["train_steps_completed"], 1)
            self.assertEqual(root["development_count"], 0)
            self.assertEqual(root["means"], {})
            self.assertTrue((output / root["checkpoint_file"]).is_file())
            failed = json.loads((output / "root_0" / "failed_episode.json").read_text())
            self.assertEqual(failed["source_id"], fixtures["development"].source_id)
            self.assertEqual(failed["old"], list(fixtures["development"].old))
            self.assertEqual(failed["phase"], "development")
            self.assertEqual(failed["score_context"]["stage"], 2)

    def test_registration_rejects_changed_budget_and_training_selection(self):
        original = runner.load_registration()
        for key, replacement in (("steps", 801), ("batch_size", True)):
            changed = copy.deepcopy(original)
            changed["training"][key] = replacement
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "registration.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.load_registration(path)
        changed = copy.deepcopy(original)
        changed["budget"]["wall_seconds"] = 901
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                runner.load_registration(path)
        changed = copy.deepcopy(original)
        changed["roots"][0] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                runner.load_registration(path)


if __name__ == "__main__":
    unittest.main()
