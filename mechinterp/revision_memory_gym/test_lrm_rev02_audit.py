"""Hand-authored audit receipts; no models or train/development sources run.

Tests reduce only audit count constants and replace the frozen generator
loader with a fixture oracle. Bayes values come from direct per-bit products.
"""
from __future__ import annotations

import copy
from fractions import Fraction as Q
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lrm_rev02 as audit

REAL_SOURCE_LOADER = audit._load_source_identity


CONTROL_SLOTS = ([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11],
                 [0, 1, 6, 7, 10, 11], [2, 3, 4, 5, 8, 9])


def hand_episode(root, index, phase):
    """An explicitly artificial source, unrelated to the pilot generator."""
    return SimpleNamespace(source_id=hashlib.sha256(f"audit-fixture/{phase}/{root}/{index}".encode()).hexdigest(),
                           old=(0,) * 12, correction_slots=tuple(range(6)),
                           replacements=(0,) * 6, true_regime=2)


def zero_report_posterior(count):
    priors = (Q(9, 20), Q(7, 20), Q(1, 5))
    match = (Q(19, 20), Q(13, 20), Q(3, 20))
    weights = []
    for prior, t in zip(priors, match):
        weight = prior
        for _ in range(count):
            weight *= 1 - t
        weights.append(weight)
    return [float(w / sum(weights)) for w in weights]


def source_record(index, phase):
    e = hand_episode(0, index, phase)
    return {"root": 0, "episode_index": index, "source_id": e.source_id,
            "old": list(e.old), "correction_slots": list(e.correction_slots),
            "replacements": list(e.replacements), "true_regime": e.true_regime,
            "target_initial": zero_report_posterior(6),
            "target_full": zero_report_posterior(12)}


def development_record(index):
    row = {"schema_version": 1, **source_record(index, "development")}
    q = row["target_full"]
    bayes = 1 - math.fsum(x * x for x in q)
    row.update(initial_probability=list(row["target_initial"]), initial_excess=0.0,
               prefix_replay_logits_abs_max=0.0, prefix_replay_hidden_abs_max=0.0,
               prefix_replay_probability_abs_max=0.0)
    arms = {}
    for a, name in enumerate(audit.ARMS):
        curve = []
        decrement = 0.004 if a == 0 else 0.0035 - 0.0005 * a
        for r in range(7):
            error = 0.03 - decrement * r
            p = [q[0] + error, q[1], q[2] - error]
            excess = math.fsum((x - y) ** 2 for x, y in zip(p, q))
            curve.append({"r": r, "probability": p, "excess": excess,
                          "expected_brier": bayes + excess,
                          "hidden_displacement_l2": r * 0.01 * (a + 1)})
        arms[name] = {"slots": list(range(6)) if a == 0 else CONTROL_SLOTS[a - 1], "curve": curve}
    row["arms"] = arms
    row["prefix_replay_probability"] = list(arms["corrected"]["curve"][0]["probability"])
    row["paired_effect"] = math.fsum(arms[name]["curve"][-1]["excess"] for name in audit.ARMS[1:]) / 4 - arms["corrected"]["curve"][-1]["excess"]
    return row


def aggregate_hand_rows(rows):
    average = lambda values: math.fsum(values) / len(rows)
    q = rows[0]["target_full"]
    bayes = 1 - math.fsum(x * x for x in q)
    candidate = average(row["arms"]["corrected"]["curve"][-1]["excess"] for row in rows)
    control = average(math.fsum(row["arms"][name]["curve"][-1]["excess"] for name in audit.ARMS[1:]) / 4 for row in rows)
    means = {"initial_excess": 0.0,
             "base_excess": average(row["arms"]["corrected"]["curve"][0]["excess"] for row in rows),
             "candidate_excess": candidate, "control_mean_excess": control,
             "paired_effect": control - candidate, "bayes_risk": bayes,
             "candidate_expected_brier": candidate + bayes,
             "control_mean_expected_brier": control + bayes}
    curves = {name: [{"r": stage, **{key: average(row["arms"][name]["curve"][stage][key] for row in rows)
                                    for key in ("excess", "expected_brier", "hidden_displacement_l2")}}
                     for stage in range(7)] for name in audit.ARMS}
    probabilities = [row["initial_probability"] for row in rows]
    probabilities += [point["probability"] for row in rows for arm in row["arms"].values() for point in arm["curve"]]
    diag = {"probability_abs_sum_error_max": max(abs(math.fsum(p) - 1) for p in probabilities),
            "prefix_replay_logits_abs_max": 0.0, "prefix_replay_hidden_abs_max": 0.0,
            "prefix_replay_probability_abs_max": 0.0, "r0_abs_difference_max": 0.0,
            "initial_competence": True, "candidate_competence": candidate <= 0.02,
            "numeric_pass": True, "engineering_competence_pass": candidate <= 0.02}
    return means, curves, diag


class ReceiptAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual((audit.ROOTS, audit.TRAIN_COUNT, audit.DEVELOPMENT_COUNT, audit.TRAIN_STEPS),
                         ([0, 1, 2, 3], 2048, 512, 800))
        bounds = patch.multiple(audit, ROOTS=[0], TRAIN_COUNT=2, DEVELOPMENT_COUNT=2, TRAIN_STEPS=1)
        bounds.start()
        self.addCleanup(bounds.stop)
        identity = {"make_pilot_episode": hand_episode,
                    "control_slots": lambda episode, index: tuple(CONTROL_SLOTS[index]),
                    "training_slots": lambda episode: tuple(range(6))}
        loader = patch.object(audit, "_load_source_identity", return_value=identity)
        loader.start()
        self.addCleanup(loader.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix="lrm-audit-hand-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repository"
        self.out = Path(self.temporary.name) / "receipt"
        self.repo.mkdir()
        (self.out / "root_0").mkdir(parents=True)
        self.config = {"experiment_id": "LRM-REV-02", "roots": [0],
                       "evidence_law": {"n": 12, "j": 6, "priors": [[9, 20], [7, 20], [1, 5]],
                                        "match_probabilities": [[19, 20], [13, 20], [3, 20]]},
                       "training": {"steps": 1, "episodes_per_root": 2},
                       "development": {"episodes_per_root": 2, "controls": 4},
                       "diagnostics": {"initial_excess_max": 0.01, "candidate_excess_max": 0.02, "replay_abs_max": 1e-6},
                       "source_files": [audit.REGISTRATION, audit.FROZEN_GYM, audit.FROZEN_FIXTURE]}
        self.write_json(self.repo / audit.REGISTRATION, self.config)
        for path in (audit.FROZEN_GYM, audit.FROZEN_FIXTURE):
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# Hand-fixture provenance stub; generator loader is mocked.\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Audit Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "hand-authored receipt fixture")
        commit, tree = self.git("rev-parse", "HEAD"), self.git("rev-parse", "HEAD^{tree}")
        hashes = {path: self.sha(self.repo / path) for path in self.config["source_files"]}
        self.write_json(self.out / "source_manifest.json", {"schema_version": 1, "source_commit": commit,
                                                            "source_tree_sha": tree, "source_hashes": hashes})
        self.rows = [development_record(0), development_record(1)]
        train = [{**source_record(i, "train"), "reminder_slots": list(range(6))} for i in range(2)]
        self.write_gzip(self.out / "root_0/training_episodes.jsonl.gz", train)
        self.write_gzip(self.out / "root_0/development_outcomes.jsonl.gz", self.rows)
        (self.out / "root_0/training_trace.jsonl").write_text(json.dumps({"step": 1, "total_loss": 0.1}) + "\n")
        checkpoint = self.out / "root_0/checkpoint_final.pt"
        checkpoint.write_bytes(b"hand-authored checkpoint fixture; no model")
        means, curves, diagnostics = aggregate_hand_rows(self.rows)
        self.root = {"schema_version": 1, "root": 0, "status": "COMPLETE", "model_seed": 1, "sampler_seed": 2,
                     "train_steps_completed": 1, "n_train_episodes": 2, "development_count": 2,
                     "model_parameters": 4995, "checkpoint_file": "root_0/checkpoint_final.pt",
                     "checkpoint_sha256": self.sha(checkpoint), "weights_tensor_sha256": "0" * 64,
                     "development_outcomes_file": "root_0/development_outcomes.jsonl.gz",
                     "training_archive_sha256": self.sha(self.out / "root_0/training_episodes.jsonl.gz"),
                     "means": means, "curve_means": curves, "diagnostics": diagnostics}
        self.summary = {"schema_version": 1, "experiment_id": "LRM-REV-02", "verdict": "ENGINEERING_PILOT_ONLY",
                        "preflight_status": audit.PREFLIGHT_STATUS, "run_status": "COMPLETE", "source_commit": commit,
                        "source_hashes": hashes, "registration": self.config, "planned_roots": [0],
                        "completed_roots": [0], "n_units_completed": 1, "roots": [self.root]}
        self.refresh()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    @staticmethod
    def write_gzip(path, rows):
        with gzip.open(path, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL).strip()

    def refresh(self):
        self.write_json(self.out / "root_0/root_summary.json", self.root)
        self.summary["artifact_sha256"] = {str(path.relative_to(self.out)): self.sha(path)
                                           for path in self.out.rglob("*") if path.is_file() and path.name != "summary.json"}
        self.write_json(self.out / "summary.json", self.summary)

    def audit(self):
        return audit.audit_run(self.out, self.repo)

    def rewrite_rows(self):
        self.write_gzip(self.out / "root_0/development_outcomes.jsonl.gz", self.rows)
        self.refresh()

    def assert_rejected(self, phrase):
        result = self.audit()
        self.assertFalse(result["passed"], result)
        self.assertIn(phrase, " ".join(result["failures"]))

    def test_complete_hand_fixture_and_frozen_provenance(self):
        result = self.audit()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["complete"])
        self.assertEqual(result["development_rows_checked"], 2)
        self.assertAlmostEqual(result["completed_root_mean_effect"], self.root["means"]["paired_effect"])

    def test_interrupted_outcomes_preserve_status(self):
        for status in ("BUDGET_STOP", "FAILED"):
            with self.subTest(status=status):
                self.root.update(status=status, error="hand fixture interruption after scoring",
                                 failure_context={"phase": "development_complete", "root": 0, "episode_index": 2})
                self.summary.update(run_status=status, completed_roots=[], n_units_completed=0)
                self.refresh()
                result = self.audit()
                self.assertTrue(result["passed"], result)
                self.assertFalse(result["complete"])
                self.assertEqual(result["run_status"], status)

    def test_tampered_target_is_detected_after_rehashing(self):
        self.rows[0]["target_full"][0] += 0.01
        self.rows[0]["target_full"][2] -= 0.01
        self.rewrite_rows()
        self.assert_rejected("target_full")

    def test_failed_development_source_is_retained_and_checked(self):
        context = {key: value for key, value in source_record(1, "development").items()
                   if not key.startswith("target_")}
        context["phase"] = "development"
        context["score_context"] = {"phase": "reminders", "arm": "random_0", "stage": 2}
        self.root.update(status="FAILED", error="hand fixture scoring failure",
                         failure_context=context, development_count=1)
        self.summary.update(run_status="FAILED", completed_roots=[], n_units_completed=0)
        self.rows = self.rows[:1]
        self.write_json(self.out / "root_0/failed_episode.json", context)
        self.rewrite_rows()
        result = self.audit()
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["complete"])
        self.assertEqual(result["development_rows_checked"], 1)
        context["old"][0] = 1
        self.write_json(self.out / "root_0/failed_episode.json", context)
        self.refresh()
        self.assert_rejected("failure source differs")

    def test_tampered_score_is_detected_after_rehashing(self):
        self.rows[0]["arms"]["corrected"]["curve"][2]["expected_brier"] += 0.01
        self.rewrite_rows()
        self.assert_rejected("expected_brier")

    def test_wrong_valid_control_set_is_detected(self):
        self.rows[0]["arms"]["random_0"]["slots"] = list(range(6))
        self.rewrite_rows()
        self.assert_rejected("frozen control policy")

    def test_duplicate_root_is_detected(self):
        self.summary["roots"].append(copy.deepcopy(self.root))
        self.summary.update(completed_roots=[0, 0], n_units_completed=2)
        self.refresh()
        self.assert_rejected("duplicate")

    def test_tampered_root_mean_is_detected_after_rehashing(self):
        self.root["means"]["paired_effect"] += 0.01
        self.refresh()
        self.assert_rejected("means.paired_effect")

    def test_checkpoint_bytes_are_hashed(self):
        with (self.out / "root_0/checkpoint_final.pt").open("ab") as handle:
            handle.write(b"altered")
        self.assert_rejected("artifact hash mismatch")

    def test_wrong_source_world_is_detected_after_rehashing(self):
        self.rows[0]["old"][11] = 1
        self.rewrite_rows()
        self.assert_rejected("frozen root/index/phase generator")

    def test_replay_probability_scalar_is_recomputed(self):
        self.rows[0]["prefix_replay_probability_abs_max"] = 1e-5
        self.rewrite_rows()
        self.assert_rejected("prefix_replay_probability_abs_max")

    def test_frozen_loader_definitions_use_fixture_phase_only(self):
        # Exercise the real loader with the reviewed definitions, while keeping
        # all requests inside the existing public unit-test fixture namespace.
        source_root = Path(__file__).resolve().parents[2]
        for relative in (audit.FROZEN_GYM, audit.FROZEN_FIXTURE):
            (self.repo / relative).write_bytes((source_root / relative).read_bytes())
        self.git("add", ".")
        self.git("-c", "user.name=Audit Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "reviewed generator definitions for fixture-only loader test")
        functions = REAL_SOURCE_LOADER(self.repo, self.git("rev-parse", "HEAD"))
        episode = functions["make_pilot_episode"](0, 0, "fixture")
        self.assertEqual((len(episode.old), len(episode.correction_slots), len(episode.replacements)), (12, 6, 6))
        self.assertEqual(len(set(functions["control_slots"](episode, 0))), 6)
        self.assertEqual(len(functions["training_slots"](episode)), 6)


if __name__ == "__main__":
    unittest.main()
