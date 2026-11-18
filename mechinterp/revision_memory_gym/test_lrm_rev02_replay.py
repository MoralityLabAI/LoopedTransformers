"""Hand-built checkpoint fixtures; no pilot source generator or training."""
import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_lrm_rev02 as replay


def make_fixture(folder):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    model = nn.Module()
    model.gru = nn.GRU(17, 32, batch_first=True, device="cpu", dtype=torch.float32)
    model.head = nn.Linear(32, 3, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(((torch.arange(parameter.numel()).reshape(parameter.shape) % 17) - 8) * .003 + index * .001)
    old, revised, new = [0, 1] * 6, list(range(6)), [1, 1, 0, 1, 0, 1]
    row = {"root": 0, "episode_index": 0, "old": old, "correction_slots": revised,
           "replacements": new, "target_full": [.6, .3, .1], "arms": {}}
    positions = [revised, list(range(6, 12)), list(range(0, 12, 2)),
                 list(range(1, 12, 2)), [0, 1, 6, 7, 8, 9]]
    losses = []
    for name, slots in zip(replay.ARMS, positions):
        # Construct the fixture stream directly, independently of replay.encode.
        features = torch.zeros((1, 24, 17))
        events = list(zip([0] * 12, range(12), old)) + list(zip([1] * 6, revised, new))
        events += list(zip([2] * 6, slots, [old[i] for i in slots]))
        for step, (kind, position, bit) in enumerate(events):
            features[0, step, [kind, 3 + position, 15 + bit]] = 1
        with torch.no_grad():
            states, _ = model.gru(features)
            probabilities = model.head(states)[0].double().softmax(-1).tolist()
        row["initial_probability"] = probabilities[11]
        curve = [{"r": r, "probability": probabilities[17 + r],
                  "hidden_displacement_l2": float(torch.linalg.vector_norm(states[0, 17 + r].double() - states[0, 17].double()))}
                 for r in range(7)]
        row["arms"][name] = {"slots": slots, "curve": curve}
        losses.append(math.fsum((p - q) ** 2 for p, q in zip(probabilities[23], row["target_full"])))
    source = {"source_commit": "a" * 40, "source_hashes": {"fixture.py": "b" * 64}}
    (folder / "root_0").mkdir()
    checkpoint = folder / "root_0" / "checkpoint_final.pt"
    torch.save({**source, "root": 0, "steps": 800, "model_state_dict": model.state_dict()}, checkpoint)
    archive = folder / "root_0" / "development_outcomes.jsonl.gz"
    with gzip.open(archive, "wt") as handle:
        handle.write(json.dumps(row) + "\n")
    (folder / "source_manifest.json").write_text(json.dumps(source))
    root = {"root": 0, "status": "BUDGET_STOP", "development_count": 1, "train_steps_completed": 800,
            "checkpoint_file": "root_0/checkpoint_final.pt", "checkpoint_sha256": replay.digest(checkpoint),
            "development_outcomes_file": "root_0/development_outcomes.jsonl.gz",
            "means": {"paired_effect": math.fsum(losses[1:]) / 4 - losses[0]}}
    summary = {**source, "run_status": "BUDGET_STOP", "planned_roots": [0, 1, 2, 3],
               "completed_roots": [], "n_units_completed": 0, "roots": [root],
               "registration": {"diagnostics": {"checkpoint_replay_probability_abs_max": 1e-6,
                                                  "checkpoint_replay_effect_abs_max": 1e-6}},
               "artifact_sha256": {str(p.relative_to(folder)): replay.digest(p)
                                   for p in (checkpoint, archive, folder / "source_manifest.json")}}
    (folder / "summary.json").write_text(json.dumps(summary))
    return row, summary


class IndependentReplayTests(unittest.TestCase):
    def test_partial_fixture_replays_and_changed_prediction_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            row, summary = make_fixture(folder)
            result = replay.replay_run(folder)
            self.assertTrue(result["passed"], result)
            self.assertFalse(result["complete"])
            self.assertEqual(result["run_status"], "BUDGET_STOP")
            self.assertEqual(result["development_rows_checked"], 1)
            self.assertLessEqual(result["roots"][0]["probability_abs_max"], 1e-6)
            original = copy.deepcopy(row)
            row["arms"]["corrected"]["curve"][6]["probability"][0] += .05
            row["arms"]["corrected"]["curve"][6]["probability"][1] -= .05
            archive = folder / summary["roots"][0]["development_outcomes_file"]
            with gzip.open(archive, "wt") as handle:
                handle.write(json.dumps(row) + "\n")
            summary["artifact_sha256"][str(archive.relative_to(folder))] = replay.digest(archive)
            (folder / "summary.json").write_text(json.dumps(summary))
            self.assertFalse(replay.replay_run(folder)["passed"])
            self.assertNotEqual(row, original)

    def test_checkpoint_mutation_and_false_completion_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _, summary = make_fixture(folder)
            summary["run_status"] = "COMPLETE"
            (folder / "summary.json").write_text(json.dumps(summary))
            self.assertFalse(replay.replay_run(folder)["passed"])
            summary["run_status"] = "BUDGET_STOP"
            (folder / "summary.json").write_text(json.dumps(summary))
            checkpoint = folder / summary["roots"][0]["checkpoint_file"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            result = replay.replay_run(folder)
            self.assertFalse(result["passed"])
            self.assertTrue(any("SHA-256 mismatch" in failure for failure in result["failures"]))


if __name__ == "__main__":
    unittest.main()
