import itertools
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from expert_lattice import ExpertLattice
from gym_data import build
from inventory import MODEL, REV
from routing import parse_rows
from stage1_controller import CONTROLLERS, PassController, controls, evaluate, refresh_decision
from stage1_data import (ACTIVE, ARMS, HORIZONS, SYNTHETIC, Dataset, Turn, Window,
                         entries, extract_pilot, prepare, sha, source_receipt, write_json)
from stage1_memory import MIB, MemoryGuard


class Guard:
    def check(self):
        pass


def make_fixture(folder):
    for arm in ARMS:
        directory = folder / arm
        directory.mkdir(parents=True)
        roster, rows = [], []
        for i, split in enumerate(("train", "dev")):
            roster.append({"root": split, "turn": arm, "family": split, "split": split,
                           "prompt_len": 2, "token_ids": list(range(35))})
            for token, (layer, pass_id) in itertools.product(range(35), zip(*np.nonzero(ACTIVE))):
                use = 16 * (pass_id + (token > 2) + (token == 34) + i)
                rows.append({"root": split, "turn": arm, "token": token, "layer": int(layer),
                             "pass": int(pass_id), "experts": list(range(use, use + 8)),
                             "weights": [1] * 8, "phase": "prompt" if token < 2 else "continuation"})
        for name, records in (("roster", roster), ("trace", rows)):
            (directory / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf8",
            )
        parsed = parse_rows(map(json.dumps, rows), strict=True)
        for h in HORIZONS:
            np.savez(directory / f"demand_H{h}.npz", **build(roster, parsed, h))
    write_json(folder / "manifest.json", {"kind": SYNTHETIC, "files": entries(folder)})


class Stage1DataTests(unittest.TestCase):
    def test_rebuilds_published_shapes_and_excludes_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            make_fixture(source)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            self.assertGreater(source_receipt(source, synthetic=True)["files"], 1)
            original, train_frequency = prepare([source], ARMS[0], first, teacher=True, guard=Guard())
            path = source / ARMS[0] / "trace.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
            for row in rows:
                if row["root"] == "dev" and row["token"] == 34:
                    row["experts"] = list(range(116, 124))
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                source_receipt(source, synthetic=True)
            modified, frequency = prepare([source], ARMS[0], second, teacher=False, guard=Guard())
            try:
                np.testing.assert_array_equal(train_frequency, frequency)
                np.testing.assert_array_equal(original[32].x, modified[32].x)
                self.assertFalse(np.array_equal(original[32].y[1], modified[32].y[1]))
                dev = np.array([0, 1])
                baseline, _, _ = controls(original[32], dev, frequency)
                changed, _, _ = controls(modified[32], dev, frequency)
                np.testing.assert_array_equal(baseline["previous_window"], changed["previous_window"])
            finally:
                for dataset in (*original.values(), *modified.values()):
                    dataset.close()

    def test_archive_rejects_path_traversal_and_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, destination = root / "pilot.tar.gz", root / "extract"
            with tarfile.open(archive, "w:gz") as file:
                item = tarfile.TarInfo("pilot/../escape.txt")
                item.size = 1
                file.addfile(item, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "size or SHA"):
                extract_pilot(archive, destination)
            with (patch("stage1_data.PILOT_BYTES", archive.stat().st_size),
                  patch("stage1_data.PILOT_SHA256", sha(archive))):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    extract_pilot(archive, destination)
            self.assertFalse(destination.exists())
            self.assertFalse((root / "escape.txt").exists())

    def test_archive_verifies_nested_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, archive = root / "source", root / "pilot.tar.gz"
            destination = root / "fresh" / "extract"
            source.mkdir()
            for i in range(18):
                (source / f"file{i}.json").write_text(str(i), encoding="utf8")
            write_json(source / "manifest.json", {
                "model": MODEL, "revision": REV, "files": entries(source),
            })
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(source, arcname="routing")
            with (patch("stage1_data.PILOT_BYTES", archive.stat().st_size),
                  patch("stage1_data.PILOT_SHA256", sha(archive))):
                receipt = extract_pilot(archive, destination)
            self.assertEqual(receipt["verified_files"], 18)
            self.assertEqual(entries(source), entries(destination))


class Stage1ControllerTests(unittest.TestCase):
    def test_cuda_allocator_uses_an_explicit_index(self):
        fake_torch = Mock()
        fake_torch.cuda.current_device.return_value = 0
        fake_torch.cuda.mem_get_info.return_value = (3962 * MIB, 4096 * MIB)
        guard = MemoryGuard()
        guard.configure_cuda(fake_torch, torch.device("cuda"))
        fake_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(1536 / 4096, 0)
        self.assertEqual(guard.device, 0)

    def test_lattice_projection_matches_brute_force_and_prefers_fewer_swaps(self):
        lattice = ExpertLattice(8)
        old = frozenset({0, 1, 2})
        self.assertEqual(lattice.meet(old, lattice.top), old)
        self.assertEqual(lattice.join(old, lattice.bottom), old)
        choices = [frozenset(items) for items in itertools.combinations(lattice.top, 3)]
        rng = np.random.default_rng(17)
        for budget, _ in itertools.product(range(4), range(8)):
            scores = rng.integers(0, 5, size=8)
            chosen = lattice.admit(scores, old, 3, budget)
            objective = lambda node: (scores[sorted(node)].sum(), -len(node - old))
            self.assertEqual(objective(chosen), max(objective(node) for node in choices
                                                   if len(node - old) <= budget))
        self.assertEqual(lattice.admit(np.ones(8), old, 3, 3), old)

    def test_matched_control_cannot_observe_previous_pass(self):
        torch.manual_seed(17)
        models = {name: PassController(12, *flags) for name, flags in CONTROLLERS.items()}
        self.assertEqual(len({sum(p.numel() for p in model.parameters())
                              for model in models.values()}), 1)
        x = torch.zeros(1, 24, 3, 256)
        changed = x.clone()
        changed[:, 8, 0, :16] = 1
        with torch.no_grad():
            for name, model in models.items():
                original, updated = model(x), model(changed)
                self.assertEqual(torch.equal(original[:, 8, 1], updated[:, 8, 1]),
                                 name != "pass_mesh")
                self.assertTrue(torch.equal(original[:, 8, 2], updated[:, 8, 2]))

    def test_one_physical_cache_counts_cold_and_replacement_bytes(self):
        turn = Turn("root", "arm", "family", "dev", 2, 4)
        targets = np.zeros((2, 24, 3, 128), dtype=np.float32)
        targets[0, :, :, 0] = ACTIVE
        targets[1, :, :, 8] = ACTIVE
        dataset = Dataset(None, targets, [Window(turn, 2, 3), Window(turn, 3, 4)])
        scores = targets.sum(axis=2) / ACTIVE.sum(axis=1)[:, None]
        result = evaluate(dataset, [0, 1], scores, 8, 2, list(range(1, 25)), Guard())
        self.assertEqual(result["root_mean_mass"], 1)
        self.assertEqual(result["roots"]["root"]["visits"], 96)
        self.assertEqual(result["roots"]["root"]["retained_mass_by_pass"], [1, 1, 1])
        self.assertEqual(result["cold_admissions"], 24 * 8)
        self.assertEqual(result["admissions"], 24 * 9)
        self.assertEqual(result["estimated_transfer_bytes"], 9 * sum(range(1, 25)))
        frozen = evaluate(dataset, [0, 1], scores, 8, 0, [1] * 24, Guard())
        self.assertEqual(frozen["root_mean_mass"], 0.5)

    def test_adaptive_refresh_rejects_cost_or_lack_of_drift(self):
        before, after = np.eye(128)[[0, 1]]
        old, proposed = frozenset({0}), frozenset({1})
        self.assertTrue(refresh_decision(after, before, old, proposed, 64, 0.1, 1))
        self.assertFalse(refresh_decision(after, after, old, proposed, 64, 0.1, 1))
        self.assertFalse(refresh_decision(after, before, old, proposed, 1 << 27, 0.1, 1))


if __name__ == "__main__":
    unittest.main()
