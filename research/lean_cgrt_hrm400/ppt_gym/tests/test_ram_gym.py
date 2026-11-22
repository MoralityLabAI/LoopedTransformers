from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from ppt_gym import ram_curriculum, ram_store
from ppt_gym.ram_campaign import native_certificate
from ppt_gym.ram_learning import Online, build_model, credit, tensor_hash
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.live_search import replay_search, search_episode


class RamGymTests(unittest.TestCase):
    def test_resident_sampler_recovers_from_a_refused_allocation(self):
        sampler = TreeBudget.__new__(TreeBudget)
        sampler.job = SimpleNamespace(pids=lambda: [1])
        sampler.psutil = Mock(NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
        sampler.psutil.Process.return_value.memory_full_info.side_effect = [
            MemoryError(), SimpleNamespace(rss=1024, uss=512)
        ]
        sampler.peak_rss = sampler.peak_private_rss = sampler.samples = sampler.dropped_samples = 0
        self.assertEqual(sampler._resident(), (None, None))
        self.assertEqual(sampler._resident(), (1024, 512))
        self.assertEqual((sampler.samples, sampler.dropped_samples), (1, 1))

    def test_preset_and_cpu_fast_state_round_trip(self):
        for plan in ("a", "b"):
            for preset, minimum, maximum in (("micro", 100_000, 500_000),
                                             ("tiny", 1_000_000, 3_000_000)):
                model = build_model(plan, preset)
                self.assertTrue(minimum <= model.parameter_count() <= maximum)
                self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32
                                    for p in model.parameters()))
        first, resumed = Online(24), Online(24)
        features = torch.randn(4, 24, generator=torch.Generator().manual_seed(19))
        update = first.update(features[0], 0, 1)
        self.assertNotEqual(update["before"], update["after"])
        resumed.restore(ram_store.unpack(ram_store.pack(first.state())))
        for i in range(1, 4):
            self.assertEqual(first.update(features[i], 1, 0), resumed.update(features[i], 1, 0))
        self.assertEqual(tensor_hash(first.weights()), tensor_hash(resumed.weights()))
        self.assertTrue(torch.equal(first.cov, resumed.cov))

    def test_abandoned_branches_and_uncertified_paths(self):
        ep = {"verified_closed": True, "events": [
            {"kind": "tactic", "status": "accepted_nonterminal"},
            {"kind": "tactic", "status": "accepted_nonterminal"},
            {"kind": "tactic", "status": "rejected"},
            {"kind": "rollback", "restore_script_length": 1},
            {"kind": "tactic", "status": "kernel_closed"},
        ]}
        self.assertEqual(credit(ep), [(0, (1, "independently_certified_path")),
                                     (1, (0, "abandoned_branch")), (2, (0, "lean_rejection")),
                                     (4, (1, "independently_certified_path"))])
        self.assertEqual(credit({**ep, "verified_closed": False}),
                         [(1, (0, "abandoned_branch")), (2, (0, "lean_rejection"))])

    def test_family_custody_and_atomic_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curriculum.json"
            ram_curriculum.freeze(path, 11, 2, 2, 1)
            record, groups = ram_curriculum.load(path)
            self.assertFalse({c.family for c in groups["support"]} & {c.family for c in groups["ood"]})
            items = ram_curriculum.schedule(record, groups, support=16, rounds=1, persistent=False,
                                            meta_steps=2, backbone_every=0, arms=["sgd"], depths=[2],
                                            final_only=False)
            self.assertEqual({task["dose"] for task in items if task["kind"] == "query"},
                             {0, 1, 2, 4, 8, 16})
            self.assertEqual(len({task["root"] for task in items if task["kind"] == "support"}), 32)
            with ram_store.open_store(Path(tmp) / "session.sqlite", write=True) as db:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    with db:
                        ram_store.append(db, {"kind": "support"}, {"rng": random.Random(1).getstate()})
                        raise RuntimeError("interrupted")
                self.assertEqual(list(ram_store.ledger(db)), [])
                self.assertIsNone(ram_store.checkpoint(db))
                with db:
                    ram_store.append(db, {"kind": "support"}, {"rng": random.Random(1).getstate()})
                self.assertEqual(len(list(ram_store.ledger(db))), 1)
                db.execute("UPDATE ledger SET json = '{}'")
                with self.assertRaisesRegex((ValueError, KeyError), "ledger|seq"):
                    list(ram_store.ledger(db))

    @unittest.skipUnless(os.environ.get("PPT_NATIVE_TEST") == "1", "requires the pinned native Lean build")
    def test_live_probe_backtracking_kernel_and_compilation(self):
        self.assertTrue(config.LEAN_EXE.is_file() and config.REPL_EXE.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curriculum.json"
            ram_curriculum.freeze(path, 904, 1, 1, 1)
            _, groups = ram_curriculum.load(path)
            with LeanRepl(timeout=30) as repl:
                self.addCleanup(repl.proc.stdout.close)
                self.assertTrue(repl.version.startswith("4.19.0"))
                successful = None
                for case in groups["warmup"]:
                    ep = search_episode(repl, case, None, policy="auto8", probe=True)
                    self.assertLessEqual(ep["tactic_checks"], 8)
                    self.assertTrue(replay_search(repl, case, ep)["replay_passed"])
                    if ep["live_kernel"]:
                        successful = native_certificate(case, ep, 30)
                        self.assertEqual(successful["status"], "passed")
                        self.assertIn("theorem", successful["source"])
                        self.assertEqual(successful, native_certificate(case, ep, 30))
                self.assertIsNotNone(successful)
