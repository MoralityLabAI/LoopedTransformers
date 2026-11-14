"""Mechanism and provenance checks on train/development fixtures only."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for the neural pilot")
class ModelTests(unittest.TestCase):
    def setUp(self):
        from experiments.activation_funnel import gym
        from experiments.activation_funnel.model import RecurrentTeacher
        from experiments.activation_funnel.train import tensor_data
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.gym = gym
        self.episodes = gym.generate_split("train", count=36)
        self.x, self.target, self.audit = tensor_data(self.episodes)
        self.teacher = RecurrentTeacher()

    def test_repeated_codec_mediates_every_visit(self):
        class Erase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def encode(self, h):
                self.calls += 1
                return h * 0
            def decode(self, z):
                return z
        codec = Erase()
        trace = self.teacher.rollout(self.x, codec)
        self.assertEqual(codec.calls, 6)
        self.assertEqual(tuple(trace.states.shape), (36, 6, 32))
        self.assertTrue(torch.equal(trace.states, torch.zeros_like(trace.states)))
        self.assertTrue(torch.equal(trace.task_logits, self.teacher.task_head.bias.expand(36, -1)))

    def test_zero_control_resets_between_every_call(self):
        trace = self.teacher.rollout(self.x, zero_between_calls=True)
        zero = torch.zeros(36, 32)
        for visit in range(6):
            expected = self.teacher.cell(self.x[:, visit // 2], zero)
            self.assertTrue(torch.equal(trace.states[:, visit], expected))

    def test_codec_receives_gradient_while_source_stays_frozen(self):
        from experiments.activation_funnel.model import LearnedCodec
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        codec = LearnedCodec()
        loss = F.cross_entropy(self.teacher.rollout(self.x, codec).task_logits, self.target)
        loss.backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in codec.parameters()))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in self.teacher.state_dict().items()))

    def test_donor_context_and_direct_hybrid_replay(self):
        from experiments.activation_funnel.model import (matched_donors, paired_continuation,
                                                        composite_audits)
        donor = matched_donors(self.x, torch.Generator().manual_seed(93))
        self.assertTrue(torch.equal(self.x[:, 0, 8], self.x[donor, 0, 8]))
        hybrid = self.x.clone()
        hybrid[:, 0] = self.x[donor, 0]
        patched = paired_continuation(self.teacher, self.x, donor)
        direct = self.teacher.rollout(hybrid)
        self.assertTrue(torch.allclose(patched.task_logits, direct.task_logits, atol=1e-7, rtol=0))
        expected = composite_audits(self.x, self.audit, donor)
        self.assertTrue(torch.equal(expected[:, 0], self.audit[donor, 0]))
        self.assertTrue(torch.equal(expected[:, 1], self.audit[:, 1]))

    def test_off_grammar_donor_depth_is_rejected(self):
        from experiments.activation_funnel.model import composite_audits
        donor = torch.arange(len(self.x))
        pair = next((i, j) for i in range(len(self.x)) for j in range(len(self.x))
                    if self.x[i, 0, 8] != self.x[j, 0, 8])
        donor[pair[0]] = pair[1]
        with self.assertRaisesRegex(ValueError, "public depth"):
            composite_audits(self.x, self.audit, donor)

    def test_compact_transition_uses_four_carried_coordinates(self):
        from experiments.activation_funnel.model import CompactTransition, resources
        compact = CompactTransition(self.teacher)
        trace = compact.rollout(self.x)
        self.assertEqual(tuple(trace.carried.shape), (36, 6, 4))
        self.assertEqual(tuple(trace.states.shape), (36, 6, 32))
        self.assertEqual(len([m for m in compact.modules() if isinstance(m, torch.nn.GRUCell)]), 1)
        self.assertEqual(compact.cell.hidden_size, 4)
        self.assertLess(resources(compact)["dense_matrix_macs_per_episode"],
                        resources(self.teacher)["dense_matrix_macs_per_episode"])
        self.assertTrue(all(not p.requires_grad for p in compact.task_head.parameters()))

    def test_fixed_projection_uses_train_mean_and_idempotent_projection(self):
        from experiments.activation_funnel.model import FixedCodec, fit_basis
        with torch.no_grad():
            bank = self.teacher.rollout(self.x).states
            mean, pca, _ = fit_basis(bank)
        codec = FixedCodec(mean, pca)
        h = bank.reshape(-1, 32)
        self.assertTrue(torch.allclose(codec(codec(h)), codec(h), atol=1e-6, rtol=0))
        self.assertEqual(tuple(codec.encode(h).shape), (216, 4))

    def test_counterfactual_enumeration_has_meaningful_pairs(self):
        from experiments.activation_funnel.evaluate import counterfactuals
        rows, summary = counterfactuals(self.teacher, None, self.episodes,
                                       self.x, self.audit, 91, "teacher")
        self.assertGreater(summary["changed_status_pairs"], 0)
        self.assertEqual(summary["direct_hybrid_replay_agreement"], 1.)
        self.assertEqual(summary["changed_status_pairs"], sum(r["changed_prefix_status"] for r in rows))

    def test_shuffled_control_maps_duplicate_patterns_identically(self):
        from experiments.activation_funnel.evaluate import control_inputs
        from experiments.activation_funnel.train import canonical_hash
        shuffled = control_inputs(self.x)
        observed = {}
        for original, changed in zip(self.x.tolist(), shuffled.tolist()):
            key = canonical_hash(original)
            if key in observed:
                self.assertEqual(observed[key], changed)
            observed[key] = changed
        self.assertTrue(torch.equal(self.x[:, 0, 8], shuffled[:, 0, 8]))
        self.assertTrue(torch.equal(self.x[:, -1], shuffled[:, -1]))
        self.assertFalse(torch.equal(self.x, shuffled))

    def test_trace_records_actual_predecode_code_and_resumes_from_decoded_state(self):
        from experiments.activation_funnel.model import LearnedCodec, matched_donors, paired_continuation
        codec = LearnedCodec()
        trace = self.teacher.rollout(self.x, codec)
        self.assertEqual(tuple(trace.carried.shape), (36, 6, 4))
        self.assertTrue(torch.allclose(codec.decode(trace.carried), trace.states, atol=1e-7, rtol=0))
        # Re-encoding a decoded state is a different operation and cannot stand
        # in for the code that was actually used at the boundary.
        self.assertFalse(torch.allclose(codec.encode(trace.states), trace.carried))
        donor = matched_donors(self.x, torch.Generator().manual_seed(95))
        patched = paired_continuation(self.teacher, self.x, donor, codec)
        hybrid = self.x.clone()
        hybrid[:, 0] = self.x[donor, 0]
        direct = self.teacher.rollout(hybrid, codec)
        self.assertTrue(torch.allclose(patched.task_logits, direct.task_logits, atol=1e-7, rtol=0))

    def test_rehashed_manifest_cannot_omit_checkpoint_or_seed(self):
        from experiments.activation_funnel.evaluate import load_frozen
        from experiments.activation_funnel.model import CODEC_ARMS
        from experiments.activation_funnel.train import CONFIG, canonical_hash, sha256_file, source_hashes
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            files = {}
            for seed in CONFIG["seeds"]:
                for arm in ("teacher",) + CODEC_ARMS + ("compact_transition",):
                    name = f"{arm}_s{seed}.pt"
                    (folder / name).write_bytes(b"hash-only fixture; never deserialized")
                    files[name] = {"sha256": sha256_file(folder / name)}
            manifest = {"scope": "registered_primary", "training_complete": True,
                        "configuration": CONFIG, "configuration_sha256": canonical_hash(CONFIG),
                        "source_hashes": source_hashes(), "files": files,
                        "seeds": [{"seed": seed} for seed in CONFIG["seeds"]]}
            def write():
                path = folder / "train_manifest.json"
                path.write_text(json.dumps(manifest))
                (folder / "train_manifest.sha256").write_text(sha256_file(path))
            write()
            self.assertEqual(load_frozen(folder)["configuration"], CONFIG)
            saved = files.pop("task_s17.pt")
            write()
            with self.assertRaisesRegex(ValueError, "checkpoint file roster"):
                load_frozen(folder)
            files["task_s17.pt"] = saved
            manifest["seeds"].pop()
            write()
            with self.assertRaisesRegex(ValueError, "seed metadata roster"):
                load_frozen(folder)
            manifest["seeds"] = [{"seed": seed} for seed in CONFIG["seeds"]]
            manifest["source_hashes"]["evaluate.py"] = "0" * 64
            write()
            with self.assertRaisesRegex(ValueError, "evaluate.py"):
                load_frozen(folder)

    def test_source_qualification_counts_distinct_inputs_equally(self):
        from experiments.activation_funnel.model import Trace
        from experiments.activation_funnel.train import accuracy
        inputs = torch.zeros(100, 3, 9)
        inputs[-1, :, 8] = .5
        target = torch.zeros(100, dtype=torch.long)
        audit = torch.zeros(100, 2, dtype=torch.long)
        task_logits = torch.zeros(100, 3)
        task_logits[:, 0] = 1
        task_logits[-1, 1] = 2
        audit_logits = torch.zeros(100, 2, 3)
        audit_logits[:, :, 0] = 1
        trace = Trace(task_logits, audit_logits, torch.zeros(100, 6, 32), torch.zeros(100, 6, 32))
        result = accuracy(trace, target, audit, inputs)
        self.assertEqual(result["exact_feature_patterns"], 2)
        self.assertEqual(result["task_accuracy"], .5)
        self.assertGreater(result["row_weighted_diagnostic"]["task_accuracy"], .95)


if __name__ == "__main__":
    unittest.main()
