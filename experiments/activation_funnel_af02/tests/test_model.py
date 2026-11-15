"""AF-02 structural and provenance regressions using train/dev fixtures only."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for neural fixtures")
class ModelTests(unittest.TestCase):
    def setUp(self):
        from experiments.activation_funnel_af02 import gym
        from experiments.activation_funnel_af02.model import make_model
        from experiments.activation_funnel_af02.train import tensor_data
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.gym = gym
        self.train = gym.canonical_cases("train")
        self.dev = gym.canonical_cases("dev")
        self.cases = [c for c in self.train if c.schedule == "PPG"]
        self.x, self.task, self.audit = tensor_data(self.cases)
        self.models = {name: make_model(name) for name in ("gru", "native")}

    def test_gru_reuses_exact_af01_parameters_and_recurrence(self):
        from experiments.activation_funnel.model import RecurrentTeacher
        from experiments.activation_funnel_af02.model import make_model
        torch.manual_seed(93)
        old = RecurrentTeacher()
        torch.manual_seed(93)
        new = make_model("gru")
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[key], new.state_dict()[key]))
        left, right = old.rollout(self.x), new.rollout(self.x)
        self.assertTrue(torch.equal(left.states, right.states))
        self.assertTrue(torch.equal(left.task_logits, right.task_logits))
        self.assertTrue(torch.equal(left.audit_logits, right.audit_logits))

    def test_native_class_is_pinned_verbatim_import(self):
        from experiments.activation_funnel_af02.model import native_cell_class
        cell_class = native_cell_class()
        self.assertIs(type(self.models["native"].cell), cell_class)
        self.assertEqual(cell_class.__module__, "_af02_pinned_lr01_train")
        self.assertEqual(sum(p.numel() for p in self.models["native"].cell.parameters()), 5024)

    def test_repeated_codec_records_actual_codes_and_decoded_carry(self):
        from experiments.activation_funnel_af02.model import make_codec
        for name, model in self.models.items():
            with self.subTest(architecture=name):
                codec = make_codec(name)
                trace = model.rollout(self.x, codec)
                width = 32 if name == "gru" else 80
                code = 4 if name == "gru" else 20
                self.assertEqual(tuple(trace.states.shape), (36, 6, width))
                self.assertEqual(tuple(trace.carried.shape), (36, 6, code))
                self.assertTrue(torch.equal(trace.before[:, 1:], trace.states[:, :-1]))
                self.assertTrue(torch.allclose(codec.decode(trace.carried), trace.states, atol=1e-7, rtol=0))
                self.assertFalse(torch.allclose(codec.encode(trace.states), trace.carried))
                for visit in range(6):
                    expected = model.one_step(trace.before[:, visit], self.x[:, visit // 2])
                    self.assertTrue(torch.equal(expected, trace.raw[:, visit]))

    def test_erasing_codec_controls_all_visits_and_final_readout(self):
        class Erase(torch.nn.Module):
            def __init__(self, state_width, code_width):
                super().__init__()
                self.state_width, self.code_width, self.calls = state_width, code_width, 0
            def encode(self, h):
                self.calls += 1
                return h.new_zeros((len(h), self.code_width))
            def decode(self, z):
                return z.new_zeros((len(z), self.state_width))
        for name, model in self.models.items():
            codec = Erase(model.state_width, 4 if name == "gru" else 20)
            trace = model.rollout(self.x, codec)
            self.assertEqual(codec.calls, 6)
            self.assertEqual(float(trace.states.abs().sum()), 0.)
            expected = model.readout(torch.zeros(len(self.x), model.state_width))
            self.assertTrue(torch.equal(trace.task_logits, expected[0]))
            self.assertTrue(torch.equal(trace.audit_logits, expected[1]))

    def test_zero_between_calls_preserves_current_observation_access(self):
        for model in self.models.values():
            trace = model.rollout(self.x, zero_between_calls=True)
            zero = torch.zeros(len(self.x), model.state_width)
            self.assertEqual(float(trace.before.abs().sum()), 0.)
            for visit in range(6):
                expected = model.one_step(zero, self.x[:, visit // 2])
                self.assertTrue(torch.equal(trace.states[:, visit], expected))

    def test_native_writer_address_payload_and_erasure(self):
        class IdentityCell(torch.nn.Module):
            def forward(self, a, r, prem, head):
                return a, r
        model = self.models["native"]
        model.cell = IdentityCell()
        x = self.x[:1, 0].clone()
        x[:, 2:4] = torch.tensor([[1., 0.]])
        payload = x.clone()
        payload[:, 2:4] = 0
        atom_write, rule_write = model.atom_writer(payload), model.rule_writer(payload)
        raw = model.one_step(torch.zeros(1, 80), x).reshape(1, 5, 16)
        self.assertTrue(torch.equal(raw[:, 0], atom_write))
        self.assertTrue(torch.equal(raw[:, 3], rule_write))
        self.assertEqual(float(raw[:, [1, 2, 4]].abs().sum()), 0.)
        x[:, 2:4] = 0
        erased = model.one_step(torch.zeros(1, 80), x).reshape(1, 5, 16)
        self.assertTrue(torch.equal(erased[:, 2], atom_write))
        self.assertEqual(float(erased[:, [0, 1, 3, 4]].abs().sum()), 0.)

    def test_native_all_layouts_transport_canonical_states_and_logits(self):
        from experiments.activation_funnel_af02.model import make_codec
        model = self.models["native"]
        for codec in (None, make_codec("native")):
            base = model.rollout(self.x, codec)
            for layout in self.gym.graph_permutations():
                trace = model.rollout(self.x, codec, layout=layout)
                for name in ("before", "raw", "states", "carried", "task_logits", "audit_logits"):
                    self.assertTrue(torch.allclose(getattr(trace, name), getattr(base, name),
                                                   atol=1e-5, rtol=0), (layout, name))

    def test_source_stays_frozen_while_joint_codec_has_gradients(self):
        from experiments.activation_funnel_af02.model import make_codec, commutation_loss, audit_loss
        for name, model in self.models.items():
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            codec = make_codec(name)
            with torch.no_grad():
                native = model.rollout(self.x)
            trace = model.rollout(self.x, codec)
            loss = F.cross_entropy(trace.task_logits, self.task) + .5 * audit_loss(trace.audit_logits, self.audit)
            loss = loss + .5 * commutation_loss(model, codec, self.x, native)
            loss.backward()
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in codec.parameters()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))

    def test_wait_prefix_donors_resume_at_actual_first_preview(self):
        from experiments.activation_funnel_af02.model import make_codec, matched_donors, paired_continuation, composite_audits
        from experiments.activation_funnel_af02.train import tensor_data
        cases = [c for c in self.train if len(c.schedule) == 4] + self.dev
        x, _, audit = tensor_data(cases)
        donor = matched_donors(cases, torch.Generator().manual_seed(95))
        hybrid = x.clone()
        for i, j in enumerate(donor.tolist()):
            cut = self.gym.first_probe_cut(cases[i])
            self.assertEqual(self.gym.donor_stratum(cases[i]), self.gym.donor_stratum(cases[j]))
            hybrid[i, :cut] = x[j, :cut]
        labels = composite_audits(cases, audit, donor)
        for i, j in enumerate(donor.tolist()):
            first = cases[i].probe_order[0]
            self.assertEqual(int(labels[i, first]), int(audit[j, first]))
            self.assertEqual(int(labels[i, 1 - first]), int(audit[i, 1 - first]))
        for name, model in self.models.items():
            for codec in (None, make_codec(name)):
                patch = paired_continuation(model, x, cases, donor, codec)
                direct = model.rollout(hybrid, codec)
                self.assertTrue(torch.allclose(patch.task_logits, direct.task_logits, atol=1e-6, rtol=0))
                self.assertTrue(torch.allclose(patch.audit_logits, direct.audit_logits, atol=1e-6, rtol=0))

    def test_donor_guard_checks_depth_address_and_complete_schedule(self):
        from experiments.activation_funnel_af02.model import composite_audits
        audit = torch.tensor([self.gym.audit_targets(c) for c in self.train])
        for attribute in ("depth", "probe_order", "schedule"):
            donor = torch.arange(len(self.train))
            i, j = next((i, j) for i in range(len(self.train)) for j in range(len(self.train))
                        if getattr(self.train[i], attribute) != getattr(self.train[j], attribute))
            donor[i] = j
            with self.assertRaisesRegex(ValueError, "complete schedule"):
                composite_audits(self.train, audit, donor)

    def test_weighted_pca_assigns_equal_mass_to_each_input(self):
        from experiments.activation_funnel_af02.train import fit_bases
        short = torch.ones(1, 32) * 2
        long = torch.zeros(9, 32)
        bases = fit_bases("gru", [short, long])
        self.assertTrue(torch.allclose(bases["state"]["mean"], torch.ones(32), atol=1e-6, rtol=0))
        one = torch.zeros(1, 5, 16)
        one[:, :3] = 2
        one[:, 3:] = 10
        many = torch.zeros(7, 5, 16)
        bases = fit_bases("native", [one.reshape(1, 80), many.reshape(7, 80)])
        self.assertTrue(torch.allclose(bases["atom"]["mean"], torch.ones(16), atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(bases["rule"]["mean"], torch.ones(16) * 5, atol=1e-6, rtol=0))

    def test_length_bucket_loss_matches_per_input_mean(self):
        from experiments.activation_funnel_af02.train import _Data, bucket_indices, tensor_data
        data = _Data(self.train)
        selected = [0, 0, 1, 2, 18, 19, 20, 36, 37]
        self.assertEqual({len(self.train[i].schedule) for i in selected}, {3, 4})
        with self.assertRaisesRegex(ValueError, "true-length"):
            tensor_data([self.train[0], self.train[18]])
        with torch.no_grad():
            for model in self.models.values():
                bucket_loss = torch.zeros(())
                for indices in bucket_indices(self.train, selected).values():
                    x, task, _ = data.batch(indices)
                    bucket_loss += len(indices) / len(selected) * F.cross_entropy(model.rollout(x).task_logits, task)
                singleton = []
                for i in selected:
                    x, task, _ = data.batch([i])
                    singleton.append(F.cross_entropy(model.rollout(x).task_logits, task))
                self.assertTrue(torch.allclose(bucket_loss, torch.stack(singleton).mean(), atol=1e-6, rtol=0))

    def test_source_qualification_checks_each_order_and_each_branch(self):
        from experiments.activation_funnel_af02.train import source_qualified
        good = {"by_order": {"0,1": {"task_accuracy": 1., "audit_branch_accuracy": [1., 1.]},
                              "1,0": {"task_accuracy": 1., "audit_branch_accuracy": [1., 1.]}}}
        bad = json.loads(json.dumps(good))
        bad["by_order"]["1,0"]["audit_branch_accuracy"] = [.94, 1.]
        self.assertTrue(source_qualified(good, good))
        self.assertFalse(source_qualified(good, bad))
        self.assertFalse(source_qualified(bad, good))

    def test_corpus_binding_checks_self_digest_and_actual_training_inputs(self):
        from experiments.activation_funnel_af02 import train as training
        datasets = {"train": training._Data(self.train), "dev": training._Data(self.dev)}
        receipt = json.loads(Path(training.__file__).with_name("corpus_manifest.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corpus.json"
            claimed = receipt.pop("manifest_sha256")
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "self-digest"):
                training.validate_corpus_inputs(path, datasets)
            receipt["manifest_sha256"] = claimed
            receipt["splits"]["train"]["canonical_cases_sha256"] = "0" * 64
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "self-digest"):
                training.validate_corpus_inputs(path, datasets)
            receipt["manifest_sha256"] = training.canonical_hash(
                {key: value for key, value in receipt.items() if key != "manifest_sha256"})
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "train canonical_cases_sha256"):
                training.validate_corpus_inputs(path, datasets)

    def test_resource_accounting_includes_writers_types_and_actual_wait_visits(self):
        from experiments.activation_funnel_af02.model import make_codec, resources
        native = resources(self.models["native"], make_codec("native"), events=3)
        self.assertEqual(native["model_parameters"], 5590)
        self.assertEqual(native["model_buffer_floats"], 12)
        self.assertEqual(native["codec_parameters"], 296)
        self.assertEqual(native["carried_code_bits"], 640)
        self.assertEqual(native["dense_matrix_macs_per_episode"], 6 * (14688 + 640) + 434)
        waited = resources(self.models["native"], make_codec("native"), events=4)
        self.assertEqual(waited["dense_matrix_macs_per_episode"] - native["dense_matrix_macs_per_episode"], 2 * (14688 + 640))
        gru = resources(self.models["gru"], make_codec("gru"), events=3)
        self.assertEqual(gru["dense_matrix_macs_per_episode"], 25440)
        self.assertEqual(gru["carried_code_bits"], 128)

    def test_native_float64_step_supports_reference_state_derivatives(self):
        model = self.models["native"].double()
        h = torch.randn(1, 80, dtype=torch.float64, requires_grad=True)
        out = model.one_step(h, self.x[:1, -1].double())
        gradient = torch.autograd.grad(out[0, 0], h)[0]
        self.assertEqual(out.dtype, torch.float64)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_named_array_export_matches_each_checkpoint_tensor(self):
        from experiments.activation_funnel_af02.train import CONFIG, canonical_hash, save_checkpoint, _load_weights, _file_entry
        with tempfile.TemporaryDirectory() as temp:
            folder, files = Path(temp), {}
            save_checkpoint(folder, "native", 17, "teacher", self.models["native"], CONFIG, files)
            manifest = {"configuration_sha256": canonical_hash(CONFIG), "files": files}
            state = _load_weights(folder, "native", 17, "teacher", manifest)
            self.assertEqual(set(state), set(self.models["native"].state_dict()))
            export = folder / "native_teacher_s17.npz"
            with np.load(export, allow_pickle=False) as old:
                values = {key: old[key].copy() for key in old.files}
            values["task_head.bias"] += 1.
            np.savez(export, **values)
            files[export.name] = _file_entry(export)
            with self.assertRaisesRegex(ValueError, "array value mismatch"):
                _load_weights(folder, "native", 17, "teacher", manifest)

    def test_rehashed_manifest_cannot_omit_weights_runs_or_prediction_rows(self):
        from experiments.activation_funnel_af02 import train as training
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = {}
            for name in training.expected_file_roster():
                (folder / name).write_bytes(b"hash-only fixture; never deserialize")
                files[name] = training._file_entry(folder / name)
            rows = [{"architecture": a, "seed": s, "arm": arm, "split": split,
                     "canonical_id": case.canonical_id}
                    for a in training.CONFIG["architectures"] for s in training.CONFIG["seeds"]
                    for arm in training.CONFIG["arms"]
                    for split, cases in (("train", self.train), ("dev", self.dev)) for case in cases]
            prediction = folder / training.PREDICTION_FILE
            prediction.write_text("".join(json.dumps(row) + "\n" for row in rows))
            hashes = {"model.py": "fixture-model-hash"}
            manifest = {"experiment": "AF-02", "scope": "registered_primary", "training_complete": True,
                        "configuration": training.CONFIG,
                        "configuration_sha256": training.canonical_hash(training.CONFIG),
                        "protocol_sha256": training.REGISTERED_PROTOCOL_SHA256,
                        "dataset_access": ["train", "dev"], "torch_version": "2.5.1+cpu", "threads": 2,
                        "runs": [{"architecture": a, "seed": s} for a in training.CONFIG["architectures"]
                                 for s in training.CONFIG["seeds"]],
                        "datasets": {"train": training._dataset_summary(training._Data(self.train)),
                                     "dev": training._dataset_summary(training._Data(self.dev))},
                        "source_hashes": dict(hashes), "files": files,
                        "frozen_corpus_sha256": training.sha256_file(Path(training.__file__).with_name("corpus_manifest.json")),
                        "prediction_files": {training.PREDICTION_FILE: {**training._file_entry(prediction), "rows": len(rows)}}}
            def write():
                path = folder / "train_manifest.json"
                path.write_text(json.dumps(manifest))
                (folder / "train_manifest.sha256").write_text(training.sha256_file(path))
            with mock.patch.object(training, "source_hashes", return_value=hashes):
                with mock.patch.object(torch, "load", side_effect=AssertionError("unexpected tensor load")):
                    write()
                    self.assertEqual(training.load_frozen(folder)["configuration"], training.CONFIG)
                    omitted = files.pop("native_joint_s43.npz")
                    write()
                    with self.assertRaisesRegex(ValueError, "checkpoint file roster"):
                        training.load_frozen(folder)
                    files["native_joint_s43.npz"] = omitted
                    saved_run = manifest["runs"].pop()
                    write()
                    with self.assertRaisesRegex(ValueError, "metadata roster"):
                        training.load_frozen(folder)
                    manifest["runs"].append(saved_run)
                    manifest["source_hashes"]["model.py"] = "changed"
                    write()
                    with self.assertRaisesRegex(ValueError, "model.py"):
                        training.load_frozen(folder)
                    manifest["source_hashes"] = dict(hashes)
                    prediction.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
                    manifest["prediction_files"][training.PREDICTION_FILE] = {
                        **training._file_entry(prediction), "rows": len(rows) - 1}
                    write()
                    with self.assertRaisesRegex(ValueError, "prediction row roster"):
                        training.load_frozen(folder)


if __name__ == "__main__":
    unittest.main()
