"""Executable contracts for the fixed-target probe study."""
import copy
import inspect
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_gym.analysis.bounded_probes import (
    ProbeSpec, Standardizer, array_hash, canonical_permutation, evaluate_selected,
    fit_probe, fixed_target, group_means, json_hash, make_graph_splits, paired_summary,
    partition_types, permutation_signature, rename_permutation, scores,
    select_probes, validate_graph_splits,
)


class GraphSplitTests(unittest.TestCase):
    def setUp(self):
        self.splits = make_graph_splits(8, [6, 3, 3], 5, 75)

    def test_partition_counts(self):
        self.assertEqual([len(partition_types(n)) for n in range(1, 9)], [1, 2, 3, 5, 7, 11, 15, 22])

    def test_all_permutations_have_canonical_type(self):
        self.assertEqual({permutation_signature(p) for p in itertools.permutations(range(4))}, set(partition_types(4)))

    def test_exhaustive_renaming_invariance(self):
        for p in itertools.permutations(range(4)):
            for rename in itertools.permutations(range(4)):
                self.assertEqual(permutation_signature(p), permutation_signature(rename_permutation(p, rename)))

    def test_cycle_type_is_complete_invariant_n4(self):
        all_p = list(itertools.permutations(range(4)))
        for p in all_p:
            orbit = {tuple(rename_permutation(p, r)) for r in all_p}
            same_type = {q for q in all_p if permutation_signature(p) == permutation_signature(q)}
            self.assertEqual(orbit, same_type)

    def test_invalid_permutations(self):
        for p in ([], [0, 0], [-1, 0], [0.0, 1.0], [[0, 1]], [1, 2]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                permutation_signature(p)

    def test_enumeration_bound(self):
        for n in (0, 33, True, 3.5):
            with self.assertRaises(ValueError):
                partition_types(n)

    def test_invalid_cycle_lengths(self):
        for lengths in ([], [0], [-1, 2], [True, 2], [1.5]):
            with self.assertRaises(ValueError):
                canonical_permutation(lengths)

    def test_rename_size_error(self):
        with self.assertRaises(ValueError):
            rename_permutation([0, 1], [0, 1, 2])

    def test_fixed_target_is_independent_of_visit(self):
        p = [1, 2, 0]
        self.assertEqual(fixed_target(p, 0, 8), 2)
        self.assertEqual(fixed_target(p, 0, 16), 1)
        self.assertEqual(fixed_target(p, 1, 0), 1)

    def test_target_validation(self):
        for start, horizon in ((-1, 2), (2, 1), (0, -1), (0, 1.5), (True, 1)):
            with self.assertRaises(ValueError):
                fixed_target([1, 0], start, horizon)

    def test_split_determinism(self):
        self.assertEqual(self.splits, make_graph_splits(8, [6, 3, 3], 5, 75))

    def test_seed_changes_split_or_items(self):
        self.assertNotEqual(self.splits, make_graph_splits(8, [6, 3, 3], 5, 76))

    def test_no_structural_overlap(self):
        result = validate_graph_splits(self.splits)
        self.assertTrue(result['passed'])
        self.assertEqual(result['split_overlap_groups'], 0)

    def test_renamed_graph_leak_rejected(self):
        bad = copy.deepcopy(self.splits)
        row = copy.deepcopy(bad['train'][0])
        row['id'], row['split'] = 'new_test_item', 'test'
        bad['test'][0] = row
        with self.assertRaisesRegex(ValueError, 'cross splits'):
            validate_graph_splits(bad)

    def test_tampered_tokens_rejected(self):
        bad = copy.deepcopy(self.splits)
        bad['train'][0]['tokens'][0] = -1
        with self.assertRaisesRegex(ValueError, 'tokens'):
            validate_graph_splits(bad)

    def test_tampered_hash_rejected(self):
        bad = copy.deepcopy(self.splits)
        bad['train'][0]['graph_hash'] = 'forged'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_graph_splits(bad)

    def test_duplicate_id_rejected(self):
        bad = copy.deepcopy(self.splits)
        bad['train'][1]['id'] = bad['train'][0]['id']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_graph_splits(bad)

    def test_group_count_limits(self):
        for counts in ([2, 2, 2], [0, 1, 1], [1, 1], [True, 1, 1]):
            with self.assertRaises(ValueError):
                make_graph_splits(4, counts, 2, 1)

    def test_group_equal_weighting(self):
        keys, means = group_means([1, 1, 1, 9], ['a', 'a', 'a', 'b'])
        self.assertEqual(keys, ['a', 'b'])
        self.assertEqual(means.mean(), 5)


class ScoringTests(unittest.TestCase):
    def test_train_only_standardization(self):
        x = np.arange(24).reshape(2, 4, 3)
        fitted = Standardizer.fit(x)
        before = fitted.receipt()
        fitted.transform(x + 1e6)
        self.assertEqual(before, fitted.receipt())
        np.testing.assert_allclose(fitted.transform(x).reshape(-1, 3).mean(0), 0, atol=1e-14)

    def test_constant_dimensions_stay_finite(self):
        x = np.ones((2, 4, 3))
        z = Standardizer.fit(x).transform(x)
        np.testing.assert_array_equal(z, 0)

    def test_nonfinite_features_rejected(self):
        for x in (np.zeros((2, 3)), np.full((1, 2, 3), np.nan), np.zeros((1, 0, 2))):
            with self.assertRaises(ValueError):
                Standardizer.fit(x)

    def test_width_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Standardizer.fit(np.zeros((2, 3, 4))).transform(np.zeros((2, 3, 5)))

    def test_stable_logit_loss(self):
        losses, correct = scores([[1000, 1000], [-1000, -1001]], np.array([0, 0]))
        np.testing.assert_allclose(losses, [np.log(2), np.log1p(np.exp(-1))])
        self.assertTrue(correct.all())

    def test_invalid_labels_rejected(self):
        for labels in (np.array([2]), np.array([0.0]), np.array([])):
            with self.assertRaises(ValueError):
                scores(np.zeros((1, 2)), labels)

    def test_paired_summary_clusters(self):
        out = paired_summary([2, 2, 2, 6], [1, 1, 1, 2], ['a', 'a', 'a', 'b'], seed=5)
        self.assertEqual(out['independent_groups'], 2)
        self.assertEqual(out['mean_gain_nats'], 2.5)

    def test_single_group_has_no_interval(self):
        out = paired_summary([2, 2], [1, 1], ['a', 'a'], seed=1)
        self.assertIsNone(out['bootstrap_percentile_95'])
        self.assertEqual(out['ci_status'], 'INSUFFICIENT_GROUPS')

    def test_bootstrap_reproducible(self):
        kw = dict(baseline_losses=[1, 3, 4], visit_losses=[0, 1, 3], groups=['a', 'b', 'c'], seed=50)
        self.assertEqual(paired_summary(**kw), paired_summary(**kw))

    def test_zero_gain_for_equal_losses(self):
        out = paired_summary([1, 2], [1, 2], ['a', 'b'], seed=1)
        self.assertEqual(out['bootstrap_percentile_95'], [0.0, 0.0])

    def test_hash_sensitive_to_dtype_and_shape(self):
        a = np.array([1, 2], dtype=np.int32)
        self.assertNotEqual(array_hash({'a': a}), array_hash({'a': a.astype(np.int64)}))
        self.assertNotEqual(array_hash({'a': a}), array_hash({'a': a.reshape(1, 2)}))

    def test_json_hash_ignores_key_order(self):
        self.assertEqual(json_hash({'a': 1, 'b': 2}), json_hash({'b': 2, 'a': 1}))


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        rng = np.random.default_rng(8)
        self.x = rng.standard_normal((2, 16, 4))
        self.y = (self.x[0, :, 0] > 0).astype(np.int64)
        self.groups = ['a'] * 8 + ['b'] * 8
        self.spec = ProbeSpec(steps=12, batch_size=16, learning_rate=.02)

    def test_fit_deterministic(self):
        a = fit_probe(self.x, self.y, 2, self.spec, 16)
        b = fit_probe(self.x, self.y, 2, self.spec, 16)
        self.assertEqual(a.receipt(), b.receipt())

    def test_exact_budget(self):
        probe = fit_probe(self.x, self.y, 2, self.spec, 8)
        self.assertEqual(probe.training['examples_processed'], 192)
        self.assertEqual(probe.training['steps'], 12)
        self.assertEqual(probe.receipt()['parameters'], 10)

    def test_mlp_budget_and_count(self):
        spec = ProbeSpec(family='mlp', hidden=5, steps=3, batch_size=4)
        probe = fit_probe(self.x, self.y, 2, spec, 8)
        self.assertEqual(probe.receipt()['parameters'], 37)
        self.assertEqual(probe.receipt()['linear_multiply_accumulates_per_item'], 30)

    def test_test_data_absent_from_selection_interface(self):
        self.assertFalse(any('test' in key for key in inspect.signature(select_probes).parameters))

    def test_selection_unchanged_after_test_label_change(self):
        selected = select_probes(self.x, self.y, self.x, self.y, self.groups, [0, 1], 2,
                                 [self.spec], mode='separate', seed=7)
        seal = json_hash(selected.receipt())
        evaluate_selected(selected, self.x, self.y, self.groups, seed=2, bootstrap=10)
        evaluate_selected(selected, self.x, 1 - self.y, self.groups, seed=2, bootstrap=10)
        self.assertEqual(seal, json_hash(selected.receipt()))

    def test_identical_visits_give_zero_separate_gain(self):
        x = np.stack([self.x[0], self.x[0]])
        selected = select_probes(x, self.y, x, self.y, self.groups, [0, 1], 2,
                                 [self.spec], mode='separate', seed=7)
        out = evaluate_selected(selected, x, self.y, self.groups, seed=7)
        self.assertEqual(out['rows'][1]['gain_from_input_only']['mean_gain_nats'], 0)

    def test_shared_uses_one_decoder(self):
        selected = select_probes(self.x, self.y, self.x, self.y, self.groups, [0, 1], 2,
                                 [self.spec], mode='shared', seed=7)
        self.assertEqual(len(selected.probes), 1)
        self.assertEqual(selected.probes[0].training['examples_processed'], 192)

    def test_shuffle_is_consistent_and_changes_training_labels(self):
        kw = dict(training_x=self.x, training_y=self.y, validation_x=self.x,
                  validation_y=self.y, validation_groups=self.groups, visits=[0, 1],
                  n_classes=2, specs=[self.spec], mode='separate', seed=7)
        normal = select_probes(**kw)
        shuffled = select_probes(**kw, shuffle_labels=True)
        hashes = {p.training['training_labels_sha256'] for p in shuffled.probes}
        self.assertEqual(len(hashes), 1)
        self.assertNotEqual(next(iter(hashes)), normal.probes[0].training['training_labels_sha256'])

    def test_invalid_probe_specs(self):
        for spec in (ProbeSpec(family='other'), ProbeSpec(steps=0), ProbeSpec(learning_rate=np.nan),
                     ProbeSpec(weight_decay=-1), ProbeSpec(dtype='int32'), ProbeSpec(hidden=True)):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_prediction_is_pure(self):
        fitted = fit_probe(self.x, self.y, 2, self.spec, 8)
        receipt = fitted.receipt()
        fitted.predict_logits(self.x[0])
        self.assertEqual(receipt, fitted.receipt())

    def test_xor_accessibility_control(self):
        bits = np.array(list(itertools.product([-1., 1.], repeat=2)) * 16)
        y = (bits.prod(1) > 0).astype(np.int64)
        early = np.column_stack([bits, np.zeros(len(bits))])
        late = np.column_stack([bits, bits.prod(1)])
        x = np.stack([early, late])
        spec = ProbeSpec(steps=150, batch_size=128, learning_rate=.05)
        selected = select_probes(x, y, x, y, ['a'] * 32 + ['b'] * 32, [0, 1], 2,
                                 [spec], mode='separate', seed=20)
        result = evaluate_selected(selected, x, y, ['a'] * 32 + ['b'] * 32, seed=20)
        self.assertGreater(result['rows'][1]['gain_from_input_only']['mean_gain_nats'], .60)


class NativeInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        try:
            from research_gym.scripts.loopscope_semantic_echo import load_model
            cls.load_model = staticmethod(load_model)
        except ImportError:
            raise unittest.SkipTest('native LoopedChase module is not installed')

    def test_native_feature_layout_and_zero_baseline(self):
        from research_gym.scripts.loopscope_bounded_probes import extract_native_features
        model, cfg = self.load_model(None, smoke=True, seed=91, device='cpu', dtype='float64')
        splits = make_graph_splits(4, [3, 1, 1], 2, 91)
        x, logits, parity = extract_native_features(model, splits['train'], [0, 1, 2, 4], device='cpu', dtype='float64')
        self.assertEqual(x.shape, (4, 6, 160))
        self.assertEqual(logits.shape, (4, 6, 4))
        self.assertTrue(all(row['passed'] for row in parity))
        np.testing.assert_array_equal(x[0, :, 80:], 0)
        for visit in range(4):
            np.testing.assert_array_equal(x[visit, :, :80], x[0, :, :80])

    def test_checkpoint_round_trip_loads_native_interface(self):
        import torch
        model, cfg = self.load_model(None, smoke=True, seed=91, device='cpu', dtype='float32')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'untrained_fixture.pt'
            torch.save({'cfg': cfg, 'model': model.state_dict()}, path)
            loaded, loaded_cfg = self.load_model(path, smoke=False, seed=999, device='cpu', dtype='float32')
        self.assertEqual(cfg, loaded_cfg)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))

    def test_variable_depth_checkpoint_rejected(self):
        import torch
        model, cfg = self.load_model(None, smoke=True, seed=91, device='cpu', dtype='float32')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad_fixture.pt'
            torch.save({'cfg': {**cfg, 'fixed_depth': False}, 'model': model.state_dict()}, path)
            with self.assertRaises(ValueError):
                self.load_model(path, smoke=False, seed=999, device='cpu', dtype='float32')


if __name__ == '__main__':
    unittest.main()
