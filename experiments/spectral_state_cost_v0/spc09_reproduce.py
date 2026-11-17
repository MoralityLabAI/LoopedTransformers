"""A: original SPC-08 cohorts and thresholds, with bounded inference batches."""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import numpy as np

from experiments.spectral_state_cost_v0 import spc09_common as common


@contextlib.contextmanager
def bounded_original(batch_size=8, budget=None):
    """Scope and restore adapters; preserve original prompt generation and FD epsilon."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from experiments.spectral_state_cost_v0 import spc05_phase_subspace as phase
    from experiments.spectral_state_cost_v0 import spc07_transfer as transfer
    from experiments.spectral_state_cost_v0 import spc08_phase_readout as original
    from experiments.spectral_state_cost_v0.spc05_phase_interventions import prepare_step
    if not 4 <= batch_size <= 16: raise ValueError('microbatch outside hardware limits')

    def bounded_step(model, tokens):
        chunks = list(tokens.split(batch_size))
        pairs = [prepare_step(model, t) for t in chunks]
        def advance(h):
            result = torch.cat([pair[0](part) for pair, part in zip(pairs, h.split(batch_size))])
            if budget: budget.tick()
            return result
        return advance, torch.cat([p[1] for p in pairs])

    def finite(model, tokens, depth=32, horizon=8, eps=2e-4):
        if len(tokens) != 1: raise ValueError('one derivative prompt required')
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            step, x = bounded_step(model, tokens)
            for _ in range(depth): x = step(x)
            offsets = torch.cat((torch.eye(48, dtype=x.dtype), -torch.eye(48, dtype=x.dtype))) * eps
            chunks = []
            for off in offsets.split(batch_size):
                varied = x.repeat(len(off), 1, 1); varied[:, -1] += off
                advance, _ = bounded_step(model, tokens.repeat(len(off), 1))
                outputs = []
                for t in range(horizon + 1):
                    logit = model.readout(varied[:, -1])
                    outputs.append(logit[:, :-1] - logit[:, -1:])
                    if t < horizon: varied = advance(varied)
                chunks.append(torch.stack(outputs, 1))
            values = torch.cat(chunks)
            return ((values[:48] - values[48:]) / (2 * eps)).permute(1, 2, 0).reshape(-1, 48), x

    replacements = [(module, 'prepare_step', bounded_step) for module in (phase, transfer, original)]
    replacements += [(original, 'finite_temporal_jacobian', finite),
                     (transfer, 'finite_temporal_jacobian', finite)]
    old = [(m, name, getattr(m, name)) for m, name, _ in replacements]
    try:
        for m, name, value in replacements: setattr(m, name, value)
        yield
    finally:
        for m, name, value in reversed(old): setattr(m, name, value)


def compare_numbers(expected, actual, tolerance, prefix=''):
    differences = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            path = prefix + '/' + str(key)
            if key not in actual: differences.append({'metric': path, 'status': 'missing'})
            else: differences.extend(compare_numbers(value, actual[key], tolerance, path))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)):
            differences.append({'metric': prefix, 'status': 'type_mismatch'})
        elif (expected != actual if isinstance(expected, int) else abs(expected - actual) > tolerance):
            differences.append({'metric': prefix, 'expected': expected, 'actual': actual,
                                'absolute_difference': abs(expected - actual)})
    elif isinstance(expected, list):
        if len(expected) != len(actual): differences.append({'metric': prefix, 'status': 'length_mismatch'})
        else:
            for i, (x, y) in enumerate(zip(expected, actual)):
                differences.extend(compare_numbers(x, y, tolerance, prefix + '/' + str(i)))
    return differences


def initial_control(seed):
    """A seed alone cannot attest to bit-identical initialization across hosts."""
    from experiments.spectral_state_cost_v0.spc04_trained_dynamics import new_model, weight_sha256
    model = new_model(seed)
    actual = weight_sha256(model)
    expected = common.pinned_manifest()[seed]['initial_weight_sha256']
    info = {'same_initialization': actual == expected,
            'expected_initial_weight_sha256': expected,
            'actual_initial_weight_sha256': actual}
    if actual != expected:
        info.update(status='original_initialization_unavailable',
                    reason='seeded native initialization differs from original tensor hash; no matched control substituted')
        return None, info
    return model, info


def run(checkpoints, config, budget, args):
    """Replay original rank-eight geometry, derivatives, exact fibers and controls."""
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    from experiments.spectral_state_cost_v0 import spc08_phase_readout as original
    from experiments.spectral_state_cost_v0.spc04_trained_dynamics import new_model, weight_sha256
    reference = json.loads((common.ROOT / 'receipts/spc08_summary.json').read_text())
    from unittest.mock import patch
    progress = {'fits': [], 'heldout_batches': []}
    progress_path = args.out.with_name(args.out.stem + '_partial.json')
    original_fit, original_batch = original.fit_causal_decomposition, original.causal_phase_batch
    def fit_with_receipt(*a, **kw):
        result = original_fit(*a, **kw)
        progress['fits'].append(result[1]); common.save_json(progress_path, progress)
        return result
    def batch_with_receipt(*a, **kw):
        result = original_batch(*a, **kw)
        progress['heldout_batches'].append(result); common.save_json(progress_path, progress)
        return result
    with bounded_original(config['batch_size'], budget), patch.object(original,'fit_causal_decomposition',fit_with_receipt), patch.object(original,'causal_phase_batch',batch_with_receipt):
        data = original.campaign(Path(checkpoints))
        summarized = original.summarize(data)
        derivative, controls = {}, {}
        for seed in config['model_seeds']:
            model = common.load_model(checkpoints, seed)
            bases, _, _ = original.fit_causal_decomposition(model)
            derivative[str(seed)] = original.heldout_local_sensitivity(model, bases, 910709, 12, 32, 8)
            untrained, initialization = initial_control(seed)
            if untrained is None:
                controls[str(seed)] = initialization
                continue
            untrained = untrained.eval().double()
            for p in untrained.parameters(): p.requires_grad_(False)
            entries = []
            for prompt_seed in original.EVAL_SEEDS:
                tokens, hops = make_batch(8, original.EVAL_BATCH, prompt_seed, 65)
                correct = []
                for tok, labels in zip(tokens.split(config['batch_size']), hops.split(config['batch_size'])):
                    _, state = common.state_at(untrained, tok, 64)
                    correct.extend((untrained.readout(state[:, -1]).argmax(-1) == labels[:, 63]).tolist())
                entries.append({'prompt_seed': prompt_seed, 'correct': sum(correct), 'n': len(correct)})
            untrained_phase = original_batch(untrained, bases, seed=910701, batch=64)
            controls[str(seed)] = {**initialization, 'status': 'measured', 'untrained_depth64': entries,
                'untrained_phase_control': untrained_phase,
                'basis_note': 'trained target basis applied to matched untrained initialization'}
    differences = compare_numbers(reference['paired_results'], summarized,
                                config['decisions']['reproduction_absolute_tolerance'], 'paired_results')
    for seed, fit in data['fitted'].items():
        for key in ('captures', 'projector_overlaps'):
            differences += compare_numbers(reference['fitted'][seed][key], fit[key],
                            config['decisions']['reproduction_absolute_tolerance'], f'fitted/{seed}/{key}')
        prior_derivative = reference['heldout_jacobian']['rows'][seed]['heldout_local_jacobian']
        for key in ('basis_capture','local_rank_rel_1e-3'):
            differences += compare_numbers(prior_derivative[key], derivative[seed][key],
                            config['decisions']['reproduction_absolute_tolerance'], f'heldout/{seed}/{key}')
    # Audit original cohorts independently of their original exact-prompt audit.
    fit_graphs = set()
    for prompt_seed, count in ((904001, 256), (920731, 16)):
        t, _ = make_batch(8, count, prompt_seed, 1)
        fit_graphs.update(common.graph_id(x) for x in t)
    overlap = {}
    for prompt_seed in original.EVAL_SEEDS:
        t, _ = make_batch(8, 64, prompt_seed, 1)
        overlap[str(prompt_seed)] = len(fit_graphs & {common.graph_id(x) for x in t})
    raw_path = args.out.with_name(args.out.stem + '_original_full.json')
    common.save_json(raw_path, data)
    trained_differences = list(differences)
    for seed, record in controls.items():
        if not record['same_initialization']:
            differences.append({'metric': f'untrained/{seed}/initial_weight_sha256',
                'status': record['status'], 'expected': record['expected_initial_weight_sha256'],
                'actual': record['actual_initial_weight_sha256']})
    return {'reproduction_pass': not differences, 'divergences': differences,
            'frozen_trained_reproduction_pass': not trained_differences,
            'frozen_trained_divergences': trained_differences,
            'original_full_receipt': raw_path.name, 'original_full_sha256': common.sha256(raw_path),
            'summary': summarized, 'fitted': data['fitted'], 'heldout_derivatives': derivative,
            'untrained_controls': controls, 'original_graph_overlap_counts': overlap,
            'split_note': 'A preserves original prompt cohorts; B-E enforce canonical graph partitions',
            'paired_bootstrap': original.paired_bootstrap(data)}


if __name__ == '__main__':
    raise SystemExit(common.stage_main('A', run))
