"""Shared bounded SPC-09 numerics, provenance and graph-disjoint cohorts."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/spc09_frozen.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    if c['schema'] != 1 or not 4 <= c['batch_size'] <= 16:
        raise ValueError('schema or microbatch outside registered bounds')
    if not c['ranks'] or any(r < 1 or r > 48 or r % 2 for r in c['ranks']):
        raise ValueError('positive even ranks <=48 required')
    if min(c['depths']) < 3 or c['future_horizon'] < 1 or c['fit_horizon'] < 1:
        raise ValueError('invalid recurrence horizons')
    if not 1 <= c['continuation_updates'] <= 300 or c['checkpoint_every'] != 50:
        raise ValueError('continuation budget exceeded')
    if c['gpu_allocated_bytes'] > 3200000000 or c['additional_rss_bytes'] > 8000000000:
        raise ValueError('memory ceiling exceeded')
    if not c['radius_grid'] or any(not np.isfinite(x) or x <= 0 for x in c['radius_grid']):
        raise ValueError('invalid radius grid')
    if len(set(c['eval_seeds'])) != len(c['eval_seeds']):
        raise ValueError('duplicate evaluation seeds')
    if set(c['eval_seeds']) & {c['fit_seed'], c['validation_seed']}:
        raise ValueError('seed leakage')
    return c


def graph_id(row):
    """Canonical graph, independent of edge order AND query/start node."""
    a = row.detach().cpu().numpy() if hasattr(row, 'detach') else np.asarray(row)
    edges = np.asarray(a[:16], dtype=np.int64).reshape(8, 2)
    edges = edges[np.argsort(edges[:, 0])]
    return hashlib.sha256(edges.astype('<i8').tobytes()).hexdigest()


def partition(identity):
    bucket = int(identity, 16) % 10
    return 'train' if bucket < 6 else ('validation' if bucket < 8 else 'test')


def cohort(split, n, seed, horizon):
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if split not in ('train', 'validation', 'test') or n < 1:
        raise ValueError('invalid cohort')
    rows, labels, seen = [], [], set()
    for attempt in range(10000):
        t, y = make_batch(8, min(16, max(4, n)), seed + attempt * 104729, horizon)
        for row, lab in zip(t, y):
            key = graph_id(row)
            if partition(key) == split and key not in seen:
                rows.append(row); labels.append(lab); seen.add(key)
                if len(rows) == n:
                    return torch.stack(rows), torch.stack(labels)
    raise RuntimeError('could not form unique graph-disjoint cohort')


def fingerprints(tokens):
    return {'graph_ids': [graph_id(t) for t in tokens],
            'prompt_sha256': [hashlib.sha256(t.cpu().numpy().astype('<i8').tobytes()).hexdigest()
                              for t in tokens]}


def rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            counters = Counters(); counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo')
            return counters.WorkingSetSize
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if platform.system() == 'Darwin' else value * 1024


class Budget:
    """Sampled resident-memory guard; bounded kernels are checked between calls."""
    def __init__(self, config):
        self.config = config; self.start = time.perf_counter()
        self.baseline = rss_bytes(); self.peak = self.baseline
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()

    def _sample(self):
        while not self.stop.wait(.05):
            self.peak = max(self.peak, rss_bytes())

    def tick(self):
        import torch
        self.peak = max(self.peak, rss_bytes())
        if self.peak - self.baseline > self.config['additional_rss_bytes']:
            raise MemoryError('additional resident-memory ceiling exceeded')
        if time.perf_counter() - self.start > self.config['max_seconds_per_stage']:
            raise TimeoutError('registered stage runtime budget exhausted')
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > self.config['gpu_allocated_bytes']:
            raise MemoryError('GPU allocated-memory ceiling exceeded')

    def finish(self):
        import torch
        self.stop.set(); self.worker.join(timeout=1)
        self.peak = max(self.peak, rss_bytes())
        return {'runtime_seconds': time.perf_counter() - self.start,
                'baseline_rss_bytes': self.baseline, 'sampled_peak_rss_bytes': self.peak,
                'additional_sampled_rss_bytes': self.peak - self.baseline,
                'rss_sampling_seconds': .05,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version()}


def device_for(config):
    import torch
    if config['device'] == 'cpu' or not torch.cuda.is_available():
        return 'cpu', 'requested CPU' if config['device'] == 'cpu' else 'CUDA unavailable; automatic CPU fallback'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(config['gpu_allocated_bytes'] / total, .99))
    torch.cuda.reset_peak_memory_stats()
    return 'cuda', None


def pinned_manifest():
    old = json.loads((ROOT / 'receipts/spc04_cpu_two_seed_summary.json').read_text())
    return {r['seed']: {'file_sha256': r['checkpoint_file_sha256'],
            'weight_sha256': r['train_receipt']['trained_weights_sha256'],
            'initial_weight_sha256': r['train_receipt']['initial_weights_sha256']}
            for r in old['rows']}


def verify_checkpoints(directory, seeds):
    manifest = pinned_manifest(); rows = []; ready = True
    for seed in seeds:
        path = Path(directory) / f'spc04_seed{seed}.pt'
        row = {'seed': seed, 'filename': path.name, 'expected': manifest[seed]}
        if not path.is_file():
            row['status'] = 'missing'; ready = False
        else:
            row['file_sha256'] = sha256(path)
            row['status'] = 'verified' if row['file_sha256'] == manifest[seed]['file_sha256'] else 'hash_mismatch'
            ready &= row['status'] == 'verified'
        rows.append(row)
    return ready, rows


def load_model(directory, seed):
    from experiments.spectral_state_cost_v0.spc05_phase_interventions import model_from_receipt
    ready, rows = verify_checkpoints(directory, [seed])
    if not ready:
        raise ValueError(f'checkpoint pre-load hash verification failed: {rows}')
    model, digest = model_from_receipt(Path(directory) / f'spc04_seed{seed}.pt', seed)
    if digest != pinned_manifest()[seed]['weight_sha256']:
        raise ValueError('weights differ from independent SPC-04 receipt')
    for p in model.parameters(): p.requires_grad_(False)
    return model.cpu().double()


def native_step(model, tokens):
    from experiments.spectral_state_cost_v0.spc05_phase_interventions import prepare_step
    return prepare_step(model, tokens.to(next(model.parameters()).device))


def state_at(model, tokens, depth):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        step, state = native_step(model, tokens)
        for _ in range(depth): state = step(state)
    return step, state.detach()


def row_basis(matrix, relative=1e-3, absolute=1e-10):
    import torch
    _, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    cutoff = max(absolute, float(s[0]) * relative) if len(s) else absolute
    return vh[s > cutoff].detach(), s.detach()


def observation_function(model, step, state, horizon):
    import torch
    def observe(answer):
        x = torch.cat((state[:, :-1], answer.reshape(1, 1, -1)), dim=1)
        out = []
        for t in range(horizon + 1):
            logits = model.readout(x[:, -1])
            out.append((logits[:, :-1] - logits[:, -1:]).reshape(-1))
            if t < horizon: x = step(x)
        return torch.cat(out)
    return observe


def temporal_jacobian(model, tokens, depth, horizon, budget=None):
    """Sequential answer-injection JVPs. Never builds a full-state Jacobian."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if len(tokens) != 1 or next(model.parameters()).device.type != 'cpu':
        raise ValueError('one prompt on CPU required for float64 derivatives')
    step, state = state_at(model, tokens, depth)
    f = observation_function(model, step, state, horizon)
    answer = state[0, -1]
    columns = []
    with sdpa_kernel([SDPBackend.MATH]), torch.enable_grad():
        for i in range(answer.numel()):
            v = torch.zeros_like(answer); v[i] = 1
            columns.append(torch.autograd.functional.jvp(f, answer, v, create_graph=False)[1].detach())
            if budget: budget.tick()
    return torch.stack(columns, 1), step, state


def orthogonal_rows(rows, rank=None):
    basis, _ = row_basis(rows, relative=1e-10)
    if rank is not None and len(basis) < rank:
        raise ValueError('requested subspace is rank deficient')
    return basis if rank is None else basis[:rank]


def fit_bases(model, config, budget=None):
    import torch
    horizon = max(config['phase_depths']) + 2
    tokens, _ = cohort('train', config['phase_prompts'], config['fit_seed'], horizon)
    increments, states = [], []
    for chunk in tokens.split(config['batch_size']):
        with torch.no_grad():
            step, h = native_step(model, chunk)
            for d in range(max(config['phase_depths']) + 1):
                nxt = step(h)
                if d in config['phase_depths']:
                    increments.append((nxt[:, -1] - h[:, -1]).detach())
                    states.append(h[:, -1].detach())
                h = nxt
        if budget: budget.tick()
    X = torch.cat(states); mean = X.mean(0)
    phase, _ = row_basis(torch.cat(increments), relative=1e-12)
    pca, _ = row_basis(X - mean, relative=1e-12)
    dtokens, _ = cohort('train', config['fit_prompts'], config['fit_seed'] + 17,
                         config['fit_depth'] + config['fit_horizon'] + 1)
    gram = torch.zeros((48, 48), dtype=torch.float64)
    for token in dtokens:
        O, _, _ = temporal_jacobian(model, token[None], config['fit_depth'], config['fit_horizon'], budget)
        row, _ = row_basis(O[:7], 1e-10)
        null = torch.eye(48, dtype=torch.float64) - row.T @ row
        deferred = O[7:] @ null
        gram += deferred.T @ deferred / O.square().sum().clamp_min(1e-30)
    eig, vec = torch.linalg.eigh(gram / len(dtokens))
    return {'P': phase, 'D': vec.flip(1).T.detach(), 'PCA': pca, 'mean': mean.detach(),
            'deferred_eigenvalues': eig.flip(0).clamp_min(0).tolist(),
            'fit_graphs': sorted(set(fingerprints(tokens)['graph_ids'] + fingerprints(dtokens)['graph_ids']))}


def random_rows(rank, seed):
    import torch
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(48, rank, generator=generator, dtype=torch.float64)).Q.T


def paired_interval(clusters, repeats=4000, seed=930999):
    """Clusters are paired (sum of outcome differences, number of items)."""
    rows = np.asarray(clusters, dtype=float)
    if rows.size == 0 or rows[:, 1].sum() == 0:
        return {'difference': None, 'interval95': None, 'clusters': len(clusters), 'n': 0}
    rng = np.random.default_rng(seed); draws = []
    for _ in range(repeats):
        pick = rows[rng.integers(len(rows), size=len(rows))]
        if pick[:, 1].sum(): draws.append(pick[:, 0].sum() / pick[:, 1].sum())
    return {'difference': float(rows[:, 0].sum() / rows[:, 1].sum()),
            'interval95': np.quantile(draws, [.025, .975]).tolist(),
            'clusters': len(rows), 'n': int(rows[:, 1].sum()), 'seed': seed,
            'scope': 'paired prompt-seed clusters conditional on fixed checkpoints; not a seed-population interval'}


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def stage_main(stage, runner):
    import torch
    parser = argparse.ArgumentParser(description=runner.__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--validated-receipts', type=Path)
    args = parser.parse_args(); config = load_config(args.config)
    torch.set_num_threads(config['cpu_threads'])
    budget = Budget(config)
    receipt = {'campaign': 'SPC-09', 'stage': stage, 'status': 'incomplete',
               'config_sha256': sha256(args.config), 'config': config,
               'source_sha256': {p.name: sha256(p) for p in sorted(ROOT.glob('spc09_*.py'))},
               'replication': {'independent_original_seeds': config['model_seeds'],
                               'third_seed': 'not available; incomplete replication'}}
    ready, receipt['checkpoints'] = verify_checkpoints(args.checkpoint_dir, config['model_seeds'])
    try:
        if not ready:
            receipt['reason'] = 'original frozen checkpoint files missing or SHA-256 mismatch; no inference performed'
        else:
            try:
                receipt['result'] = runner(args.checkpoint_dir, config, budget, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                receipt['cpu_retry_reason'] = 'CUDA allocation failed; retrying unchanged protocol on CPU'
                receipt['result'] = runner(args.checkpoint_dir, dict(config, device='cpu'), budget, args)
            receipt['status'] = 'complete'
    except (MemoryError, TimeoutError) as exc:
        receipt['reason'] = f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        receipt['status'] = 'failed'; receipt['reason'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['resources'] = budget.finish()
        save_json(args.out, receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(args.out), 'reason': receipt.get('reason')}))
    return 0 if receipt['status'] == 'complete' else 2
