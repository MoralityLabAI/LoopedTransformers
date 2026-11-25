"""Matched-budget point versus interval-aware discrete search.

Selection scores are numerical diagnostics. Only audit.verify_scalar issues
exact-real invariant certificates. All continuous fits use the LS13 objective.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

import repair13 as parent

SEEDS = tuple(range(445, 469))
DOSE = 15
ARMS = ('point_single', 'bound_single', 'point_paired', 'bound_paired')
ROUNDS = 16
FIT_SLOTS_PER_ANCHOR = 5
SCORE_POINTS = 17
TARGET = 3 / 32
TOL = 1e-12
CASES = np.array(list(itertools.product(range(4), (0, 1), (0, 1), (0, 1))))
GRID = np.linspace(-1 / 8, 1 / 8, SCORE_POINTS)
X = np.column_stack(((CASES[:, 0, None] - 2 + GRID).ravel(),
                     np.repeat(CASES[:, 1:], SCORE_POINTS, axis=0))).astype(float)
Y = np.repeat((CASES[:, :3].sum(1) % 4) - 2, SCORE_POINTS).astype(float)
DELTA = 1 / (4 * (SCORE_POINTS - 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_proposals(c, paired):
    """Stable proposal slots. Illegal moves become no-op; duplicates stay.

    This fixes profile budgets across seeds and selection rules. Shortlisting
    later removes identical codebooks before choosing four non-no-op fits.
    """
    candidates, names = [c.copy()], ['noop']

    def add(q, name):
        if not np.all(np.abs(q) <= 7):
            q, name = c.copy(), 'invalid:' + name
        candidates.append(q)
        names.append(name)

    for i in range(len(c)):
        for d in (-1, 1):
            q = c.copy()
            q[i] += d
            add(q, f'{i}:{d:+}')
    if paired:
        for i in range(0, 8, 2):
            a, b = 4 + 4 * i, 4 + 4 * (i + 1)
            for source, target in ((a, b), (b, a)):
                q = c.copy()
                q[target:target + 4] = c[source:source + 4]
                add(q, f'copy_up:{source}:{target}')
            for d in (-1, 1):
                for j in range(4):
                    q = c.copy()
                    q[a + j] += d
                    q[b + j] += d
                    add(q, f'joint_up:{i}:{j}:{d:+}')
                q = c.copy()
                q[36 + i] += d
                q[36 + i + 1] += d
                add(q, f'joint_bias:{i}:{d:+}')
                q = c.copy()
                q[44 + i] += d
                q[44 + i + 1] -= d
                add(q, f'opp_down:{i}:{d:+}')
            for i0, i1, sign in ((i, i + 1, 1), (i + 1, i, -1)):
                q = c.copy()
                q[36 + i1] = c[36 + i0] - 2 * sign
                add(q, f'gap2:{i0}:{i1}')
                q = c.copy()
                q[44 + i1] = -c[44 + i0]
                add(q, f'opp:{i0}:{i1}')
    assert len(candidates) == (177 if paired else 105)
    return np.stack(candidates), names


def score_batch(codes, scales, offsets, down_bias):
    """Evaluate the stored FP32 decode in float64 at 17 endpoints per case.

    g''(z)=(2-z*z)*phi(z). Its absolute maximum on an interval occurs at
    endpoints or a contained -2, 0, 2. Linear interpolation remainder is
    at most delta**2 * sup|F''| / 8. No directed rounding is performed here.
    """
    codes = np.asarray(codes, dtype=np.int64)
    scales = np.asarray(scales, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    assert codes.ndim == 2 and scales.shape == (len(codes), 4)
    sk, up, bias, down = parent.unpack(codes)

    def decoded(v, s):
        return (v.astype(np.float32) * s.astype(np.float32)).astype(float)

    sk = decoded(sk, scales[:, :1])
    up = decoded(up, scales[:, 1, None, None])
    bias = decoded(bias, scales[:, 2, None])
    down = decoded(down, scales[:, 3, None])
    offset = (offsets - .2 * down_bias).astype(np.float32).astype(float)
    offset += .2 * float(np.float32(down_bias))
    pre = np.einsum('kij,nj->kni', up, X) + bias[:, None, :]
    out = X[None, :, 0] + sk @ X.T + offset[:, None]
    out += .2 * np.einsum('kni,ki->kn', pre * ndtr(pre), down)
    error = out - Y
    point = np.mean(error ** 2 + 10 * np.maximum(abs(error) - TARGET, 0) ** 2, axis=1)
    abs_error = np.abs(error).reshape(len(codes), 32, SCORE_POINTS)
    pre = pre.reshape(len(codes), 32, SCORE_POINTS, 8)
    lo = np.minimum(pre[:, :, :-1], pre[:, :, 1:])
    hi = np.maximum(pre[:, :, :-1], pre[:, :, 1:])

    def curvature(z):
        return np.abs(2 - z * z) * np.exp(-.5 * z * z) / math.sqrt(2 * math.pi)

    curv = np.maximum(curvature(lo), curvature(hi))
    for z in (-2., 0., 2.):
        curv = np.maximum(curv, np.where((lo <= z) & (z <= hi), curvature(z), 0))
    second = .2 * np.sum(curv * (np.abs(down) * up[:, :, 0] ** 2)[:, None, None, :], axis=-1)
    pad = DELTA ** 2 / 8 * second
    segments = np.maximum(abs_error[:, :, :-1], abs_error[:, :, 1:]) + pad
    result = {'point': point, 'grid_max': abs_error.max(axis=(1, 2)),
              'bound': segments.max(axis=(1, 2)), 'max_curvature_pad': pad.max(axis=(1, 2))}
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def scores(c, s, o, db):
    return {key: float(value[0]) for key, value in
            score_batch(c[None], s[None], np.array([o]), db).items()}


def shortlist(candidates, values):
    chosen = [0]
    seen = {candidates[0].tobytes()}
    for j in np.argsort(values, kind='stable'):
        key = candidates[j].tobytes()
        if key not in seen:
            seen.add(key)
            chosen.append(int(j))
        if len(chosen) == FIT_SLOTS_PER_ANCHOR:
            break
    # There are always many legal distinct single moves for this architecture.
    assert len(chosen) == FIT_SLOTS_PER_ANCHOR
    return chosen


def state(c, s, o, db):
    return {'codes': c.tolist(), 'scales': s.tolist(), 'offset': o,
            'scores': scores(c, s, o, db)}


def initialize(seed):
    c, s, o, db = parent.setup(seed, DOSE)
    return {'seed': seed, 'dose': DOSE, 'codes': c.tolist(), 'scales': s.tolist(),
            'offset': o, 'down_bias': db, 'weights': parent.decoded(c, s, o, db)}


def search(initial, arm):
    assert arm in ARMS
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    c = np.array(initial['codes'], dtype=np.int64)
    s = np.array(initial['scales'], dtype=float)
    o, db = initial['offset'], initial['down_bias']
    initial_c, initial_s = c.copy(), s.copy()
    selection = arm.split('_')[0]
    paired = arm.endswith('_paired')
    # One common initial fit, accepted identically in every arm.
    s, o, fit = parent.fit(c, s, o)
    selected = state(c, s, o, db)
    ledger = [{'round': 0, 'move': 'common_initial_fit', 'fit': fit, 'state': selected}]
    fit_count, fit_calls, profile_count = fit['nfev'], 1, 0
    for round_no in range(1, ROUNDS + 1):
        candidates, names = fixed_proposals(c, paired)
        incumbent = (selected['scores'][selection], c.copy(), s.copy(), o, fit, selected, 'incumbent')
        best = incumbent
        anchors = []
        for anchor, anchor_s in (('current', s), ('initial', initial_s)):
            _, coefficients, offsets = parent.profile(candidates, anchor_s)
            candidate_s = np.repeat(anchor_s[None], len(candidates), axis=0)
            candidate_s[:, [0, 3]] = coefficients
            metrics = score_batch(candidates, candidate_s, offsets, db)
            ranking = metrics[selection]
            chosen = shortlist(candidates, ranking)
            fits = []
            for j in chosen:
                ns, no, record = parent.fit(candidates[j], candidate_s[j], float(offsets[j]))
                candidate_state = state(candidates[j], ns, no, db)
                fit_count += record['nfev']
                fit_calls += 1
                fits.append({'slot': j, 'move': names[j], 'fit': record,
                             'scales': ns.tolist(), 'offset': no, 'scores': candidate_state['scores']})
                score = candidate_state['scores'][selection]
                if score < best[0] - TOL:
                    best = (score, candidates[j].copy(), ns, no, record, candidate_state,
                            f'{names[j]}@{anchor}')
            anchors.append({'name': anchor, 'profile_scores': ranking.tolist(), 'fits': fits})
            profile_count += len(candidates)
        _, c, s, o, fit, selected, move = best
        ledger.append({'round': round_no, 'move': move, 'state': selected, 'anchors': anchors})
    assert fit_calls == 1 + ROUNDS * 2 * FIT_SLOTS_PER_ANCHOR == 161
    assert profile_count == ROUNDS * 2 * (177 if paired else 105)
    return {'seed': initial['seed'], 'dose': DOSE, 'arm': arm, 'selection': selection,
            'initial_scores': scores(initial_c, initial_s, initial['offset'], db),
            'final': selected | {'weights': parent.decoded(c, s, o, db)},
            'hamming': int(np.count_nonzero(c != initial_c)),
            'code_l1': int(np.abs(c - initial_c).sum()), 'fit_calls': fit_calls,
            'fit_nfev': fit_count, 'profile_slots': profile_count,
            'ledger': ledger, 'seconds': time.monotonic() - started}


def run_job(job):
    seed, arm, root = job
    root = Path(root)
    path = root / 'initial' / f'{seed}.json'
    initial = json.loads(path.read_text())
    target = root / 'results' / f'{arm}-{seed}.json'
    assert not target.exists(), target
    result = search(initial, arm)
    result.update(initial_sha256=sha(path), protocol_sha256=sha(root / 'protocol.md'),
                  source_lock_sha256=sha(root / 'lock.sha256'))
    target.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print('completed', seed, arm, 'seconds', round(result['seconds'], 2), flush=True)
    return str(target)


def main():
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    assert 1 <= args.workers <= 4
    for line in (root / 'lock.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(root / name.strip()) == digest, name
    assert not (root / 'initial').exists() and not (root / 'results').exists()
    (root / 'initial').mkdir()
    (root / 'results').mkdir()
    for seed in SEEDS:
        (root / 'initial' / f'{seed}.json').write_text(json.dumps(initialize(seed), separators=(',', ':')) + '\n')
    jobs = [(seed, arm, str(root)) for seed in SEEDS for arm in ARMS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run_job, jobs, chunksize=1))


if __name__ == '__main__':
    main()
