"""Replay LS14 receipts and independently verify the fixed-region invariant.

The replay checks every stored shortlist and ranking, using direct tensor
evaluation to check the selected numerical scores. It does not rerun SciPy
optimization trajectories. Exact certificates use the unchanged LS12 verifier.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
import torch

import search14 as experiment
import repair13 as parent
from audit import verify_scalar
from construction import scalar_weights
from parent_train import NAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10), (label, a, b)


def weights_digest(w):
    return hashlib.sha256(json.dumps(w, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def input_grid(points):
    cases = np.array(list(itertools.product(range(4), (0, 1), (0, 1), (0, 1))))
    h = cases[:, 0, None] - 2 + np.linspace(-.125, .125, points)
    x = np.column_stack((h.ravel(), np.repeat(cases[:, 1:], points, axis=0)))
    target = np.repeat((cases[:, :3].sum(axis=1) % 4) - 2, points)
    return x, target


def direct_values(w, points=17):
    """Direct stored-weight evaluation, separate from score_batch's decoder."""
    x, target = input_grid(points)
    up = np.asarray(w['up'], dtype=float)
    pre = x @ up.T + np.asarray(w['up_bias'])
    output = (x[:, 0] + x @ np.asarray(w['skip']) + w['skip_bias'][0]
              + .2 * ((pre * ndtr(pre)) @ np.asarray(w['down']) + w['down_bias'][0]))
    return output - target, pre


def direct_scores(w):
    """Independent numerical implementation of the interpolation formula.

    This is a consistency check in floating arithmetic, not an extra proof.
    """
    errors, pre = direct_values(w, 17)
    errors = errors.reshape(32, 17)
    pre = pre.reshape(32, 17, 8)
    left, right = pre[:, :-1], pre[:, 1:]
    lo, hi = np.minimum(left, right), np.maximum(left, right)
    candidates = [lo, hi]
    for critical in (-2., 0., 2.):
        candidates.append(np.clip(np.full_like(lo, critical), lo, hi))
    curvature = np.max(np.stack([
        np.abs(2 - z ** 2) * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        for z in candidates]), axis=0)
    coefficients = np.abs(np.asarray(w['down'])) * np.asarray(w['up'])[:, 0] ** 2
    remainder = np.sum(curvature * coefficients, axis=-1) / (40 * 64 ** 2)
    interval_error = np.maximum(abs(errors[:, :-1]), abs(errors[:, 1:])) + remainder
    return {'point': float(np.mean(errors ** 2 + 10 * np.maximum(abs(errors) - 3 / 32, 0) ** 2)),
            'grid_max': float(abs(errors).max()), 'bound': float(interval_error.max()),
            'max_curvature_pad': float(remainder.max())}


def check_scores(record, w, label):
    checked = direct_scores(w)
    assert set(record) == set(checked), label
    for name in checked:
        close(record[name], checked[name], f'{label}.{name}')
    return checked


def check_state(state, down_bias, label):
    c = np.asarray(state['codes'])
    s = np.asarray(state['scales'], dtype=float)
    assert c.shape == (52,) and np.equal(c, np.rint(c)).all() and np.abs(c).max() <= 7, label
    assert s.shape == (4,) and np.isfinite(s).all() and (s >= parent.LO - 1e-14).all()
    assert (s <= parent.HI + 1e-10).all() and -8 - 1e-10 <= state['offset'] <= 8 + 1e-10
    w = parent.decoded(c.astype(np.int64), s, state['offset'], down_bias)
    check_scores(state['scores'], w, label)
    return w


def check_fit(record, codes, scales, offset, label):
    assert isinstance(record['nfev'], int) and 1 <= record['nfev'] <= 40, label
    assert isinstance(record['njev'], int) and 1 <= record['njev'] <= record['nfev'], label
    assert record['status'] in (0, 1, 2, 3, 4), label
    sk, up, bias, down = parent.unpack(np.asarray(codes, dtype=np.int64))
    s = np.asarray(scales)
    z = s[1] * (parent.X @ up.T) + s[2] * bias
    e = parent.H + s[0] * (parent.X @ sk) + .2 * s[3] * ((z * ndtr(z)) @ down) + offset - parent.Y
    close(record['loss'], parent.score(e), label + '.loss')


def apply_named_move(codes, name):
    """Read the logged operation independently of fixed_proposals."""
    q = np.asarray(codes, dtype=np.int64).copy()
    if name == 'noop':
        return q
    parts = name.split(':')
    kind = parts[0]
    args = list(map(int, parts[1:]))
    if kind.isdigit():
        assert len(args) == 1 and args[0] in (-1, 1)
        q[int(kind)] += args[0]
    elif kind == 'copy_up':
        source, destination = args
        assert source in range(4, 36, 4) and destination in range(4, 36, 4)
        assert ((source - 4) // 8) == ((destination - 4) // 8) and source != destination
        q[destination:destination + 4] = q[source:source + 4]
    elif kind == 'joint_up':
        i, j, delta = args
        assert i in (0, 2, 4, 6) and 0 <= j < 4 and delta in (-1, 1)
        q[4 + 4 * i + j] += delta
        q[4 + 4 * (i + 1) + j] += delta
    elif kind in ('joint_bias', 'opp_down'):
        i, delta = args
        assert i in (0, 2, 4, 6) and delta in (-1, 1)
        base = 36 if kind == 'joint_bias' else 44
        q[base + i] += delta
        q[base + i + 1] += delta if kind == 'joint_bias' else -delta
    elif kind in ('gap2', 'opp'):
        source, destination = args
        assert 0 <= source < 8 and source // 2 == destination // 2 and source != destination
        if kind == 'gap2':
            q[36 + destination] = q[36 + source] - (2 if destination > source else -2)
        else:
            q[44 + destination] = -q[44 + source]
    else:
        raise AssertionError(name)
    return q


def check_proposals(codes, paired):
    candidates, names = experiment.fixed_proposals(codes, paired)
    assert len(candidates) == (177 if paired else 105)
    for q, name in zip(candidates, names):
        invalid = name.startswith('invalid:')
        intended = apply_named_move(codes, name[len('invalid:'):] if invalid else name)
        if invalid:
            assert (abs(intended) > 7).any() and np.array_equal(q, codes)
        else:
            assert (abs(intended) <= 7).all() and np.array_equal(q, intended)
    assert np.array_equal(candidates[0], codes) and names[0] == 'noop'
    return candidates, names


def choose_shortlist(candidates, ranking):
    # Python's stable sort supplies an independent deterministic ordering.
    result = [0]
    seen = {tuple(candidates[0])}
    for j in sorted(range(len(candidates)), key=lambda i: ranking[i]):
        key = tuple(candidates[j])
        if key not in seen:
            result.append(j)
            seen.add(key)
        if len(result) == 5:
            break
    assert len(result) == 5
    return result


def audit_receipt(receipt, initial, root, path):
    arm, seed = receipt['arm'], receipt['seed']
    assert arm in experiment.ARMS and seed == initial['seed'] and receipt['dose'] == 15
    assert receipt['initial_sha256'] == sha(root / 'initial' / f'{seed}.json')
    assert receipt['protocol_sha256'] == sha(root / 'protocol.md')
    assert receipt['source_lock_sha256'] == sha(root / 'lock.sha256')
    selection = 'bound' if arm.startswith('bound_') else 'point'
    assert receipt['selection'] == selection
    paired = arm.endswith('_paired')
    db = initial['down_bias']
    check_scores(receipt['initial_scores'], initial['weights'], 'initial')
    ledger = receipt['ledger']
    assert len(ledger) == 17 and ledger[0]['round'] == 0
    assert ledger[0]['move'] == 'common_initial_fit'
    previous = ledger[0]['state']
    assert previous['codes'] == initial['codes']
    check_state(previous, db, 'initial fit')
    check_fit(ledger[0]['fit'], previous['codes'], previous['scales'], previous['offset'], 'initial fit')
    fit_calls, fit_nfev, profile_slots = 1, ledger[0]['fit']['nfev'], 0
    accepted_edits = accepted_updates = 0
    for round_no, row in enumerate(ledger[1:], 1):
        assert row['round'] == round_no and len(row['anchors']) == 2
        old_codes = np.asarray(previous['codes'], dtype=np.int64)
        candidates, names = check_proposals(old_codes, paired)
        best_state, best_score, best_move = previous, previous['scores'][selection], 'incumbent'
        for anchor_no, anchor in enumerate(row['anchors']):
            anchor_name = ('current', 'initial')[anchor_no]
            assert anchor['name'] == anchor_name
            ss = np.asarray(previous['scales'] if anchor_no == 0 else initial['scales'])
            _, coeffs, offsets = parent.profile(candidates, ss)
            candidate_s = np.broadcast_to(ss, (len(candidates), 4)).copy()
            candidate_s[:, 0] = coeffs[:, 0]
            candidate_s[:, 3] = coeffs[:, 1]
            recomputed = experiment.score_batch(candidates, candidate_s, offsets, db)[selection]
            stored = anchor['profile_scores']
            close(stored, recomputed, 'profile ranking')
            chosen = choose_shortlist(candidates, stored)
            assert [f['slot'] for f in anchor['fits']] == chosen
            assert len(anchor['fits']) == 5
            profile_slots += len(candidates)
            for fit in anchor['fits']:
                j = fit['slot']
                assert fit['move'] == names[j]
                candidate_state = {'codes': candidates[j].tolist(), 'scales': fit['scales'],
                                   'offset': fit['offset'], 'scores': fit['scores']}
                check_state(candidate_state, db, f'{round_no}/{anchor_name}/{j}')
                check_fit(fit['fit'], candidates[j], fit['scales'], fit['offset'], 'shortlisted fit')
                fit_calls += 1
                fit_nfev += fit['fit']['nfev']
                value = fit['scores'][selection]
                if value < best_score - 1e-12:
                    best_state, best_score = candidate_state, value
                    best_move = f'{fit["move"]}@{anchor_name}'
        assert row['state'] == best_state and row['move'] == best_move
        assert row['state']['scores'][selection] <= previous['scores'][selection]
        accepted_updates += best_move != 'incumbent'
        accepted_edits += row['state']['codes'] != previous['codes']
        previous = row['state']
    assert fit_calls == receipt['fit_calls'] == 161
    assert fit_nfev == receipt['fit_nfev'] and fit_nfev <= 6440
    assert profile_slots == receipt['profile_slots'] == (5664 if paired else 3360)
    final = receipt['final']
    assert {key: final[key] for key in previous} == previous
    expected_weights = check_state(final, db, 'final')
    assert final['weights'] == expected_weights
    nc, ic = np.asarray(final['codes']), np.asarray(initial['codes'])
    assert receipt['hamming'] == int(np.count_nonzero(nc != ic))
    assert receipt['code_l1'] == int(np.abs(nc - ic).sum())
    dense_max = float(np.abs(direct_values(final['weights'], 257)[0]).max())
    assert dense_max <= final['scores']['bound'] + 1e-9, (arm, seed, dense_max, final['scores'])
    return {'receipt_sha256': sha(path), 'accepted_edits': accepted_edits,
            'accepted_updates': accepted_updates, 'dense_grid_max': dense_max,
            'profile_slots': profile_slots, 'fit_calls': fit_calls, 'fit_nfev': fit_nfev}


def numerical_evaluation(w, seed):
    q = [torch.tensor(w[key], dtype=torch.float32)[None] for key in NAMES]
    return parent.old.evaluate(q, seed)


def audit_seed(job):
    seed, root_string = job
    root = Path(root_string)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial_path = root / 'initial' / f'{seed}.json'
    initial = json.loads(initial_path.read_text())
    assert initial == experiment.initialize(seed)
    initial_cert = verify_scalar(initial['weights'])
    initial_eval = numerical_evaluation(initial['weights'], seed)
    rows = []
    common_fit = None
    for arm in experiment.ARMS:
        path = root / 'results' / f'{arm}-{seed}.json'
        receipt = json.loads(path.read_text())
        assert receipt['seed'] == seed and receipt['arm'] == arm
        if common_fit is None:
            common_fit = receipt['ledger'][0]
        else:
            assert receipt['ledger'][0] == common_fit
        replay = audit_receipt(receipt, initial, root, path)
        final_weights = receipt['final']['weights']
        final_cert = verify_scalar(final_weights)
        final_eval = numerical_evaluation(final_weights, seed)
        rows.append({'seed': seed, 'dose': 15, 'arm': arm, 'initial': initial_cert,
                     'final': final_cert, 'initial_eval': initial_eval, 'final_eval': final_eval,
                     'initial_scores': receipt['initial_scores'], 'final_scores': receipt['final']['scores'],
                     'hamming': receipt['hamming'], 'code_l1': receipt['code_l1'],
                     'weights_sha256': weights_digest(final_weights), 'search_seconds': receipt['seconds'],
                     **replay})
    return {'seed': seed, 'receipt_sha256': sha(initial_path), 'certificate': initial_cert,
            'eval': initial_eval, 'weights_sha256': weights_digest(initial['weights'])}, rows


def summarize(rows):
    summaries = []
    for arm in experiment.ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        n = len(selected)
        assert n == 24
        initial_cert = lambda r: r['initial']['verdict'] == 'CERTIFIED'
        final_cert = lambda r: r['final']['verdict'] == 'CERTIFIED'
        new_cert = lambda r: r['initial']['verdict'] == 'POINT_VIOLATION' and final_cert(r)
        proxy_pass = lambda r: r['final_scores']['bound'] <= 3 / 32
        summary = {'arm': arm, 'n_units': n, 'initial': sum(map(initial_cert, selected)),
                   'final': sum(map(final_cert, selected)), 'new': sum(map(new_cert, selected)),
                   'acquired_from_noncertified': sum(not initial_cert(r) and final_cert(r) for r in selected),
                   'acquired_from_unresolved': sum(r['initial']['verdict'] == 'UNRESOLVED' and final_cert(r) for r in selected),
                   'retained': sum(initial_cert(r) and final_cert(r) for r in selected),
                   'lost': sum(initial_cert(r) and not final_cert(r) for r in selected),
                   'initial_verdicts': dict(Counter(r['initial']['verdict'] for r in selected)),
                   'final_verdicts': dict(Counter(r['final']['verdict'] for r in selected)),
                   'new_seeds': [r['seed'] for r in selected if new_cert(r)],
                   'certificate_seeds': [r['seed'] for r in selected if final_cert(r)],
                   'numerical_bound_passes': sum(map(proxy_pass, selected)),
                   'proxy_pass_without_certificate': sum(proxy_pass(r) and not final_cert(r) for r in selected),
                   'certificate_with_proxy_above_target': sum(final_cert(r) and not proxy_pass(r) for r in selected),
                   'T8': sum(r['final_eval']['8']['accuracy'] for r in selected) / n,
                   'T128': sum(r['final_eval']['128']['accuracy'] for r in selected) / n,
                   'T128_all': sum(r['final_eval']['128']['all_prefix'] for r in selected) / n,
                   'hamming_mean': sum(r['hamming'] for r in selected) / n,
                   'accepted_edits': sum(r['accepted_edits'] for r in selected),
                   'accepted_updates': sum(r['accepted_updates'] for r in selected),
                   'profile_slots': sum(r['profile_slots'] for r in selected),
                   'fit_calls': sum(r['fit_calls'] for r in selected),
                   'fit_nfev': sum(r['fit_nfev'] for r in selected),
                   'search_seconds_sum': sum(r['search_seconds'] for r in selected)}
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    assert 1 <= args.workers <= 4
    started = time.monotonic()
    for line in (root / 'lock.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(root / name.strip()) == digest, name
    expected = {(seed, arm) for seed in experiment.SEEDS for arm in experiment.ARMS}
    assert len(expected) == 96
    assert {p.name for p in (root / 'initial').glob('*.json')} == {f'{seed}.json' for seed in experiment.SEEDS}
    assert {p.name for p in (root / 'results').glob('*.json')} == {f'{arm}-{seed}.json' for seed, arm in expected}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    control = verify_scalar(scalar_weights(4)[0])
    assert control['verdict'] == 'CERTIFIED'
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        checked = list(executor.map(audit_seed, [(seed, str(root)) for seed in experiment.SEEDS], chunksize=1))
    initials = [result[0] for result in checked]
    rows = [row for result in checked for row in result[1]]
    assert {(r['seed'], r['arm']) for r in rows} == expected
    certificates = {'experiment': 'LS-NEURON-14', 'semantics': 'Exact real GELU with stored FP32 dyadic weights and beta=1/5',
                    'protocol_sha256': sha(root / 'protocol.md'), 'source_lock_sha256': sha(root / 'lock.sha256'),
                    'positive_control': control, 'unique_initial_conditions': initials, 'rows': rows}
    certificate_path = root / 'certificates.json'
    certificate_path.write_text(json.dumps(certificates, indent=2) + '\n')
    results = {'verdict': 'PASS', 'experiment': 'LS-NEURON-14', 'n_units': 24, 'paired_rows': 96,
               'unique_initial_conditions': 24, 'exact_verifier_calls_including_control': 121,
               'initial_verdicts': dict(Counter(r['certificate']['verdict'] for r in initials)),
               'final_verdicts': dict(Counter(r['final']['verdict'] for r in rows)),
               'numerical_horizon_conditions': 240, 'stored_receipt_files': 120,
               'profile_slots_replayed': sum(r['profile_slots'] for r in rows),
               'fitted_candidate_states_checked': sum(r['fit_calls'] for r in rows),
               'accepted_edits': sum(r['accepted_edits'] for r in rows),
               'accepted_updates': sum(r['accepted_updates'] for r in rows),
               'positive_control': 'CERTIFIED', 'certificate_sha256': sha(certificate_path),
               'scope': 'Source and receipt hashes; complete profile ranking replay; independent stored-weight score checks; legal moves; deterministic shortlists and acceptance; fit budgets and losses; shared initial fit; final decode; 257-point diagnostic; float32 rollout vectors; unchanged rational invariant verifier. SciPy optimization trajectories were not retrained. Numerical diagnostics do not prove floating-kernel invariance.',
               'summary': summarize(rows), 'audit_seconds': time.monotonic() - started}
    (root / 'audit_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
