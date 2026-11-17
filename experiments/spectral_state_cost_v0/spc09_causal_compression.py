"""B: one-shot answer-state compression with matched-magnitude controls."""
from __future__ import annotations
import numpy as np
from experiments.spectral_state_cost_v0 import spc09_common as common


def retention_map(retain, suppress, condition_limit=1e6):
    """Column operator Q: Q retain.T=retain.T and Q suppress.T=0."""
    import torch
    W = torch.cat((retain, suppress)).T
    s = torch.linalg.svdvals(W)
    if len(s) != W.shape[1] or float(s[-1]) <= 0 or float(s[0] / s[-1]) > condition_limit:
        raise ValueError('overlapping or ill-conditioned retain/suppress subspaces')
    dual = torch.linalg.pinv(W)
    Q = retain.T @ dual[:len(retain)]
    if not torch.allclose(Q @ retain.T, retain.T, atol=1e-8) or not torch.allclose(Q @ suppress.T, torch.zeros_like(suppress.T), atol=1e-8):
        raise RuntimeError('oblique retention identities failed')
    return Q


def compression_maps(bases, rank, seed, immediate):
    import torch
    if rank > min(len(bases['P']), len(bases['D']), len(bases['PCA'])):
        raise ValueError('rank exceeds fitted dimension')
    P = bases['P'][:rank]; D = bases['D'][:rank]
    union = common.orthogonal_rows(torch.cat((P[:rank//2], D[:rank//2])), rank)
    random = common.random_rows(rank, seed)
    maps = {'baseline': torch.eye(48, dtype=torch.float64),
            'retain_union': union.T @ union, 'retain_pca': bases['PCA'][:rank].T @ bases['PCA'][:rank],
            'retain_random': random.T @ random, 'suppress_immediate_invisible': immediate.T @ immediate}
    invalid = {}
    for label, a, b in (('retain_P_suppress_D', P, D), ('retain_D_suppress_P', D, P)):
        try: maps[label] = retention_map(a, b)
        except ValueError as exc: invalid[label] = str(exc)
    return maps, invalid


def compression_displacement(answer, mean, Q):
    return (answer - mean) @ Q.T + mean - answer


def magnitude_control(delta, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(delta.shape, generator=g, dtype=delta.dtype).to(delta.device)
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-30) * delta.norm(dim=-1, keepdim=True)


def score_displacement(model, step, state, delta, hops, depth, follow, period, refs, registered):
    import torch
    with torch.no_grad():
        varied = state.clone(); varied[:, -1] += delta
        original_logits = model.readout(state[:, -1])
        current_logits = model.readout(varied[:, -1])
        current_correct = current_logits.argmax(-1) == hops[:, depth-1]
        for _ in range(follow): varied = step(varied)
        future_logits = model.readout(varied[:, -1])
        reference_logits = model.readout(refs[0][:, -1])
        dist = torch.stack([(varied - r).flatten(1).norm(dim=1) for r in refs], 1)
        phase = [int(dist[i, :p].argmin()) if valid else None for i, (p, valid) in enumerate(zip(period, registered))]
        return {'current_correct': current_correct.int().cpu().tolist(),
                'future_correct': (future_logits.argmax(-1) == hops[:, depth+follow-1]).int().cpu().tolist(),
                'future_shifted_correct': (future_logits.argmax(-1) == hops[:, depth+follow]).int().cpu().tolist(),
                'phase_zero': [int(p == 0) if p is not None else None for p in phase],
                'phase_plus_one': [int(p == 1) if p is not None else None for p in phase],
                'phase_offset': phase, 'registered': registered,
                'intervention_norm': delta.norm(dim=-1).cpu().tolist(),
                'immediate_logit_distortion': (current_logits-original_logits).norm(dim=-1).cpu().tolist(),
                'future_logit_distortion': (future_logits-reference_logits).norm(dim=-1).cpu().tolist()}


def evaluate(model, bases, tokens, hops, config, rank, depth, budget=None):
    import torch
    from experiments.spectral_state_cost_v0.spc05_phase_interventions import pointer_periods
    from experiments.spectral_state_cost_v0.spc06_readout_observability import logit_margin_jacobian
    if len(tokens) > config['batch_size']: raise ValueError('inference microbatch exceeded')
    device = next(model.parameters()).device
    tokens, hops = tokens.to(device), hops.to(device)
    step, state = common.state_at(model, tokens, depth)
    with torch.no_grad():
        trajectory = [state]; h = state
        for _ in range(config['future_horizon'] + 8):
            h = step(h); trajectory.append(h)
    follow = config['future_horizon']; refs = trajectory[follow:follow+8]
    period = pointer_periods(tokens.cpu()); registered = []
    for i, p in enumerate(period):
        scale = refs[0][i].norm().clamp_min(1e-30)
        closure = float((trajectory[follow+p][i] - refs[0][i]).norm() / scale)
        lag = float((refs[1][i] - refs[0][i]).norm() / scale)
        gap = min(float((trajectory[j][i] - state[i]).norm()) for j in range(1, p)) if p > 1 else 0
        registered.append(p > 1 and closure < 1e-4 and lag > .05 and gap > 1e-8)
    j = logit_margin_jacobian(model, state[:, -1]).detach().cpu()
    # Per-prompt immediate rows differ: all other maps are fixed train-only.
    maps, invalid = compression_maps(bases, rank, config['random_seed'], common.row_basis(j[0], 1e-10)[0])
    out = {'rank_budget': rank, 'depth': depth, 'invalid_arms': invalid,
           'prompts': common.fingerprints(tokens.cpu()), 'conditions': {}}
    with torch.no_grad():
        natural = trajectory[1][:, -1] - state[:, -1]
        for label, Qcpu in maps.items():
            if label == 'suppress_immediate_invisible':
                local = [common.row_basis(m, 1e-10)[0] for m in j]
                Q = torch.stack([b.T @ b for b in local]).to(device)
                v = state[:, -1] - bases['mean'].to(device)
                delta = torch.einsum('bij,bj->bi', Q, v) - v
                projected_update = torch.einsum('bij,bj->bi', Q, natural)
                ranks = [len(b) for b in local]
                opnorm = float(torch.linalg.matrix_norm(Q, 2).max())
            else:
                Q = Qcpu.to(device)
                delta = compression_displacement(state[:, -1], bases['mean'].to(device), Q)
                projected_update = natural @ Q.T
                ranks = [int(torch.linalg.matrix_rank(Q))] * len(tokens)
                opnorm = float(torch.linalg.matrix_norm(Q, 2))
            z = score_displacement(model, step, state, delta, hops, depth, follow, period, refs, registered)
            z.update({'actual_rank': ranks, 'operator_norm': opnorm,
                      'removed_state_energy': delta.square().sum(-1).cpu().tolist(),
                      'removed_update_energy': (natural-projected_update).square().sum(-1).cpu().tolist(),
                      'natural_update_energy': natural.square().sum(-1).cpu().tolist()})
            out['conditions'][label] = z
            if label != 'baseline':
                control = magnitude_control(delta, config['random_seed'] + rank * 101 + depth)
                out['conditions'][label + '_magnitude_control'] = score_displacement(
                    model, step, state, control, hops, depth, follow, period, refs, registered)
            if budget: budget.tick()
    return out


def summarize(rows, config):
    results = {}
    for rank in config['ranks']:
        for depth in config['depths']:
            selected = [r for r in rows if r['rank_budget'] == rank and r['depth'] == depth]
            names = sorted(set(k for r in selected for k in r['conditions']))
            for name in names:
                clusters, phase_clusters, correct, future = [], [], [], []
                # Keep every microbatch in its original prompt-seed cluster.
                by_cluster = {}
                for r in selected:
                    if name not in r['conditions']: continue
                    z = r['conditions'][name]; b = r['conditions']['baseline']
                    key = (r['model_seed'], r['prompt_seed'])
                    a = by_cluster.setdefault(key, [0, 0, 0, 0])
                    a[0] += sum(z['current_correct']) - sum(b['current_correct']); a[1] += len(z['current_correct'])
                    pairs = [(x, y) for x, y in zip(z['phase_zero'], b['phase_zero']) if x is not None and y is not None]
                    a[2] += sum(x-y for x, y in pairs); a[3] += len(pairs)
                    correct += z['current_correct']; future += z['future_correct']
                clusters = [(a[0], a[1]) for a in by_cluster.values()]
                phase_clusters = [(a[2], a[3]) for a in by_cluster.values()]
                ci = common.paired_interval(clusters, config['bootstrap_repeats'], config['random_seed'])
                pi = common.paired_interval(phase_clusters, config['bootstrap_repeats'], config['random_seed'])
                results[f'r{rank}_d{depth}_{name}'] = {
                    'current_accuracy': float(np.mean(correct)), 'future_accuracy': float(np.mean(future)),
                    'accuracy_difference': ci, 'phase_difference': pi,
                    'accuracy_preserved': ci['interval95'] is not None and ci['interval95'][0] >= -config['decisions']['accuracy_noninferiority_absolute'],
                    'phase_preserved': pi['interval95'] is not None and pi['interval95'][0] >= -config['decisions']['phase_noninferiority_absolute']}
    return results


def run(checkpoints, config, budget, args):
    """Evaluate every frozen compression arm/rank/depth without test selection."""
    import torch
    rows = []; device, fallback = common.device_for(config)
    for seed in config['model_seeds']:
        model = common.load_model(checkpoints, seed)
        bases = common.fit_bases(model, config, budget)
        model.to(device)
        for prompt_seed in config['eval_seeds']:
            tokens, hops = common.cohort('test', config['eval_items_per_seed'], prompt_seed,
                    max(config['depths']) + config['future_horizon'] + 9)
            if set(common.fingerprints(tokens)['graph_ids']) & set(bases['fit_graphs']):
                raise RuntimeError('graph leakage')
            for tok, labels in zip(tokens.split(config['batch_size']), hops.split(config['batch_size'])):
                for rank in config['ranks']:
                    for depth in config['depths']:
                        try:
                            row = evaluate(model, bases, tok, labels, config, rank, depth, budget)
                        except torch.cuda.OutOfMemoryError:
                            model.cpu(); torch.cuda.empty_cache(); device = 'cpu'; fallback = 'CUDA OOM; resumed on CPU'
                            row = evaluate(model, bases, tok, labels, config, rank, depth, budget)
                        row.update(model_seed=seed, prompt_seed=prompt_seed); rows.append(row)
                        common.save_json(args.out.with_name(args.out.stem + '_partial.json'), {'rows': rows})
    summary = summarize(rows, config)
    return {'device': device, 'fallback': fallback, 'rows': rows, 'summary': summary,
            'frozen_interventions_validated': bool(rows) and all(
                not r['invalid_arms'] and max(r['conditions']['baseline']['immediate_logit_distortion']) < 1e-9
                for r in rows)}


if __name__ == '__main__':
    raise SystemExit(common.stage_main('B', run))
