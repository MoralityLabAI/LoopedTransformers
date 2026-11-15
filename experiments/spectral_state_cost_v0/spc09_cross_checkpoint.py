"""D: bidirectional Procrustes transfer at identical ranks and intervention norms."""
from __future__ import annotations
import numpy as np
from experiments.spectral_state_cost_v0 import spc09_common as common


def procrustes(source, target):
    import torch
    if source.shape != target.shape or source.ndim != 2 or source.shape[0] < source.shape[1]:
        raise ValueError('matched trajectories with at least width rows required')
    u, _, vh = torch.linalg.svd(source.T @ target)
    return u @ vh


def alignment_data(model, tokens, config):
    import torch
    rows = []
    for chunk in tokens.split(config['batch_size']):
        with torch.no_grad():
            step, h = common.native_step(model, chunk)
            for depth in range(max(config['phase_depths']) + 1):
                nxt = step(h)
                if depth in config['phase_depths']: rows.append(nxt[:, -1] - h[:, -1])
                h = nxt
    return torch.cat(rows).detach()


def transfer_bases(source, target, Q, shuffled, rank, seed):
    random = common.random_rows(48, seed).T
    return {kind: {'raw': source[kind][:rank], 'paired': source[kind][:rank] @ Q,
                   'shuffled': source[kind][:rank] @ shuffled,
                   'target_fitted': target[kind][:rank], 'random_alignment': source[kind][:rank] @ random}
            for kind in ('P', 'D')}


def matched_update(natural, basis):
    projected = natural @ basis.T @ basis
    norm = projected.norm(dim=-1, keepdim=True)
    if bool((norm <= 1e-12).any()): raise ValueError('zero projected update; norm matching undefined')
    return projected / norm * natural.norm(dim=-1, keepdim=True)


def run(checkpoints, config, budget, args):
    """Fit only training trajectories; record every direction and failed transfer."""
    import torch
    from experiments.spectral_state_cost_v0.spc05_phase_interventions import pointer_periods
    from experiments.spectral_state_cost_v0.spc09_causal_compression import score_displacement
    models = {s: common.load_model(checkpoints, s) for s in config['model_seeds']}
    bases = {s: common.fit_bases(m, config, budget) for s, m in models.items()}
    tokens, _ = common.cohort('train', config['phase_prompts'], config['fit_seed'] + 31, max(config['phase_depths']) + 2)
    activations = {s: alignment_data(m, tokens, config) for s, m in models.items()}
    a, b = config['model_seeds'][:2]; A, B = activations[a], activations[b]
    Q = procrustes(A, B)
    g = torch.Generator().manual_seed(config['random_seed'])
    shuffled = procrustes(A, B[torch.randperm(len(B), generator=g)])
    rows = []
    for source, target in ((a, b), (b, a)):
        q = Q if source == a else Q.T
        nq = shuffled if source == a else shuffled.T
        model = models[target]
        for prompt_seed in config['eval_seeds']:
            ts, hops = common.cohort('test', config['eval_items_per_seed'], prompt_seed,
                         max(config['depths']) + config['future_horizon'] + 9)
            for tok, labels in zip(ts.split(config['batch_size']), hops.split(config['batch_size'])):
                for depth in config['depths']:
                    step, state = common.state_at(model, tok, depth)
                    with torch.no_grad():
                        h = state; trajectory = [h]
                        for _ in range(config['future_horizon'] + 8): h = step(h); trajectory.append(h)
                    natural = trajectory[1][:, -1] - state[:, -1]
                    follow = config['future_horizon']; refs = trajectory[follow:follow+8]
                    periods = pointer_periods(tok); registered = []
                    for i, p in enumerate(periods):
                        scale = refs[0][i].norm().clamp_min(1e-30)
                        gap = min(float((trajectory[j][i]-state[i]).norm()) for j in range(1,p)) if p > 1 else 0
                        registered.append(p > 1 and gap > 1e-8 and float((trajectory[follow+p][i]-refs[0][i]).norm()/scale) < 1e-4
                                          and float((refs[1][i]-refs[0][i]).norm()/scale) > .05)
                    for rank in config['ranks']:
                        conditions = {}; failures = {}
                        for kind, arms in transfer_bases(bases[source], bases[target], q, nq, rank, config['random_seed']).items():
                            for name, basis in arms.items():
                                label = kind + '_' + name
                                try: delta = matched_update(natural, basis)
                                except ValueError as exc: failures[label] = str(exc); continue
                                z = score_displacement(model, step, state, delta, labels, depth, follow, periods, refs, registered)
                                z['rank'] = len(basis)
                                z['energy_match_max_error'] = float((delta.norm(dim=-1)-natural.norm(dim=-1)).abs().max())
                                conditions[label] = z
                        row = {'source': source, 'target': target, 'prompt_seed': prompt_seed,
                            'depth': depth, 'rank': rank, 'prompts': common.fingerprints(tok),
                            'conditions': conditions, 'failures': failures}
                        rows.append(row); budget.tick()
                        common.save_json(args.out.with_name(args.out.stem + '_partial.json'), {'rows': rows})
    comparisons = {}
    for source, target in ((a, b), (b, a)):
        for depth in config['depths']:
            for rank in config['ranks']:
                for kind in ('P', 'D'):
                    for control in ('raw', 'shuffled', 'target_fitted', 'random_alignment'):
                        for metric in ('future_correct','future_shifted_correct','phase_plus_one'):
                            clusters = {}
                            for r in rows:
                                if (r['source'],r['target'],r['depth'],r['rank']) != (source,target,depth,rank): continue
                                chosen = r['conditions'].get(kind+'_paired'); other = r['conditions'].get(kind+'_'+control)
                                if chosen is None or other is None: continue
                                pairs = [(x,y) for x,y in zip(chosen[metric],other[metric]) if x is not None and y is not None]
                                delta = sum(x-y for x,y in pairs)
                                x = clusters.setdefault(r['prompt_seed'], [0,0]); x[0] += delta; x[1] += len(pairs)
                            comparisons[f'{source}_to_{target}_{kind}_r{rank}_d{depth}_paired_minus_{control}_{metric}'] = common.paired_interval(
                                list(clusters.values()), config['bootstrap_repeats'], config['random_seed'])
    asymmetry = {}
    for rank in config['ranks']:
        for depth in config['depths']:
            for kind in ('P','D'):
                key = f'{kind}_r{rank}_d{depth}_paired_minus_raw_phase_plus_one'
                x = comparisons[f'{a}_to_{b}_{key}']['difference']; y = comparisons[f'{b}_to_{a}_{key}']['difference']
                asymmetry[key] = x-y if x is not None and y is not None else None
    return {'alignment_training': common.fingerprints(tokens), 'rows': rows,
            'comparisons': comparisons, 'directional_asymmetry_of_alignment_gain': asymmetry,
            'coordinate_convention': 'row-vector source @ Q = target; reverse uses Q.T',
            'paired_fit_relative_error': float((A@Q-B).norm()/B.norm())}


if __name__ == '__main__':
    raise SystemExit(common.stage_main('D', run))
