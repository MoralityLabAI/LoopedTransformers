"""C: sequential float64 JVP steering predictions checked against native replay."""
from __future__ import annotations
import numpy as np
from experiments.spectral_state_cost_v0 import spc09_common as common


def local_decomposition(O, relative=1e-3, absolute=1e-10):
    import torch
    width = O.shape[1]
    immediate, s0 = common.row_basis(O[:7], 1e-10, absolute)
    N = torch.eye(width, dtype=O.dtype) - immediate.T @ immediate
    future, sf = common.row_basis(O[7:] @ N, relative, absolute)
    combined = torch.cat((immediate, future))
    _, singular, vh = torch.linalg.svd(combined, full_matrices=True)
    rank = int((singular > max(absolute, float(singular[0]) * 1e-10)).sum()) if len(singular) else 0
    invisible = vh[rank:]
    return {'immediate': immediate, 'future_in_current_null': future,
            'horizon_approximately_invisible': invisible, 'null_projector': N,
            'singular_immediate': s0.tolist(), 'singular_deferred': sf.tolist(),
            'numerical_observation_rank': len(common.row_basis(O, relative, absolute)[0])}


def compare_prediction(predicted, measured, zero_tolerance=1e-10):
    p = np.asarray(predicted, dtype=float); m = np.asarray(measured, dtype=float)
    magnitude = float(np.linalg.norm(p)); valid = np.abs(p) > zero_tolerance
    return {'relative_error': float(np.linalg.norm(p-m) / magnitude) if magnitude > zero_tolerance else None,
            'absolute_error': float(np.linalg.norm(p-m)), 'predicted_norm': magnitude,
            'measured_norm': float(np.linalg.norm(m)),
            'sign_agreement': float(np.mean(np.sign(p[valid]) == np.sign(m[valid]))) if valid.any() else None,
            'nonzero_predicted_components': int(valid.sum()), 'zero_predicted_components': int((~valid).sum())}


def steer_prompt(model, token, depth, config, budget=None, return_bases=False):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    O, step, state = common.temporal_jacobian(model, token, depth, config['future_horizon'], budget)
    split = local_decomposition(O, config['relative_rank_tolerance'], config['absolute_rank_tolerance'])
    f = common.observation_function(model, step, state, config['future_horizon'])
    answer = state[0, -1]; scale = max(float(answer.norm()), 1e-12)
    rows = []
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        baseline = f(answer)
        for kind in ('immediate', 'future_in_current_null', 'horizon_approximately_invisible'):
            basis = split[kind]
            if not len(basis):
                rows.append({'direction': kind, 'status': 'empty numerical subspace'}); continue
            direction = basis[0]
            for radius in config['radius_grid']:
                for sign in (-1, 1):
                    delta = direction * (sign * radius * scale)
                    predicted = O @ delta
                    measured = f(answer + delta) - baseline
                    rows.append({'direction': kind, 'relative_radius': radius, 'sign': sign,
                        'intervention_norm': float(delta.norm()),
                        'predicted_margins': predicted.tolist(), 'measured_margin_changes': measured.tolist(),
                        'immediate': compare_prediction(predicted[:7], measured[:7]),
                        'future': compare_prediction(predicted[7:], measured[7:]), 'status': 'measured'})
                    if budget: budget.tick()
    result = {'depth': depth, 'prompt': common.fingerprints(token),
            'dimensions': {k: len(split[k]) for k in ('immediate', 'future_in_current_null', 'horizon_approximately_invisible')},
            'numerical_observation_rank': split['numerical_observation_rank'],
            'singular_immediate': split['singular_immediate'], 'singular_deferred': split['singular_deferred'],
            'rows': rows, 'exact_rank_claim': False}
    if return_bases:
        return result, {kind: split[kind] for kind in ('immediate','future_in_current_null','horizon_approximately_invisible')}
    return result


def run(checkpoints, config, budget, args):
    """Evaluate every depth and radius; do not choose a radius from held-out results."""
    results = []
    for seed in config['model_seeds']:
        model = common.load_model(checkpoints, seed)
        for prompt_seed in config['eval_seeds']:
            tokens, _ = common.cohort('test', config['steering_prompts_per_seed'], prompt_seed,
                    max(config['depths']) + config['future_horizon'] + 1)
            for token in tokens:
                previous = None
                for depth in config['depths']:
                    row, spaces = steer_prompt(model, token[None], depth, config, budget, return_bases=True)
                    if previous is not None:
                        import torch
                        previous_depth, prior = previous
                        changes = {}
                        for kind, basis in spaces.items():
                            before = prior[kind]
                            cosines = torch.linalg.svdvals(before @ basis.T).clamp(0,1)
                            changes[kind] = {'previous_rank':len(before),'rank':len(basis),
                                'principal_cosines':cosines.tolist(),
                                'projector_frobenius_change':float((before.T@before-basis.T@basis).norm())}
                        row['subspace_change_from_previous_depth'] = {'previous_depth':previous_depth,'changes':changes}
                    previous = depth, spaces
                    row.update(model_seed=seed, prompt_seed=prompt_seed); results.append(row)
                    common.save_json(args.out.with_name(args.out.stem + '_partial.json'), {'rows': results})
    summary = {}
    for radius in config['radius_grid']:
        for kind in ('immediate', 'future_in_current_null', 'horizon_approximately_invisible'):
            selected = [r['future'] for item in results for r in item['rows']
                        if r['status'] == 'measured' and r['direction'] == kind and r['relative_radius'] == radius]
            errors = [r['relative_error'] for r in selected if r['relative_error'] is not None]
            signs = [r['sign_agreement'] for r in selected if r['sign_agreement'] is not None]
            err = float(np.median(errors)) if errors else None
            sign = float(np.mean(signs)) if signs else None
            summary[f'{kind}_{radius}'] = {'median_relative_error': err, 'mean_sign_agreement': sign,
                'zero_prediction_cases': len(selected)-len(errors), 'n': len(selected),
                'prediction_criterion_pass': err is not None and sign is not None and
                    err <= config['decisions']['steering_relative_prediction_error'] and
                    sign >= config['decisions']['steering_sign_agreement']}
    return {'device': 'cpu', 'precision': 'float64', 'rows': results, 'summary': summary}


if __name__ == '__main__':
    raise SystemExit(common.stage_main('C', run))
