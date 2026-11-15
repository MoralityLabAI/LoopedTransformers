"""E: gated, bounded continuation with frozen phase/deferred teacher losses."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from experiments.spectral_state_cost_v0 import spc09_common as common


def require_validation(directory, config_sha, current_sources, checkpoints):
    if directory is None: raise ValueError('A-D validation receipts are required before continuation')
    for stage in ('A', 'B', 'C', 'D'):
        path = Path(directory) / f'spc09_{stage}.json'
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('stage') != stage or receipt.get('status') != 'complete' or receipt.get('config_sha256') != config_sha:
            raise ValueError(f'{stage} is incomplete or uses a different frozen configuration')
        if receipt.get('source_sha256') != current_sources or receipt.get('checkpoints') != checkpoints:
            raise ValueError(f'{stage} source/checkpoint provenance differs')
        if stage == 'A' and not receipt['result'].get('reproduction_pass'):
            raise ValueError('SPC-08 reproduction has divergences')
        if stage == 'B' and not receipt['result'].get('frozen_interventions_validated'):
            raise ValueError('frozen compression interventions did not validate')


def auxiliary_losses(student, teacher, phase_basis, deferred_basis):
    import torch
    if student.shape != teacher.shape: raise ValueError('teacher/student shape mismatch')
    zero = torch.zeros_like(student[:1])
    ds = student - torch.cat((zero, student[:-1]))
    dt = teacher - torch.cat((zero, teacher[:-1]))
    p_target = dt @ phase_basis.T
    d_target = teacher @ deferred_basis.T
    phase = ((ds-dt) @ phase_basis.T).square().mean() / (p_target.square().mean().detach() + 1e-8)
    deferred = ((student-teacher) @ deferred_basis.T).square().mean() / (d_target.square().mean().detach() + 1e-8)
    return phase, deferred


def capability(model, config):
    import torch
    device = next(model.parameters()).device; rows = []
    model.eval()
    with torch.no_grad():
        for seed in config['eval_seeds']:
            tokens, hops = common.cohort('test', config['eval_items_per_seed'], seed, max(config['depths']))
            counts = {str(d): 0 for d in config['depths']}
            bits = {str(d): [] for d in config['depths']}
            for tok, labels in zip(tokens.split(config['batch_size']), hops.split(config['batch_size'])):
                states = model(tok.to(device), max(config['depths']))
                for depth in config['depths']:
                    correct = (model.readout(states[depth-1]).argmax(-1).cpu() == labels[:, depth-1]).int().tolist()
                    counts[str(depth)] += sum(correct); bits[str(depth)] += correct
            rows.append({'prompt_seed': seed, 'n': len(tokens), 'correct': counts,
                         'correct_bits': bits, 'prompts': common.fingerprints(tokens)})
    return rows


def compressed_snapshot(model, path, metadata):
    from experiments.spectral_state_cost_v0.spc04_trained_dynamics import weight_sha256
    arrays = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    row = dict(metadata, filename=path.name, file_sha256=common.sha256(path), weight_sha256=weight_sha256(model))
    common.save_json(path.with_suffix('.json'), row)
    return row


def run(checkpoints, config, budget, args):
    """Continue only after source/config/checkpoint-bound A-D validation receipts."""
    import torch
    import torch.nn.functional as F
    sources = {p.name: common.sha256(p) for p in sorted(common.ROOT.glob('spc09_*.py'))}
    _, verified = common.verify_checkpoints(checkpoints, config['model_seeds'])
    require_validation(args.validated_receipts, common.sha256(args.config), sources, verified)
    device, fallback = common.device_for(config); output = []
    for seed in config['model_seeds']:
        frozen = common.load_model(checkpoints, seed)
        bases = common.fit_bases(frozen, config, budget)
        teacher = frozen.float().to(device)
        P = bases['P'][:8].float().to(device); D = bases['D'][:8].float().to(device)
        for arm, use_phase, use_deferred in (('ce',False,False), ('ce_phase',True,False),
                                              ('ce_deferred',False,True), ('ce_both',True,True)):
            model = copy.deepcopy(teacher)
            for parameter in model.parameters(): parameter.requires_grad_(True)
            optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            snapshots = [{'update': 0, 'heldout': capability(model, config)}]; training = []
            for update in range(1, config['continuation_updates'] + 1):
                tokens, hops = common.cohort('train', config['batch_size'],
                    config['fit_seed'] + 100000 + update * 8191, config['continuation_depth'])
                tokens, hops = tokens.to(device), hops.to(device)
                with torch.no_grad(): reference = teacher(tokens, config['continuation_depth'])
                model.train(); states = model(tokens, config['continuation_depth'])
                ce = F.cross_entropy(model.readout(states).reshape(-1, 8), hops.T.reshape(-1))
                phase, deferred = auxiliary_losses(states, reference, P, D)
                loss = ce + config['phase_loss_weight'] * phase * use_phase + config['deferred_loss_weight'] * deferred * use_deferred
                optim.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
                training.append({'update': update, 'ce': float(ce.detach()), 'phase_loss': float(phase.detach()),
                                 'deferred_loss': float(deferred.detach()), 'loss': float(loss.detach())})
                if update % config['checkpoint_every'] == 0 or update == config['continuation_updates']:
                    filename = args.out.parent / 'checkpoints' / f'spc09_seed{seed}_{arm}_u{update}.npz'
                    ckpt = compressed_snapshot(model, filename, {'seed': seed, 'arm': arm, 'update': update})
                    snapshots.append({'update': update, 'checkpoint': ckpt, 'heldout': capability(model, config)})
                budget.tick()
                common.save_json(args.out.with_name(args.out.stem + '_partial.json'), {
                    'completed_arms': output, 'active_seed': seed, 'active_arm': arm,
                    'training': training, 'snapshots': snapshots})
            output.append({'seed': seed, 'arm': arm, 'training': training, 'snapshots': snapshots})
            del model, optim
    comparisons = {}
    for arm in ('ce_phase','ce_deferred','ce_both'):
        for step in sorted({s['update'] for row in output for s in row['snapshots']}):
            for depth in config['depths']:
                clusters = []
                for seed in config['model_seeds']:
                    candidate = next(r for r in output if r['seed']==seed and r['arm']==arm)
                    baseline = next(r for r in output if r['seed']==seed and r['arm']=='ce')
                    x = next(s for s in candidate['snapshots'] if s['update']==step)['heldout']
                    y = next(s for s in baseline['snapshots'] if s['update']==step)['heldout']
                    for a,b in zip(x,y): clusters.append((a['correct'][str(depth)]-b['correct'][str(depth)], a['n']))
                comparisons[f'{arm}_u{step}_d{depth}_minus_ce'] = common.paired_interval(
                    clusters, config['bootstrap_repeats'], config['random_seed'])
    return {'device': device, 'fallback': fallback, 'arms': output, 'comparisons': comparisons,
            'interpretation': 'provisional two-checkpoint continuation; no independent third-seed replication'}


if __name__ == '__main__':
    raise SystemExit(common.stage_main('E', run))
