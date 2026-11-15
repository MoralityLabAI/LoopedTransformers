"""Exact-current-output interventions for existing fixed-depth LoopedChase checkpoints.

Uses the repository's native Semantic Echo trace and checkpoint loader. An
edge-only edit leaves the cursor head unchanged. Future differences test the
sufficiency of that head on an explicitly intervened state set. Smoke mode is
engineering validation only and never produces a trained-model conclusion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

import torch

from research_gym.scripts.loopscope_semantic_echo import (
    load_model, native_trace, rollout,
)
from research_gym.scripts.loopscope_toy import make_batch

SCHEMA = "semantic_echo.output_preserving_edge_witness.v1"
DOSES = (0.25, 0.5, 1.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def probabilities(logits):
    require(bool(torch.isfinite(logits).all()), "nonfinite head output")
    return logits.double().softmax(dim=-1)


def tv(a, b):
    require(a.shape == b.shape, "probability shape mismatch")
    return (a - b).abs().sum(dim=-1) / 2


def check_close(a, b, *, rtol, atol, name):
    require(a.shape == b.shape, name + ": shape mismatch")
    require(bool(torch.isfinite(a).all() & torch.isfinite(b).all()), name + ": nonfinite")
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        raise RuntimeError(name + ": native/control replay mismatch")
    return float((a - b).abs().max())


def select_pairs(p, oracle, *, horizon, training_loops, tau, per_item, seed):
    """Select from endpoint data only; indices stored as completed loop counts.

    Caps apply independently per item and per depth stratum. No continuation
    outcomes, gradients, or intervention effects enter this function.
    """
    require(p.ndim == 3 and oracle.shape == (p.shape[1], p.shape[0]), "trace shapes")
    require(horizon >= 1 and p.shape[0] >= horizon + 2, "insufficient trace horizon")
    require(0 <= tau <= 1 and per_item >= 1, "invalid selection budget/threshold")
    require(bool(torch.isfinite(p).all()), "nonfinite probabilities")
    rng = random.Random(seed)
    prediction = p.argmax(-1)
    selected, coverage = [], []
    for item in range(p.shape[1]):
        buckets = defaultdict(list)
        count = dict(item=item, same_semantic_pairs=0, both_correct_pairs=0,
                     distribution_matched_pairs=0, selected_pairs=0)
        for t in range(1, p.shape[0] - horizon + 1):
            for u in range(t + 1, p.shape[0] - horizon + 1):
                target = int(oracle[item, t - 1])
                if target != int(oracle[item, u - 1]):
                    continue
                count['same_semantic_pairs'] += 1
                if int(prediction[t - 1, item]) != target or int(prediction[u - 1, item]) != target:
                    continue
                count['both_correct_pairs'] += 1
                gap = float(tv(p[t - 1, item], p[u - 1, item]))
                if gap > tau:
                    continue
                count['distribution_matched_pairs'] += 1
                stratum = 'training_depth' if u + horizon <= training_loops else 'depth_extrapolation'
                buckets[stratum].append(dict(item=item, t=t, u=u, target=target,
                                              endpoint_tv=gap, stratum=stratum))
        for stratum in sorted(buckets):
            candidates = buckets[stratum]
            rng.shuffle(candidates)
            chosen = sorted(candidates[:per_item], key=lambda pair: (pair['t'], pair['u']))
            selected.extend(chosen)
            count['selected_pairs'] += len(chosen)
        coverage.append(count)
    return selected, coverage


def edge_patch(base, donor, n_edges, dose, kind, seed):
    """Keep the cursor unchanged; match random/gauge raw L2 PER edge token."""
    require(base.shape == donor.shape and base.ndim == 3 and base.shape[0] == 1,
            "expected a single [1,tokens,width] state pair")
    require(0 < n_edges < base.shape[1] and 0 <= dose <= 1, "invalid edit scope/dose")
    delta = dose * (donor[:, :n_edges] - base[:, :n_edges])
    norms = delta.norm(dim=-1, keepdim=True)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    if kind == 'sham':
        delta = torch.zeros_like(delta)
    elif kind == 'edge_donor':
        pass
    elif kind in ('edge_random', 'edge_layernorm_gauge'):
        if kind == 'edge_random':
            direction = torch.randn(delta.shape, generator=generator, dtype=torch.float64)
            direction = direction.to(device=base.device, dtype=base.dtype)
            direction = direction - direction.mean(dim=-1, keepdim=True)
        else:
            signs = 2 * torch.randint(0, 2, (*delta.shape[:-1], 1), generator=generator) - 1
            direction = signs.to(device=base.device, dtype=base.dtype).expand_as(delta)
        delta = direction * norms / direction.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    else:
        raise ValueError('unknown patch kind: ' + kind)
    edited = base.clone()
    edited[:, :n_edges] += delta
    require(torch.equal(edited[:, n_edges:], base[:, n_edges:]), "edit changed a non-edge token")
    return edited


@torch.inference_mode()
def audit_pair(model, states, embedding, pair, oracle, *, horizon, random_controls,
               seed, rtol, atol, effect_tv):
    item, t, u = pair['item'], pair['t'], pair['u']
    require(1 <= t < u and u + horizon < len(states), "invalid pair/horizon")
    require(random_controls >= 1, "at least one random control required")
    base, donor = states[u, item:item+1], states[t, item:item+1]
    e = embedding[item:item+1]
    reference = rollout(model, base, e, horizon)
    donor_future = rollout(model, donor, e, horizon)
    controls = {
        'receiver_replay_max_abs': check_close(reference, states[u+1:u+horizon+1, item:item+1],
                                               rtol=rtol, atol=atol, name='receiver'),
        'full_donor_replay_max_abs': check_close(donor_future, states[t+1:t+horizon+1, item:item+1],
                                                 rtol=rtol, atol=atol, name='full-state donor'),
    }
    current_logits = model.readout(base[:, -1])
    ref_logits = model.readout(reference[:, 0, -1])
    ref_prob = probabilities(ref_logits)
    target = oracle[item, u:u+horizon].to(base.device)
    require(target.shape == (horizon,), 'missing oracle continuation')
    ref_labels = ref_logits.argmax(-1)
    full_labels = model.readout(states[u+1:u+horizon+1, item, -1]).argmax(-1)
    require(torch.equal(ref_labels, full_labels), 'receiver label replay failed')
    specifications = [('sham', 0.0, 0)]
    for dose in DOSES:
        specifications.extend([('edge_donor', dose, 0), ('edge_layernorm_gauge', dose, 0)])
        specifications.extend(('edge_random', dose, k) for k in range(random_controls))
    outcomes = []
    for kind, dose, replicate in specifications:
        edited = edge_patch(base, donor, model.n_nodes, dose, kind,
                            seed + replicate + (10000 if kind == 'edge_layernorm_gauge' else 0))
        edited_logits = model.readout(edited[:, -1])
        # Exactly identical head arguments in the same batch shape. No tolerance gate.
        require(torch.equal(current_logits, edited_logits), 'current logits changed')
        future = rollout(model, edited, e, horizon)
        if kind in ('sham', 'edge_layernorm_gauge'):
            err = check_close(future, reference, rtol=rtol, atol=atol, name=kind)
        else:
            err = None
        logits = model.readout(future[:, 0, -1])
        prob = probabilities(logits)
        labels = logits.argmax(-1)
        if kind in ('sham', 'edge_layernorm_gauge'):
            require(torch.equal(labels, ref_labels), kind + ': future decision changed')
        delta = edited - base
        distance = tv(prob, ref_prob)
        outcomes.append(dict(
            kind=kind, dose=dose, replicate=replicate, current_logits_bitwise_equal=True,
            raw_edge_l2_per_token=delta[:, :model.n_nodes].norm(dim=-1).flatten().tolist(),
            injected_state_delta_l2=float((model.inject(edited)-model.inject(base)).norm()),
            base_edge_l2=float(base[:, :model.n_nodes].norm()),
            applied_l2=float(delta.norm()),
            future_tv=distance.tolist(), future_logits=logits.cpu().tolist(),
            future_labels=labels.tolist(), future_oracle_correct=(labels == target).tolist(),
            future_decision_changed=(labels != ref_labels).tolist(),
            numerical_control_max_abs=err,
            exceeds_effect_tv=bool((distance >= effect_tv).any()),
        ))
    return dict(**pair, controls=controls,
                original_endpoint_logits=model.readout(torch.cat((donor[:, -1], base[:, -1]))).cpu().tolist(),
                natural_future_tv=tv(probabilities(model.readout(donor_future[:, 0, -1])),ref_prob).tolist(),
                oracle_future=target.tolist(), receiver_future_labels=ref_labels.tolist(),
                receiver_future_logits=ref_logits.cpu().tolist(),
                receiver_future_oracle_correct=(ref_labels == target).tolist(), interventions=outcomes)


def summarize(rows, seed):
    """Checkpoint-local, equal-weight item means. Bootstrap independent inputs."""
    result = {}
    for stratum in ('training_depth', 'depth_extrapolation'):
        by_item = defaultdict(list)
        for row in rows:
            if row['stratum'] != stratum:
                continue
            donor = next(x for x in row['interventions'] if x['kind']=='edge_donor' and x['dose']==1.0)
            nulls = [x for x in row['interventions'] if x['kind']=='edge_random' and x['dose']==1.0]
            effect = sum(donor['future_tv'])/len(donor['future_tv'])
            null = sum(sum(x['future_tv'])/len(x['future_tv']) for x in nulls)/len(nulls)
            by_item[row['item']].append((effect, effect-null, float(any(donor['future_decision_changed']))))
        means = [[sum(a[k] for a in values)/len(values) for k in range(3)] for values in by_item.values()]
        if not means:
            result[stratum] = dict(items=0, pairs=0, mean_future_tv=None,
                                    donor_minus_random_tv=None, item_bootstrap_95ci=None)
            continue
        mean = [sum(x[k] for x in means)/len(means) for k in range(3)]
        interval = None
        if len(means) >= 2:
            rng = random.Random(seed)
            bs = sorted(sum(rng.choice(means)[1] for _ in means)/len(means) for _ in range(1000))
            interval = [bs[24], bs[974]]
        result[stratum] = dict(items=len(means), pairs=sum(map(len, by_item.values())),
            mean_future_tv=mean[0], donor_minus_random_tv=mean[1],
            pair_flip_rate_equal_item_weight=mean[2], item_bootstrap_95ci=interval,
            inference_scope='conditional on endpoint-eligible independent input items; one checkpoint')
    return result


def resolve_checkpoints(checkpoints=None, directories=None, smoke=False):
    if smoke:
        require(not checkpoints and not directories, 'smoke must be separate from checkpoints')
        return [None]
    found = [Path(x) for x in (checkpoints or [])]
    for name in directories or []:
        directory = Path(name)
        require(directory.is_dir(), 'checkpoint directory missing: '+str(directory))
        matches = sorted(directory.glob('*.final.pt'))
        require(bool(matches), 'no *.final.pt checkpoints in '+str(directory))
        found.extend(matches)
    require(bool(found), 'supply checkpoints or explicitly select --smoke')
    for path in found:
        require(path.is_file(), 'checkpoint missing: '+str(path))
    return list(dict.fromkeys(path.resolve() for path in found))


def evaluate(checkpoint, args):
    model, cfg = load_model(checkpoint, smoke=args.smoke, seed=args.seed,
                            device=args.device, dtype=args.dtype)
    model.eval()
    training_loops = cfg.get('loops')
    require(type(training_loops) is int and training_loops >= 1, 'checkpoint needs cfg.loops')
    tokens, oracle = make_batch(cfg['n_nodes'], args.items, args.seed, args.max_loops, depth=None)
    tokens = tokens.to(args.device)
    rtol, atol = (2e-5, 2e-6) if args.dtype=='float32' else (1e-10, 1e-12)
    with torch.inference_mode():
        embedding, states = native_trace(model, tokens, args.max_loops)
        native = model(tokens, args.max_loops)
        parity = check_close(states[1:, :, -1], native, rtol=rtol, atol=atol, name='native forward')
        logits = model.readout(states[1:, :, -1])
        require(torch.equal(logits.argmax(-1), model.readout(native).argmax(-1)), 'native head parity')
        p = probabilities(logits).cpu()
        pairs, coverage = select_pairs(p, oracle, horizon=args.horizon, training_loops=training_loops,
                                       tau=args.endpoint_tv, per_item=args.pairs_per_item, seed=args.seed+1)
        selected_hash = json_sha(pairs)  # fixed before any intervention replay
        rows = [audit_pair(model,states,embedding,pair,oracle,horizon=args.horizon,
                    random_controls=args.random_controls,seed=args.seed+100+i,
                    rtol=rtol,atol=atol,effect_tv=args.effect_tv) for i,pair in enumerate(pairs)]
        engineering = None
        if args.smoke:
            engineering = audit_pair(model,states,embedding,
                dict(item=0,t=1,u=2,stratum='UNTRAINED_ENGINEERING_CONTROL',endpoint_gate=False),oracle,
                horizon=args.horizon,random_controls=args.random_controls,seed=args.seed,
                rtol=rtol,atol=atol,effect_tv=args.effect_tv)
    import research_gym.scripts.loopscope_toy as source_model
    import research_gym.scripts.loopscope_semantic_echo as source_adapter
    sources = {Path(p).name:file_sha(p) for p in (__file__,source_model.__file__,source_adapter.__file__)}
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,
                                           text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):
        revision = None
    return dict(schema=SCHEMA,
        status='UNTRAINED_SMOKE' if args.smoke else ('CHECKPOINT_EVALUATION' if pairs else 'NO_ELIGIBLE_PAIRS'),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=None if checkpoint is None else str(checkpoint),
        checkpoint_sha256=None if checkpoint is None else file_sha(checkpoint),checkpoint_cfg=cfg,
        configuration={k:v for k,v in vars(args).items() if k not in ('checkpoint','checkpoint_dir','out')},
        provenance=dict(code_sha256=sources,git_commit=revision,torch=torch.__version__,
                        python=platform.python_version(),device=args.device,threads=torch.get_num_threads()),
        inputs=tokens.cpu().tolist(), oracle=oracle.tolist(), inputs_sha256=json_sha(tokens.cpu().tolist()),
        current_head_accuracy_by_loop=(p.argmax(-1)==oracle.T).double().mean(dim=1).tolist(),
        native_forward_max_abs=parity,selected_pairs_sha256=selected_hash,coverage=coverage,
        pairs=rows,summary=None if args.smoke else summarize(rows,args.seed+2),
        engineering_control=engineering,
        scope='same-input edge interventions; hybrids may be off-manifold; no consciousness claim')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', nargs='+')
    group.add_argument('--checkpoint-dir', nargs='+')
    group.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', choices=('float32','float64'), default='float32')
    ap.add_argument('--items', type=int, default=64)
    ap.add_argument('--max-loops', type=int, default=32)
    ap.add_argument('--horizon', type=int, default=4)
    ap.add_argument('--pairs-per-item', type=int, default=2)
    ap.add_argument('--random-controls', type=int, default=8)
    ap.add_argument('--endpoint-tv', type=float, default=0.01)
    ap.add_argument('--effect-tv', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=940217)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args(argv)
    require(args.items>0 and args.threads>0 and args.random_controls>=1, 'invalid run size')
    require(args.horizon>=1 and args.max_loops>=args.horizon+2, 'invalid continuation horizon')
    require(0<=args.endpoint_tv<=1 and 0<args.effect_tv<=1, 'invalid TV threshold')
    files = resolve_checkpoints(args.checkpoint,args.checkpoint_dir,args.smoke)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = dict(schema=SCHEMA,complete=False,runs=[])
    manifest_path = args.out/'manifest.json'
    require(not manifest_path.exists(), 'output already exists; use a fresh run directory')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    for index, path in enumerate(files):
        result = evaluate(path,args)
        dest = args.out/f'run_{index:03d}.json'
        dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        manifest['runs'].append(dict(file=dest.name,sha256=file_sha(dest),status=result['status']))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(file=str(dest),status=result['status'],selected_pairs=len(result['pairs']))),flush=True)
    manifest['complete']=True
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
