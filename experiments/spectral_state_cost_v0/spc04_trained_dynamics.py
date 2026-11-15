"""SPC-04: bounded trained vs same-initialization untrained LoopedChase dynamics.

Uses the repository's native LoopedChase and make_batch, not a surrogate.
No internet, pretrained weights, GPU, or hidden external datasets required.
The task is synthetic n-node pointer-chase, with a frozen 3-hop supervision
curriculum, matched held-out prompts, and an independent full-state recurrence.

All claims are limited to this miniature trained model. In particular, better
accuracy and Jacobian changes do not show general reasoning, interpretability,
or safety transfer to modern large models. Checkpoint provenance is recorded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (41522, 41523)
EVAL_SEEDS = (881001, 881002, 881003)
N_NODES = 8
D_MODEL = 48
N_LAYERS = 1
HEADS = 4
SUPERVISED_LOOPS = 3
TRAIN_STEPS = 1200
BATCH = 64
LR = 0.001
WEIGHT_DECAY = 0.005
JAC_DEPTHS = (3, 30)
RECURRENCE_LOOPS = 42
# Third-hop label is tested at trained depth and at longer, unseen horizons.
READOUT_DEPTHS = (1, 2, 3, 6, 12, 24, 42)


def weight_sha256(model: Any) -> str:
    """Deterministic hash of tensor values and named shapes, excluding ZIP metadata."""
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        val = value.detach().cpu().contiguous()
        h.update(key.encode('utf-8') + b'\x00')
        h.update(str(tuple(val.shape)).encode('ascii') + b'\x00')
        h.update(str(val.dtype).encode('ascii') + b'\x00')
        h.update(val.numpy().tobytes())
    return h.hexdigest()


def new_model(seed: int):
    import torch
    from research_gym.scripts.loopscope_toy import LoopedChase
    torch.manual_seed(seed)
    return LoopedChase(n_nodes=N_NODES, d=D_MODEL, layers=N_LAYERS, heads=HEADS)


def train_model(seed: int, steps: int = TRAIN_STEPS, batch: int = BATCH):
    import torch
    import torch.nn.functional as F
    from research_gym.scripts.loopscope_toy import make_batch
    if steps < 0 or batch < 1:
        raise ValueError('invalid training size')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    trained = new_model(seed)
    initialized = copy.deepcopy(trained).eval()
    init_sha = weight_sha256(trained)
    optim = torch.optim.AdamW(trained.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trained.train()
    last_loss = None
    for step in range(steps):
        tokens, hops = make_batch(N_NODES, batch, seed * 100003 + step, SUPERVISED_LOOPS)
        states = trained(tokens, SUPERVISED_LOOPS)
        logits = trained.readout(states)
        # One correct hop per loop. No future-hop label is provided in tokens.
        loss = F.cross_entropy(logits.reshape(-1, N_NODES), hops.T.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained.parameters(), 1.0)
        optim.step()
        last_loss = float(loss.detach())
    trained.eval()
    return trained, initialized, {
        'seed':seed, 'steps':steps, 'batch_size':batch,
        'train_last_loss':last_loss, 'initial_weights_sha256':init_sha,
        'trained_weights_sha256':weight_sha256(trained),
        'optimizer':'AdamW', 'lr':LR, 'weight_decay':WEIGHT_DECAY,
        'supervised_hops':[1,2,3],
    }


def evaluate_capability(model: Any, eval_items: int = 512,
                        eval_seeds: tuple[int,...] = EVAL_SEEDS) -> dict[str,Any]:
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if eval_items < 1 or not eval_seeds: raise ValueError('invalid eval settings')
    model.eval()
    result=[]
    with torch.no_grad():
        for seed in eval_seeds:
            tokens,hops=make_batch(N_NODES,eval_items,seed,RECURRENCE_LOOPS)
            states=model(tokens,RECURRENCE_LOOPS)
            selected=model.readout(states)
            per_hop=[]
            for hop in range(3):
                pred=selected[hop].argmax(-1)
                per_hop.append(float((pred==hops[:,hop]).float().mean()))
            late={str(d):float((selected[d-1].argmax(-1)==hops[:,2]).float().mean())
                    for d in READOUT_DEPTHS if d>=3}
            current={str(d):float((selected[d-1].argmax(-1)==hops[:,d-1]).float().mean())
                    for d in READOUT_DEPTHS}
            result.append({'eval_seed':seed,'n_items':eval_items,
                           'hop_accuracies_at_aligned_loops':per_hop,
                           'hop3_accuracy_at_loop':late,
                           'current_hop_accuracy_at_loop':current})
    return {
        'per_seed':result,
        'mean_aligned_hop_accuracies':[float(np.mean([v['hop_accuracies_at_aligned_loops'][k] for v in result])) for k in range(3)],
        'mean_hop3_accuracy_by_depth':{str(t):float(np.mean([v['hop3_accuracy_at_loop'][str(t)] for v in result])) for t in READOUT_DEPTHS if t>=3},
        'mean_current_hop_accuracy_by_depth':{str(t):float(np.mean([v['current_hop_accuracy_at_loop'][str(t)] for v in result])) for t in READOUT_DEPTHS},
        'chance_accuracy':1/N_NODES,
    }


def prompt_and_recurrence(model: Any, seed: int, loops: int = RECURRENCE_LOOPS):
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    tokens, hops = make_batch(N_NODES,1,seed,SUPERVISED_LOOPS)
    edges = tokens[:, :2 * N_NODES].reshape(1, N_NODES, 2)
    orig = torch.cat((model.tok(edges[...,0])+model.dst(edges[...,1]),
                      model.tok(tokens[:,2*N_NODES:])),dim=1)
    e = orig+model.pos(torch.arange(orig.shape[1]))
    def step(h):
        y = model.inject(h)+e
        for block in model.core:
            y = block(y)
        return y
    h=torch.zeros_like(e)
    seq=[h]
    with torch.no_grad():
        for _ in range(loops):
            h=step(h)
            seq.append(h)
        native=model(tokens,loops)
    gap=float((native-torch.stack([v[:,-1] for v in seq[1:]])).abs().max())
    return step,seq,tokens,hops,gap


def gramian_from_jvps(step, state, horizon: int, vectors, readout=None):
    """Discrete tangent energy for supplied directions; no arbitrary-rank claim."""
    import torch
    if horizon < 1:raise ValueError('horizon must be positive')
    rows=[]
    for direction in vectors:
        tangent=direction
        energy=0.0
        x=state
        for _ in range(horizon):
            # All-token state is propagated; answer-token norm is an observation.
            if readout is None:
                y=tangent[:,-1]
            else:
                y=readout(tangent)
            energy += float((y*y).sum())
            tangent=torch.autograd.functional.jvp(step,x,tangent,create_graph=False)[1]
            x=step(x).detach()
        rows.append(energy)
    return rows


def diagnostic_trajectory(model: Any, seed: int=982001,
                          jacobian: bool=True) -> dict[str,Any]:
    """Full-state native recurrence, optionally with frozen JVP/Arnoldi probes."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
    model=copy.deepcopy(model).eval().double()
    for param in model.parameters(): param.requires_grad_(False)
    with sdpa_kernel([SDPBackend.MATH]):
        step,seq,tokens,hops,gap=prompt_and_recurrence(model,seed)
        if gap>1e-11:raise AssertionError('native full-state reconstruction mismatch')
        states=np.stack([v[:,-1].detach().numpy()[0] for v in seq[1:]],axis=0)
        raw=float(np.sum(states**2))
        dc=float(np.sum(np.mean(states,axis=0)**2)*len(states)/max(raw,1e-30))
        delta=np.diff(states,axis=0)
        singular=np.linalg.svd(delta,full_matrices=False,compute_uv=False)
        squared=singular**2
        effective=float(np.sum(squared)**2 / max(np.sum(squared**2),1e-50))
        norms=np.linalg.norm(delta,axis=1)
        rr={
            'seed':seed, 'n_tokens':int(seq[-1].shape[1]),
            'full_state_dim':int(seq[-1].numel()),
            'max_native_forward_gap':gap,
            'temporal_dc_energy_fraction':dc,
            'delta_participation_rank':effective,
            'delta_norm_early':float(norms[0]),
            'delta_norm_depth_3':float(norms[2]),
            'delta_norm_depth_12':float(norms[11]),
            'delta_norm_depth_41':float(norms[40]),
            'late_over_early_delta_norm':float(norms[40] / max(norms[0],1e-30)),
        }
        if jacobian:
            eigrows=[]
            for depth in JAC_DEPTHS:
                s=seq[depth]
                dim=s.numel(); shape=tuple(s.shape)
                calls=[0]
                def mv(v):
                    calls[0]+=1
                    direction=torch.from_numpy(np.asarray(v).reshape(shape).copy()).double()
                    return torch.autograd.functional.jvp(step,s,direction,create_graph=False)[1].detach().numpy().reshape(-1)
                lin=LinearOperator((dim,dim),matvec=mv,dtype=np.float64)
                converged=True
                try:
                    values=eigs(lin,k=3,which='LM',tol=1e-5,ncv=16,maxiter=160,
                                v0=np.linspace(1,2,dim),return_eigenvectors=False)
                except ArpackNoConvergence as err:
                    converged=False
                    values=err.eigenvalues
                sorted_vals=sorted(values,key=abs,reverse=True)
                if not sorted_vals:
                    eigrows.append({'depth':depth,'converged':False,
                                    'arnoldi_calls':calls[0],'spectral_radius_estimate':None})
                    continue
                eigrows.append({'depth':depth,'converged':converged,
                                'arnoldi_calls':calls[0],
                                'spectral_radius_estimate':float(abs(sorted_vals[0])),
                                'ritz_values':[{'real':float(v.real),'imag':float(v.imag),'modulus':float(abs(v))}
                                               for v in sorted_vals]})
            rr['jacobian_ritz'] = eigrows
        return rr


def run_campaign(seeds: tuple[int,...] = SEEDS, steps: int = TRAIN_STEPS,
                 batch: int = BATCH, eval_items: int = 512,
                 with_jacobian: bool = True, checkpoint_dir: Path|None=None) -> dict[str,Any]:
    import torch
    if not seeds:raise ValueError('seed set must be nonempty')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    rows=[]
    for seed in seeds:
        trained,untrained,receipt=train_model(seed,steps,batch)
        tr_acc=evaluate_capability(trained,eval_items)
        un_acc=evaluate_capability(untrained,eval_items)
        trained_dyn=diagnostic_trajectory(trained,982001+seed-SEEDS[0],with_jacobian)
        untrained_dyn=diagnostic_trajectory(untrained,982001+seed-SEEDS[0],with_jacobian)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True,exist_ok=True)
            torch.save({'model':trained.state_dict(),'seed':seed,
                        'hyperparameters':{'n_nodes':N_NODES,'d_model':D_MODEL,'layers':N_LAYERS,
                         'heads':HEADS,'loops':SUPERVISED_LOOPS,'steps':steps,'batch_size':batch},
                        'weight_sha256':receipt['trained_weights_sha256']},
                       checkpoint_dir/f'spc04_seed{seed}.pt')
        rows.append({'train_receipt':receipt,
                     'trained':{'capability':tr_acc,'dynamics':trained_dyn},
                     'untrained_same_init':{'capability':un_acc,'dynamics':untrained_dyn}})
    result={'campaign':'SPC-04','model':'native LoopedChase',
        'baseline':'exact same initialized tensor values and held-out prompts',
        'task':'random-permutation three-hop pointer chase; 8 nodes',
        'seeds':list(seeds),'steps':steps,'batch_size':batch,
        'train_set_generation':'make_batch(seed * 100003 + step)',
        'heldout_eval_seeds':list(EVAL_SEEDS),'heldout_items_per_seed':eval_items,
        'jacobian_probe_depths':list(JAC_DEPTHS) if with_jacobian else [],
        'precision_for_dynamics':'float64; float32 training weights converted',
        'conclusion_scope':'synthetic bounded trained checkpoints; no larger-model transfer',
        'rows':rows}
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    p.add_argument('--steps',type=int,default=TRAIN_STEPS)
    p.add_argument('--batch',type=int,default=BATCH)
    p.add_argument('--eval-items',type=int,default=512)
    p.add_argument('--no-jacobian',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--checkpoint-dir',type=Path)
    args=p.parse_args()
    r=run_campaign(tuple(args.seeds),args.steps,args.batch,args.eval_items,
                   with_jacobian=not args.no_jacobian,checkpoint_dir=args.checkpoint_dir)
    s=json.dumps(r,sort_keys=True,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(s)
    print(s)

if __name__=='__main__':main()
