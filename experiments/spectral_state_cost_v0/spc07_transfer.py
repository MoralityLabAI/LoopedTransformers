"""SPC-07: frozen cross-checkpoint transfer of temporal observability directions.

Reproduces the SPC-04 trained 8-node LoopedChase checkpoints. Fits per-model
finite-horizon *local* output-margin Gramians on a preregistered train-only
prompt cohort. Applies source eigenvectors without matching/rotating neuron
coordinates to the other model's future-phase interventions, with target-fit,
instantaneous, phase-SVD, and norm-matched random controls.

The model's hidden coordinate systems are independently learned. Direct
cross-checkpoint transfer is therefore explicitly a basis-dependent test;
negative transfer does not prove absence of an abstract shared computation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES, D_MODEL
from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    SEEDS, FOLLOW_LOOPS, MAX_PERIOD, PHASE_DEPTH, model_from_receipt,
    pointer_periods, prepare_step, _norm_batch,
)
from experiments.spectral_state_cost_v0.spc05_phase_subspace import (
    fit_phase_svd, project_increment, random_basis,
)
from experiments.spectral_state_cost_v0.spc06_readout_observability import (
    logit_margin_jacobian,
)

TRAIN_PROMPT_SEED = 920731
TRAIN_PROMPTS = 16
TRAIN_DEPTH = 32
FUTURE_HORIZON = 8
FD_EPS = 2e-4
EVAL_PROMPT_SEEDS = (910701, 910702, 910703, 910704, 910705, 910706, 910707, 910708)
EVAL_ITEMS = 64
RANKS = (8,)
RANDOM_REPEATS = 8
ALIGN_SEED = 904001
ALIGN_PROMPTS = 256
ALIGN_DEPTHS = (32, 48, 64)
SHUFFLE_SEED = 20261115


def token_hashes(tokens) -> set[str]:
    """Exact prompt partition fingerprint, stable against generated-label changes."""
    return {hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for x in tokens}


def _margin_outputs(model, z):
    logits = model.readout(z[:, -1, :])
    return logits[:, :-1] - logits[:, -1:]


def finite_temporal_jacobian(model, tokens, depth=TRAIN_DEPTH,
                             horizon=FUTURE_HORIZON, eps=FD_EPS):
    """The stacked (h+1)*7 by 48 central-difference future-output Jacobian.

    `tokens` contains exactly one prompt, and the 96 interventions are kept
    in a single batch. No parameter-gradient or native-model retraining.
    """
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if tuple(tokens.shape) != (1, 2*N_NODES+1):
        raise ValueError('one exact no-depth-token pointer prompt required')
    if depth < 3 or horizon < 1 or not (1e-6 < eps < 1e-2):
        raise ValueError('invalid derivative configuration')
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        step, x = prepare_step(model, tokens)
        for _ in range(depth):
            x = step(x)
        offsets = torch.cat((torch.eye(D_MODEL, dtype=x.dtype),
                             -torch.eye(D_MODEL, dtype=x.dtype)),dim=0) * eps
        varied = x.repeat(2*D_MODEL, 1, 1)
        varied[:,-1,:] += offsets
        advance, _ = prepare_step(model, tokens.repeat(2*D_MODEL, 1))
        jac = []
        for t in range(horizon+1):
            out = _margin_outputs(model, varied)
            jac.append(((out[:D_MODEL] - out[D_MODEL:])/(2*eps)).T.clone())
            if t < horizon:
                varied = advance(varied)
        O = torch.cat(jac, dim=0)
        if O.shape != ((horizon+1)*(N_NODES-1), D_MODEL) or not bool(torch.isfinite(O).all()):
            raise RuntimeError('temporal measurement Jacobian invalid')
        return O, x


def fit_temporal_basis(model, prompt_seed=TRAIN_PROMPT_SEED,
                       n_prompts=TRAIN_PROMPTS, depth=TRAIN_DEPTH,
                       horizon=FUTURE_HORIZON, eps=FD_EPS):
    """Uncentered finite-horizon observability Gramian averaged over prompts.

    Each prompt Jacobian is Frobenius-normalized before summation to give
    equal weight to distinct training prompts. This is preregistered.
    """
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if n_prompts < 2 or n_prompts > 128:
        raise ValueError('bounded prompt cohort required')
    model = model.eval().double()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokens, _ = make_batch(N_NODES, n_prompts, prompt_seed, depth+horizon+1)
    G = torch.zeros((D_MODEL,D_MODEL),dtype=torch.float64)
    checks=[]
    for i in range(n_prompts):
        O, x = finite_temporal_jacobian(model,tokens[i:i+1],depth,horizon,eps)
        sq_norm=float(O.square().sum())
        if sq_norm < 1e-20:
            raise RuntimeError('zero observability matrix at training prompt')
        G += (O.T @ O) / sq_norm
        if i < 2:
            other,_=finite_temporal_jacobian(model,tokens[i:i+1],depth,horizon,eps/2)
            relative=float(torch.linalg.vector_norm(O-other)/torch.linalg.vector_norm(other))
            instant=logit_margin_jacobian(model,x[:,-1,:])[0]
            instant_rel=float(torch.linalg.vector_norm(O[:N_NODES-1]-instant)/torch.linalg.vector_norm(instant))
            checks.append({'prompt_index':i, 'two_step_fd_relative_difference':relative,
                           'instant_analytic_vs_fd_relative_difference':instant_rel})
    G /= n_prompts
    val, V = torch.linalg.eigh(G)
    val = val.flip(0).clamp_min(0)
    basis = V.flip(1).T.contiguous()
    if not bool(torch.allclose(basis@basis.T,torch.eye(D_MODEL,dtype=torch.float64),atol=1e-8)):
        raise RuntimeError('eigenvectors lost orthonormality')
    return basis, G, token_hashes(tokens), {
        'prompt_seed':prompt_seed,'n_prompts':n_prompts,'depth':depth,
        'horizon':horizon,'eps':eps,'method':'equal-prompt-weighted normalized finite-horizon Gramian',
        'eigenvalues':val.cpu().tolist(),
        'energy_by_rank':{str(r):float(val[:r].sum()/val.sum()) for r in RANKS},
        'fd_checks':checks,
        'input_prompt_hashes_sha256':hashlib.sha256(''.join(sorted(token_hashes(tokens))).encode()).hexdigest(),
    }


def projector_alignment(U, V, rank: int) -> dict:
    """Gauge-sensitive overlap of two Euclidean rank-r subspaces."""
    import torch
    A=U[:rank];B=V[:rank]
    singular=torch.linalg.svdvals(A@B.T).clamp(0,1)
    return {'rank':rank,'overlap_trace_fraction':float(singular.square().mean()),
            'max_principal_angle_deg':float(torch.rad2deg(torch.acos(singular.min()))),
            'min_cosine':float(singular.min()),
            'canonical_cosines':singular.detach().cpu().tolist()}


def gramian_capture(basis, target_gramian, rank):
    import torch
    U=basis[:rank]
    return float(torch.trace(U@target_gramian@U.T)/torch.trace(target_gramian))



def paired_phase_differences(model, tokens, depths=ALIGN_DEPTHS):
    """Answer-token phase differences on precisely paired unlabeled inputs."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if not depths or min(depths)<3 or tokens.ndim!=2:
        raise ValueError('invalid paired phase calibration')
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        advance, h = prepare_step(model, tokens)
        out=[]
        for t in range(max(depths)+1):
            future = advance(h)
            if t in depths:
                out.append((future[:,-1,:]-h[:,-1,:]).clone())
            h=future
    return torch.cat(out, dim=0)


def orthogonal_procrustes(source, target):
    """Row-vector convention: minimize ||source @ Q - target||_F, Q.T Q=I.

    The optimal Q is U @ V.T for source.T @ target = U S V.T.
    The trace-norm expression is the exact linear-algebra certificate.
    """
    import torch
    if source.shape!=target.shape or source.ndim!=2 or source.shape[0]<source.shape[1]:
        raise ValueError('paired full-column-rank calibration shape required')
    U,S,Vh = torch.linalg.svd(source.T @ target)
    Q=U@Vh
    if not bool(torch.allclose(Q.T@Q,torch.eye(Q.shape[0],dtype=Q.dtype),atol=1e-9)):
        raise RuntimeError('orthogonal calibration failed')
    score=float(torch.linalg.vector_norm(source@Q-target)/torch.linalg.vector_norm(target).clamp_min(1e-30))
    return Q, score


def train_alignment(source_model, target_model, prompt_seed=ALIGN_SEED,
                    n_prompts=ALIGN_PROMPTS, depths=ALIGN_DEPTHS,
                    shuffle_seed=SHUFFLE_SEED):
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if n_prompts < D_MODEL or not depths:
        raise ValueError('at least 48 matched prompts required')
    tokens,_=make_batch(N_NODES,n_prompts,prompt_seed,max(depths)+2)
    A=paired_phase_differences(source_model,tokens,depths)
    B=paired_phase_differences(target_model,tokens,depths)
    Q,err=orthogonal_procrustes(A,B)
    g=torch.Generator(device='cpu').manual_seed(shuffle_seed)
    shuffled=B[torch.randperm(len(B),generator=g)]
    Qneg,_=orthogonal_procrustes(A,shuffled)
    errneg=float(torch.linalg.vector_norm(A@Qneg-B)/torch.linalg.vector_norm(B).clamp_min(1e-30))
    calibration={
        'prompt_seed':prompt_seed,'prompts':n_prompts,'sample_depths':list(depths),
        'paired_rows':int(A.shape[0]),'width':int(A.shape[1]),
        'pairing':'identical prompt and depth positions; no answers used',
        'relative_mse_sqrt_before_alignment':float(torch.linalg.vector_norm(A-B)/torch.linalg.vector_norm(B)),
        'paired_procrustes_relative_error':err,
        'shuffled_procrustes_error_evaluated_on_true_pairs':errneg,
        'shuffle_seed':shuffle_seed,
        'prompt_hashes_sha256':hashlib.sha256(''.join(sorted(token_hashes(tokens))).encode()).hexdigest(),
    }
    return Q,Qneg,token_hashes(tokens),calibration


def evaluate_phase_batch(model, bases:dict[str, object], target_seed:int, prompt_seed:int,
                         batch:int=EVAL_ITEMS, depth:int=PHASE_DEPTH,
                         follow:int=FOLLOW_LOOPS, ranks:Sequence[int]=RANKS,
                         random_repeats:int=RANDOM_REPEATS):
    """Paired target-model intervention on heldout prompts only.

    With one target natural phase-step delta, compare source/target O,
    phase-difference SVD and random projection directions in both raw and
    full-delta-norm-matched variants. Reuses independent target weights and
    target prompt, with no source trace pasted into target hidden states.
    """
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if batch<1 or not bases or max(ranks)>D_MODEL or random_repeats<1:
        raise ValueError('invalid heldout configuration')
    horizon=depth+follow+MAX_PERIOD+3
    tokens,hops=make_batch(N_NODES,batch,prompt_seed,horizon)
    periods=np.asarray(pointer_periods(tokens),dtype=np.int64)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        F,h=prepare_step(model,tokens)
        states=[h]
        for _ in range(horizon):
            h=F(h)
            states.append(h)
        x=states[depth]
        d=states[depth+1][:,-1,:]-x[:,-1,:]
        ref=states[depth+follow]
        phase_next=states[depth+follow+1]
        reference=[states[depth+follow+j] for j in range(MAX_PERIOD)]
        closure=np.array([float(_norm_batch((states[depth+follow+int(p)]-ref)[i:i+1]) /
                               _norm_batch(ref[i:i+1]).clamp_min(1e-30)) for i,p in enumerate(periods)])
        one=(_norm_batch(phase_next-ref)/_norm_batch(ref).clamp_min(1e-30)).numpy()
        active=(periods>1)&(closure<1e-4)&(one>0.05)
        distinct=np.array([min(float(_norm_batch((states[depth+j]-x)[i:i+1]))
                               for j in range(1,int(p))) if p>1 else 0 for i,p in enumerate(periods)])
        registered=active&(distinct>1e-8)
        idx=np.flatnonzero(registered)
        future=hops[:,depth+follow].numpy()
        true_pred=model.readout(phase_next[:,-1,:]).argmax(-1).numpy()
        delta_norm=_norm_batch(d[:,None,:]).clamp_min(1e-30)
        frozen = token_hashes(tokens)
        result={'target_seed':target_seed,'prompt_seed':prompt_seed,'batch':batch,
                'registered':int(len(idx)),'nontrivial':int(np.sum(periods>1)),
                'target_natural_next_hop_accuracy_registered':float(np.mean(true_pred[idx]==future[idx])) if len(idx) else None,
                'prompt_hashes':sorted(frozen), 'conditions':{}}
        def evaluate(name, update, raw_energy=None, scale_ratio=None):
            y=x.clone()
            y[:,-1,:]+=update
            for _ in range(follow):
                y=F(y)
            distances=torch.stack([_norm_batch(y-z) for z in reference],dim=1).numpy()
            closest=np.array([int(np.argmin(distances[i,:p])) if p>1 else -1
                              for i,p in enumerate(periods)])
            pred=model.readout(y[:,-1,:]).argmax(-1).numpy()
            result['conditions'][name]={
                'n_evaluable':int(len(idx)),
                'shift_success':int(np.sum(closest[idx]==1)),
                'phase_zero':int(np.sum(closest[idx]==0)),
                'next_hop_correct':int(np.sum(pred[idx]==future[idx])),
                'shift_and_correct':int(np.sum((closest[idx]==1)&(pred[idx]==future[idx]))),
                'shift_bits':(closest[idx]==1).astype(int).tolist(),
                'shift_and_correct_bits':((closest[idx]==1)&(pred[idx]==future[idx])).astype(int).tolist(),
                'median_raw_projection_energy':float(np.median(raw_energy[idx])) if raw_energy is not None and len(idx) else None,
                'median_norm_match_multiplier':float(np.median(scale_ratio[idx])) if scale_ratio is not None and len(idx) else None,
                'max_norm_match_multiplier':float(np.max(scale_ratio[idx])) if scale_ratio is not None and len(idx) else None,
            }
        evaluate('full_phase_step',d)
        evaluate('no_step',torch.zeros_like(d))
        for rank in ranks:
            for label,basis in bases.items():
                for mode in ('raw','norm_matched'):
                    projection = project_increment(d,basis[:rank],'raw')
                    ratio=delta_norm/_norm_batch(projection[:,None,:]).clamp_min(1e-30)
                    raw_energy=(_norm_batch(projection[:,None,:]) / delta_norm).square().numpy()
                    y=projection if mode=='raw' else projection*ratio[:,None]
                    evaluate(f'{label}_r{rank}_{mode}',y,raw_energy,ratio.numpy() if mode=='norm_matched' else None)
            for trial in range(random_repeats):
                Q=random_basis(D_MODEL,max(ranks),seed=73731+target_seed*31+trial)
                for rank in ranks:
                    for mode in ('raw','norm_matched'):
                        projection=project_increment(d,Q[:rank],'raw')
                        ratio=delta_norm/_norm_batch(projection[:,None,:]).clamp_min(1e-30)
                        raw_energy=(_norm_batch(projection[:,None,:])/delta_norm).square().numpy()
                        y=projection if mode=='raw' else projection*ratio[:,None]
                        evaluate(f'random_trial{trial}_r{rank}_{mode}',y,raw_energy,ratio.numpy() if mode=='norm_matched' else None)
    return result


def run_campaign(checkpoint_dir:Path, eval_seeds=EVAL_PROMPT_SEEDS, batch=EVAL_ITEMS,
                 n_train=TRAIN_PROMPTS, temporal_horizon=FUTURE_HORIZON,
                 ranks=RANKS, random_repeats=RANDOM_REPEATS):
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if TRAIN_PROMPT_SEED in eval_seeds or len(set(eval_seeds))!=len(eval_seeds):
        raise ValueError('disjoint prompt cohorts required')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    models={}
    for seed in SEEDS:
        models[seed]=model_from_receipt(checkpoint_dir/f'spc04_seed{seed}.pt',seed)
    fitted={}
    for seed,(model,sha) in models.items():
        O,G,train_hashes,record=fit_temporal_basis(model,n_prompts=n_train,horizon=temporal_horizon)
        P,phase_rec=fit_phase_svd(model)
        fitted[seed]=(O,P,G,train_hashes,record,phase_rec)
    model_a, model_b = models[SEEDS[0]][0], models[SEEDS[1]][0]
    aligned_Q, shuffled_Q, alignment_hashes, alignment_record = train_alignment(model_a, model_b)
    out={'campaign':'SPC-07','version':'0.1','model_seeds':list(SEEDS),
         'protocol':{'train_seed':TRAIN_PROMPT_SEED,'train_prompts':n_train,
                     'train_depth':TRAIN_DEPTH,'temporal_horizon':temporal_horizon,
                     'fd_eps':FD_EPS,'heldout_seeds':list(eval_seeds),'heldout_batch':batch,
                     'ranks':list(ranks),'random_repeats':random_repeats,
                     'transfer_coordinate_system':'raw hidden-neuron coordinates, no target alignment',
                     'basis_sources':'train-only frozen separate model weights and prompts',
                     'head_only':'same task, no finetuning, no transfer-based learning',
                     'alignment':'paired prompt/depth unsupervised orthogonal Procrustes; separate shuffled-row control'},
         'alignment_calibration':alignment_record, 'checkpoints':{str(k):{'sha256':v[1],'temporal_fit':fitted[k][4],
                                'phase_svd_train':{'train_seed':fitted[k][5]['train_seed'],
                                                   'energy_by_rank':fitted[k][5]['energy_capture_by_rank']}}
                         for k,v in models.items()}, 'directions':{}, 'heldout':{}}
    for source_seed,target_seed in itertools.product(SEEDS,repeat=2):
        source_o,source_p,_,source_hashes,_,_=fitted[source_seed]
        target_o,target_p,target_G,target_hashes,_,_=fitted[target_seed]
        Q = aligned_Q if source_seed==SEEDS[0] else aligned_Q.T
        Q_shuffle = shuffled_Q if source_seed==SEEDS[0] else shuffled_Q.T
        aligned_o = source_o @ Q
        aligned_p = source_p @ Q
        shuffled_o = source_o @ Q_shuffle
        shuffled_p = source_p @ Q_shuffle
        pair=f'{source_seed}_to_{target_seed}'
        out['directions'][pair]={
            'same_checkpoint':source_seed==target_seed,
            'by_rank':{str(rank):{
                'obs_alignment':projector_alignment(source_o,target_o,rank),
                'source_obs_captures_target_obs_gram':gramian_capture(source_o,target_G,rank),
                'target_obs_captures_target_obs_gram':gramian_capture(target_o,target_G,rank),
                'phase_alignment':projector_alignment(source_p,target_p,rank),
                'aligned_temporal_alignment':projector_alignment(aligned_o,target_o,rank),
                'aligned_phase_alignment':projector_alignment(aligned_p,target_p,rank),
                'aligned_source_obs_captures_target_gram':gramian_capture(aligned_o,target_G,rank),
                'shuffled_source_obs_captures_target_gram':gramian_capture(shuffled_o,target_G,rank),
            }for rank in ranks}}
        if source_seed==target_seed:
            continue
        bases={'source_temporal':source_o,'target_temporal':target_o,
               'aligned_temporal':aligned_o,'shuffled_temporal':shuffled_o,
               'source_phase':source_p,'target_phase':target_p,
               'aligned_phase':aligned_p,'shuffled_phase':shuffled_p}
        samples=[]
        for eval_seed in eval_seeds:
            z=evaluate_phase_batch(models[target_seed][0],bases,target_seed,
                         eval_seed,batch=batch,ranks=ranks,random_repeats=random_repeats)
            overlap=len(source_hashes & set(z['prompt_hashes']))
            overlap+=len(target_hashes & set(z['prompt_hashes']))
            overlap+=len(alignment_hashes & set(z['prompt_hashes']))
            if overlap:
                raise RuntimeError(f'train/heldout prompt leakage: {pair}, {overlap} duplicates')
            z['train_eval_overlap_count']=overlap
            # Keep exact overlap hashes only in audit receipt, not every summary.
            z['prompt_hashes_sha256']=hashlib.sha256(''.join(z.pop('prompt_hashes')).encode()).hexdigest()
            samples.append(z)
        out['heldout'][pair]=samples
    return out


def aggregate(result):
    """Aggregate paired heldout phase-shift rates with fixed denominators."""
    summary={}
    for pair,rows in result['heldout'].items():
        total=sum(r['registered'] for r in rows)
        conditions={}
        for condition in rows[0]['conditions']:
            records=[r['conditions'][condition] for r in rows]
            conditions[condition]={
                'shift':sum(r['shift_success'] for r in records),
                'shift_and_correct':sum(r['shift_and_correct'] for r in records),
                'correct':sum(r['next_hop_correct'] for r in records),
                'n':total,
                'median_raw_projection_energy':float(np.median([r['median_raw_projection_energy'] for r in records
                                  if r['median_raw_projection_energy'] is not None])) if records[0]['median_raw_projection_energy'] is not None else None,
                'median_norm_match_multiplier':float(np.median([r['median_norm_match_multiplier'] for r in records
                                  if r['median_norm_match_multiplier'] is not None])) if records[0]['median_norm_match_multiplier'] is not None else None,
            }
        summary[pair]={'registered':total,'conditions':conditions}
    return summary



def paired_cluster_bootstrap(result, comparisons=None, repeats=4000, seed=771407):
    """Paired held-out rates, resampling entire input batches across models."""
    if comparisons is None:
        comparisons=(('aligned_temporal_r8_raw','source_temporal_r8_raw'),
                     ('aligned_temporal_r8_norm_matched','shuffled_temporal_r8_norm_matched'),
                     ('aligned_phase_r8_raw','shuffled_phase_r8_raw'),
                     ('aligned_phase_r8_raw','source_phase_r8_raw'))
    rng=np.random.default_rng(seed)
    clusters=[run for rows in result['heldout'].values() for run in rows]
    if not clusters:
        return {}
    out={}
    for chosen,control in comparisons:
        sample=[]
        for run in clusters:
            a=np.asarray(run['conditions'][chosen]['shift_bits'],dtype=np.int8)
            b=np.asarray(run['conditions'][control]['shift_bits'],dtype=np.int8)
            if len(a)!=len(b) or len(a)!=run['registered']:
                raise ValueError('unpaired held-out intervention conditions')
            sample.append((len(a),int((a-b).sum()),int(np.sum((a==1)&(b==0))),int(np.sum((a==0)&(b==1)))))
        total=sum(c[0] for c in sample)
        if total==0:raise ValueError('no registered heldout cycles')
        estimate=sum(c[1] for c in sample)/total
        boots=[]
        for _ in range(repeats):
            ids=rng.integers(0,len(sample),len(sample))
            denom=sum(sample[j][0] for j in ids)
            if denom:
                boots.append(sum(sample[j][1] for j in ids)/denom)
        out[f'{chosen}__vs__{control}']={
            'registered':total,'cluster_count':len(clusters),
            'difference_rate':estimate,
            'bootstrap_95pct_cluster_ci':[float(x) for x in np.quantile(boots,(0.025,0.975))],
            'discordant_gain':sum(c[2] for c in sample),
            'discordant_loss':sum(c[3] for c in sample),
            'bootstrap_repeats':repeats,'bootstrap_seed':seed,
        }
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--summary-out',type=Path)
    p.add_argument('--train-prompts',type=int,default=TRAIN_PROMPTS)
    p.add_argument('--heldout-batch',type=int,default=EVAL_ITEMS)
    p.add_argument('--random-repeats',type=int,default=RANDOM_REPEATS)
    args=p.parse_args()
    result=run_campaign(args.checkpoint_dir,n_train=args.train_prompts,
                        batch=args.heldout_batch,random_repeats=args.random_repeats)
    brief={'pairs':aggregate(result),'paired_bootstrap':paired_cluster_bootstrap(result)}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True,exist_ok=True)
        args.summary_out.write_text(json.dumps(brief,indent=2,sort_keys=True)+'\n')
    if not args.out:
        print(json.dumps(brief,indent=2,sort_keys=True))

if __name__=='__main__':main()
