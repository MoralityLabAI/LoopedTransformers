"""SPC-08: phase-driving vs current-readout and deferred-output directions.

Frozen SPC-04 LoopedChase weights only. All fitting uses predefined independent
prompt seeds. A local output-margin derivative defines the instantaneous
observable rowspace, and future-output derivatives restricted to its nullspace
define the delayed-only Gramian. No claim of global readout nullity or
unique mechanistic features follows from such local linear decompositions.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Sequence
import numpy as np

from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    SEEDS, PHASE_DEPTH, FOLLOW_LOOPS, MAX_PERIOD, model_from_receipt,
    prepare_step, pointer_periods, _norm_batch,
)
from experiments.spectral_state_cost_v0.spc05_phase_subspace import (
    fit_phase_svd, project_increment,
)
from experiments.spectral_state_cost_v0.spc06_readout_observability import (
    logit_margin_jacobian, exact_readout_fiber_rotate,
    random_equal_chord_fiber,
)
from experiments.spectral_state_cost_v0.spc07_transfer import (
    finite_temporal_jacobian, train_alignment, gramian_capture,
    projector_alignment, token_hashes,
)
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES,D_MODEL

TRAIN_SEED=920731
TRAIN_PROMPTS=16
TRAIN_HORIZON=8
EVAL_SEEDS=(910701,910702,910703,910704,910705,910706,910707,910708)
EVAL_BATCH=64
RANK=8
FIBER_SCALE=8.0


def row_null(delta, jac):
    """Orthogonal decomposition into instantaneous margin row and null spaces.

    delta: [batch, width]; jac: [batch, seven margins, width].
    """
    import torch
    if delta.ndim!=2 or jac.shape!=(delta.shape[0],N_NODES-1,delta.shape[1]):
        raise ValueError('invalid jacobian dimensions')
    _,singular,Vh=torch.linalg.svd(jac,full_matrices=False)
    rank=(singular > singular[:,:1]*1e-10).sum(-1)
    if not bool(torch.all(rank == N_NODES-1)):
        raise RuntimeError('current margin derivative lost row rank')
    row=((delta[:,None,:]@Vh.transpose(1,2))@Vh)[:,0,:]
    null=delta-row
    if float(torch.linalg.vector_norm((jac@null[:,:,None]).squeeze(-1),dim=-1).max())>1e-8:
        raise RuntimeError('nullspace projection failed')
    return row,null


def eigh_basis(gram):
    import torch
    eig,vec=torch.linalg.eigh(gram)
    eig=torch.clamp(eig.flip(0),min=0)
    basis=vec.flip(1).T.contiguous()
    return basis,eig


def fit_causal_decomposition(model, prompt_seed=TRAIN_SEED,
                             n_prompts=TRAIN_PROMPTS,horizon=TRAIN_HORIZON):
    """Three matched Gramians; delayed excludes current readout-row directions.

    G_current = E[J0^T J0 / ||O||_F^2]
    G_total   = E[O^T O / ||O||_F^2]
    G_deferred= E[(O_future N0)^T (O_future N0) / ||O||_F^2]
    where N0 projects onto ker J0 *for the particular prompt*.
    """
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if n_prompts<2 or n_prompts>128 or horizon<1:raise ValueError('invalid fit cohort')
    tokens,_=make_batch(N_NODES,n_prompts,prompt_seed,32+horizon+1)
    G0=torch.zeros((D_MODEL,D_MODEL),dtype=torch.float64)
    Gf=torch.zeros_like(G0)
    Gtot=torch.zeros_like(G0)
    checks=[]
    for i in range(n_prompts):
        O,x=finite_temporal_jacobian(model,tokens[i:i+1],depth=32,horizon=horizon)
        J=O[:N_NODES-1]
        future=O[N_NODES-1:]
        _,sv,Vh=torch.linalg.svd(J,full_matrices=False)
        if int((sv>sv[0]*1e-10).sum())!=N_NODES-1:raise RuntimeError('singular current output')
        N=torch.eye(D_MODEL,dtype=torch.float64)-Vh.T@Vh
        deferred=future@N
        scale=O.square().sum().clamp_min(1e-30)
        G0+=J.T@J/scale
        Gf+=deferred.T@deferred/scale
        Gtot+=O.T@O/scale
        if i<2:
            direct=logit_margin_jacobian(model,x[:,-1])[0]
            checks.append(float((torch.linalg.vector_norm(direct-J)/torch.linalg.vector_norm(direct)).detach()))
    G0/=n_prompts;Gf/=n_prompts;Gtot/=n_prompts
    bases={}
    for kind,gram in [('immediate',G0),('delayed_null',Gf),('temporal',Gtot)]:
        b,e=eigh_basis(gram)
        bases[kind]=b
    phase,phase_info=fit_phase_svd(model)
    bases['phase']=phase
    diag={'seed':prompt_seed,'prompts':n_prompts,'horizon':horizon,'output_margins':N_NODES-1,
          'analytic_jacobian_max_relative_gap':max(checks),
          'fit_prompt_fingerprints_sha256':hashlib.sha256(''.join(sorted(token_hashes(tokens))).encode()).hexdigest(),
          'phase_fit':{'seed':phase_info['train_seed'],'count':phase_info['train_prompts'],
                       'depths':phase_info['sample_depths'], 'rank8_energy':phase_info['energy_capture_by_rank']['8']},
          'spectra':{},'captures':{},'projector_overlaps':{}}
    grams={'immediate':G0,'delayed_null':Gf,'temporal':Gtot}
    for kind,gram in grams.items():
        _,ev=eigh_basis(gram)
        diag['spectra'][kind]={'rank7_energy_fraction':float(ev[:7].sum()/ev.sum()),
                               'rank8_energy_fraction':float(ev[:8].sum()/ev.sum()),
                               'top_eigenvalues':ev[:12].tolist(),
                               'positive_trace':float(torch.trace(gram))}
        diag['captures'][kind]={name:gramian_capture(b,gram,RANK) for name,b in bases.items()}
    for kind in ('temporal','delayed_null','immediate'):
        diag['projector_overlaps']['phase_vs_'+kind]=projector_alignment(bases['phase'],bases[kind],RANK)
    return bases,diag,token_hashes(tokens)



def heldout_local_sensitivity(model, bases:dict, prompt_seed:int=910708,
                              n_prompts:int=8, depth:int=32, horizon:int=8):
    """Independent, local held-out readout gains for frozen fitted subspaces.

    Reports the fraction of Frobenius squared output-gradient energy captured
    at the present time and by later measurements after current-row removal.
    The latter is conditional on current invisibility, not a causal phase test.
    """
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if not (1 <= n_prompts <= 64 and depth >= 3 and horizon >= 1):
        raise ValueError('invalid heldout Jacobian cohort')
    tokens,_=make_batch(N_NODES,n_prompts,prompt_seed,depth+horizon+1)
    fractions={k: {'current': [],'deferred_given_current_null': []}
               for k in ('phase','immediate','temporal','delayed_null')}
    ranks=[];analytic=[]
    for i in range(n_prompts):
        O, state=finite_temporal_jacobian(model,tokens[i:i+1],depth=depth,horizon=horizon)
        J=O[:N_NODES-1]
        _,s,Vh=torch.linalg.svd(J,full_matrices=False)
        N=torch.eye(D_MODEL,dtype=torch.float64)-Vh.T@Vh
        Deferred=O[N_NODES-1:]@N
        if float(torch.linalg.vector_norm(Deferred))<1e-12:
            raise RuntimeError('heldout delayed output derivative collapsed')
        sv=torch.linalg.svdvals(O)
        rank=int((sv>sv[0]*1e-3).sum())
        ranks.append(rank)
        analytic.append(float((torch.linalg.vector_norm(J-logit_margin_jacobian(model,state[:,-1,:])[0])/
                               torch.linalg.vector_norm(J)).detach()))
        for key in fractions:
            B=bases[key][:RANK].T
            fractions[key]['current'].append(float(((J@B).square().sum()/J.square().sum()).detach()))
            fractions[key]['deferred_given_current_null'].append(float(((Deferred@B).square().sum()/Deferred.square().sum()).detach()))
    return {'prompt_seed':prompt_seed,'n_heldout_prompts':n_prompts,
            'depth':depth,'future_horizon':horizon,
            'local_rank_rel_1e-3':ranks,
            'max_analytic_jacobian_disagreement':max(analytic),
            'basis_capture':{k:{metric:{'mean':float(np.mean(vals)),'median':float(np.median(vals))}
                                 for metric,vals in v.items()}
                            for k,v in fractions.items()},
            'sha256_prompt_set':hashlib.sha256(''.join(sorted(token_hashes(tokens))).encode()).hexdigest()}


def causal_phase_batch(model,bases:dict,seed:int,batch:int=EVAL_BATCH,
                       depth:int=PHASE_DEPTH,follow:int=FOLLOW_LOOPS,
                       fiber_scale:float=FIBER_SCALE,random_trial:int=0):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if batch<1 or depth<3 or follow<1 or fiber_scale<0:
        raise ValueError('invalid phase evaluation parameters')
    tokens,hops=make_batch(N_NODES,batch,seed,depth+follow+MAX_PERIOD+3)
    periods=np.asarray(pointer_periods(tokens),dtype=np.int64)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        F,h=prepare_step(model,tokens)
        states=[h]
        for _ in range(depth+follow+MAX_PERIOD+2):
            h=F(h);states.append(h)
        start=states[depth]; natural_delta=states[depth+1][:,-1,:]-start[:,-1,:]
        reference=states[depth+follow]
        refs=states[depth+follow:depth+follow+MAX_PERIOD]
        perr=np.array([float(_norm_batch((states[depth+follow+int(p)]-reference)[i:i+1])/_norm_batch(reference[i:i+1]).clamp_min(1e-30)) for i,p in enumerate(periods)])
        one=(_norm_batch(states[depth+follow+1]-reference)/_norm_batch(reference).clamp_min(1e-30)).numpy()
        sep=np.array([min(float(_norm_batch((states[depth+j]-start)[i:i+1])) for j in range(1,int(p))) if p>1 else 0. for i,p in enumerate(periods)])
        registered=(periods>1)&(perr<1e-4)&(one>0.05)&(sep>1e-8)
        idx=np.flatnonzero(registered)
        j0=logit_margin_jacobian(model,start[:,-1])
        readout_row,readout_null=row_null(natural_delta,j0)
        phase_proj=project_increment(natural_delta,bases['phase'][:RANK],'raw')
        delayed_proj=project_increment(natural_delta,bases['delayed_null'][:RANK],'raw')
        temporal_proj=project_increment(natural_delta,bases['temporal'][:RANK],'raw')
        p_row,p_null=row_null(phase_proj,j0)
        d_row,d_null=row_null(delayed_proj,j0)
        zero=torch.zeros_like(natural_delta)
        conditions={
            'natural_phase':natural_delta,
            'unperturbed':zero,
            'target_phase8':phase_proj,
            'target_temporal8':temporal_proj,
            'target_delayed8':delayed_proj,
            'immediate_row':readout_row,
            'immediate_null':readout_null,
            'phase8_row':p_row,
            'phase8_null':p_null,
            'delayed8_row':d_row,
            'delayed8_null':d_null,
        }
        for kind,projected in (('phase8',phase_proj),('temporal8',temporal_proj),('delayed8',delayed_proj)):
            norm=_norm_batch(projected[:,None,:]).clamp_min(1e-30)
            conditions[kind+'_normmatched']=projected*( _norm_batch(natural_delta[:,None,:]) / norm)[:,None]
        for kind in ('aligned_phase8','aligned_temporal8','aligned_delayed8','shuffled_phase8','shuffled_delayed8'):
            if kind in bases:
                projected=project_increment(natural_delta,bases[kind][:RANK],'raw')
                conditions[kind]=projected
                if kind in ('aligned_delayed8','aligned_temporal8'):
                    conditions[kind+'_normmatched']=projected*( _norm_batch(natural_delta[:,None,:]) / _norm_batch(projected[:,None,:]).clamp_min(1e-30))[:,None]
        phase_fiber=exact_readout_fiber_rotate(model,start[:,-1],phase_proj,fiber_scale,'logits')
        delayed_fiber=exact_readout_fiber_rotate(model,start[:,-1],delayed_proj,fiber_scale,'logits')
        natural_fiber=exact_readout_fiber_rotate(model,start[:,-1],natural_delta,fiber_scale,'logits')
        conditions['fiber_phase8']=phase_fiber-start[:,-1]
        conditions['fiber_delayed8']=delayed_fiber-start[:,-1]
        conditions['fiber_natural']=natural_fiber-start[:,-1]
        for trial in range(4):
            conditions[f'fiber_phase8_random_trial{trial}']=random_equal_chord_fiber(model,start[:,-1],phase_fiber,
                                                   810830+seed*17+random_trial*11+trial)-start[:,-1]
            conditions[f'fiber_delayed8_random_trial{trial}']=random_equal_chord_fiber(model,start[:,-1],delayed_fiber,
                                                   910830+seed*19+random_trial*13+trial)-start[:,-1]
        original_logits=model.readout(start[:,-1])
        natural_norm=_norm_batch(natural_delta[:,None,:]).clamp_min(1e-30)
        future_hops=hops[:,depth+follow].numpy()
        response={'prompt_seed':seed,'n_items':batch,'registered':int(len(idx)),'sha256_prompt_set':hashlib.sha256(''.join(sorted(token_hashes(tokens))).encode()).hexdigest(),
                  'conditions':{},'energy_shares':{}}
        for name,v in [('readout_row',readout_row),('readout_null',readout_null),('phase8_row',p_row),('phase8_null',p_null),('delayed8_row',d_row),('delayed8_null',d_null)]:
            share=v.square().sum(-1)/natural_norm.square()
            response['energy_shares'][name]=float(np.median(share.numpy()[idx])) if len(idx) else None
        for name,change in conditions.items():
            h=start.clone();h[:,-1,:]+=change
            current=model.readout(h[:,-1,:]); logit_change=(current-original_logits).abs().max(dim=1).values
            for _ in range(follow):h=F(h)
            distances=torch.stack([_norm_batch(h-r) for r in refs],dim=1).numpy()
            matched=np.asarray([int(np.argmin(distances[i,:p])) if p>1 else -1 for i,p in enumerate(periods)])
            pred=model.readout(h[:,-1]).argmax(-1).numpy()
            subidx=idx
            rr={
                'shift_bits':(matched[subidx]==1).astype(np.int8).tolist(),
                'correct_bits':(pred[subidx]==future_hops[subidx]).astype(np.int8).tolist(),
                'shift':int(np.sum(matched[subidx]==1)),
                'shift_correct':int(np.sum((matched[subidx]==1)&(pred[subidx]==future_hops[subidx]))),
                'next_hop_correct':int(np.sum(pred[subidx]==future_hops[subidx])),
                'median_norm_fraction':float(np.median((_norm_batch(change[:,None,:])/natural_norm).numpy()[idx])) if len(idx) else None,
                'median_immediate_max_abs_logit_change':float(np.median(logit_change.numpy()[idx])) if len(idx) else None,
                'max_immediate_abs_logit_change':float(logit_change.max()),
            }
            if name.startswith('fiber_') and rr['max_immediate_abs_logit_change']>1e-9:
                raise RuntimeError('exact-readout fiber lost current logits')
            response['conditions'][name]=rr
        return response


def campaign(checkpoint_dir:Path,eval_seeds=EVAL_SEEDS,batch:int=EVAL_BATCH,
             train_prompts:int=TRAIN_PROMPTS,fiber_scale:float=FIBER_SCALE):
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    models={seed:model_from_receipt(checkpoint_dir/f'spc04_seed{seed}.pt',seed) for seed in SEEDS}
    bases={}; diagnostics={}; train_hashes={}
    for seed,(model,sha) in models.items():
        b,d,h=fit_causal_decomposition(model,n_prompts=train_prompts)
        bases[seed]=b;diagnostics[str(seed)]={'checkpoint_hash':sha,**d};train_hashes[seed]=h
    qa,qn,align_hash,record=train_alignment(models[SEEDS[0]][0],models[SEEDS[1]][0])
    allruns={}
    for source,target in ((SEEDS[0],SEEDS[1]),(SEEDS[1],SEEDS[0])):
        q=qa if source==SEEDS[0] else qa.T
        nq=qn if source==SEEDS[0] else qn.T
        tmp=dict(bases[target]);tmp.update({
           'aligned_phase8':bases[source]['phase']@q,
           'aligned_temporal8':bases[source]['temporal']@q,
           'aligned_delayed8':bases[source]['delayed_null']@q,
           'shuffled_phase8':bases[source]['phase']@nq,
           'shuffled_delayed8':bases[source]['delayed_null']@nq,
        })
        rows=[]
        for es in eval_seeds:
            r=causal_phase_batch(models[target][0],tmp,seed=es,batch=batch,
                                 fiber_scale=fiber_scale)
            # The source/target phase training sets, derivative-fit sets, and
            # unlabeled alignment cohort are disjoint from test inputs.
            tokens,_=__import__('research_gym.scripts.loopscope_toy',fromlist=['make_batch']).make_batch(N_NODES,batch,es,PHASE_DEPTH+FOLLOW_LOOPS+MAX_PERIOD+3)
            hashed=token_hashes(tokens)
            overlap=sum(len(hashed & ss) for ss in (train_hashes[source],train_hashes[target],align_hash))
            if overlap:raise RuntimeError(f'train/heldout graph overlap {overlap}')
            r['train_eval_prompt_overlap']=overlap
            rows.append(r)
        allruns[f'{source}_to_{target}']=rows
    return {'campaign':'SPC-08','model':'trained 8-node LoopedChase','note':'synthetic pointer graphs; locally linearized readout masks and exact nonlinear fiber controls; frozen checkpoint weights',
            'protocol':{'eval_seeds':list(eval_seeds),'eval_batch':batch,'train_prompts':train_prompts,
                        'phase_fit':'SPC-05 fixed 256 prompts + depths 32/48/64',
                        'derivative_fit_seed':TRAIN_SEED,'derivative_horizon':TRAIN_HORIZON,
                        'alignment':'SPC-07 paired Procrustes and shuffled control',
                        'phase_depth':PHASE_DEPTH,'follow':FOLLOW_LOOPS,'fiber_scale':fiber_scale},
            'fitted':diagnostics,'alignment':record,'heldout':allruns}


def summarize(data):
    result={}
    for label,rows in data['heldout'].items():
        n=sum(r['registered'] for r in rows)
        result[label]={'registered':n,'conditions':{},'energy_shares':{}}
        for name in rows[0]['conditions']:
            z=[r['conditions'][name] for r in rows]
            result[label]['conditions'][name]={
                'shift':sum(r['shift'] for r in z),
                'shift_correct':sum(r['shift_correct'] for r in z),
                'next_hop_correct':sum(r['next_hop_correct'] for r in z),
                'n':n,
                'median_intervention_norm_fraction':float(np.median([w['median_norm_fraction'] for w in z])) if n else None,
                'median_immediate_max_abs_logit_change':float(np.median([w['median_immediate_max_abs_logit_change'] for w in z])) if n else None,
                'max_immediate_abs_logit_change':max(w['max_immediate_abs_logit_change'] for w in z),
            }
        for name in rows[0]['energy_shares']:
            result[label]['energy_shares'][name]=float(np.median([r['energy_shares'][name] for r in rows]))
    return result


def paired_bootstrap(data, comparisons=(('target_phase8','target_delayed8'),('target_phase8','target_temporal8'),('fiber_phase8','fiber_phase8_random_trial0'),('aligned_phase8','aligned_delayed8'),('aligned_phase8','shuffled_phase8')),n_repeat=4000,seed=811508):
    rng=np.random.default_rng(seed)
    clusters=[r for rows in data['heldout'].values() for r in rows]
    if not clusters:raise ValueError('empty heldout')
    answer={}
    for a,b in comparisons:
        observations=[]
        for r in clusters:
            x=np.asarray(r['conditions'][a]['shift_bits'],dtype=int)
            y=np.asarray(r['conditions'][b]['shift_bits'],dtype=int)
            if len(x)!=len(y) or len(x)!=r['registered']:raise RuntimeError('paired outcomes mismatch')
            observations.append((len(x),int((x-y).sum()),int(((x==1)&(y==0)).sum()),int(((x==0)&(y==1)).sum())))
        total=sum(z[0] for z in observations)
        point=sum(z[1] for z in observations)/total
        draws=[]
        for _ in range(n_repeat):
            pick=rng.integers(0,len(observations),len(observations))
            den=sum(observations[k][0] for k in pick)
            if den:draws.append(sum(observations[k][1] for k in pick)/den)
        answer[f'{a}_minus_{b}']={'difference_rate':point,'registered':total,
                    'cluster_count':len(observations),
                    'cluster_bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist(),
                    'discordant_gain':sum(z[2] for z in observations),
                    'discordant_loss':sum(z[3] for z in observations)}
    return answer


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--checkpoint-dir',type=Path,required=True)
    a.add_argument('--out',type=Path)
    a.add_argument('--summary',type=Path)
    a.add_argument('--train-prompts',type=int,default=TRAIN_PROMPTS)
    a.add_argument('--batch',type=int,default=EVAL_BATCH)
    a.add_argument('--eval-seeds',nargs='+',type=int,default=list(EVAL_SEEDS))
    a.add_argument('--fiber-scale',type=float,default=FIBER_SCALE)
    args=a.parse_args()
    result=campaign(args.checkpoint_dir,args.eval_seeds,args.batch,args.train_prompts,args.fiber_scale)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    info={'summary':summarize(result),'paired_cluster_bootstrap':paired_bootstrap(result)}
    if args.summary:
        args.summary.parent.mkdir(parents=True,exist_ok=True);args.summary.write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
    else:print(json.dumps(info,indent=2))
if __name__=='__main__':main()
