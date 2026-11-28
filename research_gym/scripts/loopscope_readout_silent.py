"""Readout-silent edge-state patches in native trained LoopedChase checkpoints.

The answer-token state stays fixed, so the entire current native logit vector
is identical. Pairs share input and oracle phase; future outcomes never select
pairs. This tests readout sufficiency under finite interventions, not a claim
that transplanted states are natural trajectories. See LOCK.md for the pilot.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from research_gym.scripts.loopscope_toy import LoopedChase, make_batch

ARMS=('sham','edges_025','random_025','edges_100','random_100','layernorm_null','full_donor')
DOSES={'sham':0.0,'edges_025':.25,'random_025':.25,'edges_100':1.,'random_100':1.,'layernorm_null':1.,'full_donor':1.}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def embedding(model,tokens):
    n=model.n_nodes
    edges=tokens[:,:2*n].reshape(tokens.shape[0],n,2)
    e=torch.cat([model.tok(edges[...,0])+model.dst(edges[...,1]),model.tok(tokens[:,2*n:])],1)
    return e+model.pos(torch.arange(e.shape[1],device=e.device))

def step(model,h,e):
    h=model.inject(h)+e
    for block in model.core:h=block(h)
    return h

def trace(model,tokens,loops):
    e=embedding(model,tokens);h=torch.zeros_like(e);states=[h]
    for _ in range(loops):
        h=step(model,h,e);states.append(h)
    return e,torch.stack(states)

def parity(a,b,name,rtol=2e-5,atol=2e-6):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError(name+': shape/nonfinite failure')
    if not torch.allclose(a,b,rtol=rtol,atol=atol):
        # LayerNorm cancellation / batching are certified by per-vector L2,
        # avoiding unstable relative tests on scalar coordinates near zero.
        err=torch.linalg.vector_norm((a-b).flatten(-2),dim=-1)
        scale=torch.linalg.vector_norm(b.flatten(-2),dim=-1)
        if bool((err>atol+rtol*scale).any()):
            raise RuntimeError(name+': native replay mismatch')
    return float((a-b).abs().max())

def select_pairs(predictions,oracle,last_visit,max_pairs,seed):
    """Read endpoint data only. One pair/item, preventing within-item pseudoreplication."""
    rng=np.random.default_rng(seed);eligible=[];candidate_count=0
    for item in range(predictions.shape[1]):
        candidates=[]
        for t in range(1,last_visit):
            for u in range(t+1,last_visit+1):
                label=int(oracle[t-1,item])
                if label==oracle[u-1,item] and predictions[t-1,item]==label and predictions[u-1,item]==label:
                    candidates.append((item,t,u,label))
        candidate_count+=len(candidates)
        if candidates:eligible.append(candidates[int(rng.integers(len(candidates)))])
    coverage=len(eligible)
    if len(eligible)>max_pairs:
        keep=sorted(rng.choice(len(eligible),max_pairs,replace=False).tolist())
        eligible=[eligible[i] for i in keep]
    return eligible,coverage,candidate_count

def make_patches(base,donor,n_edges,seed):
    if base.shape!=donor.shape or base.ndim!=3 or n_edges!=base.shape[1]-1:
        raise ValueError('Expected all edge tokens plus one answer token')
    rng=np.random.default_rng(seed)
    delta=donor[:,:n_edges]-base[:,:n_edges]
    norm=torch.linalg.vector_norm(delta.flatten(1),dim=1)
    def normalized(raw):
        v=torch.as_tensor(raw,device=base.device,dtype=base.dtype)
        lengths=torch.linalg.vector_norm(v.flatten(1),dim=1).clamp_min(1e-30)
        return v*(norm/lengths)[:,None,None]
    noise=normalized(rng.standard_normal(tuple(delta.shape)))
    null=normalized(np.broadcast_to(rng.standard_normal((*delta.shape[:2],1)),tuple(delta.shape)).copy())
    out=[]
    for arm in ARMS:
        h=base.clone()
        if arm=='full_donor':h=donor.clone()
        elif arm=='layernorm_null':h[:,:n_edges]+=null
        elif arm.startswith('edges_'):h[:,:n_edges]+=DOSES[arm]*delta
        elif arm.startswith('random_'):h[:,:n_edges]+=DOSES[arm]*noise
        out.append(h)
    return torch.stack(out),norm

def summarize_arrays(data,training_loops):
    arms=list(ARMS);pairs=data['pairs'];future=data['future_logits']
    clean=data['receiver_logits'];donor=data['donor_logits'];expected=data['expected']
    def softmax(x):
        p=np.exp(x-x.max(axis=-1,keepdims=True));return p/p.sum(axis=-1,keepdims=True)
    baseline=clean.argmax(-1);dlabels=donor.argmax(-1);pred=future.argmax(-1)
    prob=softmax(future);bp=softmax(clean)
    tv=.5*np.abs(prob-bp[None]).sum(axis=-1)
    cp=(baseline==expected).all(-1)&(dlabels==expected).all(-1)
    masks={'all':np.ones(len(pairs),dtype=bool),
           'both_within_training':pairs[:,2]<=training_loops,
           'cross_training_boundary':(pairs[:,1]<=training_loops)&(pairs[:,2]>training_loops),
           'both_beyond_training':pairs[:,1]>training_loops,
           'posthoc_both_clean_futures_correct':cp}
    summary={}
    for label,mask in masks.items():
        count=int(mask.sum());entry={'pairs':count,'arms':{}}
        for a,arm in enumerate(arms):
            changed=(pred[a]!=baseline).any(-1)
            entry['arms'][arm]={
              'future_changed_pairs':int(changed[mask].sum()),
              'future_changed_rate':float(changed[mask].mean()) if count else None,
              'oracle_correct_steps':int((pred[a,mask]==expected[mask]).sum()),
              'oracle_comparisons':int(count*expected.shape[-1]),
              'mean_max_future_probability_tv':float(tv[a,mask].max(-1).mean()) if count else None,
              'max_future_probability_tv':float(tv[a,mask].max()) if count else None}
        entry['baseline_oracle_correct_steps']=int((baseline[mask]==expected[mask]).sum())
        summary[label]=entry
    return summary

@torch.inference_mode()
def evaluate(checkpoint:Path,out:Path,items=512,loops=16,horizon=4,max_pairs=256,seed=941117,device='cpu'):
    if items<1 or loops<=horizon+1 or horizon<1 or max_pairs<1:raise ValueError('invalid evaluation dimensions')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=saved['cfg']
    if cfg.get('fixed_depth') is not True:raise ValueError('requires fixed_depth=True (no requested-depth token)')
    model=LoopedChase(cfg['n_nodes'],cfg['d'],cfg['layers'],cfg['heads'])
    model.load_state_dict(saved['model'],strict=True);model.to(device=device,dtype=torch.float32).eval()
    tokens,hops=make_batch(cfg['n_nodes'],items,seed,loops)
    tokens=tokens.to(device)
    e,states=trace(model,tokens,loops)
    native=model(tokens,loops)
    controls={'native_cursor_max_abs':parity(states[1:,:,-1],native,'native forward')}
    logits=model.readout(states[1:,:,-1]);native_logits=model.readout(native)
    controls['native_logit_max_abs']=parity(logits,native_logits,'native head')
    pred=logits.argmax(-1).cpu().numpy();oracle=hops.T.numpy()
    rows,coverage,candidates=select_pairs(pred,oracle,loops-horizon,max_pairs,seed+1)
    accuracy=[float((pred[i]==oracle[i]).mean()) for i in range(loops)]
    training_loops=int(cfg.get('loops',cfg['max_train_depth']))
    receipt={'schema':'LOOPSCOPE-READOUT-SILENT-V1','checkpoint_sha256':sha(checkpoint),
             'cfg':cfg,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,
             'items':items,'loops':loops,'horizon':horizon,'max_pairs':max_pairs,
             'eligible_items':coverage,'candidate_pairs':candidates,'selected_pairs':len(rows),
             'oracle_accuracy_by_loop':accuracy,
             'capability_gate_passed':training_loops<=loops and accuracy[training_loops-1]>=.9,
             'selection':'one pair per item; same oracle node and correct endpoints; no future outcome selection',
             'device':device,'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version(),
             'arms':list(ARMS),'controls':controls}
    out.mkdir(parents=True,exist_ok=True)
    if not rows:
        receipt['status']='NO_ELIGIBLE_PAIRS'
        (out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
    pairs=np.asarray(rows,dtype=np.int64)
    b=torch.as_tensor(pairs[:,0],device=device);t=torch.as_tensor(pairs[:,1],device=device);u=torch.as_tensor(pairs[:,2],device=device)
    base=states[u,b];donor=states[t,b];prompt=e[b]
    patches,donornorm=make_patches(base,donor,cfg['n_nodes'],seed+2)
    # Identical shape and cursor inputs make this exact, not merely top-1 matching.
    now=model.readout(patches[:,:,-1]);clean_now=model.readout(base[:,-1])
    silent_max=float((now[:-1]-clean_now).abs().max())
    if silent_max!=0.0:raise RuntimeError(f'non-silent patch: current logits changed {silent_max}')
    controls['all_silent_current_logits_max_abs']=silent_max
    B=len(rows);A=len(ARMS)
    h=patches.reshape(A*B,*base.shape[1:]);ee=prompt.repeat(A,1,1);future_states=[]
    for _ in range(horizon):
        h=step(model,h,ee);future_states.append(h.reshape(A,B,*base.shape[1:]))
    future_states=torch.stack(future_states,dim=2) # [arm,item,future,token,width]
    shifts=torch.arange(1,horizon+1,device=device)
    clean_future=states[u[:,None]+shifts,b[:,None]]
    donor_future=states[t[:,None]+shifts,b[:,None]]
    controls['sham_max_abs']=parity(future_states[0],clean_future,'sham')
    controls['full_donor_max_abs']=parity(future_states[-1],donor_future,'donor replay')
    controls['null_max_abs']=parity(future_states[5],clean_future,'LayerNorm null',rtol=5e-5,atol=5e-6)
    future_logits=model.readout(future_states[:,:,:,-1]);clean_logits=model.readout(clean_future[:,:,-1]);donor_logits=model.readout(donor_future[:,:,-1])
    for index,name in [(0,'sham'),(5,'null')]:
        count=int((future_logits[index].argmax(-1)!=clean_logits.argmax(-1)).sum())
        controls[name+'_answer_changes']=count
        if count:raise RuntimeError(name+' control changed discrete answers')
    data={'pairs':pairs,'current_logits':now.cpu().numpy(),'receiver_current_logits':clean_now.cpu().numpy(),
          'future_logits':future_logits.cpu().numpy(),'receiver_logits':clean_logits.cpu().numpy(),
          'donor_logits':donor_logits.cpu().numpy(),
          'expected':hops[pairs[:,0,None],pairs[:,2,None]+np.arange(horizon)].numpy(),
          'applied_l2':torch.linalg.vector_norm((patches-base).flatten(2),dim=-1).cpu().numpy(),
          'donor_edge_l2':donornorm.cpu().numpy(),
          'receiver_l2':torch.linalg.vector_norm(base.flatten(1),dim=-1).cpu().numpy(),
          'oracle':oracle,'predictions':pred,'tokens':tokens.cpu().numpy()}
    summary=summarize_arrays(data,training_loops)
    receipt.update(status='MEASURED_TRAINED_CHECKPOINT',summary=summary,
                   mean_relative_edge_patch_l2=float((donornorm/torch.linalg.vector_norm(base.flatten(1),dim=-1)).mean()),
                   median_relative_edge_patch_l2=float((donornorm/torch.linalg.vector_norm(base.flatten(1),dim=-1)).median()),
                   wall_seconds=time.monotonic()-started)
    np.savez_compressed(out/'raw.npz',**data)
    receipt['raw_sha256']=sha(out/'raw.npz')
    receipt['source_sha256']={'runner':sha(__file__),
      'model_module':sha(__import__('inspect').getfile(LoopedChase))}
    (out/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checkpoint':str(checkpoint),'accuracy_depth4':accuracy[3],
                     'selected':B,'controls':controls,'all':summary['all'],'correct_future_stratum':summary['posthoc_both_clean_futures_correct']},indent=2),flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint',type=Path)
    group.add_argument('--checkpoint-dir',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--items',type=int,default=512);p.add_argument('--loops',type=int,default=16)
    p.add_argument('--horizon',type=int,default=4);p.add_argument('--max-pairs',type=int,default=256)
    p.add_argument('--seed',type=int,default=941117);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    paths=[a.checkpoint] if a.checkpoint else sorted(a.checkpoint_dir.glob('*.final.pt'))
    if not paths:raise FileNotFoundError('No *.final.pt checkpoints; trained Q4/Q5 weights are external to Git')
    for path in paths:
        dest=a.out if a.checkpoint else a.out/path.stem
        evaluate(path,dest,a.items,a.loops,a.horizon,a.max_pairs,a.seed,a.device)
if __name__=='__main__':main()
