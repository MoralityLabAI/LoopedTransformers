"""Post-hoc cross-task positive-control attempt; does not replace the locked test."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from research_gym.scripts.loopscope_toy import LoopedChase,make_batch
from research_gym.scripts.loopscope_readout_silent import embedding,step,trace,make_patches,ARMS,sha

@torch.inference_mode()
def run(checkpoint,out):
    torch.set_num_threads(1)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=payload['cfg']
    m=LoopedChase(cfg['n_nodes'],cfg['d'],cfg['layers'],cfg['heads']);m.load_state_dict(payload['model']);m.eval()
    tokens,hops=make_batch(cfg['n_nodes'],512,941117,8);e,states=trace(m,tokens,8)
    t=4;h=states[t];now=m.readout(h[:,-1]);native=m.readout(states[t+1:,:,-1]).permute(1,0,2)
    pred=now.argmax(-1).numpy();oracle=hops[:,t-1].numpy();nextoracle=hops[:,t].numpy()
    # Sorted edge map, independent of prompt ordering.
    edges=tokens[:,:2*cfg['n_nodes']].reshape(512,cfg['n_nodes'],2).numpy()
    graph=np.zeros((512,cfg['n_nodes']),dtype=int)
    for b in range(512):graph[b,edges[b,:,0]]=edges[b,:,1]
    rng=np.random.default_rng(941217);pairs=[]
    for b in range(512):
        if pred[b]!=oracle[b]:continue
        eligible=[d for d in range(512) if d!=b and oracle[d]==oracle[b] and pred[d]==oracle[d]
                  and nextoracle[d]!=nextoracle[b] and not np.array_equal(graph[b],graph[d])]
        if eligible:pairs.append((b,int(rng.choice(eligible))))
    pairs=[pairs[i] for i in sorted(rng.choice(len(pairs),min(256,len(pairs)),replace=False))]
    b=torch.tensor([x[0] for x in pairs]);d=torch.tensor([x[1] for x in pairs]);B=len(pairs)
    patches,norm=make_patches(h[b],h[d],cfg['n_nodes'],941218)
    current=m.readout(patches[:,:,-1]);pre=m.readout(h[b,-1])
    err=float((current[:-1]-pre).abs().max())
    if err!=0.:raise AssertionError('initial readout changed')
    hh=patches.reshape(len(ARMS)*B,*h.shape[1:]);ee=e[b].repeat(len(ARMS),1,1);futures=[]
    for _ in range(4):
        hh=step(m,hh,ee);futures.append(m.readout(hh[:,-1]).reshape(len(ARMS),B,-1))
    future=torch.stack(futures,dim=2);labels=future.argmax(-1);base=native[b].argmax(-1)
    rows={}
    for a,name in enumerate(ARMS[:-1]):
        rows[name]={'future_changed_pairs':int((labels[a]!=base).any(-1).sum()),
                    'next_recipient_agreement':int((labels[a,:,0]==hops[b,t]).sum()),
                    'next_donor_agreement':int((labels[a,:,0]==hops[d,t]).sum())}
    if rows['sham']['future_changed_pairs'] or rows['layernorm_null']['future_changed_pairs']:
        raise AssertionError('null-control failure')
    result={'status':'POSTHOC_CROSS_TASK_MANIPULATION_CHECK','checkpoint_sha256':sha(checkpoint),
            'selection_seed':941217,'patch_seed':941218,'items':512,'loop':t,'pairs':B,
            'zero_initial_logits_max_abs':err,'arms':rows,
            'mean_relative_full_patch_l2':float((norm/torch.linalg.vector_norm(h[b].flatten(1),dim=1)).mean()),
            'scope':'receiver embedding retained; cross-input edge donor; possible off-manifold conflict'}
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'calibration.npz',pairs=np.array(pairs),current_logits=current.numpy(),
          receiver_current_logits=pre.numpy(),future_logits=future.numpy(),receiver_logits=native[b].numpy(),
          receiver_expected=hops[b,t:t+4].numpy(),donor_expected=hops[d,t:t+4].numpy(),norm=norm.numpy())
    result['raw_sha256']=sha(out/'calibration.npz')
    (out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.checkpoint,a.out)
