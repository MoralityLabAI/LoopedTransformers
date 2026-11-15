"""Fresh small CPU LoopedChase pilot, distinct from the Q4/Q5 checkpoint cohort."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from research_gym.scripts.loopscope_toy import LoopedChase, make_batch

def train(seed: int, out: Path, steps: int=2000):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    cfg=dict(n_nodes=6,d=64,layers=1,heads=4,fixed_depth=True,max_train_depth=4,
             loops=4,seed=seed,lam=1.0,steps=steps,batch=64,
             training_status='FRESH_CPU_ARCHITECTURE_PILOT',lr=0.002)
    model=LoopedChase(cfg['n_nodes'],cfg['d'],cfg['layers'],cfg['heads'])
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01,betas=(0.9,0.98))
    start=time.monotonic(); log=[]
    for step in range(steps):
        tokens,hops=make_batch(6,64,seed*1000003+step,4)
        warm=100
        lr=.002*(min(1,(step+1)/warm) if step<warm else
                 (.05+.95*.5*(1+math.cos(math.pi*(step-warm)/(steps-warm)))))
        for g in opt.param_groups:g['lr']=lr
        logits=model.readout(model(tokens,4))
        loss=F.cross_entropy(logits.reshape(-1,6),hops.T.reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (step+1)%250==0:
            entry=dict(step=step+1,loss=float(loss.detach()),elapsed=time.monotonic()-start)
            log.append(entry);print(seed,entry,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    path=out/f'cpu_s{seed}.final.pt'
    torch.save(dict(cfg=cfg,model=model.state_dict()),path)
    receipt=dict(cfg=cfg,parameters=sum(p.numel() for p in model.parameters()),
                 torch=str(torch.__version__),device='cpu',threads=1,
                 wall_seconds=time.monotonic()-start,log=log,
                 checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();train(a.seed,a.out,a.steps)
