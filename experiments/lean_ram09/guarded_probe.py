"""Posthoc RAM-09G: frozen-trunk, bounded residual lookahead correction.

Designed after observing RAM-09 stage0 unguarded fine-tuning regressions;
this run is a retrospective methodological repair, not preregistered.
"""
from __future__ import annotations
import argparse,hashlib,json,time,random,torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict,Counter
from offline_graph import ObservedClosureGraph,load_archive,source_splits,key
from offline_fit import originals,split_internal,pairs_from_graph,encode,score,ARCHIVE_SHA

ALPHA=.35
STEPS=200
BATCH=32
SEED=20261115

def embedded(model,batch):
    d=model.config.width
    fixed=model.fuse(torch.cat([model.mean(batch[k]) for k in ('context','goal','pending','action')]+[model.family(batch['family']),batch['counts']],-1))
    z=fixed
    for _ in range(model.config.loops):z=model.norm(z+model.cell(torch.cat([z,fixed],-1)))
    return z

@torch.no_grad()
def score_new(model,head,rows):
    baseline=[];new=[]
    for i in range(0,len(rows),64):
        batch=encode(rows[i:i+64])
        z=embedded(model,batch)
        a=score(model,batch)
        b=a+ALPHA*torch.tanh(head(z).squeeze(-1))
        if not torch.all((b-a).abs()<=ALPHA+1e-6):raise AssertionError('rank-bound violated')
        baseline.extend(a.tolist());new.extend(b.tolist())
    return baseline,new

def stats(graph,rows,vals):
    by=defaultdict(list)
    for r,s in zip(rows,vals):by[key(r)].append((r,s))
    result=Counter()
    for root,rr in by.items():
        best=max(rr,key=lambda a:a[1])[0]
        result['states']+=1
        result['top1_useful']+=int(best['label']>0)
        result['top1_closed']+=int(best['label']==3)
        d=graph.action_distance(best)
        result['top1_observed_witness']+=int(d is not None and d<=4)
        if root in graph.distance:
            result['states_with_witness']+=1
            result['selected_witness']+=int(d is not None and d<=4)
    return dict(result)

def run(archive,out):
    if out.exists():raise ValueError('out already exists')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,_=load_archive(archive);tr,dev=source_splits(rows)
    fit,hold=split_internal(tr)
    gfit=ObservedClosureGraph(fit)
    ghold=ObservedClosureGraph(hold)
    gdev=ObservedClosureGraph(dev)
    pairs=pairs_from_graph(gfit)
    result=dict(campaign='RAM-09G_posthoc_bounded_residual',note='Posthoc after observed stage0 unguarded loss of witness selection', alpha=ALPHA,steps=STEPS,batch=BATCH,seed=SEED,pairs=len(pairs),PEF_cases=0,models={})
    out.mkdir(parents=True,exist_ok=False)
    for name,model,sha in originals(archive):
        torch.manual_seed(SEED)
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
        with torch.no_grad():
            pfeat=torch.cat([embedded(model,encode([p[0] for p in pairs[i:i+64]])) for i in range(0,len(pairs),64)])
            nfeat=torch.cat([embedded(model,encode([p[1] for p in pairs[i:i+64]])) for i in range(0,len(pairs),64)])
            pbase=torch.cat([score(model,encode([p[0] for p in pairs[i:i+64]])) for i in range(0,len(pairs),64)])
            nbase=torch.cat([score(model,encode([p[1] for p in pairs[i:i+64]])) for i in range(0,len(pairs),64)])
        head=nn.Linear(model.config.width,1)
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
        rng=random.Random(SEED);losses=[];began=time.perf_counter()
        for step in range(1,STEPS+1):
            ids=torch.tensor([rng.randrange(len(pairs)) for _ in range(BATCH)])
            pp=torch.tanh(head(pfeat[ids]).squeeze(-1))
            nnv=torch.tanh(head(nfeat[ids]).squeeze(-1))
            spread=(pbase[ids]-nbase[ids])+ALPHA*(pp-nnv)
            loss=F.softplus(-spread).mean()+.02*(pp.square().mean()+nnv.square().mean())
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%50==0:losses.append([step,round(float(loss.detach()),6)])
        metrics={}
        for split,graph in [('holdback',ghold),('old_development',gdev)]:
            b,n=score_new(model,head,graph.rows)
            original=stats(graph,graph.rows,b);guarded=stats(graph,graph.rows,n)
            # Verify bound and top1 preservation on high-margin original states.
            grouped=defaultdict(list)
            for row,bs,ns in zip(graph.rows,b,n):grouped[key(row)].append((bs,ns,row['script']))
            safe=0;safe_preserved=0
            for candidates in grouped.values():
                original_top=max(range(len(candidates)),key=lambda i:candidates[i][0])
                ordered=sorted([x[0] for x in candidates],reverse=True)
                if len(ordered)>1 and ordered[0]-ordered[1]>2*ALPHA:
                    safe+=1
                    safe_preserved+=int(original_top==max(range(len(candidates)),key=lambda i:candidates[i][1]))
            if safe!=safe_preserved:raise AssertionError('bounded-margin ranking theorem failed')
            metrics[split]=dict(original=original,guarded=guarded,high_margin_states=safe,guaranteed_preserved=safe_preserved)
        path=out/f'{name}_guarded_head.pt'
        torch.save(dict(schema='RAM09G-posthoc-v1',base_sha256=sha,alpha=ALPHA,head_state_dict=head.state_dict()),path)
        result['models'][name]=dict(base_sha256=sha,head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fit_seconds=round(time.perf_counter()-began,3),loss_snapshots=losses,**metrics)
        print('RAM09G',name,json.dumps(result['models'][name],sort_keys=True),flush=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('archive',type=Path);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();run(args.archive,args.out)
