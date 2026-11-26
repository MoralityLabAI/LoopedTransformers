"""MMS development follow-on to LBF-19 and LBF-20D/F.

Reuse native models, task generation, exact labels and search evaluators.
Historical registered trainers and receipts are untouched. No protected
test data enters the bank. Each run is one CPU seed and one treatment.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from research_gym.interp.mms import MechanisticMemory, SharedPagePool, MultiSiteMemory


def memory_kwargs(args):
    return dict(mib=args.mms_mib, mode=args.mms_mode, rank=args.rank,
                topk=args.topk, ratio=args.ratio, refresh=args.refresh,
                max_age=args.max_age)


def native_fit(args):
    if args.gym == '20d':
        import lbf20d_learning as gym
        cohorts = gym.tasks()
        train = cohorts['train'][:args.train_roots]
        rng = random.Random(2026112510)
        data = [gym.dataset(p,128,rng) for p in train]
        x,y,m = (torch.cat([d[i] for d in data]) for i in (1,2,3))
        held = [(p,gym.dataset(p),gym.analytic(p)) for p in cohorts['dev6'][:args.dev_roots]]
        model, visits, lr, batch = gym.Net('recur3'),3,.002,gym.BATCH
        def loss(ids):
            pred=model(x[ids]); valid=m[ids]
            return ((pred-y[ids]).square()*valid).sum()/valid.sum()
        def evaluate():
            rows=gym.evaluate(model,held)
            return dict(rows=rows,mean_regret=statistics.mean(r['regret'] for r in rows),
                        mean_expected_queries=statistics.mean(r['expected_queries'] for r in rows))
    else:
        import lbf20f_learning as gym
        train=gym.generate('train',6,args.train_roots,2026112515,allow_unsat=True)
        held=gym.generate('dev',8,args.dev_roots,2026112516,allow_unsat=True)
        assert not ({(c.n,c.domains,c.edges) for c in train}&{(c.n,c.domains,c.edges) for c in held})
        x,rel,y,m,roots=gym.frames(train)
        info=[gym.root_info(c) for c in held]
        model, visits, lr, batch = gym.Net('recur4'),4,.003,gym.BATCH
        def loss(ids):
            pred=model(x[ids],rel[ids]); valid=m[ids]
            return F.binary_cross_entropy_with_logits(pred[valid],y[ids][valid])
        def evaluate():
            rows=gym.evaluate(held,info,'lcv',model)
            return dict(rows=rows,mean_nodes=statistics.mean(r['nodes'] for r in rows),
                        statuses={k:sum(r['status']==k for r in rows) for k in ('solved','refuted','budget')})
    mem=MechanisticMemory(model.blocks[0],**memory_kwargs(args)) if args.mms_mib else None
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    gen=torch.Generator().manual_seed(args.seed+(7919 if args.gym=='20d' else 3001))
    stream=hashlib.sha256()
    wall,cpu=time.perf_counter(),time.process_time()
    count=0
    for step in range(args.updates):
        if args.cpu_seconds is not None and time.process_time()-cpu>=args.cpu_seconds: break
        ids=torch.randint(len(x),(batch,),generator=gen)
        stream.update(ids.numpy().tobytes())
        opt.zero_grad(set_to_none=True)
        with (mem.capture(ids,step,visits//2,visits) if mem else nullcontext()):
            value=loss(ids)
            if not bool(torch.isfinite(value)): raise FloatingPointError('nonfinite LBF loss')
            value.backward()
        opt.step();count+=1
    result=dict(updates=count,train_rows=len(x),draws=count*batch,
                executed_visits=count*batch*visits,train_wall_seconds=time.perf_counter()-wall,
                train_process_cpu_seconds=time.process_time()-cpu,
                stream_sha256=stream.hexdigest(),memory=mem.receipt() if mem else None)
    model.eval()
    result['development']=evaluate()
    return model,result


def transfer_fit(args):
    import lbf19_learned_transfer as gym
    # Reserve the full original test and validation lineage sets before
    # selecting train roots. Only the public validation subset is scored.
    splits=gym.make_splits(n_train=args.train_roots,n_val=80,n_test=100)
    items=gym.build_items(splits['train'],3,lambda rng:[rng.randint(*gym.TRAIN_L)],gym.KTRAIN,gym.SEED_TRAIN)
    val=gym.build_items(splits['val'][:args.dev_roots],2,lambda rng:[rng.randint(*gym.TRAIN_L)],gym.KTRAIN,gym.SEED_VAL)
    buckets=gym.bucket_by_L(items); lengths=sorted(buckets)
    rng=random.Random(args.seed*7919+13)
    model=gym.make_arm('rec_prior_hs')
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.updates)
    pool=SharedPagePool(args.mms_mib) if args.mms_mib else None
    # Separate row shapes by length, while sharing one physical budget.
    memories={}
    if pool:
        for length in lengths:
            for name,site in [('candidate',model.blocks[0].cand),('gate',model.blocks[0].gate)]:
                key=f'{length}/{name}'
                mem=MechanisticMemory(site,**memory_kwargs(args))
                mem.page_pool,mem.site_name=pool,key
                memories[key]=mem
    stream=hashlib.sha256()
    wall,cpu=time.perf_counter(),time.process_time()
    count=0; choices=dict.fromkeys(lengths,0)
    for step in range(args.updates):
        if args.cpu_seconds is not None and time.process_time()-cpu>=args.cpu_seconds: break
        length=rng.choice(lengths);bucket=buckets[length]
        ids=[rng.randrange(len(bucket)) for _ in range(64)]
        stream.update(json.dumps([length,ids],separators=(',',':')).encode())
        bt=gym.to_batch([bucket[i] for i in ids])
        site=('candidate','gate')[choices[length]%2]; choices[length]+=1
        mem=memories.get(f'{length}/{site}')
        opt.zero_grad(set_to_none=True)
        with (mem.capture(torch.tensor(ids),step,gym.KTRAIN//2,gym.KTRAIN) if mem else nullcontext()):
            pred=model(bt,gym.KTRAIN)
            tgt=bt.Y[:,1:gym.KTRAIN+1].transpose(0,1)
            value=F.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),tgt)
            value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step();sched.step();count+=1
    result=dict(updates=count,train_rows=len(items),draws=count*64,
        executed_visits=count*64*gym.KTRAIN,train_wall_seconds=time.perf_counter()-wall,
        train_process_cpu_seconds=time.process_time()-cpu,stream_sha256=stream.hexdigest(),
        memory=None if not pool else dict(budget_bytes=pool.budget_bytes,allocated_bytes=pool.used,
            peak_bank_bytes=pool.peak,evictions=pool.evictions,sites={k:m.receipt() for k,m in memories.items()}))
    model.eval()
    tp=positives=predicted=0
    with torch.no_grad():
        for bucket in gym.bucket_by_L(val).values():
            for start in range(0,len(bucket),64):
                bt=gym.to_batch(bucket[start:start+64])
                actual=model(bt,gym.KTRAIN)>=.5
                truth=bt.Y[:,1:gym.KTRAIN+1].transpose(0,1).bool()
                tp+=int((actual&truth).sum());positives+=int(truth.sum());predicted+=int(actual.sum())
    result['development']=dict(recall=tp/max(1,positives),precision=tp/max(1,predicted),
                                true_positive=tp,positives=positives,predicted=predicted,
                                threshold=.5,scope='public validation only')
    return model,result


def main(args):
    if min(args.updates,args.threads,args.train_roots,args.dev_roots)<1 or args.mms_mib<0:
        raise ValueError('invalid LBF MMS budget')
    if args.cpu_seconds is not None and args.cpu_seconds<=0: raise ValueError('invalid CPU budget')
    torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started,cpu=time.perf_counter(),time.process_time()
    model,record=(transfer_fit if args.gym=='19' else native_fit)(args)
    args.out.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),args.out/'model.pt')
    record.update(scope='development_only',gym=args.gym,seed=args.seed,threads=args.threads,
        total_wall_seconds=time.perf_counter()-started,total_process_cpu_seconds=time.process_time()-cpu,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        map_scope='SVD proxy, not a learned causal BPD map')
    (args.out/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='development'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gym',choices=('19','20d','20f'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=107)
    p.add_argument('--updates',type=int,default=400)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--train-roots',type=int,default=96)
    p.add_argument('--dev-roots',type=int,default=12)
    p.add_argument('--cpu-seconds',type=float)
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--topk',type=int,default=4)
    p.add_argument('--ratio',type=float,default=.2)
    p.add_argument('--refresh',type=int,default=128)
    p.add_argument('--max-age',type=int,default=32768)
    main(p.parse_args())
