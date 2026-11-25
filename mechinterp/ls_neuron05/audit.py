"""Independent scalar label and checkpoint-backed replay audit; no retraining."""
from experiment import *

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    n=0;dyncount=0;qeval=0;scalar=0;maxerr=0.
    for seed in SEEDS:
        ctx=[set(split(seed)[k][2]) for k in ('train','cal','dev')];assert len(set.union(*ctx))==sum(map(len,ctx))==16
        for part,t,count,domain in [('cal',16,128,1500000),('dev',16,128,1600000)]+[('dev',t,512,1800000) for t in H]:
            x,y,cs=data(seed,part,t,count,domain)
            for bits,labels in zip(x.tolist(),y.tolist()):
                assert sum(int(bits[j][2])<<j for j in range(4)) in cs
                v=0
                for (a,b,_),p in zip(bits,labels):v^=int(a)^int(b);assert v==p;scalar+=1
        for arm in PARENT_ARMS:
            m,dig=load_parent(seed,arm);r=json.loads(Path(f'diag/{seed}-{arm}.json').read_text());assert dig==r['parent_sha'];n+=1
            for k,net in [('fp',m),('q4',quant(m,4))]:
                now=diagnostics(net,seed);assert now==r[k];dyncount+=1;maxerr=max(maxerr,now['jacobian_max_error'])
        samplers=set()
        for arm in QARMS:
            r=json.loads(Path(f'qat/{seed}-{arm}.json').read_text());p=Path(f'qat/{seed}-{arm}.pt');assert sha(p.read_bytes())==r['checkpoint_sha'];m=fresh(4,seed);m.load_state_dict(torch.load(p,weights_only=True));assert sum(p.numel() for p in m.parameters())==513;n+=1;samplers.add(r['sampler_sha'])
            parent,dig=load_parent(seed,'curriculum');assert dig==r['parent_sha'];assert qa_eval(parent,seed)==r['initial'];assert qa_eval(m,seed)==r['final'];qeval+=40
            for k,net in [('fp_dynamics',m),('q4_dynamics',quant(m,4))]:
                now=diagnostics(net,seed);assert now==r[k];dyncount+=1;maxerr=max(maxerr,now['jacobian_max_error'])
        assert len(samplers)==1
    assert (n,dyncount,qeval)==(30,60,360)
    print(f'PASS: {n} checkpoint hashes, {dyncount} dynamics replays, {qeval} initial/final precision-horizon replays, {scalar} scalar XOR prefix labels, matched sampler hashes, max analytic/autograd Jacobian error {maxerr:.9g}')
if __name__=='__main__':main()
