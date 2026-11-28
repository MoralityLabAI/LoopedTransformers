"""Replay training and measure complete MLP removal, including output bias."""
from pilot import *

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
    for seed in SEEDS:
        for arm,(banks,width) in {'tied':(1,W),'untied':(T,W),'budget':(T,W//T)}.items():
            torch.manual_seed(seed);model=Model(banks,width);opt=torch.optim.AdamW(model.parameters(),lr=.003)
            train=worlds(1000+seed,4096);cal=worlds(2000+seed,256);test=worlds(3000+seed,512);g=torch.Generator().manual_seed(4000+seed)
            for _ in range(240):
                ix=torch.randint(len(train[0]),(64,),generator=g);batch=tuple(x[ix] for x in train);z=logits(model,batch)
                loss=F.binary_cross_entropy_with_logits(z,torch.tensor([1.,0.]).expand_as(z));opt.zero_grad();loss.backward();opt.step()
            model.eval();sal=ranks(model,cal).detach()
            with torch.no_grad():
                r=dict(seed=seed,arm=arm,dense=outcomes(logits(model,test)),zero_neurons_bias_retained=outcomes(logits(model,test,[torch.zeros(width)]*T)),absolute_counts={})
                handles=[m.register_forward_hook(lambda m,i,o: torch.zeros_like(o)) for m in model.down]
                r['zero_mlp']=outcomes(logits(model,test))
                for h in handles:h.remove()
                for k in (1,2,3,4,6,8,16):
                    r['absolute_counts'][str(k)]=outcomes(logits(model,test,masks_for(sal,banks,k,'shared',0)))
            rows.append(r);print(seed,arm,[(k,sum(r[k])/1024) for k in ('dense','zero_neurons_bias_retained','zero_mlp')],flush=True)
    Path('zero_control_results.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
if __name__=='__main__':main()
