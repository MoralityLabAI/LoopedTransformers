"""Dense-function-matched untie control and trained sparse width sweep."""
from pilot import *
import copy

def clone(source):
    target=Model('untied','affine');target.init.data.copy_(source.init.data);target.read.load_state_dict(source.read.state_dict());target.skip.load_state_dict(source.skip.state_dict())
    for u,d in zip(target.up,target.down):u.load_state_dict(source.up[0].state_dict());d.load_state_dict(source.down[0].state_dict())
    return target

def tune(model,data,seed,steps):
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0);x,y,_=data;g=torch.Generator().manual_seed(31000+seed)
    for _ in range(steps):
        ids=torch.randint(len(x),(128,),generator=g);z,_=model(x[ids]);loss=F.binary_cross_entropy_with_logits(z,y[ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()

def measure(model,d,seed):
    model.eval();x,y,_=d['cal'];z,a=model(x,trace=True);grad=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,y),a);sal=torch.stack([(v*g).abs().sum(0) for v,g in zip(a,grad)]).detach();rows=[]
    for kind in ('global','local','random'):
        for k in COUNTS:
            if k>model.width:continue
            for rep in range(3 if kind=='random' and k not in (0,model.width) else 1):
                m=masks(sal,model.banks,k,kind,41000+seed+rep);unique=int(torch.stack(m).amax(0).sum()) if model.banks==1 and kind=='local' else k*model.banks
                rows.append({'kind':kind,'keep':k,'unique':unique,'rep':rep,**evaluate(model,d['dev'],m)})
    return {'dense':evaluate(model,d['dev']),'remove_mlp':evaluate(model,d['dev'],remove=True),'rows':rows}

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out=Path('fork');out.mkdir(exist_ok=True);rows=[]
    for seed in SEEDS[:3]:
        d=split(seed);src=Model('tied','affine');p=Path('affine')/f'affine-tied-{seed}.pt';src.load_state_dict(torch.load(p,weights_only=True));target=clone(src)
        with torch.no_grad():delta=float((src(d['cal'][0])[0]-target(d['cal'][0])[0]).abs().max())
        assert delta==0.,delta
        for name,m in [('tied',copy.deepcopy(src)),('untied',target)]:
            tune(m,d['train'],seed,600);cp=out/f'{seed}-{name}.pt';torch.save(m.state_dict(),cp);r={'seed':seed,'arm':name,'initial_logit_max_difference':delta,'parent_checkpoint_sha256':sha(p.read_bytes()),'checkpoint_sha256':sha(cp.read_bytes()),**measure(m,d,seed)};rows.append(r)
            print(seed,name,r['dense']['acc'],[(a['keep'],round(a['acc'],4)) for a in r['rows'] if a['kind']=='global'],flush=True)
            (out/'results.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
if __name__=='__main__':main()
