"""Verify scalar parity, context ownership, and checkpoint-backed outcomes."""
from pilot import *

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);count=0
    for seed in SEEDS[:3]:
        d=split(seed);c=[set(d[a][2]) for a in ('train','cal','dev')];assert sum(map(len,c))==16 and len(set.union(*c))==16
        for x,y,_ in d.values():
            for bits,labels in zip(x.tolist(),y.tolist()):
                v=0
                for (a,b,_),label in zip(bits,labels):
                    v^=int(a)^int(b);assert v==label
        for folder in ('affine','affine_long','attention','fork','narrow'):
            raw=json.loads((Path(folder)/'results.json').read_text());rs=raw if isinstance(raw,list) else raw['runs'];assert len(rs)=={'affine':12,'affine_long':12,'attention':12,'fork':6,'narrow':15}[folder]
            for r in rs:
                if r['seed']!=seed:continue
                if folder=='narrow':
                    p=Path(folder)/f"{seed}-{r['width']}.pt";m=Model('tied','affine');m.width=r['width'];m.up=nn.ModuleList([nn.Linear(D+3,m.width)]);m.down=nn.ModuleList([nn.Linear(m.width,D)])
                elif folder=='fork':p=Path(folder)/f"{seed}-{r['arm']}.pt";m=Model(r['arm'],'affine');assert r['initial_logit_max_difference']==0.
                else:p=Path(folder)/f"{r['mode']}-{r['arm']}-{seed}.pt";m=Model(r['arm'],r['mode'])
                assert sha(p.read_bytes())==r['checkpoint_sha256'];m.load_state_dict(torch.load(p,weights_only=True));assert evaluate(m,d['dev'])['correct']==r['dense']['correct'];assert evaluate(m,d['dev'],remove=True)['correct']==r['remove_mlp']['correct'];count+=1
    assert count==57
    print('PASS: scalar oracle, context ownership, 57 checkpoint hashes and 114 dense/complete-removal correctness replays.')
if __name__=='__main__':main()
