"""Train sparse-width recurrent models from scratch; distinct from pruning."""
from pilot import *
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out=Path('narrow');out.mkdir(exist_ok=True);rows=[]
    for seed in SEEDS[:3]:
        d=split(seed);x,y,_=d['train']
        for width in (2,4,8,16,32):
            torch.manual_seed(seed);m=Model('tied','affine');m.width=width;m.up=nn.ModuleList([nn.Linear(D+3,width)]);m.down=nn.ModuleList([nn.Linear(width,D)]);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0);g=torch.Generator().manual_seed(20000+seed)
            for _ in range(1200):
                ix=torch.randint(len(x),(128,),generator=g);z,_=m(x[ix]);loss=F.binary_cross_entropy_with_logits(z,y[ix]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            cp=out/f'{seed}-{width}.pt';torch.save(m.state_dict(),cp);r={'seed':seed,'width':width,'parameters':sum(p.numel() for p in m.parameters()),'checkpoint_sha256':sha(cp.read_bytes()),'dense':evaluate(m,d['dev']),'remove_mlp':evaluate(m,d['dev'],remove=True)};rows.append(r);print(seed,width,r['dense']['acc'],flush=True);(out/'results.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
if __name__=='__main__':main()
