"""Exploratory per-tensor symmetric weight fake-quantization and length transfer."""
from pilot import *
import copy

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
    for seed in SEEDS[:3]:
        for width in (2,4,8,16,32,64):
            m=Model('tied','affine');m.width=width;m.up=nn.ModuleList([nn.Linear(D+3,width)]);m.down=nn.ModuleList([nn.Linear(width,D)])
            path=Path('narrow')/f'{seed}-{width}.pt' if width<64 else Path('affine')/f'affine-tied-{seed}.pt';m.load_state_dict(torch.load(path,weights_only=True));m.eval()
            for bits in (32,8,4,2):
                q=copy.deepcopy(m)
                if bits!=32:
                    with torch.no_grad():
                        for p in q.parameters():
                            maxval=p.abs().max();levels=2**(bits-1)-1
                            if maxval>0:p.copy_((p/maxval*levels).round().clamp(-levels,levels)*maxval/levels)
                for steps in (2,4,6,8):
                    data=split(seed,steps)['dev'];rows.append({'seed':seed,'width':width,'bits':bits,'steps':steps,'checkpoint_sha256':sha(path.read_bytes()),**evaluate(q,data)})
    Path('precision_results.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
if __name__=='__main__':main()
