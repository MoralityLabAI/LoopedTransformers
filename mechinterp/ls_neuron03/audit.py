"""Replay saved checkpoint predictions and audit the greedy selection ledger."""
from recover import *

def audit(base,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);ncp=nreplay=ncand=0;maxerr=0.
    for seed in SEEDS[:3]:
        r=json.loads((out/f'results-{seed}.json').read_text());sel=json.loads((out/f'selection-{seed}.json').read_text());p=base/'affine_long'/f'affine-tied-{seed}.pt';assert sha(p.read_bytes())==r['parent_sha256']==sel['parent_sha256'];parent=Model('tied','affine');parent.load_state_dict(torch.load(p,weights_only=True));d=split(seed);x,y,_=d['cal'];x,y=x[:512],y[:512];active=list(range(W));data={k:split(seed,k)['dev'] for k in (2,4,6,8)}
        for row in sel['candidate_ledger']:
            assert row['active']==active and len(row['losses'])==len(active)
            j=min(range(len(active)),key=lambda j:(row['losses'][j],active[j]));assert row['removed']==active[j]
            # Direct, non-batched replay of selected deletion at each width.
            mask=torch.zeros(W);mask[active]=1;mask[row['removed']]=0
            with torch.no_grad():z,_=parent(x,masks=[mask]*T);v=float(F.binary_cross_entropy_with_logits(z,y))
            err=abs(v-row['losses'][j]);assert err<1e-4;maxerr=max(maxerr,err);ncand+=len(active);active.pop(j);assert active==sel['greedy_path'][str(len(active))]
        for arm in r['arms']:
            m=fresh(4,seed)
            for row in arm['rows']:
                p=out/row['checkpoint'];assert sha(p.read_bytes())==row['checkpoint_sha256'];m.load_state_dict(torch.load(p,weights_only=True));assert sum(p.numel() for p in m.parameters())==513;ncp+=1
                for k,old in row['eval'].items():
                    now=measure(m,parent,data[int(k)]);assert now['correct']==old['correct'];assert abs(now['logit_corr']-old['logit_corr'])<1e-6;nreplay+=1
            # Initial and final state equality for frozen groups.
            a,b=[torch.load(out/arm['rows'][i]['checkpoint'],weights_only=True) for i in (0,-1)]
            if arm['arm']=='freeze_mlp':assert all(torch.equal(a[k],b[k]) for k in a if k.startswith(('up.','down.')))
            if arm['arm']=='freeze_bypass':assert all(torch.equal(a[k],b[k]) for k in a if k.startswith(('skip.','read.','init')))
        assert len({tuple(q['sampler_sha256'] for q in a['rows']) for a in r['arms']})==1
    assert ncp==96 and nreplay==288 and ncand==6237,(ncp,nreplay,ncand)
    print(f'PASS: {ncp} checkpoints, {nreplay} correctness/logit-correlation replays, {ncand} candidate ledger entries, 189 independently replayed chosen removals, max loss error {maxerr:.9g}; frozen weights and sampler identities verified.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();audit(a.base,a.out)
