from train import *

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);n=nr=0;mx=0.;npurge=0
    for seed in SEEDS[:3]:
        split0=split(seed);c={k:set(split0[k][2]) for k in ('train','cal','dev')};assert len(set.union(*c.values()))==sum(map(len,c.values()))==16
        data={t:eval_data(seed,split0['dev'][2],t) for t in HORIZONS}
        for t,(x,y,ctx) in data.items():
            for xx,yy in zip(x.tolist(),y.tolist()):
                assert sum(int(xx[j][2])<<j for j in range(4)) in c['dev'];v=0
                for (a,b,_),target in zip(xx,yy):v^=int(a)^int(b);assert v==target
        teacher=Model('tied','affine');p=Path('parents/affine_long')/f'affine-tied-{seed}.pt';teacher.load_state_dict(torch.load(p,weights_only=True));td=measure(teacher,data);nr+=len(HORIZONS)
        for folder in ('results','extended'):
            f=Path(folder);rs=[json.loads((f/f'{seed}-{a}.json').read_text()) for a in ARMS];assert len({r['sampler_sha256'] for r in rs if folder=='extended' or r['arm']!='budget_curriculum'})==1
            for r in rs:
                assert sha(p.read_bytes())==r['parent_sha256'];cp=f/r['checkpoint'];assert sha(cp.read_bytes())==r['checkpoint_sha256'];m=fresh(4,seed);m.load_state_dict(torch.load(cp,weights_only=True));assert sum(p.numel() for p in m.parameters())==r['params']==513;n+=1
                assert r['train_contexts']==split0['train'][2] and r['cal_contexts']==split0['cal'][2] and r['dev_contexts']==split0['dev'][2]
                if folder=='results':
                    assert td==r['teacher'];
                    for j,row in enumerate(r['prune']):
                        width=(64,32,16,8)[j];assert len(row['keep'])==width//2==len(set(row['keep'])) and all(0<=i<width for i in row['keep']);mx=max(mx,row['compact_max_error']);npurge+=1
                else:assert r['base_sha256']==sha((Path('results')/r['checkpoint']).read_bytes())
                assert measure(m,data)==r['eval'];nr+=len(HORIZONS)
                assert evaluate(m,data[4],remove=True)==r['removed'];nr+=1
                for b in (8,4):assert measure(quant(m,b),{t:data[t] for t in (8,12)})==r['quant'][str(b)];nr+=2
            costs={r['arm']:r['student_mac_proxy'] for r in rs};assert abs(costs['prog_salience']-costs['budget_curriculum'])<=B*8*cost(4)
    assert (n,nr,npurge)==(42,435,24) and mx<.002
    print(f'PASS: {n} student checkpoint hashes, {nr} evaluation replays including 15 teacher conditions, scalar XOR/ownership, sampler identities and {npurge} logged compactions; max recorded compact error {mx:.9g}.')
if __name__=='__main__':main()
