"""Check code/scale decoding, paired sampler hashes, scalar labels, and stored outcomes."""
from train08 import *
from certify08 import certify,REGS,EPS
from verify import wf,box,Q,positive

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);ledger=json.loads(Path('certificates.json').read_text());certs={(r['arm'],r['seed']):r for r in ledger['rows']};n=0;reps=0;pv=0;cert=0;unresolved=0;labels=0;maxbounddiff=0.
    assert len(certs)==len(SEEDS)*len(ARMS);assert certify(positive())==ledger['positive_control'];cert+=1
    for seed in SEEDS:
        g=torch.Generator().manual_seed(8010000+seed);digest=hashlib.sha256()
        for _ in range(STEPS):digest.update(torch.randint(2,(BATCH,T,3),generator=g).float().numpy().tobytes())
        g=torch.Generator().manual_seed(8030000+seed)
        for t in (4,8,16,32,64,128):
            x=torch.randint(2,(1,512,t,3),generator=g);y=x[:,:,:,:2].sum(-1).cumsum(-1).remainder(2)
            for xx,yy in zip(x[0].tolist(),y[0].tolist()):
                v=0
                for (a,b,_),target in zip(xx,yy):v^=a^b;assert v==target;labels+=1
        for arm in ARMS:
            f=Path(f'results/{arm}-{seed}.json');r=json.loads(f.read_text());c=certs[arm,seed];assert r['sampler_sha256']==digest.hexdigest();assert sha(f)==c['weight_receipt_sha256'];p=[torch.tensor(r['fp32_weights'][k],dtype=torch.float32)[None] for k in NAMES];q=[torch.tensor(r['q4_weights'][k],dtype=torch.float32)[None] for k in NAMES]
            for i in QIDS:
                code=torch.tensor(r['codes'][NAMES[i]],dtype=torch.int64);s=torch.tensor(r['scales'][NAMES[i]],dtype=torch.float32);assert s>0 and code.abs().max()<=7 and torch.equal(q[i][0],code.float()*s);assert torch.equal(code,(p[i][0]/s).clamp(-7,7).round().long())
            for i in (1,5):assert torch.equal(p[i],q[i])
            assert eval08(p,seed)==r['fp32'] and eval08(q,seed)==r['q4'];reps+=12;n+=1
            h,x,y=centers(1);bd=float(enclosure(q,h,x,y).max());diff=abs(bd-r['float_bound_max']);maxbounddiff=max(maxbounddiff,diff);assert diff<1e-5
            rec=c['certificate']
            if rec['verdict']=='POINT_VIOLATION':
                z=rec['witness'];a,b,nu=z['input'];assert z['target']==z['p']^a^b;out=wf(r['q4_weights'],box(Q(z['h'])),a,b,nu);assert out.out()==z['output'];lo,hi=map(Q,z['target_inner']);assert out.lo>hi or out.hi<lo;pv+=1
            else:
                assert certify(r['q4_weights'])==rec
                if rec['verdict']=='CERTIFIED':cert+=1
                else:unresolved+=1
    assert n==48 and reps==576 and pv+cert+unresolved==49
    result={'verdict':'PASS','checkpoint_receipts':n,'numerical_replays':reps,'scalar_label_checks':labels,'point_violations_replayed':pv,'certificate_replays_including_positive_control':cert,'unresolved_replays':unresolved,'max_float_bound_replay_difference':maxbounddiff}
    Path('audit_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
