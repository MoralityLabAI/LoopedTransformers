"""Replay all weights/outcomes and exact counterexamples; certify physical compaction."""
from verify import *
from repair import codes

def check_point(w,rec):
    p,a,b,n=rec['p'],*rec['input'];assert p^a^b==rec['target'];h=Q(rec['h']);o=wf(w,box(h),a,b,n);assert o.out()==rec['output'];lo,hi=map(Q,rec['target_inner']);assert o.lo>hi or o.hi<lo

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);regs=[(-Q(1,8),Q(1,8)),(Q(7,8),Q(9,8))];reps=0;checks=0;certs=0
    v=json.loads(Path('verification.json').read_text());rep=json.loads(Path('repair_results.json').read_text());red=json.loads(Path('reduced_margin_results.json').read_text());orig={(r['arm'],r['seed']):r for r in v['rows']};fixed={(r['arm'],r['seed']):r for r in rep};reduced={(r['arm'],r['seed']):r for r in red};assert len(orig)==len(fixed)==len(reduced)==36
    for seed in SEEDS:
        g=torch.Generator().manual_seed(100000+seed);digest=hashlib.sha256()
        for _ in range(STEPS):digest.update(torch.randint(2,(BATCH,T,3),generator=g).float().numpy().tobytes())
        for arm in ARMS:
            f=Path(f'results/{arm}-{seed}.json');r=json.loads(f.read_text());key=(arm,seed);assert r['sampler_sha256']==digest.hexdigest();assert sha(f)==orig[key]['weights_sha256']==fixed[key]['parent_sha256'];p=[torch.tensor(r['fp32_weights'][k])[None] for k in NAMES];q=params(p,4);assert {k:t[0].tolist() for k,t in zip(NAMES,q)}==r['q4_weights'];assert evaluate(p,seed)==r['fp32'] and evaluate(q,seed)==r['q4'];reps+=10
            assert orig[key]['fixed']['verdict']=='POINT_VIOLATION';check_point(r['q4_weights'],orig[key]['fixed']['witness']);checks+=1
            w=fixed[key]['q4_weights'];meta=fixed[key]['meta'];c,_=codes(r['q4_weights']);assert {k:t.tolist() for k,t in c.items()}==meta['codes']
            for k,s in zip(('skip','up','up_bias','down'),meta['scales']):assert torch.equal(torch.tensor(w[k]),torch.tensor(meta['codes'][k],dtype=torch.float32)*torch.tensor(s))
            assert w['down_bias']==r['q4_weights']['down_bias'];p=[torch.tensor(w[k])[None] for k in NAMES];assert evaluate(p,seed)==fixed[key]['q4'];reps+=5
            assert fixed[key]['certificate']['verdict']=='POINT_VIOLATION';check_point(w,fixed[key]['certificate']['witness']);checks+=1
            for ep,c in reduced[key]['certificates'].items():
                if c['verdict']=='POINT_VIOLATION':check_point(w,c['witness']);checks+=1
                else:assert exact(w,regs,Q(ep))==c;certs+=1
    assert exact(positive(),regs,Q(1,32))==v['positive_control'];certs+=1
    assert (reps,checks,certs)==(540,214,3)
    w=fixed[('bound',107)]['q4_weights'];ids=[i for i,v in enumerate(w['down']) if v!=0];assert ids==[0,1,2];compact=dict(w);compact['up']=[w['up'][i] for i in ids];compact['up_bias']=[w['up_bias'][i] for i in ids];compact['down']=[w['down'][i] for i in ids];count=sum(torch.tensor(v).numel() for v in compact.values())+3;assert count==27
    cert=exact(compact,regs,Q(1,256));assert cert['verdict']=='CERTIFIED';p=[torch.tensor(compact[k])[None] for k in NAMES];assert evaluate(p,107)==fixed[('bound',107)]['q4'];g=torch.Generator().manual_seed(77107);N=2048;h=torch.zeros(1,N);par=torch.zeros(1,N,dtype=torch.long);bad=0;maxerr=0.
    with torch.no_grad():
        for _ in range(512):
            x=torch.randint(2,(1,N,3),generator=g).float();par^=x[:,:,:2].long().sum(-1).remainder(2);e=(torch.randint(2,(1,N),generator=g)*2-1).float()/256;h=step(p,h,x)+e;bad+=int(((h>.5)!=par.bool()).sum());maxerr=max(maxerr,float((h-par).abs().max()))
    assert bad==0 and maxerr<=.125
    result=dict(verification='PASS',numeric_horizon_replays=reps,point_violations_replayed=checks,certificate_replays=certs,compact_slots=count,compact_width=3,compact_weights=compact,compact_certificate=cert,stress={'histories':N,'visits':512,'transitions':N*512,'errors':bad,'max_region_error':maxerr,'disturbance':'1/256 random sign'},torch_version=torch.__version__)
    Path('audit_results.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',reps,'numeric replays;',checks,'exact point violations;',certs,'certificate replays; compact',count,'slots;',cert['leaves'],'leaves;',result['stress'],flush=True)
if __name__=='__main__':main()
