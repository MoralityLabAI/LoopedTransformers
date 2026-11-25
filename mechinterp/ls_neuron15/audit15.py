"""Replay all candidate selection and verify the exact fixed-region endpoints."""
from search15 import *
from fractions import Fraction as Q
from collections import Counter

def check_state(st,db):
    c=np.asarray(st['codes']);s=np.asarray(st['scales']);o=st['offset'];assert c.shape==(52,) and np.max(abs(c))<=7 and np.all(c==c.astype(int));assert np.isfinite(s).all() and np.all(s>=parent.LO-1e-12) and np.all(s<=parent.HI+1e-10) and abs(o)<=8+1e-10
    w=parent.decoded(c,s,o,db);met=metric(w);assert all(abs(met[k]-st['scores'][k])<=1e-10 for k in met);return w

def evaluate(w,seed):return parent.old.evaluate([torch.tensor(w[k],dtype=torch.float32)[None] for k in NAMES],seed)
def audit_seed(seed):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);initial=json.loads(Path(f'initial/{seed}.json').read_text());assert initial==init(seed);ic=verify_scalar(initial['weights']);iev=evaluate(initial['weights'],seed);db=initial['down_bias'];si=np.array(initial['scales']);rows=[];common=None
    for arm in ARMS:
        path=Path(f'results/{arm}-{seed}.json');r=json.loads(path.read_text());assert r['seed']==seed and r['arm']==arm and r['initial_sha256']==hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest();key=arm.split('_')[0];ledger=r['ledger'];assert len(ledger)==13
        if common is None:common=ledger[0]
        else:assert common==ledger[0]
        prev=ledger[0]['state'];assert prev['codes']==initial['codes'];check_state(prev,db);calls=1;secondary=slots=edits=0;nfev=ledger[0]['fit']['nfev'];lpstats=[];improved=passes=0
        for j,row in enumerate(ledger[1:],1):
            assert row['round']==j;cs,names=candidates(np.array(prev['codes']));best=prev;move='incumbent'
            for anchor,ss in zip(row['anchors'],[np.array(prev['scales']),si]):
                rank,_,_=parent.profile(cs,ss);assert np.allclose(rank,anchor['profile'],rtol=1e-10,atol=1e-10);assert [f['slot'] for f in anchor['fits']]==shortlist(cs,rank);slots+=len(cs)
                for f in anchor['fits']:
                    i=f['slot'];assert f['move']==names[i];a,b=f['first'],f['second'];assert a['codes']==b['codes']==cs[i].tolist();check_state(a,db);check_state(b,db);calls+=1;secondary+=1;nfev+=f['fit']['nfev'];assert 1<=f['fit']['nfev']<=40
                    if arm.endswith('_lp'):
                        v=f['secondary'];assert a['scales'][1:3]==b['scales'][1:3];lpstats.append(v)
                        if v['success']:
                            x=np.array(v['x']);M,rhs=lp_matrix(cs[i],np.array(a['scales']),a['offset'],db,key=='bound');assert np.max(M@x-rhs)<=1e-6;assert abs(x[3]-v['objective'])<=1e-8;assert np.allclose([b['scales'][0],b['scales'][3],b['offset']],x[:3],rtol=0,atol=1e-10)
                        else:assert a==b
                    else:assert 1<=f['secondary']['nfev']<=40;nfev+=f['secondary']['nfev']
                    choose=b['scores'][key]<a['scores'][key]-1e-12;assert choose==f['selected_second'];win=b if choose else a;improved+=choose;passes+=a['scores']['bound']>TARGET and b['scores']['bound']<=TARGET
                    if win['scores'][key]<best['scores'][key]-1e-12:best=win;move=f"{names[i]}@{anchor['anchor']}"
            assert row['state']==best and row['move']==move;edits+=best['codes']!=prev['codes'];prev=best
        assert (calls,secondary,slots,nfev)==(r['first_calls'],r['secondary_calls'],r['slots'],r['ls_nfev'])==(73,72,4248,nfev);w=check_state(prev,db);assert r['final']==prev|{'weights':w};assert r['hamming']==sum(a!=b for a,b in zip(prev['codes'],initial['codes']));cert=verify_scalar(w);ev=evaluate(w,seed);out={'seed':seed,'arm':arm,'initial':ic,'final':cert,'initial_eval':iev,'final_eval':ev,'scores':prev['scores'],'weights':w,'receipt_sha256':sha(path),'slots':slots,'first_calls':calls,'secondary_calls':secondary,'ls_nfev':nfev,'lp_calls':len(lpstats),'lp_successes':sum(x['success'] for x in lpstats),'lp_max_primal':max([abs(x['primal_violation']) for x in lpstats if x['success']] or [0]),'lp_max_dual_gap':max([abs(x['dual_gap']) for x in lpstats if x['success']] or [0]),'lp_max_dual_stationarity':max([abs(x['dual_stationarity']) for x in lpstats if x['success']] or [0]),'secondary_improvements':improved,'secondary_bound_crossings':passes,'accepted_code_edits':edits,'hamming':r['hamming'],'seconds':r['seconds']};rows.append(out)
    return rows

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert len(list(Path('initial').glob('*.json')))==24 and len(list(Path('results').glob('*.json')))==96;control=verify_scalar(scalar_weights(4)[0]);assert control['verdict']=='CERTIFIED'
    with ProcessPoolExecutor(max_workers=4) as ex:rows=[r for rs in ex.map(audit_seed,SEEDS,chunksize=1) for r in rs]
    Path('certificates.json').write_text(json.dumps({'positive_control':control,'rows':rows},indent=2)+'\n');summ=[]
    def new(r):return r['initial']['verdict']=='POINT_VIOLATION' and r['final']['verdict']=='CERTIFIED'
    for a in ARMS:
        rs=[r for r in rows if r['arm']==a];d={'arm':a,'initial':dict(Counter(r['initial']['verdict'] for r in rs)),'final':dict(Counter(r['final']['verdict'] for r in rs)),'new':sum(new(r) for r in rs),'new_seeds':[r['seed'] for r in rs if new(r)],'T8':sum(r['final_eval']['8']['accuracy'] for r in rs)/24,'T128':sum(r['final_eval']['128']['accuracy'] for r in rs)/24,'T128_all':sum(r['final_eval']['128']['all_prefix'] for r in rs)/24,'bound_passes':sum(r['scores']['bound']<=TARGET for r in rs),'proxy_agreement':sum((r['scores']['bound']<=TARGET)==(r['final']['verdict']=='CERTIFIED') for r in rs),'median_bound':float(np.median([r['scores']['bound'] for r in rs]))}
        for k in ('slots','first_calls','secondary_calls','ls_nfev','lp_calls','lp_successes','secondary_improvements','secondary_bound_crossings','accepted_code_edits','seconds'):d[k]=sum(r[k] for r in rs)
        for k in ('lp_max_primal','lp_max_dual_gap','lp_max_dual_stationarity'):d[k]=max(r[k] for r in rs)
        summ.append(d)
    comparisons=[]
    for a,b in [('max_lp','max_ls'),('bound_lp','max_lp')]:
        aa={r['seed'] for r in rows if r['arm']==a and new(r)};bb={r['seed'] for r in rows if r['arm']==b and new(r)};w,l=len(aa-bb),len(bb-aa);n=w+l;p=min(Q(1),Q(2*sum(math.comb(n,j) for j in range(min(w,l)+1)),2**n));comparisons.append({'candidate':a,'control':b,'candidate_only':sorted(aa-bb),'control_only':sorted(bb-aa),'net':len(aa)-len(bb),'p_exact':str(p),'p_bonferroni':str(min(Q(1),2*p)),'gate':len(aa)>=12 and len(aa)-len(bb)>=6})
    result={'status':'PASS','study':'PILOT_ONLY','n_seeds':24,'endpoints':96,'exact_verifier_calls':121,'numerical_horizons':240,'summary':summ,'comparisons':comparisons,'certificate_sha256':sha('certificates.json')};Path('audit_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
