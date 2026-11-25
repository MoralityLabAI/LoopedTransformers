"""Adaptive multi-anchor search, preserving the original registered endpoint."""
from repair13 import *
from concurrent.futures import ProcessPoolExecutor
EXT_ARMS=('continuous_restart','single_restart','paired_restart')

def run_restart(seed,arm):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();c,s,o,db=setup(seed,15);ic=c.copy();si=s.copy();oi=o;initial=decoded(c,s,o,db);s,o,rec=fit(c,s,o);fit_count=rec['nfev'];search=0;ledger=[dict(round=0,move='fit',fit=rec,codes=c.tolist(),scales=s.tolist(),offset=o)]
    for r in range(1,ROUNDS+1):
        cand,names=proposals(c,arm=='paired_restart') if arm!='continuous_restart' else (c[None],['noop']);best=(rec['loss'],c.copy(),s.copy(),o,rec,'noop');short=[];n=0
        for anchor,ss in [('current',s),('initial',si)]:
            l,coef,offs=profile(cand,ss);n+=len(cand)
            shortlist=list(np.argsort(l,kind='stable')[:4]);shortlist=list(dict.fromkeys([0]+shortlist))
            for j in shortlist:
                proposal=ss.copy();proposal[[0,3]]=coef[j];ns,no,nrec=fit(cand[j],proposal,float(offs[j]));fit_count+=nrec['nfev'];short.append(dict(anchor=anchor,index=int(j),move=names[j],profile_loss=float(l[j]),fit=nrec))
                if nrec['loss']<best[0]-1e-12:best=(nrec['loss'],cand[j].copy(),ns,no,nrec,names[j]+'@'+anchor)
        _,c,s,o,rec,move=best;search+=n;ledger.append(dict(round=r,move=move,candidates=n,shortlist=short,fit=rec,codes=c.tolist(),scales=s.tolist(),offset=o))
    final=decoded(c,s,o,db);row=dict(seed=seed,dose=15,arm=arm,initial=dict(codes=ic.tolist(),scales=si.tolist(),offset=oi,weights=initial,**measure(initial,seed)),final=dict(codes=c.tolist(),scales=s.tolist(),offset=o,weights=final,**measure(final,seed)),hamming=int(np.count_nonzero(c!=ic)),code_l1=int(abs(c-ic).sum()),search_evaluations=search,fit_nfev=fit_count,ledger=ledger,seconds=time.monotonic()-start)
    out=Path('restarts');out.mkdir(exist_ok=True);(out/f'15-{arm}-{seed}.json').write_text(json.dumps(row,separators=(',',':'))+'\n');print(seed,arm,'bound',round(row['final']['float_bound'],5),'moves',row['hamming'],'secs',round(row['seconds'],1),flush=True)

def job(args):run_restart(*args)
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(job,[(s,a) for s in SEEDS for a in EXT_ARMS],chunksize=1))
