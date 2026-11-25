"""Exact certificate and replay of all LS12 endpoints, independent matrix audit."""
from construction import *
from experiment import SEEDS,DOSES,ARMS,STEPS,QIDS,NAMES,init,decode,centers,enclosure,evaluate,torch,hashlib
from rational import box,I
from verify import wf
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools

def verify_scalar(w):
    cs=list(product(range(4),(0,1),(0,1),(0,1)));c=torch.tensor(cs,dtype=torch.float64);h=c[:,0,None]-2+torch.linspace(-.125,.125,257,dtype=torch.float64);x=c[:,1:,None].transpose(1,2).expand(-1,257,-1);y=c[:,:3].sum(-1).remainder(4)-2
    from parent_train import step
    p=[torch.tensor(w[k],dtype=torch.float64)[None] for k in NAMES]
    with torch.no_grad():val=step(p,h.reshape(1,-1),x.reshape(1,-1,3)).reshape(len(cs),257)
    err=(val-y[:,None]).abs();idx,j=divmod(int(err.argmax()),257);p,a,b,n=cs[idx];h=Q(p)-2-R+2*R*Q(j,256);inner=R-EPS;t=Q((p+a+b)%4)-2
    def witness(p,a,b,n,h,v):
        t=Q((p+a+b)%4)-2;return {'verdict':'POINT_VIOLATION','p':p,'input':[a,b,n],'h':str(h),'out':v.out(),'target_inner':[str(t-inner),str(t+inner)]}
    v=wf(w,box(h),a,b,n)
    if v.lo>t+inner or v.hi<t-inner:return witness(p,a,b,n,h,v)
    # Rigorous C^2 interpolation bound, |GELU''|<=1.
    C=Q(1,5)*sum(abs(Q(d))*Q(row[0])**2 for row,d in zip(w['up'],w['down']));count=leaves=0;records=[]
    for p,a,b,n in cs:
        t=Q((p+a+b)%4)-2;tl,tu=t-inner,t+inner;lo,hi=Q(p)-2-R,Q(p)-2+R;cache={};local=0;mn=mx=None
        # Prefer direct affine-ReLU enclosure when activation signs are fixed.
        zends=[[sum(Q(u)*v for u,v in zip(row,(h,a,b,n)))+Q(bias) for row,bias in zip(w['up'],w['up_bias'])] for h in (lo,hi)]
        sep=[min(abs(z0),abs(z1)) if z0*z1>=0 else Q(0) for z0,z1 in zip(*zends)]
        if all(z0*z1>=0 for z0,z1 in zip(*zends)):
            relu=[]
            for h,z in zip((lo,hi),zends):relu.append(h+sum(Q(u)*v for u,v in zip(w['skip'],(h,a,b,n)))+Q(w['skip_bias'][0])+BETA*(sum(Q(u)*max(Q(0),v) for u,v in zip(w['down'],z))+Q(w['down_bias'][0])))
            E=BETA*sum(abs(Q(v))*(Q(1,1024) if s>=4 else Q(2,5)) for v,s in zip(w['down'],sep));L=min(relu)-E;U=max(relu)+E
            if L>=tl and U<=tu:
                count+=1;leaves+=1;records.append({'p':p,'input':[a,b,n],'method':'relu_tail','leaves':1,'enclosure':[str(L),str(U)]});continue
        def ev(h):
            if h not in cache:cache[h]=wf(w,box(h),a,b,n)
            return cache[h]
        todo=[(lo,hi,0)]
        while todo:
            l,u,d=todo.pop();a0,a1=ev(l),ev(u);count+=1
            for h,v in ((l,a0),(u,a1)):
                if v.lo>tu or v.hi<tl:return witness(p,a,b,n,h,v)
            m=C*(u-l)**2/8;L=min(a0.lo,a1.lo)-m;U=max(a0.hi,a1.hi)+m
            if L>=tl and U<=tu:
                leaves+=1;local+=1;mn=L if mn is None else min(mn,L);mx=U if mx is None else max(mx,U);continue
            if d>=14 or count>20000:return {'verdict':'UNRESOLVED','p':p,'input':[a,b,n],'boxes':count}
            mid=(l+u)/2;todo.extend([(mid,u,d+1),(l,mid,d+1)])
        records.append({'p':p,'input':[a,b,n],'method':'interpolation','leaves':local,'enclosure':[str(mn),str(mx)]})
    return {'verdict':'CERTIFIED','cases':records,'boxes':count,'leaves':leaves,'epsilon':str(EPS)}

def check_matrix(mdl):
    d=mdl['d'];centers=mdl['centers'];m=len(centers);w=mdl['w'];rows=0
    for p,a,b,n in product(range(m),(0,1),(0,1),(0,1)):
        c=centers[p];x=c+[Q(a),Q(b),Q(n)];expected=centers[(p+a+b)%m]
        # Independently compute affine ReLU image and slope by actual matrices.
        zs=[sum(Q(u)*v for u,v in zip(row,x))+bias for row,bias in zip(mdl['U'],mdl['b'])]
        rad=[R*sum(abs(v) for v in row[:d]) for row in mdl['U']];assert all(z-r>=0 or z+r<=0 for z,r in zip(zs,rad))
        slopes=[int(z>=0) for z in zs]
        for i in range(d):
            val=c[i]+sum(Q(v)*q for v,q in zip(mdl['skip'][i],x))+mdl['offset'][i]+BETA*sum(v*max(z,0) for v,z in zip(mdl['V'][i],zs));assert val==expected[i]
            for j in range(d):assert int(i==j)+mdl['skip'][i][j]+BETA*sum(v*s*row[j] for v,s,row in zip(mdl['V'][i],slopes,mdl['U']))==0
        rows+=1
    return rows

def job(path):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r=json.loads(path.read_text());w=r['q4_weights'];q=[torch.tensor(w[k],dtype=torch.float32)[None] for k in NAMES];p=[torch.tensor(r['fp32_weights'][k],dtype=torch.float32)[None] for k in NAMES]
    for i in QIDS:
        c=torch.tensor(r['codes'][NAMES[i]],dtype=torch.float32);s=torch.tensor(r['scales'][NAMES[i]],dtype=torch.float32);assert c.abs().max()<=7 and s>0 and torch.equal(q[i][0],c*s)
        if r['step']==0 or r['arm']=='full':assert torch.equal(c,(p[i][0]/s).clamp(-7,7).round())
    for i in (1,5):assert torch.equal(p[i],q[i])
    if r['step']==0:assert all(torch.allclose(a[0],b,atol=1e-6,rtol=1e-6) for a,b in zip(p,init(r['seed'],r['dose'])[0]))
    if r['arm']=='scale' and r['step']==800:
        initial=json.loads(path.with_name(f"{r['dose']}-scale-{r['seed']}-0.json").read_text());assert r['codes']==initial['codes'];assert all(r['fp32_weights'][NAMES[i]]==initial['fp32_weights'][NAMES[i]] for i in QIDS)
    assert evaluate(q,r['seed'])==r['eval'];h,x,y=centers(1);assert abs(float(enclosure(q,h,x,y).max())-r['bound'])<1e-4
    c=verify_scalar(w);return {k:r[k] for k in ('dose','arm','seed','step')}|{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'certificate':c}

def stress(mdl,seed):
    d=mdl['d'];cs=torch.tensor(mdl['centers'],dtype=torch.float32);n=512;T=256;h=cs[0].expand(n,-1).clone();p=torch.zeros(n,dtype=torch.long);g=torch.Generator().manual_seed(12040000+seed);U=torch.tensor(mdl['U'],dtype=torch.float32);b=torch.tensor(mdl['b'],dtype=torch.float32);V=torch.tensor(mdl['V'],dtype=torch.float32);skip=torch.tensor(mdl['skip'],dtype=torch.float32);bias=torch.tensor(mdl['offset'],dtype=torch.float32);errors=0;maxerr=0.
    with torch.no_grad():
        for _ in range(T):
            x=torch.randint(2,(n,3),generator=g);p=(p+x[:,:2].sum(-1))%len(cs);z=torch.cat((h,x.float()),1);h=h+z@skip.T+bias+.2*(torch.nn.functional.gelu(z@U.T+b)@V.T);h+=((torch.randint(2,h.shape,generator=g)*2-1).float()/32);pred=(h[:,None,:]-cs[None,:,:]).abs().amax(-1).argmin(1);errors+=int((pred!=p).sum());maxerr=max(maxerr,float((h-cs[p]).abs().max()))
    assert errors==0 and maxerr<=float(R)
    return {'transitions':n*T,'errors':errors,'max_region_error':maxerr}

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);models={f'scalar_{m}':scalar(m) for m in (2,3,4,8,16,32,64)}|{f'binary_L{L}_k{k}':binary(k,L) for L,km in ((1,2),(4,6)) for k in range(1,km+1)}
    cons={};sc={};checks=0
    for j,(name,mdl) in enumerate(models.items()):cons[name]=certify(mdl);checks+=check_matrix(mdl);sc[name]=stress(mdl,j);print(name,cons[name]['epsilon'],sc[name],flush=True)
    Path('construction_results.json').write_text(json.dumps(cons,indent=2)+'\n')
    paths=sorted(Path('results').glob('*.json'));expected={(d,a,s,t) for d,a,s in product(DOSES,ARMS,SEEDS) for t in ((0,) if a=='frozen' else (0,800))};assert len(paths)==len(expected)==120
    w,_=scalar_weights(4);control=verify_scalar(w);assert control['verdict']=='CERTIFIED'
    # The three modes must start from the same physical checkpoint.
    for d,s in product(DOSES,SEEDS):
        r=[json.loads(Path(f'results/{d}-{a}-{s}-0.json').read_text()) for a in ARMS];assert all(q['q4_weights']==r[0]['q4_weights'] and q['codes']==r[0]['codes'] for q in r)
    with ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(job,paths,chunksize=1))
    assert {(r['dose'],r['arm'],r['seed'],r['step']) for r in rows}==expected
    Path('certificates.json').write_text(json.dumps({'positive_control':control,'rows':rows},indent=2)+'\n');receipt={'verdict':'PASS','matrix_transition_checks':checks,'constructions':len(models),'numerical_stress':sc,'endpoints':len(rows),'numerical_replays':2*len(rows),'certificate_verdicts':dict(Counter(r['certificate']['verdict'] for r in rows))};Path('audit_results.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
if __name__=='__main__':main()
