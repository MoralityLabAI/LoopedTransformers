"""Exact rational endpoint interpolation certificate, with a global GELU curvature bound."""
from train11 import *
from rational import I,box,gelu,Q,pi_lo
from verify import wf
from concurrent.futures import ProcessPoolExecutor
EPS=Q(1,32);R=Q(1,8)

def cases(task):
    _,off,m,_=base(task);return list(itertools.product(range(m),(0,1),(0,1),(0,1))),Q(off),m

def numerical_grid(task,w):
    cs,off,m=cases(task);c=torch.tensor(cs,dtype=torch.float64);h=c[:,0,None]+float(off)+torch.linspace(-.125,.125,257,dtype=torch.float64);x=c[:,1:,None].transpose(1,2).expand(-1,257,-1);y=c[:,:3].sum(-1).remainder(m)+float(off);p=[torch.tensor(w[k],dtype=torch.float64)[None] for k in NAMES]
    with torch.no_grad():v=step(p,h.reshape(1,-1),x.reshape(1,-1,3)).reshape(len(cs),257)
    return v,y

def verify(task,w):
    cs,off,m=cases(task);v,y=numerical_grid(task,w);err=(v-y[:,None]).abs();idx,j=divmod(int(err.argmax()),257);p,a,b,n=cs[idx];t=(p+a+b)%m;h=off+p-R+2*R*Q(j,256);val=wf(w,box(h),a,b,n);inner=R-EPS
    def violation(p,a,b,n,h,val):
        t=(p+a+b)%m;return {'verdict':'POINT_VIOLATION','p':p,'input':[a,b,n],'h':str(h),'target':t,'output':val.out(),'target_inner':[str(off+t-inner),str(off+t+inner)]}
    if val.lo>off+t+inner or val.hi<off+t-inner:return violation(p,a,b,n,h,val)
    # GELU''=(2-z^2)phi(z), abs <= 1 on R. Linear interpolation error <= C*(u-l)^2/8.
    C=Q(1,5)*sum(abs(Q(d))*Q(row[0])**2 for row,d in zip(w['up'],w['down']));count=leaves=0;records=[]
    for p,a,b,n in cs:
        target=off+(p+a+b)%m;tl,tu=target-inner,target+inner;lo,hi=off+p-R,off+p+R;cache={};local=0;minl=maxu=None
        def ev(h):
            if h not in cache:cache[h]=wf(w,box(h),a,b,n)
            return cache[h]
        todo=[(lo,hi,0)]
        while todo:
            l,u,d=todo.pop();a0,a1=ev(l),ev(u);count+=1
            for h,val in ((l,a0),(u,a1)):
                if val.lo>tu or val.hi<tl:return violation(p,a,b,n,h,val)
            margin=C*(u-l)**2/8;L=min(a0.lo,a1.lo)-margin;U=max(a0.hi,a1.hi)+margin
            if L>=tl and U<=tu:
                leaves+=1;local+=1;minl=L if minl is None else min(minl,L);maxu=U if maxu is None else max(maxu,U);continue
            if d>=14 or count>20000:return {'verdict':'UNRESOLVED','boxes':count,'p':p,'input':[a,b,n],'interval':[str(l),str(u)]}
            mid=(l+u)/2;todo.extend([(mid,u,d+1),(l,mid,d+1)])
        records.append({'p':p,'input':[a,b,n],'target':(p+a+b)%m,'leaves':local,'enclosure':[str(minl),str(maxu)]})
    return {'verdict':'CERTIFIED','boxes':count,'leaves':leaves,'curvature':str(C),'eps':str(EPS),'margin':'3/8','cases':records}

def constructive():
    # Direct exact proof for 10-GELU template, independent of numerical grid.
    w,off,m,ss=base('mod4');w={k:v.tolist() for k,v in zip(NAMES,w)};assert m==4 and off==-2.5
    for i,s in zip(QIDS,ss):
        x=torch.tensor(w[NAMES[i]],dtype=torch.float64);assert torch.equal(x/s,(x/s).round()) and (x/s).abs().max()<=7
    # phi(5/4) < 1/5: 1/sqrt(2pi)<2/5 and e^(25/32)>2 by positive Taylor terms.
    assert pi_lo>Q(25,8) and 1+Q(25,32)+Q(25,32)**2/2>2
    assert w['skip']==[-1.,0.,0.,0.] and w['skip_bias']==[-2.5] and w['down_bias']==[0.]
    cs,off,m=cases('mod4');rows=[];bound=Q(0)
    for p,a,b,n in cs:
        values=[];assert len(w['up'])==10
        for h in (off+p-R,off+p+R):
            z=[sum(Q(v)*s for v,s in zip(row,(h,a,b,n)))+Q(c) for row,c in zip(w['up'],w['up_bias'])]
            values.append(z)
            assert off+Q(1,5)*sum(Q(v)*max(Q(0),x) for v,x in zip(w['down'],z))==off+(p+a+b)%m
        assert all(min(x,y)>=Q(5,4) or max(x,y)<=-Q(5,4) for x,y in zip(*values))
        for i in range(0,10,2):assert Q(w['down'][i])==-Q(w['down'][i+1])
        # Both GELU-minus-ReLU errors lie in [-1/5,0], so their difference <=1/5.
        bound=Q(1,5)*sum(abs(Q(w['down'][i])) for i in range(0,10,2))*Q(1,5)
        rows.append({'p':p,'input':[a,b,n],'target':(p+a+b)%m})
    assert bound==Q(7,100) and R-bound==Q(11,200) and R-bound>=EPS
    out={'task':'mod4','width':10,'dense_slots':69,'exact_real':True,'region_radius':str(R),'error_bound':str(bound),'disturbance_radius':str(R-bound),'readout_margin':'3/8','weights':w,'scales':ss,'transitions':rows};Path('construction.json').write_text(json.dumps(out,indent=2)+'\n');return out

def job(path):
    torch.set_num_threads(1);d=json.loads(path.read_text());c=verify(d['task'],d['q4_weights']);return {k:d[k] for k in ('task','arm','seed','step')}|{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'certificate':c}

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);paths=sorted(Path('results').glob('*.json'));assert len(paths)==216;construction=constructive();controls={}
    for task in ARMS:
        p,off,m,s=base(task);w={k:v.tolist() for k,v in zip(NAMES,p)};c=verify(task,w);assert c['verdict']=='CERTIFIED',c;controls[task]=c;print(task,'control',c['boxes'],c['leaves'],flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows=[]
        for r in ex.map(job,paths,chunksize=1):
            rows.append(r)
            if len(rows)%24==0:print('verified',len(rows),flush=True)
    Path('certificates.json').write_text(json.dumps({'positive_controls':controls,'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
