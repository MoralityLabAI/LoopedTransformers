"""Rational interval verification of stored Q4 dyadic parameters; numerical region search is only a proposal."""
from train import *
from rational import Q,I,box,gelu
from itertools import product
from fractions import Fraction
import glob
CASES=list(product((0,1),repeat=4));G=1024

def qparams(w):return {k:torch.tensor(v,dtype=torch.float64)[None] for k,v in w.items()}
def wf(w,h,a,b,n):
    # Independently evaluate exact-real scalar recurrence. Merge h+skip*h algebraically.
    s=w['skip'];u=w['up'];ub=w['up_bias'];v=w['down'];xb=(a,b,n)
    pre=[box(Q(row[0]))*h+sum((Q(t)*k for t,k in zip(row[1:],xb)),Q(c)) for row,c in zip(u,ub)]
    return box(1+Q(s[0]))*h+sum((Q(t)*k for t,k in zip(s[1:],xb)),Q(w['skip_bias'][0]))+Q(1,5)*(sum((Q(t)*gelu(z) for t,z in zip(v,pre)),box(w['down_bias'][0])))

def numeric_grid(w,regions):
    ps=list(qparams(w).values()) # replaced with explicit ordering below
    ps=[qparams(w)[k] for k in NAMES]
    c=torch.tensor(CASES,dtype=torch.float64);lo=torch.tensor([float(regions[p][0]) for p,a,b,n in CASES],dtype=torch.float64);hi=torch.tensor([float(regions[p][1]) for p,a,b,n in CASES],dtype=torch.float64)
    h=lo[:,None]+(hi-lo)[:,None]*torch.linspace(0,1,257,dtype=torch.float64);x=c[:,1:,None].transpose(1,2).expand(-1,257,-1)
    with torch.no_grad():out=step(ps,h.reshape(1,-1),x.reshape(1,-1,3)).reshape(16,257)
    return out

def fixed_violation(w,regions,epsilon):
    vals=numeric_grid(w,regions);targets=torch.tensor([p^a^b for p,a,b,n in CASES],dtype=torch.float64);err=(vals-targets[:,None]).abs();i,j=divmod(int(err.argmax()),257);p,a,b,n=CASES[i];l,u=regions[p];h=l+(u-l)*Q(j,256);o=wf(w,box(h),a,b,n);y=p^a^b;target=(regions[y][0]+epsilon,regions[y][1]-epsilon)
    if o.lo>target[1] or o.hi<target[0]:return {'input':[a,b,n],'p':p,'h':str(h),'target':y,'output':o.out(),'output_decimal':[float(o.lo),float(o.hi)],'target_inner':list(map(str,target)),'grid_max_error':float(err.max())}
    return None

def exact(w,regions,epsilon,maxdepth=14):
    if regions[0][0]>0 or regions[0][1]<0 or regions[0][1]>=Q(1,2) or regions[1][0]<=Q(1,2):return {'verdict':'BAD_REGION_OR_READOUT'}
    count=0;leaves=0;worst=None;records=[]
    for p,a,b,n in CASES:
        y=p^a^b;tl,tu=regions[y][0]+epsilon,regions[y][1]-epsilon
        todo=[(*regions[p],0)];minl=None;maxu=None;local=0
        while todo:
            l,u,d=todo.pop();o=wf(w,I(l,u),a,b,n);count+=1
            if o.lo>=tl and o.hi<=tu:
                leaves+=1;local+=1;minl=o.lo if minl is None else min(minl,o.lo);maxu=o.hi if maxu is None else max(maxu,o.hi);continue
            m=(l+u)/2;q=wf(w,box(m),a,b,n)
            if q.lo>tu or q.hi<tl:return {'verdict':'POINT_VIOLATION','cases_completed':len(records),'boxes':count,'witness':{'p':p,'input':[a,b,n],'h':str(m),'target':y,'output':q.out(),'target_inner':list(map(str,(tl,tu)))}}
            if d>=maxdepth or count>20000:return {'verdict':'UNRESOLVED','cases_completed':len(records),'boxes':count,'interval':list(map(str,(l,u))),'output':o.out()}
            todo.append((m,u,d+1));todo.append((l,m,d+1))
        records.append({'p':p,'input':[a,b,n],'target':y,'leaf_count':local,'enclosure':list(map(str,(minl,maxu)))})
    return {'verdict':'CERTIFIED','boxes':count,'leaves':leaves,'cases':records,'epsilon':str(epsilon),'regions':[[str(a),str(b)] for a,b in regions],'readout_margin':str(min(Q(1,2)-regions[0][1],regions[1][0]-Q(1,2)))}

def propose(w,epsilon):
    # Iterated reachable interval hull including exact init; sample is only a proposal.
    r=[(Q(0),Q(0)),(Q(1),Q(1))]
    for it in range(64):
        o=numeric_grid(w,r);new=[]
        for y in (0,1):
            vals=o[[p^a^b==y for p,a,b,n in CASES]]
            lo=min(r[y][0],Q(math.floor((float(vals.min())-float(epsilon))*G)-1,G));hi=max(r[y][1],Q(math.ceil((float(vals.max())+float(epsilon))*G)+1,G));new.append((lo,hi))
        if new[0][1]>=Q(1,2) or new[1][0]<=Q(1,2) or any(u-l>4 for l,u in new):return None,it+1
        if new==r:return r,it+1
        r=new
    return None,64

def positive():
    p=json.loads(Path('positive_weights.json').read_text());return {'skip':[p['skip.0.weight'][0][j] for j in (0,16,17,18)],'skip_bias':[p['skip.0.bias'][0]],'up':[[row[j] for j in (0,16,17,18)] for row in p['up.0.weight']],'up_bias':p['up.0.bias'],'down':p['down.0.weight'][0],'down_bias':[p['down.0.bias'][0]]}

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);regions=[(-Q(1,8),Q(1,8)),(Q(7,8),Q(9,8))];eps=Q(1,32);pos=exact(positive(),regions,eps);assert pos['verdict']=='CERTIFIED';print('positive',pos['boxes'],pos['leaves'],flush=True)
    rows=[]
    for f in sorted(Path('results').glob('*.json')):
        r=json.loads(f.read_text());w=r['q4_weights'];v=fixed_violation(w,regions,eps);fixed={'verdict':'POINT_VIOLATION','witness':v} if v else exact(w,regions,eps);adapt={}
        for e in (Q(1,32),Q(0)):
            candidate,it=propose(w,e);key=str(e);adapt[key]={'iterations':it,'proposal':None if candidate is None else [[str(a),str(b)] for a,b in candidate],'result':None if candidate is None else exact(w,candidate,e)}
        row={'arm':r['arm'],'seed':r['seed'],'weights_sha256':sha(f),'fixed':fixed,'adaptive':adapt};rows.append(row);print(r['arm'],r['seed'],fixed['verdict'],[(k,a['result']['verdict'] if a['result'] else 'NO_PROPOSAL',a['proposal']) for k,a in adapt.items()],flush=True)
    Path('verification.json').write_text(json.dumps({'positive_control':pos,'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
