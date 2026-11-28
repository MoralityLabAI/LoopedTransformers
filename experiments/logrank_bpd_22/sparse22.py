"""22B exposed-root exact sparse-delta replacement for the candidate scorer."""
import json,sys,time
from pathlib import Path
from fractions import Fraction as F
import pilot22 as p
H=Path(__file__).resolve().parent
class SparsePlanner(p.Planner):
    def __init__(self,g,t):
        super().__init__(g,t);self.w=self.local.tolist();self.c=self.charges.tolist();self.surv={o:v.tolist() for o,v in self.survive.items()}
    def scalar(self,d,C,R):
        D=d.copy();M=[0]*self.E;filled=[False]*self.E;work=0
        for i in range(self.E-1,-1,-1):
            m=0 if C[i] and R[i] else min(D[i],1) if C[i] else D[i];M[i]=m;filled[i]=C[i] and (R[i] or D[i]>0);work+=m*self.w[i]
            if m:
                for k in self.deps[i]:D[k]+=m
        return D,M,filled,work
    def greedy(self,d,C,cap,owners):
        demand=[d.get(e,0) for e in self.edges];R=[e in C for e in self.edges];target=[False]*self.E;left=cap;steps=[];order=[];evals=0;visits=0;N=max(1,len(owners));S=[self.surv[o] for o in owners]
        while True:
            D,M,filled,now=self.scalar(demand,target,R);future=[self.scalar(demand,target,[a and b for a,b in zip(filled,s)]) for s in S];base=N*now+sum(z[3] for z in future);elig=[i for i in range(self.E) if not target[i] and self.c[i]<=left]
            if not elig:break
            best=None;evals+=1+len(elig)
            for i in elig:
                change=0;todo=[(i,0,(0,)*len(S))]
                while todo:
                    j,delta,ds=todo.pop();visits+=1;chosen=target[j] or j==i;nd=D[j]+delta;assert nd>=0
                    nm=0 if chosen and R[j] else min(nd,1) if chosen else nd;nf=chosen and (R[j] or nd>0);dm=nm-M[j];dfs=[]
                    for k,s in enumerate(S):
                        fd=future[k][0][j]+ds[k];assert fd>=0;fm=0 if chosen and nf and s[j] else min(fd,1) if chosen else fd;dfs.append(fm-future[k][1][j])
                    change+=self.w[j]*(N*dm+sum(dfs))
                    if dm or any(dfs):todo.extend((k,dm,dfs) for k in self.deps[j])
                gain=-change
                if gain>0:
                    rank=(F(gain,self.c[i]),tuple(-x for x in self.edges[i]))
                    if best is None or rank>best[0]:best=(rank,i,gain)
            if best is None:break
            _,i,gain=best;target[i]=True;left-=self.c[i];order.append(self.edges[i]);steps.append([*self.edges[i],base,base-gain])
        return order,dict(objective=base,predicted_now=now,predicted_filled=[list(e) for i,e in enumerate(self.edges) if filled[i]],steps=steps,candidate_evals=evals,scenario_owners=owners,delta_visits=visits)
def strip(x):
    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if k not in ('seconds','planning_seconds','peak_rss_kib','delta_visits')}
    if isinstance(x,list):return [strip(v) for v in x]
    return x
def main(n):
    r=json.loads((H/f'receipt{n}.json').read_text());ws={w['seed']:w for w in r['worlds']};out=[];vector=p.Planner
    for a in r['results']:
        if a['status']!='solved' or ws[a['seed']]['family']!='hub':continue
        w=ws[a['seed']];compiled=p.plan(n,w['rows'])
        for reg in a['regimes']:
            for old in reg['arms']:
                if old['policy'] not in ('resident','retention'):continue
                pair={}
                order=[('vector',vector),('sparse',SparsePlanner)]
                if len(out)%2:order.reverse()
                for label,cls in order:
                    p.Planner=cls;pair[label]=p.run_arm(w,a['noise'],reg['workload'],compiled,(old['policy'],old['kib']))
                new=pair['sparse'];again=pair['vector'];assert strip(old)==strip(new)==strip(again)
                out.append(dict(seed=w['seed'],noise=a['noise'],family=w['family'],pop=reg['pop'],locality=reg['locality'],policy=old['policy'],kib=old['kib'],original_seconds=old['seconds'],old_seconds=again['seconds'],new_seconds=new['seconds'],old_plan_seconds=sum(e['planning_seconds'] for e in again['epochs']),new_plan_seconds=sum(e['planning_seconds'] for e in new['epochs']),delta_visits=sum(e['plan']['delta_visits'] for e in new['epochs']),candidate_evals=sum(e['plan']['candidate_evals'] for e in new['epochs']),work=sum(sum(e['counts'][:2]) for e in new['epochs'])))
        print('MATCH',n,w['seed'],a['noise'],flush=True)
    (H/f'sparse{n}.json').write_text(json.dumps(dict(status='PASS',evidence='adaptive exposed-root implementation diagnostic',input_sha256=p.sha(H/f'receipt{n}.json'),source_sha256=p.sha(__file__),trajectories=len(out),rows=out),indent=2)+'\n')
if __name__=='__main__':main(int(sys.argv[1]))
