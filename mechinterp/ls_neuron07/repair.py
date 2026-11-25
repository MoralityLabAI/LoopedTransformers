"""Post-hoc fixed Q4 integer codebook scale repair. Exact verifier remains the final gate."""
from verify import *
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize

def codes(w):
    ret={};scales=[]
    for name in ('skip','up','up_bias','down'):
        a=np.array(w[name],dtype=np.float64);s=np.max(np.abs(a))/7;scales.append(s if s else 1e-6);ret[name]=np.rint(a/(s if s else 1)).astype(np.int64);assert np.max(np.abs(ret[name]))<=7
    return ret,scales

def repair(w):
    c,s=codes(w);cases=np.array(CASES);h=(cases[:,0,None]+np.linspace(-.125,.125,33)).reshape(-1);x=np.repeat(cases[:,1:],33,axis=0);q=np.column_stack((h,x));y=np.repeat(np.bitwise_xor(np.bitwise_xor(cases[:,0],cases[:,1]),cases[:,2]),33)
    sprod=q@c['skip'];uprod=q@c['up'].T;ub=c['up_bias'];down=c['down'];off=float(w['skip_bias'][0]+.2*w['down_bias'][0])
    def fn(v):
        ss,su,sb,sd,b,_=v;z=su*uprod+sb*ub;cdf=ndtr(z);g=z*cdf;dg=cdf+z*np.exp(-z*z/2)/np.sqrt(2*np.pi);gout=g@down;output=h+ss*sprod+.2*sd*gout+b
        jac=np.column_stack((sprod,.2*sd*((dg*uprod)@down),.2*sd*((dg*ub)@down),.2*gout,np.ones(len(h))))
        return output-y,jac
    z=np.array(s+[off,0.]);z[-1]=np.max(np.abs(fn(z)[0]))+1e-6
    def cons(v):
        e,j=fn(v);return np.r_[v[-1]-e,v[-1]+e]
    def cj(v):
        e,j=fn(v);return np.column_stack((np.vstack((-j,j)),np.ones(2*len(h))))
    lim=[(max(1e-8,k*.1),max(1e-7,k*5)) for k in s]+[(-4,4),(0,10)]
    r=minimize(lambda v:v[-1],z,jac=lambda v:np.array([0,0,0,0,0,1.]),bounds=lim,constraints=[{'type':'ineq','fun':cons,'jac':cj}],method='SLSQP',options={'maxiter':120,'ftol':1e-10})
    v=r.x;out={k:(np.array(c[k],dtype=np.float32)*np.float32(v[i])).tolist() for i,k in enumerate(('skip','up','up_bias','down'))};out['down_bias']=w['down_bias'];out['skip_bias']=[float(np.float32(v[4]-.2*w['down_bias'][0]))]
    initial=float(np.max(np.abs(fn(z)[0])));final=float(np.max(np.abs(fn(v)[0])));return out,dict(initial_grid_error=initial,final_grid_error=final,optimizer_success=bool(r.success),status=int(r.status),iterations=int(r.nit),codes={k:t.tolist() for k,t in c.items()},scales=[float(np.float32(t)) for t in v[:4]],offset=float(v[4]))

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);regs=[(-Q(1,8),Q(1,8)),(Q(7,8),Q(9,8))];e=Q(1,32);rows=[]
    for f in sorted(Path('results').glob('*.json')):
        r=json.loads(f.read_text());w,meta=repair(r['q4_weights']);violation=fixed_violation(w,regs,e);cert={'verdict':'POINT_VIOLATION','witness':violation} if violation else exact(w,regs,e);p=[torch.tensor(w[k],dtype=torch.float32)[None] for k in NAMES];data=evaluate(p,r['seed']);row=dict(arm=r['arm'],seed=r['seed'],parent_sha256=sha(f),q4_weights=w,meta=meta,certificate=cert,q4=data);rows.append(row);print(r['arm'],r['seed'],round(meta['initial_grid_error'],4),round(meta['final_grid_error'],4),cert['verdict'],round(data['64']['accuracy'],3),flush=True)
    Path('repair_results.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
if __name__=='__main__':main()
