"""Verify the frozen LS08 disturbance target; no region/radius adaptation."""
from verify import exact,fixed_violation,positive,Q,CASES,numeric_grid
from train08 import SEEDS,ARMS,sha,torch,Path,json
REGS=[(-Q(1,8),Q(1,8)),(Q(7,8),Q(9,8))];EPS=Q(1,32)
def certify(w):
    v=fixed_violation(w,REGS,EPS)
    return {'verdict':'POINT_VIOLATION','witness':v} if v else exact(w,REGS,EPS)
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);pos=certify(positive());assert pos['verdict']=='CERTIFIED';rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            f=Path(f'results/{arm}-{seed}.json');r=json.loads(f.read_text());w=r['q4_weights'];c=certify(w);vals=numeric_grid(w,REGS);y=torch.tensor([p^a^b for p,a,b,n in CASES],dtype=torch.float64);err=float((vals-y[:,None]).abs().max());rows.append({'arm':arm,'seed':seed,'weight_receipt_sha256':sha(f),'grid_max_error':err,'certificate':c});print(arm,seed,c['verdict'],round(err,6),flush=True)
    Path('certificates.json').write_text(json.dumps({'positive_control':pos,'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
