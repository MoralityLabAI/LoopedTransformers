"""Post-hoc weaker disturbance grid; retain all repaired models."""
from verify import *
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[];regs=[(-Q(1,8),Q(1,8)),(Q(7,8),Q(9,8))]
    for r in json.loads(Path('repair_results.json').read_text()):
        w=r['q4_weights'];certs={}
        for eps in (Q(1,64),Q(1,128),Q(1,256),Q(0)):
            v=fixed_violation(w,regs,eps);certs[str(eps)]={'verdict':'POINT_VIOLATION','witness':v} if v else exact(w,regs,eps)
        rows.append({'arm':r['arm'],'seed':r['seed'],'certificates':certs});print(r['arm'],r['seed'],[(e,c['verdict'],c.get('boxes')) for e,c in certs.items()],flush=True)
    Path('reduced_margin_results.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
