"""Adaptive diagnostic: freeze learned address map, learn code-to-memory retrieval."""
import argparse, json, time, hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def vectors(ids,k): return (((torch.tensor(ids)[:,None]>>torch.arange(k))&1).float()*2-1)

def run(inputs,out):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); results=[]
    for path in inputs:
        rec=json.loads(Path(path).read_text()); k=rec['k']
        for r in rec['results']:
            if r['arm']!='hier': continue
            code=torch.tensor(r['train']['query_map']); assert torch.equal(code,torch.tensor(r['heldout']['query_map'])); perm=torch.tensor(r['permutation']); x=vectors(r['train_ids'],k); z=vectors(r['heldout_ids'],k); y=x[:,perm]; truth=z[:,perm]; root=r['root']
            # The training learner sees address and binary task labels, not the true permutation.
            votes=torch.zeros(k,k); votes.index_add_(0,code,y.T@x); counts=torch.bincount(code,minlength=k); sparse=torch.zeros(k,k); sparse[torch.arange(k),votes.abs().argmax(1)]=votes[torch.arange(k),votes.abs().argmax(1)].sign()
            for arm in ('sparse_correlation','rank_half','rank_full'):
                torch.manual_seed(root+30000); start=time.monotonic()
                if arm=='sparse_correlation': weight=sparse; parameters=k # k coordinate indices; signs separately
                else:
                    d=k//2 if arm=='rank_half' else k; v=nn.Parameter(torch.randn(k,d)/k**.5); u=nn.Parameter(torch.randn(d,k)/d**.5); opt=torch.optim.Adam((v,u),lr=.03); g=torch.Generator().manual_seed(root+40000)
                    for _ in range(500):
                        ix=torch.randint(len(x),(128,),generator=g); logit=(x[ix]@v@u)[:,code]; loss=F.binary_cross_entropy_with_logits(logit,(y[ix]+1)/2); opt.zero_grad(); loss.backward(); opt.step()
                    weight=(v@u).detach().T; parameters=v.numel()+u.numel()
                pred=(z@weight.T)[:,code]>=0; ok=pred==(truth>0); correct=int(ok.sum()); total=ok.numel(); diag=weight[code,perm]; off=weight[code].abs().sum(1)-diag.abs(); exact_certificate=bool((diag>off).all())
                result=dict(k=k,root=root,arm=arm,parameters=parameters,query_codes=int(code.unique().numel()),correct=correct,total=total,exact_rows=int(ok.all(1).sum()),rows=len(z),strict_diagonal_dominance_certificate=exact_certificate,minimum_dominance_margin=float((diag-off).min()),seconds=time.monotonic()-start,weight=weight.tolist(),query_map=code.tolist(),predictions=''.join('1' if a else '0' for a in pred.flatten().tolist()))
                results.append(result); print(k,root,arm,correct,total,exact_certificate,flush=True)
                Path(out).write_text(json.dumps(dict(evidence='adaptive exposed-root retrieval diagnostic',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('receipts',nargs='+'); p.add_argument('--out',required=True); a=p.parse_args(); run(a.receipts,a.out)
