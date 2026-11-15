"""SPC-03J: native full-state Jacobian near the untrained LoopedChase attractor.

Explicitly reconstructs the repo model's recurrent map from its public
`tok/dst/pos/inject/core` modules. Verifies equality to model.forward on
frozen tokens before using a JVP and Arnoldi (scipy) eigenvalue estimate.
The state includes all prompt tokens. This is local Jacobian evidence, not
an empirical trained-reasoning or safety result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def run_jacobian_sweep(seeds=(13012902,13012903,13012904,13012905),
                      n_nodes=8,d_model=64,loops=42)->dict:
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from scipy.sparse.linalg import LinearOperator, eigs
    from research_gym.scripts.loopscope_toy import LoopedChase, make_batch
    if not seeds or d_model%4 or loops<20:raise ValueError('bad bounded configuration')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        model=LoopedChase(n_nodes=n_nodes,d=d_model,layers=1,heads=4).eval().double()
        for par in model.parameters():par.requires_grad_(False)
        tokens,_=make_batch(n_nodes,1,seed+1,loops)
        edges=tokens[:,:2*n_nodes].reshape(1,n_nodes,2)
        original=torch.cat((model.tok(edges[...,0])+model.dst(edges[...,1]),
                            model.tok(tokens[:,2*n_nodes:])),dim=1)
        e=original+model.pos(torch.arange(original.shape[1]))

        def recurrence(h):
            y=model.inject(h)+e
            for block in model.core:y=block(y)
            return y

        with sdpa_kernel([SDPBackend.MATH]):
            h=torch.zeros_like(e)
            seq=[h]
            with torch.no_grad():
                for _ in range(loops):
                    h=recurrence(h)
                    seq.append(h)
                native=model(tokens,loops)
            max_forward_gap=float((native-torch.stack([x[:,-1] for x in seq[1:]])).abs().max())
            if max_forward_gap>1e-10:raise AssertionError('full-state recurrence deviates from native forward')
            def jvp_at(state,direction):
                return torch.autograd.functional.jvp(recurrence,state,direction,create_graph=False)[1]
            secants=[]
            for t in (4,6,8,10,12):
                delta=seq[t+1]-seq[t]
                nxt=seq[t+2]-seq[t+1]
                pred=jvp_at(seq[t+1],delta)
                secants.append({'difference_start_depth':t,
                                'change_norm':float(delta.norm()),
                                'next_change_norm':float(nxt.norm()),
                                'relative_endpoint_jacobian_error':float((pred-nxt).norm()/nxt.norm()),
                                'scalar_rayleigh_damping':float((delta*nxt).sum()/(delta*delta).sum())})
            full_shape=tuple(h.shape)
            dim=h.numel()
            n_jvp=[0]
            def matvec(x):
                n_jvp[0]+=1
                v=torch.from_numpy(np.ascontiguousarray(x.reshape(full_shape))).to(torch.float64)
                with sdpa_kernel([SDPBackend.MATH]):
                    jv=jvp_at(seq[-1],v)
                return jv.detach().numpy().ravel().copy()
            op=LinearOperator((dim,dim),matvec=matvec,dtype=np.float64)
            # Explicit starting vector fixes the Arnoldi initialization.
            eigenvalues=eigs(op,k=4,which='LM',tol=1e-6,maxiter=700,ncv=24,
                              v0=np.linspace(1,2,dim),return_eigenvectors=False)
            eig_list=sorted(eigenvalues,key=abs,reverse=True)
            fixed_point_residual=float((recurrence(seq[-1])-seq[-1]).norm() / seq[-1].norm())
        rows.append({'seed':seed,'n_tokens':full_shape[1], 'full_state_dim':dim,
                     'max_forward_gap':max_forward_gap,
                     'fixed_point_relative_residual':fixed_point_residual,
                     'arnoldi_jvp_calls':n_jvp[0],
                     'leading_eigenvalues':[{'real':float(v.real),'imag':float(v.imag),'modulus':float(abs(v))} for v in eig_list],
                     'estimated_spectral_radius':float(abs(eig_list[0])),
                     'secants':secants})
    return {'campaign':'SPC-03J','model':'LoopedChase','trained_checkpoint':None,
            'training_performed':False,'full_state_includes_all_tokens':True,
            'precision':'float64','jacobian_method':'autograd.functional.jvp + scipy Arnoldi',
            'sdpa_backend':'math','seed_set':list(seeds),'loops':loops,
            'mean_estimated_spectral_radius':float(np.mean([v['estimated_spectral_radius'] for v in rows])),
            'mean_secant_relative_error_t12':float(np.mean([v['secants'][-1]['relative_endpoint_jacobian_error'] for v in rows])),
            'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[13012902,13012903,13012904,13012905])
    a=p.parse_args()
    r=run_jacobian_sweep(seeds=tuple(a.seeds));s=json.dumps(r,sort_keys=True,indent=2)
    if a.out:a.out.write_text(s)
    print(s)

if __name__=='__main__':main()
