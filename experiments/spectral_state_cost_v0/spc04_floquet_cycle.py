"""SPC-04C: task-cycle attractors and finite-horizon Floquet measurements.

Only the exact native LoopedChase recurrence is used. A periodic pointer
orbit is checked against its actual integer cycle length. Full-state
monodromy is a JVP composition; it is evaluated only after the observed
periodic residual is reported. Ritz eigenvalues are numerical estimates.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import (
    N_NODES, D_MODEL, RECURRENCE_LOOPS, prompt_and_recurrence,
)


def pointer_cycle_length(tokens, n_nodes:int=N_NODES) -> int:
    """Cycle containing the start node, from the tokenized permutation."""
    if tokens.shape != (1, 2*n_nodes+1):
        raise ValueError('expected one n-node prompt without depth token')
    pairs=tokens[0,:2*n_nodes].reshape(n_nodes,2).tolist()
    edges={int(s):int(t) for s,t in pairs}
    if sorted(edges.keys())!=list(range(n_nodes)) or sorted(edges.values())!=list(range(n_nodes)):
        raise ValueError('pointer data must encode one permutation')
    start=int(tokens[0,-1]); current=start; visited={}
    while current not in visited:
        visited[current]=len(visited)
        current=edges[current]
    return len(visited)-visited[current]


def phase_alignment_report(model:Any, seed:int=901004,
                           n_prompts:int=64, loops:int=128,
                           comparison_depth:int=64,
                           rel_tol:float=1e-4) -> dict:
    """Held-out cycle registration for the answer-token trajectory.

    The data seed is frozen and disjoint from training seeds. Shorter orbit
    periods are measured in the same units as actual integer pointer cycles.
    """
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    if n_prompts<1 or comparison_depth+N_NODES>=loops or rel_tol<=0:
        raise ValueError('insufficient loops for cycle comparison')
    tokens,_=make_batch(N_NODES,n_prompts,seed,loops)
    model=copy.deepcopy(model).eval().double()
    with torch.no_grad():
        states=model(tokens,loops).detach().numpy() # [loops, batch, channels]
    rows=[]
    t=comparison_depth-1
    for i in range(n_prompts):
        p=pointer_cycle_length(tokens[i:i+1])
        norm=float(np.linalg.norm(states[t,i]))
        relative=lambda shift: float(np.linalg.norm(states[t+shift,i]-states[t,i])/max(norm,1e-30))
        lag1=relative(1)
        lagp=relative(p)
        candidates=[(q,relative(q)) for q in range(1,N_NODES+1)]
        matching=[q for q,r in candidates if r<rel_tol]
        estimated=min(matching) if matching else None
        rows.append({'cycle_length':p,'lag1_relative_residual':lag1,
                     'true_cycle_relative_residual':lagp,
                     'smallest_detected_period':estimated,
                     'stable_true_cycle':bool(lagp<rel_tol),
                     'distinct_from_fixed':bool(lag1>0.05)})
    pgt=[x for x in rows if x['cycle_length']>1]
    return {'n_prompts':n_prompts,'prompt_seed':seed,
            'comparison_depth':comparison_depth,'relative_tolerance':rel_tol,
            'n_nontrivial_pointer_cycles':len(pgt),
            'n_nontrivial_stable_matching_period':sum(x['stable_true_cycle'] and x['distinct_from_fixed'] and x['smallest_detected_period']==x['cycle_length'] for x in pgt),
            'n_true_cycle_stable':sum(x['stable_true_cycle'] for x in pgt),
            'median_true_period_residual':float(np.median([x['true_cycle_relative_residual'] for x in pgt])) if pgt else None,
            'median_one_lag_residual':float(np.median([x['lag1_relative_residual'] for x in pgt])) if pgt else None,
            'cycle_lengths':{str(p):sum(x['cycle_length']==p for x in rows) for p in range(1,N_NODES+1)},
            'rows':rows}


def floquet_probe(model:Any, seed:int=982001,
                  comparison_depth:int=64, loops:int=100,
                  include_eigs:bool=True)->dict:
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from scipy.sparse.linalg import LinearOperator,eigs,ArpackNoConvergence
    if comparison_depth<3 or loops<comparison_depth+N_NODES+2:
        raise ValueError('insufficient horizon')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    m=copy.deepcopy(model).eval().double()
    for par in m.parameters():par.requires_grad_(False)
    with sdpa_kernel([SDPBackend.MATH]):
        step,seq,tokens,hops,gap=prompt_and_recurrence(m,seed,loops=loops)
        p=pointer_cycle_length(tokens)
        s=seq[comparison_depth]
        lag1=float((seq[comparison_depth+1]-s).norm()/s.norm())
        lagp=float((seq[comparison_depth+p]-s).norm()/s.norm())
        response={'prompt_seed':seed,'period_from_pointer_graph':p,
                  'full_state_dim':int(s.numel()),'max_native_forward_gap':gap,
                  'depth':comparison_depth,'one_step_relative_residual':lag1,
                  'full_cycle_relative_residual':lagp,
                  'near_cycle_at_tolerance_1e-6':bool(lagp<1e-6),
                  'is_nonfixed_at_tolerance_0p05':bool(lag1>0.05),
                  'monodromy_order':'J(F at depth+t-1) ... J(F at depth)',
                  'eigenvalue_method':'SciPy Arnoldi applied to PyTorch double-precision JVP composition'}
        if include_eigs:
            calls=[0]; shape=tuple(s.shape); dim=s.numel()
            def mv(v):
                tangent=torch.from_numpy(np.asarray(v).reshape(shape).copy()).double()
                for u in range(p):
                    calls[0]+=1
                    tangent=torch.autograd.functional.jvp(step,seq[comparison_depth+u],
                                                         tangent,create_graph=False)[1]
                return tangent.detach().numpy().reshape(-1).copy()
            op=LinearOperator((dim,dim),matvec=mv,dtype=np.float64)
            ok=True
            try:
                val=eigs(op,k=3,which='LM',tol=1e-6,ncv=18,maxiter=200,
                         v0=np.linspace(1,2,dim),return_eigenvectors=False)
            except ArpackNoConvergence as err:
                val=err.eigenvalues
                ok=False
            v=sorted(val,key=abs,reverse=True)
            response.update({'arnoldi_converged':ok,'arnoldi_jvp_calls':calls[0],
                'floquet_spectral_radius_estimate':float(abs(v[0])) if v else None,
                'floquet_ritz_eigenvalues':[{'real':float(x.real),'imag':float(x.imag),'modulus':float(abs(x))} for x in v]})
    return response
