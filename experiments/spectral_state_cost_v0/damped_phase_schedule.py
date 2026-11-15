"""SPC-03: phase-sparse measurement gains under geometrically damped modes.

For modes x[t] = rho**t * sum_r a[r] * exp(2*pi*i*r*t/d), a schedule
S and delay L gives M[L,S](s,r) = rho**(L+s)*omega**((L+s)*r).
This is an exact algebraic model. It is not a fit of true transformer
Jacobian eigenvalues; model rate 0.435 is a descriptive scalar AR proxy.
"""
from __future__ import annotations

import itertools
import json
import math
import argparse
from typing import Sequence
import numpy as np

from experiments.spectral_state_cost_v0.phase_stability import check_offsets, critical_support, phase_matrix


def _rho_delay(rho: float, delay: int) -> None:
    if not isinstance(rho,(float,int)) or isinstance(rho,bool) or not 0.0<rho<=1.0 or not math.isfinite(rho):
        raise ValueError('rho must be finite in (0,1]')
    if not isinstance(delay,int) or isinstance(delay,bool) or delay<0:
        raise ValueError('delay must be a nonnegative integer')


def damped_phase_matrix(d:int, offsets:Sequence[int],rho:float,delay:int=0)->np.ndarray:
    _rho_delay(rho,delay)
    s=check_offsets(d,offsets)
    M=phase_matrix(d,s)
    rows=np.asarray(s,dtype=np.int64)
    cols=np.arange(d,dtype=np.int64)
    return (rho**(delay+rows))[:,None]*M*np.exp(2j*np.pi*delay*cols[None,:]/d)


def worst_damped_gain(d:int, offsets:Sequence[int], k:int, rho:float,
                      delay:int=0, max_minors:int=20000)->float:
    s=check_offsets(d,offsets)
    p=critical_support(d,k)
    if len(s)<p:raise ValueError('fewer probes than sparse-identification budget')
    if math.comb(d,p)>max_minors:raise ValueError('too many support minors')
    V=damped_phase_matrix(d,s,rho,delay)
    return min(float(np.linalg.svd(V[:,J],compute_uv=False)[-1]) for J in itertools.combinations(range(d),p))


def damped_frontier(d:int, q:int, k:int, rho:float, max_cases:int=20000):
    p=critical_support(d,k)
    if q<p or q>d or math.comb(d,q)*math.comb(d,p)>max_cases:
        raise ValueError('invalid or oversized schedule enumeration')
    result=[{'offsets':list(S),'gain':worst_damped_gain(d,S,k,rho)}
            for S in itertools.combinations(range(d),q)]
    return sorted(result,key=lambda x:(-x['gain'],x['offsets']))


def crossover_schedule(rho_low:float=.5, rho_high:float=.7,
                       tol:float=1e-10)->dict[str,float]:
    """Small finite numerical diagnostic, not a symbolic optimality proof."""
    early=(0,1,2,3); spread=(0,1,2,4)
    def gap(r):return worst_damped_gain(7,early,2,r)-worst_damped_gain(7,spread,2,r)
    lo,hi=rho_low,rho_high
    if gap(lo)<=0 or gap(hi)>=0:raise ValueError('not a bracket for fixed comparator pair')
    while hi-lo>tol:
        mid=(lo+hi)/2
        if gap(mid)>0:lo=mid
        else:hi=mid
    rho=(lo+hi)/2
    return {'rho_crossover':rho, 'gain_at_crossover':worst_damped_gain(7,early,2,rho),
            'bracket_width':hi-lo}


def summary(rho:float=.4353629)->dict[str,object]:
    _rho_delay(rho,0)
    f=damped_frontier(7,4,2,rho)
    flat=damped_frontier(7,4,2,1.0)
    conv=worst_damped_gain(7,(0,1,2,3),2,rho)
    sp=worst_damped_gain(7,(0,1,2,4),2,rho)
    return {'campaign':'SPC-03-damped-phase', 'd':7,'k':2,'q':4,'rho':rho,
            'rates_are_constructed_not_native_eigenvalues':True,
            'schedule_count':len(f),'support_minors_per_schedule':math.comb(7,4),
            'optimal_damped_offsets':f[0]['offsets'],'optimal_damped_gain':f[0]['gain'],
            'consecutive_damped_gain':conv,'spread_damped_gain':sp,
            'consecutive_to_spread_gain_ratio':conv/sp,
            'optimal_undamped_offsets':flat[0]['offsets'],
            'crossover':crossover_schedule(),
            'ten_loop_delay_attentuation':rho**10,
            'twenty_loop_delay_attenuation':rho**20}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rho',type=float,default=.4353629)
    p.add_argument('--out',type=str)
    a=p.parse_args()
    s=json.dumps(summary(a.rho),indent=2,sort_keys=True)
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(s)
    print(s)

if __name__=='__main__':main()
