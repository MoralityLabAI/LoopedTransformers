"""LBF-20M RAM budget experiment: bounded exact separator-query memo.

No neural gradients, learned scores or oracle labels in the exact solver.
Charge Python payload recursively plus a per-entry container allowance;
the charge is an estimate, not a hard process RSS limit.
"""
from __future__ import annotations
import argparse
from collections import Counter, OrderedDict
from collections.abc import MutableMapping
import json
import math
from pathlib import Path
import sys
import time

import lbf20m_factor as native


def payload_size(x):
    if isinstance(x,(tuple,list)):
        return sys.getsizeof(x)+sum(payload_size(y) for y in x)
    return sys.getsizeof(x)


class CachePool:
    def __init__(self,mib):
        if not math.isfinite(mib) or mib<0:raise ValueError('invalid query cache budget')
        self.cap=int(mib*2**20);self.used=self.peak=self.evictions=self.rejected=0
        self.data=OrderedDict()

    def set(self,key,value):
        charge=payload_size((key,value))+128
        if key in self.data:
            self.used-=self.data.pop(key)[1]
        if charge>self.cap:
            self.rejected+=1;return
        while self.used+charge>self.cap:
            _,(_,old)=self.data.popitem(last=False)
            self.used-=old;self.evictions+=1
        self.data[key]=(value,charge);self.used+=charge
        self.peak=max(self.peak,self.used)


class UnitMemo(MutableMapping):
    def __init__(self,pool,site):
        self.pool,self.site=pool,site

    def __getitem__(self,key):
        address=(self.site,key)
        value,_=self.pool.data[address]
        self.pool.data.move_to_end(address)
        return value

    def __setitem__(self,key,value):
        self.pool.set((self.site,key),value)

    def __delitem__(self,key):
        _,size=self.pool.data.pop((self.site,key));self.pool.used-=size

    def __iter__(self):
        return iter([key for site,key in self.pool.data if site==self.site])

    def __len__(self):
        return sum(site==self.site for site,key in self.pool.data)

    def __contains__(self,key):
        return (self.site,key) in self.pool.data

    def values(self):
        # Storage accounting must not perturb the eviction order.
        return [value for (site,key),(value,charge) in self.pool.data.items() if site==self.site]


def run_case(c,cut,queries,mib):
    native.lift.cache_clear();native.project.cache_clear()
    solver=native.Solver(c);dom=solver.ac(c.domains)
    if dom is None:raise ValueError('development fixture must be arc-consistent')
    st=Counter();wall,cpu=time.perf_counter(),time.process_time()
    engine=native.Factorized(c,dom,cut,'mdd_query_cache',st)
    pool=CachePool(mib)
    for i,u in enumerate(engine.units):u.memo=UnitMemo(pool,i)
    setup=time.perf_counter()-wall
    answers=[engine.answer(q,with_witness=True) for q in queries]
    elapsed,cpu_used=time.perf_counter()-wall,time.process_time()-cpu
    nominal_base,nominal_query=engine.storage()
    truth=[native.dfs_support(solver,dom,q,Counter()) for q in queries]
    for (ans,w),expected,q in zip(answers,truth,queries):
        assert ans==expected and (w is not None)==bool(any(expected))
        if w is not None:
            assert c.valid(w) and all(m&(1<<w[i]) for i,m in q.items())
    return dict(root=c.root,queries=len(queries),answers=[a for a,w in answers],
                exact=True,setup_seconds=setup,wall_seconds=elapsed,process_cpu_seconds=cpu_used,
                base_bits=nominal_base,query_bits=nominal_query,charged_bytes=pool.used,
                peak_charged_bytes=pool.peak,cap_bytes=pool.cap,evictions=pool.evictions,
                rejected=pool.rejected,stats=dict(st))


def main(args):
    rows=[]
    if args.roots<1 or args.repeat<1:raise ValueError('positive workload required')
    for c,cut in native.fresh()[:args.roots]:
        q=native.fresh_queries(c,c.domains,cut)*args.repeat
        rows.append(run_case(c,cut,q,args.mms_mib))
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'receipt.json').write_text(json.dumps(dict(scope='exposed_development_exact',
        budget='estimated query-memo payload only; base tables and global caches excluded',
        rows=rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--roots',type=int,default=12)
    p.add_argument('--repeat',type=int,default=2)
    main(p.parse_args())
