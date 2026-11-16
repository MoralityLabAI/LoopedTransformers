"""Exact-rational witness: recurrence can change a feature's causal role.

This is an analytic *toy*, independent of trained SMC or LoopedChase weights.
SAE activations and all clean classifications agree in the two teachers,
while a fixed matched active-feature kill yields opposite signed effects
after recurrent continuation. Fraction arithmetic validates every result.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path


def dot(a: tuple[Q, ...], b: tuple[Q, ...]) -> Q:
    if len(a) != len(b):
        raise ValueError("vector dimension mismatch")
    return sum((x*y for x,y in zip(a,b)), Q(0))


def matvec(M: tuple[tuple[Q, ...], ...], x: tuple[Q, ...]) -> tuple[Q, ...]:
    if len(M) != len(x) or any(len(row) != len(x) for row in M):
        raise ValueError("matrix shape mismatch")
    return tuple(dot(row,x) for row in M)


def loop_margin(h: tuple[Q, ...], transition: tuple[tuple[Q, ...], ...],
                readout: tuple[Q, ...], loops: int, offset: Q = Q(10)) -> Q:
    if loops < 0 or loops > 128:
        raise ValueError("loop budget outside exact witness")
    state = h
    for _ in range(loops):
        state = matvec(transition,state)
    return offset + dot(readout,state)


def loop_gradient(transition: tuple[tuple[Q, ...], ...],
                  readout: tuple[Q, ...], loops: int) -> tuple[Q, ...]:
    """Exact gradient (W^T)^loops c for a weight-tied linear recurrence."""
    if loops < 0 or loops > 128:
        raise ValueError("invalid loop count")
    g = readout
    for _ in range(loops):
        g = tuple(sum((transition[row][col]*g[row]
                       for row in range(len(g))),Q(0)) for col in range(len(g)))
    return g


def basis(i: int, n: int = 6) -> tuple[Q, ...]:
    return tuple(Q(int(j==i)) for j in range(n))


def transition(sign: int) -> tuple[tuple[Q, ...], ...]:
    if sign not in (-1,1):
        raise ValueError("teacher sign must be fixed +/-1")
    M=[list(basis(i)) for i in range(6)]
    M[4][3]=Q(sign,2)
    return tuple(tuple(row) for row in M)


def difference(h: tuple[Q, ...], W: tuple[tuple[Q, ...], ...],
               c: tuple[Q, ...], kill: tuple[Q, ...],
               sham: tuple[Q, ...], loops: int) -> Q:
    """Kill minus sham margin loss = margin(h-sham) - margin(h-kill)."""
    after_sham=tuple(x-y for x,y in zip(h,sham))
    after_kill=tuple(x-y for x,y in zip(h,kill))
    return loop_margin(after_sham,W,c,loops)-loop_margin(after_kill,W,c,loops)


def witness() -> dict:
    h=(Q(1),Q(1),Q(1),Q(1),Q(0),Q(0))
    q=Q(5,4) # z_j=1 plus a 1/4 exact overshoot in this analytic witness
    kill=tuple(q*x for x in basis(3))
    sham=tuple(q*x for x in basis(5))
    c=basis(4)
    Wplus,Wminus=transition(+1),transition(-1)
    all_rows=[basis(j) for j in range(6)]+[basis(4) for _ in range(86)]
    # 92 inactive rows spanning the full six-dimensional hidden space:
    # six coordinate rows and 86 duplicates. A=(e0,e1,e2,e3).
    full_rank=6
    zero=Q(0)
    pre_kill=tuple(dot(row,kill) for row in all_rows)
    pre_sham=tuple(dot(row,sham) for row in all_rows)
    discrepancy_sq=sum(((a-b)*(a-b) for a,b in zip(pre_kill,pre_sham)),zero)
    # The unconstrained tangent least-squares solution is zero and the
    # floor squared is q^2: the e3 inactive measurement is unavoidable.
    floor_sq=q*q
    if discrepancy_sq != Q(2)*q*q:
        raise AssertionError("exact inactive displacement witness changed")
    values=[]
    for k in (0,1,2,4,8):
        p=difference(h,Wplus,c,kill,sham,k)
        m=difference(h,Wminus,c,kill,sham,k)
        gp=loop_gradient(Wplus,c,k)
        gm=loop_gradient(Wminus,c,k)
        if p!=dot(gp,tuple(a-b for a,b in zip(kill,sham))):
            raise AssertionError("exact positive looper gradient disagrees")
        if m!=dot(gm,tuple(a-b for a,b in zip(kill,sham))):
            raise AssertionError("exact negative looper gradient disagrees")
        if p != -m or p != Q(5*k,8):
            raise AssertionError("opposite causal effects not reproduced")
        if loop_margin(h,Wplus,c,k)<=0 or loop_margin(h,Wminus,c,k)<=0:
            raise AssertionError("clean classification differs from reference label")
        values.append({"loops":k,"positive_teacher_effect":str(p),
                       "negative_teacher_effect":str(m),
                       "positive_target_gradient":str(gp[3]),
                       "negative_target_gradient":str(gm[3])})
    return {"schema":"LS-SAE01/exact-rational-loop-depth-witness-v1",
            "status":"synthetic_math_only",
            "dimension":6,"inactive_row_count":len(all_rows),
            "inactive_encoder_rank":full_rank,
            "source_clean_code":list(map(str,h[:4])),
            "matched_feature_index":3,
            "protected_feature_indices":[0,1,2],
            "same_clean_task_classification":True,
            "kill_and_sham_squared_norm":str(q*q),
            "unavoidable_inactive_residual_squared":str(floor_sq),
            "equal_dose_sham_inactive_residual_squared":str(discrepancy_sq),
            "effects_by_loop":values,
            "provenance":"exact constructed affine encoder and linear tied recurrence; no checkpoint"}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    data=witness()
    raw=json.dumps(data,sort_keys=True,indent=2)+"\n"
    if args.out is None:
        print(raw,end="")
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(raw)
        print(args.out)


if __name__=="__main__":
    main()
