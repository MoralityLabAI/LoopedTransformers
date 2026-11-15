"""Prospective ARG-03 final-R development calibration and rank metrics.

Consumes *new development* model answer tapes and source-only oracle records.
Use a separate grader process to add a correct_program index, never expose
gold programs to the Nanbeige forward process. No sealed test use.

Each operator position has 4 bias entries (12 stored parameters); softmax
has one gauge redundancy per position, so this is 9 effective degrees of
freedom. Mean-center each row and L2 shrink to set a reproducible gauge.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, log, isfinite
from typing import Sequence

OPS="INSR"
PROGRAMS=tuple((i,j,k) for i in range(4) for j in range(4) for k in range(4))

@dataclass(frozen=True)
class DevRankRow:
    case_id: str
    split: str
    site_logits: tuple  # [16][3][4], *model-only* tape
    gold_program: int  # grader-only label in range 0..63

    def __post_init__(self) -> None:
        if self.split not in {"development_finalR_fit", "development_finalR_valid"}:
            raise ValueError("Fit and valid rows must be fresh final-R DEVELOPMENT only")
        if not (0<=self.gold_program<64):
            raise ValueError("Gold operator program must be 0..63")
        if PROGRAMS[self.gold_program][2]!=3:
            raise ValueError("Final-R development calibration requires gold final R")
        if len(self.site_logits)!=16 or any(len(ctx)!=3 for ctx in self.site_logits):
            raise ValueError("Need 16 contexts x 3 positions")
        if any(len(vec)!=4 or any(not isfinite(float(z)) for z in vec)
               for ctx in self.site_logits for vec in ctx):
            raise ValueError("Need finite four-op site logits")

def _logprob(v: Sequence[float]) -> list[float]:
    m=max(v)
    z=sum(exp(k-m) for k in v)
    return [float(k-m-log(z)) for k in v]

def program_scores(row: DevRankRow, bias: Sequence[Sequence[float]]) -> list[float]:
    if len(bias)!=3 or any(len(v)!=4 or any(not isfinite(float(x)) for x in v)
                            for v in bias):
        raise ValueError("Expected finite 3x4 operator bias matrix")
    def lp(ctx:int,site:int)->list[float]:
        return _logprob([float(row.site_logits[ctx][site][o])+float(bias[site][o])
                         for o in range(4)])
    a=lp(0,0)
    second=[lp(4*i,1) for i in range(4)]
    third=[[lp(4*i+j,2) for j in range(4)] for i in range(4)]
    return [a[i]+second[i][j]+third[i][j][k]
            for i,j,k in PROGRAMS]

def rank_of_gold(scores: Sequence[float], gold:int)->int:
    if len(scores)!=64 or not (0<=gold<64):
        raise ValueError("Require 64 program scores and a gold index")
    # Stable lower-index tie-break matches original ARG-03 grader.
    return 1+sum((float(s)>float(scores[gold]) or
                  (float(s)==float(scores[gold]) and idx<gold))
                 for idx,s in enumerate(scores) if idx!=gold)

def rank_summary(rows: Sequence[DevRankRow], bias: Sequence[Sequence[float]])->dict:
    if not rows:
        raise ValueError("Rank summary requires rows")
    rank=[rank_of_gold(program_scores(row,bias),row.gold_program) for row in rows]
    a=sorted(rank)
    return {"n":len(rows),"gold_rank_median": (a[(len(a)-1)//2]+a[len(a)//2])/2,
            "top1_correct_program":sum(r==1 for r in rank),
            "gold_in_top12":sum(r<=12 for r in rank),
            "gold_rank":rank}

def fit_position_operator_bias(rows: Sequence[DevRankRow], *,
                               steps:int=300, lr:float=0.12,
                               ridge:float=0.04) -> dict:
    """Fit per-position per-operator biases to DEV-FIT gold NLL.

    The input development-finalR-valid set cannot enter this function. Use
    frozen coefficients for it and compute rank-only *plus independent
    future@12* from the certified oracle in a separate grading process.
    """
    if not rows or len({r.case_id for r in rows})!=len(rows) or any(
        r.split!="development_finalR_fit" for r in rows):
        raise ValueError("Only unique fresh development_finalR_fit rows are allowed")
    if not (0<steps<=10000 and 0<lr<=1 and ridge>=0):
        raise ValueError("Invalid fixed optimizer parameters")
    b=[[0.0]*4 for _ in range(3)]
    for _ in range(steps):
        grad=[[0.0]*4 for _ in range(3)]
        for row in rows:
            i,j,k=PROGRAMS[row.gold_program]
            for t,ctx,truth in ((0,0,i),(1,4*i,j),(2,4*i+j,k)):
                logits=[float(row.site_logits[ctx][t][o])+b[t][o]
                        for o in range(4)]
                probs=[exp(x) for x in _logprob(logits)]
                for o in range(4):
                    grad[t][o]+=probs[o]-(1 if o==truth else 0)
        for t in range(3):
            for o in range(4):
                b[t][o]-=lr*(grad[t][o]/len(rows)+ridge*b[t][o])
            offset=sum(b[t])/4
            b[t]=[x-offset for x in b[t]]
    return {
        "schema":"arg03.finalR.devonly.bias12.v1",
        "fit_split":"development_finalR_fit",
        "fit_cases":len(rows),
        "parameters_3x4":b,
        "stored_parameters":12,
        "effective_degrees_of_freedom":9,
        "optimizer":{"steps":steps,"lr":lr,"ridge":ridge},
        "train_rank_summary":{key:value for key,value in rank_summary(rows,b).items()
                              if key!="gold_rank"},
        "lock_semantics":"Freeze before independent dev-valid and any heldout test."
    }
