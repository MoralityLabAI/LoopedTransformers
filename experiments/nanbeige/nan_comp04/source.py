"""Exact source-only finite-series oracle for NAN-COMP-04.

No Torch, model weights, registered ARG-03 test, or host-model outcomes.
Given f=1+a*x^m+c*x^(m+2), m odd >=3, N(f) and R(f) coincide
through degree 2m-1, then separate by a^2 at 2m and 2ac at 2m+2.
"""
from __future__ import annotations

OPS=("I","N","S","R")
COEFFICIENTS=(-5,-4,-3,-2,-1,1,2,3,4,5)


def parameters(m:int,a:int,c:int)->None:
    if type(m) is not int or m<3 or m%2!=1:
        raise ValueError("odd m>=3 required")
    if type(a) is not int or type(c) is not int or a not in COEFFICIENTS or c not in COEFFICIENTS:
        raise ValueError("source coefficients must be in declared support")


def make_source(m:int,a:int,c:int)->list[int]:
    parameters(m,a,c)
    f=[0]*(2*m+3)
    f[0]=1
    f[m]=a
    f[m+2]=c
    return f


def reciprocal(f:list[int])->list[int]:
    """Independent coefficient recursion: b0=1, b_n=-sum_{k=1}^n a_k b_{n-k}."""
    if not f or f[0]!=1:
        raise ValueError("unit constant coefficient required")
    b=[1]
    for n in range(1,len(f)):
        b.append(-sum(f[k]*b[n-k] for k in range(1,n+1)))
    return b


def multiply(f:list[int],g:list[int])->list[int]:
    n=min(len(f),len(g))
    return [sum(f[j]*g[k-j] for j in range(k+1)) for k in range(n)]


def transform(f:list[int],op:str)->list[int]:
    if op=="I":
        return f[:]
    if op=="N":
        return [v*(-1 if k%2 else 1) for k,v in enumerate(f)]
    if op=="S":
        return [f[k//2] if k%2==0 else 0 for k in range(len(f))]
    if op=="R":
        return reciprocal(f)
    raise ValueError("invalid operator")


def certificate(m:int,a:int,c:int)->dict:
    f=make_source(m,a,c)
    vals={op:transform(f,op) for op in OPS}
    if vals["N"][:2*m]!=vals["R"][:2*m]:
        raise AssertionError("N/R early collision failed")
    if vals["R"][2*m]-vals["N"][2*m]!=a*a:
        raise AssertionError("first distinguishing coefficient failed")
    if vals["R"][2*m+2]-vals["N"][2*m+2]!=2*a*c:
        raise AssertionError("cross-term failed")
    if multiply(f,vals["R"])!=[1]+[0]*(len(f)-1):
        raise AssertionError("power-series inverse certificate failed")
    if len({tuple(vals[op][:2*m+1]) for op in OPS})!=4:
        raise AssertionError("four-way late classification not unique")
    return {"m":m,"a":a,"c":c,
            "first_distinguishing_degree":2*m,
            "early_collision_through":2*m-1,
            "difference_at_2m":a*a,
            "difference_at_2m_plus_2":2*a*c,
            "series":vals,"source":f}


def signed_digits(value:int)->str:
    """Fixed character/tokenizable numeral representation for the prompt."""
    if type(value) is not int or abs(value)>999:
        raise ValueError("three-digit signed decimal bound")
    n=abs(value)
    return f'{"P" if value>=0 else "M"} {n//100} {(n//10)%10} {n%10}'


def format_series(values:list[int])->str:
    return " | ".join(signed_digits(x) for x in values)


def degrees(m:int)->dict[str,int]:
    if m not in (3,5):
        raise ValueError("only registered source exponents m=3,5")
    return {"early":2*m-1,"first":2*m,"cross":2*m+2}
