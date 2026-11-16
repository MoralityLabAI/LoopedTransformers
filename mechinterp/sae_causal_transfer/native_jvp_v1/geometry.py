"""LS-SAE03 frozen vector controls, JVP norm bookkeeping, and puzzle bootstrap."""
from __future__ import annotations
import hashlib
import math
import random
import numpy as np

RADIUS=0.2
DEPTHS=(0,1,2)
N_RANK_NULL=128
N_BOOTSTRAPS=2000
MIN_INITIAL_SLOPE=.05
MAX_ATTENUATION_RATIO=.10
MIN_SURVIVING_GLOBAL_GAIN=.25


def sha_seed(*items):
    material="|".join(map(str,("LS-SAE03",)+items))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8],"big") & ((1<<63)-1)


def unit(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    length=float(np.linalg.norm(x))
    if not math.isfinite(length) or length<=1e-8:
        return None
    return x/length


def choose_directions(reduced,physical_rows,physical_bias,
                      decoder_columns,basis512,head9,clean_logits9,
                      family,seed):
    """Pre-outcome directions use only clean activations/visible ID/prediction."""
    from mechinterp.sae_causal_transfer.observer import encode_top4
    x=np.asarray(reduced,dtype=np.float64)
    if x.shape!=(48,) or basis512.shape!=(512,48) or head9.shape!=(9,512):
        raise ValueError("frozen LS-SAE02 48D PCA/512D native head required")
    z,_=encode_top4(x,physical_rows,physical_bias)
    ordered=np.sort(z)[::-1]
    atom=int(np.argmax(z))
    result={"sae_feature":atom,"sae_status":"eligible","sae_code":float(z[atom])}
    sae=None
    if ordered[0]<=0 or ordered[0]-ordered[1]<=1e-8:
        result["sae_status"]="top1_inactive_or_tied"
    else:
        sae=unit(np.asarray(decoder_columns[atom])@basis512.T)
        if sae is None:
            result["sae_status"]="zero_native_decoder_direction"
    winner=int(np.argmax(clean_logits9))
    top=np.argsort(-np.asarray(clean_logits9),kind="stable")
    runner=int(top[1])
    task=unit(head9[winner]-head9[runner])
    if task is None:
        raise ArithmeticError("clean top2 task-only direction degenerate")
    head_null=None
    if sae is not None:
        _,sing,vh=np.linalg.svd(head9,full_matrices=False)
        if float(sing[-1])<=1e-10:
            raise ArithmeticError("9 digit head rows rank deficient")
        head_null=unit(sae-vh.T@(vh@sae))
    rand=unit(np.random.default_rng(sha_seed(family,seed,"random-jvp")).normal(size=512))
    if rand is None:raise AssertionError("impossible seeded random direction")
    return {
      "meta":{**result,"clean_winner":winner,"clean_runner":runner},
      "directions":{"sae":sae,"task_head":task,
                    "head_null":head_null,"random":rand}
    }


def random_rank_baseline(gradient,sae,family,seed):
    """128 independent rank-one observer directions; separate from JVP null."""
    g=np.asarray(gradient,dtype=np.float64)
    gn=float(np.linalg.norm(g))
    if gn<=1e-12 or sae is None:
        return {"rank_p":None,"sae_cosine":None,"random_cosines":[]}
    sae_cos=abs(float(g@sae))/gn
    null=[]
    for j in range(N_RANK_NULL):
        u=unit(np.random.default_rng(sha_seed(family,seed,"rank-null",j)).normal(size=512))
        null.append(abs(float(g@u))/gn)
    p=(1+sum(z>=sae_cos-1e-15 for z in null))/(N_RANK_NULL+1)
    return {"rank_p":p,"sae_cosine":sae_cos,"random_cosines":null}


def tangent_stats(tangent,token,readout):
    """Full tokenwise JVP diagnostic, with exact readout from target 512-vector."""
    x=np.asarray(tangent,dtype=np.float64)
    if x.shape!=(1,97,512) or not np.all(np.isfinite(x)):
        raise ValueError("native JVP must cover every one of 97×512 state coordinates")
    if not 16<=token<97:
        raise ValueError("expected native board token index")
    norms=np.linalg.norm(x[0],axis=1)
    global_gain=float(np.linalg.norm(norms))
    goal=x[0,token]
    goal_gain=float(norms[token])
    off_gain=math.sqrt(max(0,global_gain*global_gain-goal_gain*goal_gain))
    readout=np.asarray(readout,dtype=np.float64)
    slope=float(goal@readout)
    den=goal_gain*float(np.linalg.norm(readout))
    cosine=abs(slope)/den if den>1e-12 else None
    return {
        "all97_token_norms":norms.tolist(),
        "full_state_gain":global_gain,
        "goal_state_gain":goal_gain,
        "other_tokens_gain":off_gain,
        "target_energy_fraction":goal_gain**2/global_gain**2 if global_gain>1e-12 else None,
        "goal_tangent_512":goal.tolist(),
        "readout_row_512":readout.tolist(),
        "task_margin_slope":slope,
        "task_readout_abs_cosine":cosine
    }


def attenuation_event(direction):
    if direction is None:
        return False
    a=direction["0"]["task_margin_slope"]
    b=direction["2"]["task_margin_slope"]
    gain=direction["2"]["full_state_gain"]
    return bool(abs(a)>=MIN_INITIAL_SLOPE and
                abs(b)/(abs(a)+1e-6)<=MAX_ATTENUATION_RATIO and
                gain>=MIN_SURVIVING_GLOBAL_GAIN)


def family_score(rows,family):
    if family not in ("test_30","test_34"):
        raise ValueError("unregistered test family")
    events=[]
    admitted=0
    for row in rows:
        if row["sae_admitted"]:admitted+=1
        sae=attenuation_event(row["directions"].get("sae"))
        rnd=attenuation_event(row["directions"].get("random"))
        events.append((sae,rnd))
    rng=random.Random(202622130+len(family))
    draws=[]
    for _ in range(N_BOOTSTRAPS):
        sample=[events[rng.randrange(len(rows))] for _ in rows]
        if sample:
            draws.append(sum(int(a)-int(b) for a,b in sample)/len(sample))
    draws.sort()
    ci=[draws[50],draws[1949]] if len(draws)==N_BOOTSTRAPS else None
    sae_count=sum(a for a,_ in events)
    random_count=sum(b for _,b in events)
    diff=(sae_count-random_count)/len(rows) if rows else None
    passed=(len(rows)==12 and admitted>=10 and sae_count>=8 and
            len(draws)==N_BOOTSTRAPS and ci is not None and ci[0]>0)
    return {
        "n_puzzles":len(rows),
        "sae_admitted":admitted,
        "sae_attenuation_with_survival_events":sae_count,
        "random_attenuation_with_survival_events":random_count,
        "paired_event_fraction_difference":diff,
        "paired_puzzle_bootstrap95":ci,
        "valid_bootstrap_draws":len(draws),
        "registered_success":bool(passed)
    }
