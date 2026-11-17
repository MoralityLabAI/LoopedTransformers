"""LS-SAE05 matrix-free native 512D power iteration of recurrent J^T J.

Source and all candidate directions are frozen before task labels. Uses actual
Torch forward JVP and reverse VJP of the same two-ACT-step 97x512 output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as PACK
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as HIST
from mechinterp.sae_causal_transfer.native_singular_v1 import families as FAMILY
from mechinterp.sae_causal_transfer.native_jvp_v1 import geometry as GEO
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT
from mechinterp.sae_causal_transfer.observer import encode_top4

SOURCE_MANIFEST="cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a"
SOURCE_NPZ="073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25"
SOURCE_ARRAY="83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec"
SOURCE_PHYSICAL="14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb"
CHECKPOINT="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
PROTOCOL_COMMIT="36009cb071ab9d283cfae86d40f7e46cfbc608a6"
SCHEMA="LS-SAE05/native-512D-matrix-free-singular-power-v1"
POWER_UPDATES=4
RANDOM_GAIN_CONTROLS=8
SYMMETRIC_RADIUS=.05
BOOTSTRAPS=2000


def sha_seed(*items):
    material="LS-SAE05|"+"|".join(map(str,items))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8],"big")&((1<<63)-1)


def normed(x):
    a=np.asarray(x,dtype=np.float64)
    n=float(np.linalg.norm(a))
    if n<=1e-12 or not np.isfinite(n):
        return None
    return a/n


def frozen_output_from_site(model,anchor,token,site):
    """F2:R^512 -> R^(1,97,512), no in-place mutations of a primal tensor."""
    original=anchor["zh"].detach()
    zh=torch.cat((original[:,:token,:],site.reshape(1,1,-1),original[:,token+1:,:]),dim=1)
    zl=anchor["zl"].detach()
    x=anchor["x_emb"].detach()
    for _ in range(2):
        zh,zl=model.act_step(zh,zl,x)
    return zh


def one_jvp(model,anchor,token,site,u):
    v=torch.tensor(u,dtype=site.dtype,device=site.device)
    with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        clean,tangent=torch.func.jvp(
            lambda h:frozen_output_from_site(model,anchor,token,h),
            (site.detach(),),(v,))
    if tangent.shape!=(1,97,512) or not bool(torch.isfinite(tangent).all()):
        raise ArithmeticError("native full-state JVP invalid")
    return clean.detach(),tangent.detach()


def one_vjp(model,anchor,token,site,cotangent):
    if cotangent.shape!=(1,97,512):
        raise ValueError("full output cotangent shape")
    with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        variable=site.detach().clone().requires_grad_(True)
        out=frozen_output_from_site(model,anchor,token,variable)
        grad=torch.autograd.grad(out,variable,grad_outputs=cotangent.detach())[0]
    if grad.shape!=(512,) or not bool(torch.isfinite(grad).all()):
        raise ArithmeticError("native reverse-mode VJP invalid")
    return grad.detach().double().cpu().numpy()


def power_direction(model,anchor,token,family,seed):
    """Exactly four JVP/VJP updates; no labels, task margins or output rivals."""
    site=anchor["zh"].detach()[0,token].clone()
    init=np.random.default_rng(sha_seed(family,seed,"power-start")).normal(size=512)
    u=normed(init)
    if u is None:
        return None,[{"status":"power_seed_zero"}]
    history=[]
    for t in range(POWER_UPDATES):
        _,v=one_jvp(model,anchor,token,site,u)
        v_norm=float(torch.linalg.vector_norm(v))
        if v_norm<=1e-12 or not math.isfinite(v_norm):
            history.append({"step":t,"status":"forward_tangent_zero",
                            "input_direction_512":u.tolist(),"output_gain":v_norm})
            return None,history
        w=one_vjp(model,anchor,token,site,v)
        w_norm=float(np.linalg.norm(w))
        if w_norm<=1e-12 or not math.isfinite(w_norm):
            history.append({"step":t,"status":"reverse_cotangent_zero",
                            "input_direction_512":u.tolist(),
                            "output_gain":v_norm,"reverse_gradient_norm":w_norm})
            return None,history
        history.append({"step":t,"status":"eligible",
                        "input_direction_512":u.tolist(),
                        "output_gain":v_norm,
                        "reverse_gradient_norm":w_norm,
                        "reverse_gradient_512":w.tolist(),
                        "jvp_dot_vjp":float(u@w),
                        "squared_gain":v_norm*v_norm})
        u=w/w_norm
    return u,history


def native_head_and_sae(model,anchor,fit,position):
    """Pre-outcome controls sourced from original clean step1 and frozen dictionary."""
    pos=model.puzzle_emb_len+position
    head=model.lm_head.weight.detach().double().cpu().numpy()[2:11]
    with torch.no_grad():
        first=model.digit_logits(anchor["zh"])[0,position].detach().double().cpu().numpy()
    order=np.argsort(-first,kind="stable")
    task=normed(head[int(order[0])]-head[int(order[1])])
    if task is None:
        raise ArithmeticError("predicted task-only head direction is zero")
    site=anchor["zh"].detach()[0,pos].cpu().numpy().astype(np.float64)
    reduced=fit.encode(site)
    code,_=encode_top4(reduced,fit.physical_rows,fit.physical_bias)
    top=int(np.argmax(code))
    sorted_z=np.sort(code)[::-1]
    why="eligible"
    if sorted_z[0]<=0 or sorted_z[0]-sorted_z[1]<=1e-8:
        sae=None;why="top1_inactive_or_tied"
    else:
        sae=normed(fit.physical_decoder[top]@fit.basis.T)
        if sae is None:why="decoder_direction_zero"
    return sae,task,{"sae_feature":top,"sae_status":why,"sae_code":float(code[top]),
                     "clean_predicted_winner":int(order[0]),
                     "clean_predicted_runner":int(order[1])}


def score_direction(model,anchor,token,site,u):
    if u is None:return None
    clean,tan=one_jvp(model,anchor,token,site,u)
    vectors=tan[0].double().cpu().numpy()
    token_norm=np.linalg.norm(vectors,axis=1)
    goal=vectors[token]
    total=float(np.linalg.norm(token_norm))
    goalnorm=float(np.linalg.norm(goal))
    off=math.sqrt(max(0,total**2-goalnorm**2))
    return {"direction_512":u.tolist(),
            "all97_token_norms":token_norm.tolist(),
            "goal_tangent_512":goal.tolist(),
            "full_jvp_gain":total,"goal_jvp_gain":goalnorm,
            "other_tokens_gain":off,
            "goal_energy_fraction":goalnorm**2/total**2 if total>1e-12 else None}


def finite_margin(model,anchor,token,y,rival,u):
    site=anchor["zh"].detach()[0,token]
    v=torch.tensor(u,dtype=site.dtype,device=site.device)
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        a=frozen_output_from_site(model,anchor,token,site+SYMMETRIC_RADIUS*v)
        b=frozen_output_from_site(model,anchor,token,site-SYMMETRIC_RADIUS*v)
        ca=model.digit_logits(a)[0,token-model.puzzle_emb_len].float()
        cb=model.digit_logits(b)[0,token-model.puzzle_emb_len].float()
        p=float(ca[y]-ca[rival]);m=float(cb[y]-cb[rival])
    return {"plus_margin":p,"minus_margin":m,
            "finite_secant":(p-m)/(2*SYMMETRIC_RADIUS)}


def score_goal(model,anchor,fit,family,row,pos):
    token=model.puzzle_emb_len+pos
    site=anchor["zh"].detach()[0,token].clone()
    power,progress=power_direction(model,anchor,token,family,row["seed"])
    sae,task,source=native_head_and_sae(model,anchor,fit,pos)
    randoms=[normed(np.random.default_rng(sha_seed(
        family,row["seed"],"random-gain",j)).normal(size=512))
             for j in range(RANDOM_GAIN_CONTROLS)]
    if any(v is None for v in randoms):
        raise AssertionError("registered random JVP control is degenerate")
    directions={"power":power,"sae":sae,"task_head":task}
    for j,v in enumerate(randoms):
        directions[f"random_{j}"]=v
    scored={name:score_direction(model,anchor,token,site,v)
            for name,v in directions.items()}
    # Outcomes read only after all input directions and power iterations freeze.
    y=int(row["target"][pos])-1
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        clean=frozen_output_from_site(model,anchor,token,site)
        clean_logits=model.digit_logits(clean)[0,pos].detach().double().cpu().numpy()
    c=clean_logits.copy();c[y]=-np.inf
    rival=int(np.argmax(c))
    head=model.lm_head.weight.detach().double().cpu().numpy()[2:11]
    readout=head[y]-head[rival]
    with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        site_grad=site.detach().clone().requires_grad_(True)
        logits=model.digit_logits(frozen_output_from_site(model,anchor,token,site_grad))[0,pos]
        grad=torch.autograd.grad(logits[y]-logits[rival],site_grad)[0]
    source_gradient=grad.detach().double().cpu().numpy()
    for name,entry in scored.items():
        if entry is None:continue
        slope=float(readout@np.asarray(entry["goal_tangent_512"]))
        reverse=float(source_gradient@np.asarray(entry["direction_512"]))
        if abs(slope-reverse)>max(.02,.05*abs(slope)):
            raise ArithmeticError("native forward/reverse margin-gradient mismatch")
        entry["task_margin_slope"]=slope
        entry["reverse_mode_slope"]=reverse
        den=entry["goal_jvp_gain"]*float(np.linalg.norm(readout))
        entry["task_readout_abs_cosine"]=abs(slope)/den if den>1e-12 else None
    for name in ("power","sae","task_head","random_0"):
        if scored[name] is not None:
            finite=finite_margin(model,anchor,token,y,rival,
                                np.asarray(scored[name]["direction_512"]))
            finite["secant_minus_jvp"]=finite["finite_secant"]-scored[name]["task_margin_slope"]
            scored[name]["finite_margin"]=finite
    null_gains=[scored[f"random_{j}"]["full_jvp_gain"] for j in range(RANDOM_GAIN_CONTROLS)]
    power_gain=scored["power"]["full_jvp_gain"] if scored["power"] is not None else None
    best_rand=max(null_gains)
    event=bool(power_gain is not None and power_gain>=.25 and power_gain>=1.20*best_rand)
    return {"source_metadata":source,"seed":row["seed"],"goal_position":pos,
            "true_digit_zero_based":y,"rival_digit_zero_based":rival,
            "clean_logits9":clean_logits.tolist(),
            "clean_readout_row_512":readout.tolist(),
            "clean_gradient_512":source_gradient.tolist(),
            "power_iterations":progress,
            "directions":scored,
            "best_random_full_jvp_gain":best_rand,
            "power_full_jvp_gain":power_gain,
            "gain_difference":power_gain-best_rand if power_gain is not None else None,
            "event":event}


def family_score(rows,family):
    draws=[]
    pairs=[(r["gain_difference"] if r["gain_difference"] is not None else
            -r["best_random_full_jvp_gain"]) for r in rows]
    events=sum(bool(r["event"]) for r in rows)
    admissible=sum(r["power_full_jvp_gain"] is not None for r in rows)
    rng=random.Random(202625130+len(family))
    for _ in range(BOOTSTRAPS):
        picks=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws.append(sum(picks)/len(picks))
    draws.sort()
    ci=[draws[50],draws[1949]]
    passed=(len(rows)==8 and events>=6 and len(draws)==2000 and ci[0]>0)
    return {"n_puzzles":len(rows),"eligible_power":admissible,
            "power_gain_events":events,"mean_power_minus_best_random_gain":sum(pairs)/len(pairs),
            "puzzle_bootstrap95":ci,"bootstrap_draws":len(draws),
            "registered_success":passed}


def run(*,pilot=False,source_dir,out=None):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    from mechinterp.sae_causal_transfer.native_jvp_v2.fit_pack import load_pack
    source_dir=Path(source_dir)
    fit,manifest,manifest_sha=load_pack(
        source_dir/"frozen_calibration.npz",
        source_dir/"frozen_calibration_manifest.json")
    if (manifest_sha!=SOURCE_MANIFEST or
        manifest["npz_file_sha256"]!=SOURCE_NPZ or
        fit.fingerprint!=SOURCE_ARRAY or
        manifest["physical_dictionary_sha256"]!=SOURCE_PHYSICAL):
        raise ValueError("original LS-SAE04 preheldout source artifact changed")
    original,source_sha=TRM.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if source_sha!=CHECKPOINT:
        raise ValueError("pretrained network byte SHA mismatch")
    original.eval().requires_grad_(False)
    all_manifest=FAMILY.manifest()
    result={}
    names=("pilot_30","pilot_34") if pilot else ("test_30","test_34")
    for fam in names:
        rows=FAMILY.roster(fam)
        _,anchors=FIT.stage_one(original,rows,retain_zl=True)
        cases=[]
        for i,(r,anchor) in enumerate(zip(rows,anchors)):
            pos=FAMILY.median_blank(r["x"])
            one=score_goal(original,anchor,fit,
                           fam.replace("pilot_","test_"),r,pos)
            one["puzzle_index"]=i
            cases.append(one)
            print("LS_SAE05_CASE",fam,i,one["event"],one["power_full_jvp_gain"],
                  one["best_random_full_jvp_gain"],flush=True)
        score=family_score(cases,fam.replace("pilot_","test_"))
        result[fam]={"per_puzzle":cases,"statistics":score}
        print("LS_SAE05_PRIMARY",fam,json.dumps(score,sort_keys=True),flush=True)
    report={
      "schema":SCHEMA,"protocol_commit":PROTOCOL_COMMIT,
      "mode":"pilot" if pilot else "full",
      "source_checkpoint_sha256":source_sha,
      "source_calibration_npz_sha256":manifest["npz_file_sha256"],
      "source_calibration_manifest_sha256":manifest_sha,
      "source_fit_fingerprint_sha256":fit.fingerprint,
      "all_roster_sha256":all_manifest,
      "evaluated_families":result,
      "power_updates":POWER_UPDATES,"random_gain_controls":RANDOM_GAIN_CONTROLS,
      "finite_margin_radius":SYMMETRIC_RADIUS,
      "full_protocol":not pilot,
      "registered_success_both_families":bool(not pilot and
        all(result[f]["statistics"]["registered_success"] for f in ("test_30","test_34"))),
      "limitations":"Finite matrix-free forward/reverse native JVP numerics at a frozen checkpoint. Power search per heldout puzzle is label-blind but unequal compute to source SAE/head baselines."
    }
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run(pilot=args.pilot,source_dir=args.source_dir,out=args.out)
    print(json.dumps({"out":str(args.out),"pass":result["registered_success_both_families"],
                      "stats":{f:v["statistics"] for f,v in result["evaluated_families"].items()}},indent=2))


if __name__=="__main__":
    main()
