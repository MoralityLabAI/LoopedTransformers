"""LS-SAE11 same-forward state and logits on original + solver-hard Sudoku.

No fitted parameters, model-error selection or cross-run tolerance acceptance.
Every physically edited logit and hidden transport metric shares a forward pass.
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
from torch.nn.attention import SDPBackend, sdpa_kernel

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as STAGE
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as BASE
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import scoring as SCORE
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as FIXED
from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as AUDIT8
from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as OLD
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as HARD

PROTOCOL = "e94a4439db78eacdb4988ecd167bcfb62a91230f"
SCHEMA = "LS-SAE11/within-forward-original-and-solver-hard-native-v1"
ORIGINAL_SHA = "69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"
MODES = {"pilot":("pilot_champagne",), "reference":("test_52","test_54"),
         "hard":("test_champagne_a","test_champagne_b")}
HORIZONS = ("ACT3","ACT6","ACT12")
RADII = ("0.20","2.00","8.00")
ARMS = ("state_power","logit_power","sae","random")
TOL_IDENTITY = 2e-4


def sha(b):return hashlib.sha256(b).hexdigest()


def load_original(path):
    raw=Path(path).read_bytes()
    if sha(raw)!=ORIGINAL_SHA:
        raise ValueError("LS-SAE08 original full evidence SHA differs from frozen declaration")
    data=json.loads(raw)
    AUDIT8.audit(data)
    if data["mode"]!="full" or not data["full_protocol"]:
        raise ValueError("original LS-SAE08 full test missing")
    return data


def load_hard_source(path):
    raw=Path(path).read_bytes()
    data=json.loads(raw)
    if (data["schema"]!="LS-SAE11/source-only-Champagne-20-v1"
        or data["registration_commit"]!=PROTOCOL
        or data["external_repo_commit"]!=HARD.COMMIT
        or data["source_file_hashes"]!=[list(x) for x in HARD.SOURCE_FILES]):
        raise ValueError("LS-SAE11 source-only precursor or file hashes differ")
    HARD.admitted(data["rosters"])
    if {k:HARD.fingerprint(v) for k,v in data["rosters"].items()}!=data["fingerprints"]:
        raise ValueError("source-only roster fingerprint changed")
    return data,sha(raw)


def readout(model):
    weights=model.lm_head.weight.detach().double().cpu().numpy()[2:11]
    wc=weights-weights.mean(axis=0,keepdims=True)
    _,sv,vt=np.linalg.svd(wc,full_matrices=False)
    rank=int(np.count_nonzero(sv > sv[0]*1e-7))
    if rank<1 or rank>8 or sv[0]<=0:
        raise ValueError("invalid centered 9-digit native readout rank")
    q=vt[:rank].T.copy()
    return wc,q,sv[:rank].tolist()


def median_blank(grid):
    indices=[i for i,v in enumerate(grid) if v==0]
    if not indices:raise ValueError("hard source has no blank cells")
    return indices[len(indices)//2]


def donor_for(mode,group,index):
    if mode=="reference":
        return group,index
    if mode=="pilot":
        return ("test_52",index) if index<2 else ("test_54",index-2)
    if group=="test_champagne_a":return "test_52",index
    if group=="test_champagne_b":return "test_54",index
    raise ValueError("unknown donor grouping")


def get_dirs(donor):
    geometry=donor["direction_metadata"]["geometry"]
    status=donor["source_direction_status"]
    result={}
    for arm in ARMS:
        g=geometry[arm]
        if status[arm]!=(g is not None):
            raise ValueError("old LS-SAE08 direction status differs")
        if g is None:
            result[arm]=None
            continue
        arr=np.asarray(g["direction_512"],dtype=np.float64)
        if arr.shape!=(512,) or not np.isfinite(arr).all() or abs(np.linalg.norm(arr)-1)>2e-5:
            raise ValueError("original donor vector is not unit 512D")
        result[arm]=arr
    return result,status


def signed_order(directions):
    seq=[("clean",None,None,0)]
    for arm in ARMS:
        if directions[arm] is None:continue
        for dose in FIXED.RADII:
            for sign in ("plus","minus"):
                seq.append((arm,f"{dose:.2f}",sign,1 if sign=="plus" else -1))
    return seq


def certificates(clean,edited,blanks):
    base=np.asarray(clean,dtype=np.float64)
    pert=np.asarray(edited,dtype=np.float64)
    if base.shape!=(81,9) or pert.shape!=(81,9):
        raise ValueError("complete 81x9 Sudoku logits mandatory")
    # Match native argmax's first-index tie rule; ties are not strict certificates.
    winner=np.argmax(base,axis=1)
    margin=np.asarray([base[i,winner[i]]-max(base[i,j] for j in range(9) if j!=winner[i])
                       for i in blanks],dtype=np.float64)
    m=float(np.min(margin))
    eps=float(np.max(np.abs(pert-base)))
    sufficient=bool(m>2*eps)
    strict=[]
    for i in blanks:
        p=int(winner[i])
        strict.append(all(float(pert[i,p]-pert[i,j])>0 for j in range(9) if j!=p))
    flipped=int(np.count_nonzero(np.argmax(base,axis=1)[blanks]!=np.argmax(pert,axis=1)[blanks]))
    if (sufficient or all(strict)) and flipped!=0:
        raise AssertionError("finite margin certificate contradicted actual argmax")
    return {
       "minimum_clean_blank_winner_margin":m,
       "all81_absolute_epsilon":eps,
       "strict_global_margin_certificate":sufficient,
       "strict_all_blank_pairwise_certificate":bool(all(strict)),
       "strict_pairwise_certified_blank_count":int(sum(strict)),
       "actual_blank_digit_flips":flipped
    }


def metric(dh,dl,base_logits,edited_logits,source,site,dose,wc,q):
    """Independent-source-score-friendly scalars; all native inputs same batch."""
    puzzle=np.asarray(source["x"])
    blanks=np.flatnonzero(puzzle==0)
    clues=np.flatnonzero(puzzle!=0)
    board=dh[16:,:]
    full_h=float(np.linalg.norm(dh))
    full_l=float(np.linalg.norm(dl))
    source_site=float(np.linalg.norm(dh[16+site]))
    other=float(np.sqrt(max(0.,full_h**2-source_site**2)))
    board_norm=float(np.linalg.norm(board))
    projection=board@q
    row=float(np.linalg.norm(projection))
    residual=float(np.linalg.norm(board-projection@q.T))
    fraction=row/board_norm if board_norm>1e-12 else 0.
    centered=edited_logits-base_logits
    centered-=centered.mean(axis=1,keepdims=True)
    identity=float(np.linalg.norm(centered-board@wc.T))
    if identity>TOL_IDENTITY:
        raise ArithmeticError(f"same-run centered native readout identity failed: {identity}")
    cert=certificates(base_logits,edited_logits,blanks)
    return {
        "h_global_l2":full_h,"l_global_l2":full_l,
        "h_gain":full_h/dose,"l_gain":full_l/dose,
        "h_source_site_l2":source_site,"h_non_site_l2":other,
        "h_board_l2":board_norm,
        "h_blank_l2":float(np.linalg.norm(board[blanks])),
        "h_given_l2":float(np.linalg.norm(board[clues])),
        "h_readout_l2":row,"h_null_l2":residual,
        "readout_fraction":fraction,
        "centered_logit_fro":float(np.linalg.norm(centered)),
        "centered_readout_reconstruction_error":identity,
        "classification":{
           "contracted":full_h/dose<0.1,
           "persistent_output_silent":full_h/dose>=1.0 and cert["actual_blank_digit_flips"]==0,
           "readout_small":board_norm>1e-8 and fraction<0.1,
        },
        **cert,
    }


def archive_drift(archived,now):
    orig=np.asarray(archived,dtype=np.float64)
    cur=np.asarray(now,dtype=np.float64)
    if orig.shape!=(81,9) or cur.shape!=orig.shape:return None
    return {
        "max_absolute_logit_drift":float(np.max(np.abs(orig-cur))),
        "all81_argmax_changes":int(np.count_nonzero(orig.argmax(axis=1)!=cur.argmax(axis=1))),
    }


def run_case(model,source,anchor,donor,donor_family,donor_index,wc,q,mode):
    site=median_blank(source["x"])
    token=model.puzzle_emb_len+site
    if mode=="reference" and site!=donor["visible_goal_position"]:
        raise ValueError("original LS-SAE08 median blank site moved")
    directions,status=get_dirs(donor)
    order=signed_order(directions)
    n=len(order)
    base_norm=float(torch.linalg.vector_norm(anchor["zh"][0,token].float()))
    if not math.isfinite(base_norm) or base_norm<=0:
        raise ValueError("invalid original ACT1 zH source token")
    outputs={}
    measures={}
    scores={}
    diagnostics={}
    blanks=np.flatnonzero(np.asarray(source["x"])==0)
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        zh=anchor["zh"].detach().repeat(n,1,1)
        zl=anchor["zl"].detach().repeat(n,1,1)
        x_emb=anchor["x_emb"].detach().repeat(n,1,1)
        for idx,(arm,r,sign,s) in enumerate(order[1:],1):
            unit=torch.tensor(directions[arm],dtype=zh.dtype,device=zh.device)
            zh[idx,token]+=float(r)*s*unit
        for step in range(1,12):
            zh,zl=model.act_step(zh,zl,x_emb)
            if step not in (2,5,11):continue
            horizon=f"ACT{step+1}"
            arr=model.digit_logits(zh).detach().double().cpu().numpy()
            if arr.shape!=(n,81,9) or not np.isfinite(arr).all():
                raise ArithmeticError("native same-forward 81x9 logits missing or nonfinite")
            raw={"clean":arr[0].tolist()}
            cm=np.argmax(arr[0],axis=1)+1
            original_metrics=SCORE.digit_metrics(arr[0],source["x"],source["target"],site,cm.tolist())
            meta={"clean":original_metrics,"arms":{}}
            vrows={"arms":{}}
            diag={"clean":None,"arms":{}}
            if mode=="reference":
                diag["clean"]=archive_drift(donor["native_81x9_logits"][horizon]["clean"],arr[0])
            for idx,(arm,dose,sign,_) in enumerate(order[1:],1):
                ref_idx=idx+1
                signed=arr[ref_idx]
                key=dose
                raw.setdefault(arm,{}).setdefault(key,{})[sign]=signed.tolist()
                # Keep state and complete logits from same exact ACT step and 25-variant batch.
                dh=(zh[ref_idx].float()-zh[0].float()).detach().double().cpu().numpy()
                dl=(zl[ref_idx].float()-zl[0].float()).detach().double().cpu().numpy()
                geom=metric(dh,dl,arr[0],signed,source,site,float(dose),wc,q)
                vrows["arms"].setdefault(arm,{}).setdefault(key,{})[sign]=geom
                sign_score=SCORE.digit_metrics(signed,source["x"],source["target"],site,cm.tolist())
                meta["arms"].setdefault(arm,{}).setdefault(key,{})[sign]=sign_score
                if mode=="reference":
                    orig=donor["native_81x9_logits"][horizon][arm][key][sign]
                    diag["arms"].setdefault(arm,{}).setdefault(key,{})[sign]=archive_drift(orig,signed)
            outputs[horizon]=raw
            scores[horizon]=meta
            measures[horizon]=vrows
            if mode=="reference":diagnostics[horizon]=diag
    if set(outputs)!=set(HORIZONS):
        raise AssertionError("all registered native ACT horizons missing")
    return {
       "source_seed":source["seed"],"source_file":source.get("source_file"),
       "source_site":site,"source_ACT1_zH_norm":base_norm,
       "donor_family":donor_family,"donor_index":donor_index,
       "source_direction_status":status,
       "native_81x9_logits":outputs,
       "same_run_transport":measures,
       "native_task_scores":scores,
       "cross_run_original_drift":diagnostics if mode=="reference" else None
    }


def family_summary(cases,family):
    if family not in ("test_champagne_a","test_champagne_b"):
        raise ValueError("registered prospective hard cohort expected")
    if len(cases)!=8:
        raise ValueError("hard family denominator deviates from eight")
    diffs=[];events=0;clean_inexact=0
    for case in cases:
        sc=case["native_task_scores"]["ACT6"]
        clean=sc["clean"]
        clean_inexact+=int(clean["blank_correct"]<clean["blank_total"])
        def gain(arm):
            signs=sc["arms"].get(arm,{}).get("8.00")
            if signs is None:
                return 0.
            return .5*(signs["plus"]["blank_correct"]+signs["minus"]["blank_correct"])-clean["blank_correct"]
        d=float(gain("state_power")-gain("random"))
        diffs.append(d);events+=int(d>=1)
    rng=random.Random(202611170+len(family))
    draw=[]
    for _ in range(2000):
        chosen=[rng.randrange(8) for _ in range(8)]
        draw.append(sum(diffs[j] for j in chosen)/8.)
    draw.sort()
    return {
      "n_puzzles":8,"clean_ACT6_inexact":clean_inexact,
      "paired_state_minus_random_blank_correctness_gain":diffs,
      "events_D_at_least_one":events,
      "mean_paired_gain":float(sum(diffs)/8),
      "puzzle_bootstrap95":[draw[50],draw[1949]],
      "bootstrap_draws":2000,
      "registered_success":bool(clean_inexact>=4 and events>=6 and draw[50]>0)
    }


def run(mode,baseline,source_data,out):
    if mode not in MODES:
        raise ValueError("unregistered run mode")
    torch.set_num_threads(4)
    torch.manual_seed(71)
    original=load_original(baseline)
    hard,hard_sha=load_hard_source(source_data)
    model,model_sha=TRM.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if model_sha!=BASE.MODEL_SHA:raise ValueError("original pretrained checkpoint SHA mismatch")
    model.eval().requires_grad_(False)
    wc,q,values=readout(model)
    families={}
    for group in MODES[mode]:
        rows=(original["families"][group]["source_roster"]
              if mode=="reference" else hard["rosters"][group])
        if mode=="reference" and rows!=OLD.roster(group):
            raise ValueError("original first-admitted reference source roster changed")
        _,anchors=STAGE.stage_one(model,rows,batch_size=2,retain_zl=True)
        scored=[]
        for i,(source,anchor) in enumerate(zip(rows,anchors)):
            df,di=donor_for(mode,group,i)
            donor=original["families"][df]["per_puzzle"][di]
            row=run_case(model,source,anchor,donor,df,di,wc,q,mode)
            scored.append(row)
            clean6=row["native_task_scores"]["ACT6"]["clean"]
            p=row["same_run_transport"]["ACT6"]["arms"]["state_power"]["8.00"] \
                if donor["source_direction_status"]["state_power"] else None
            print("LS_SAE11_CASE",mode,group,i,source["seed"],
                "ACT6_clean_correct",clean6["blank_correct"],"/",clean6["blank_total"],
                "ACT6_state_power_H_gain",[round(p[s]["h_gain"],5) for s in ("plus","minus")] if p else None,
                "ACT6_state_power_margin_cert",[p[s]["strict_global_margin_certificate"] for s in ("plus","minus")] if p else None,
                flush=True)
        fam={"source_roster":rows,"per_puzzle":scored}
        if mode=="hard":fam["statistics"]=family_summary(scored,group)
        families[group]=fam
    gate=mode=="hard" and all(families[k]["statistics"]["registered_success"] for k in MODES["hard"])
    result={
       "schema":SCHEMA,"preregistration_commit":PROTOCOL,"mode":mode,
       "source_ls_sae08_sha256":ORIGINAL_SHA,
       "source_original_checkpoint_sha256":model_sha,
       "external_source_receipt_sha256":hard_sha,
       "external_source_roster_fingerprints":hard["fingerprints"],
       "readout_centered_rank":int(q.shape[1]),
       "readout_nonzero_singular_values":values,
       "horizons":list(HORIZONS),"physical_radii":list(RADII),
       "source_direction_arm_order":list(ARMS),
       "same_run_centered_readout_tolerance":TOL_IDENTITY,
       "families":families,"registered_hard_scientific_success":gate,
       "scope":"Pretrained native within-forward state and logits. Historical logit replay drift only diagnostic; external hard puzzles could overlap pretraining. Independent auditor cannot reconstruct hidden tensors."
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=list(MODES),required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.mode,a.original,a.source,a.out)
    print(json.dumps({"mode":a.mode,"out":str(a.out),"registered_hard_success":result["registered_hard_scientific_success"],
        "cases":{k:len(v["per_puzzle"]) for k,v in result["families"].items()},
        "stats":{k:v.get("statistics") for k,v in result["families"].items()}},
        sort_keys=True,indent=2))


if __name__=="__main__":
    main()
