"""LS-SAE03 full 512D native TRM Jacobian-vector transport.

Every task-label lookup follows checkpoint, SAE-fit and direction locks.
The JVP differentiates through exact native ACT calls with math SDPA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from mechinterp.loopscope02.scripts import trm_sudoku as TS
from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as NEW
from mechinterp.sae_causal_transfer.native_jvp_v1 import geometry as G
from mechinterp.sae_causal_transfer.native_jvp_v1 import fit_lock as LOCK

SOURCE_CHECKPOINT=TS.TRM_CHECKPOINTS["gaoxin492"]["sha256"]
OLD_FIT_FINGERPRINT="f217453dc17e078b055e99f3cebc3cf736af7e7cfae2ab120353c91bf2554c9d"
OLD_PHYSICAL_FINGERPRINT="decf90fd73137bd232ba40104cf387e894c4a3d1f4f007036c591d173f3e238c"
PROTOCOL_COMMIT="07490150318b94ca5d035d928ff3255c9d6b60ff"


def _digest(t:torch.Tensor):
    b=t.detach().cpu().contiguous().numpy().astype("<f4",copy=False).tobytes(order="C")
    return hashlib.sha256(b).hexdigest()


def clean_trajectory(model,zh,zl,xemb):
    """Callable compatible with torch.func.jvp: no in-place mutation."""
    states=[zh]
    for _ in range(2):
        zh,zl=model.act_step(zh,zl,xemb)
        states.append(zh)
    return tuple(states)


def _target_readouts(model,clean_states,position,truth):
    """Ground-truth label and clean rival are introduced only after directions freeze."""
    head9=model.lm_head.weight.detach()[2:11].double().cpu().numpy()
    rows=[]
    with torch.no_grad():
        for k,z in enumerate(clean_states):
            logits=model.digit_logits(z)[0,position].detach().double().cpu().numpy()
            other=logits.copy();other[truth]=-np.inf
            rival=int(np.argmax(other))
            rows.append({"depth":k,"target":truth,"rival":rival,
                         "clean_logits9":logits.tolist(),
                         "readout":(head9[truth]-head9[rival]).tolist()})
    return rows


def _finite_arm_logits(model,anchor,position,directions):
    """All ±0.2 unit directions share the same frozen ACT1 anchor/other tokens."""
    names=[name for name in ("sae","task_head","head_null","random")
           if directions.get(name) is not None]
    batch=1+2*len(names)
    token=model.puzzle_emb_len+position
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        zh=anchor["zh"].detach().repeat(batch,1,1)
        zl=anchor["zl"].detach().repeat(batch,1,1)
        xemb=anchor["x_emb"].detach().repeat(batch,1,1)
        for j,name in enumerate(names):
            vec=torch.tensor(directions[name],dtype=zh.dtype,device=zh.device)
            zh[1+2*j,token]+=G.RADIUS*vec
            zh[2+2*j,token]-=G.RADIUS*vec
        logits=[model.digit_logits(zh)[:,position,:].detach().double().cpu().numpy()]
        for _ in range(2):
            zh,zl=model.act_step(zh,zl,xemb)
            logits.append(model.digit_logits(zh)[:,position,:].detach().double().cpu().numpy())
    return names,logits


def _reverse_gradient(model,anchor,position,y,rival2):
    token=model.puzzle_emb_len+position
    with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        source=anchor["zh"].detach().clone().requires_grad_(True)
        zh,zl=source,anchor["zl"].detach()
        for _ in range(2):
            zh,zl=model.act_step(zh,zl,anchor["x_emb"].detach())
        logits=model.digit_logits(zh)[0,position]
        m=logits[y]-logits[rival2]
        gradient=torch.autograd.grad(m,source)[0][0,token].detach().double().cpu().numpy()
    if gradient.shape!=(512,) or not np.all(np.isfinite(gradient)):
        raise ArithmeticError("bad 512D source full-BPTT task gradient")
    return gradient


def measure_case(model,anchor,reduced,fit,row,family,position):
    token=model.puzzle_emb_len+position
    head9=model.lm_head.weight.detach()[2:11].double().cpu().numpy()
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        first_logits=model.digit_logits(anchor["zh"])[0,position].detach().double().cpu().numpy()
    # Freeze all label-independent directions prior to accessing row["target"].
    selected=G.choose_directions(reduced,fit.physical_rows,fit.physical_bias,
        fit.physical_decoder,fit.basis,head9,first_logits,family,row["seed"])
    vectors=selected["directions"]
    names,finite_logits=_finite_arm_logits(model,anchor,position,vectors)
    y=row["target"][position]-1
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        clean_states=clean_trajectory(model,anchor["zh"].detach(),
                                     anchor["zl"].detach(),anchor["x_emb"].detach())
        expected=_target_readouts(model,clean_states,position,y)
    for k in G.DEPTHS:
        if not np.allclose(finite_logits[k][0],expected[k]["clean_logits9"],
                           rtol=0,atol=2e-4):
            raise ArithmeticError("finite-arm clean forward disagrees with clean readout")
    gradient=_reverse_gradient(model,anchor,position,y,expected[2]["rival"])
    rank=G.random_rank_baseline(gradient,vectors["sae"],family,row["seed"])
    observed={}
    for name,u in vectors.items():
        if u is None:
            observed[name]=None
            continue
        direction=torch.zeros_like(anchor["zh"])
        direction[0,token]=torch.tensor(u,dtype=direction.dtype,device=direction.device)
        with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
            clean,full_jvp=torch.func.jvp(
                lambda h:clean_trajectory(model,h,anchor["zl"].detach(),
                                          anchor["x_emb"].detach()),
                (anchor["zh"].detach(),),(direction,))
        effects={}
        j=names.index(name)
        for k in G.DEPTHS:
            t=full_jvp[k]
            info=G.tangent_stats(t.detach().cpu().numpy(),token,
                                 expected[k]["readout"])
            info["full_tangent_float32_sha256"]=_digest(t)
            logit_plus=finite_logits[k][1+2*j]
            logit_minus=finite_logits[k][2+2*j]
            read=expected[k]
            yp,opp=read["target"],read["rival"]
            mp=float(logit_plus[yp]-logit_plus[opp])
            mm=float(logit_minus[yp]-logit_minus[opp])
            finite=(mp-mm)/(2*G.RADIUS)
            info["finite_plus_margin"]=mp
            info["finite_minus_margin"]=mm
            info["symmetric_finite_slope"]=finite
            info["finite_minus_jvp_slope"]=finite-info["task_margin_slope"]
            if k==2:
                cross=float(gradient@u)
                info["reverse_mode_slope"]=cross
                tolerance=max(.02,.05*abs(cross))
                if abs(cross-info["task_margin_slope"])>tolerance:
                    raise ArithmeticError("full native forward-JVP vs reverse-VJP conflict")
            effects[str(k)]=info
        observed[name]=effects
    return {
        "goal_position":int(position),"true_digit_zero_based":y,
        "source_directions":{k:(None if v is None else v.tolist()) for k,v in vectors.items()},
        "source_direction_metadata":selected["meta"],
        "sae_admitted":bool(vectors["sae"] is not None),
        "full_bptt_gradient_512":gradient.tolist(),
        "rank_one_nulls":rank,
        "clean_readouts":expected,
        "directions":observed
    }


def run(*,pilot=False,out=None,fit_out=None,fit_in=None):
    if (pilot and (fit_out is None or fit_in is not None)) or (not pilot and (fit_in is None or fit_out is not None)):
        raise ValueError("pilot must archive one source fit; full must use the archived pilot fit")
    torch.set_num_threads(4)
    torch.manual_seed(71)
    # Use old 32 registered fit puzzles in both pilot and full; holdout never enters this fit.
    old,_=OLD.build_rosters(pilot=False)
    fit_roster=old["fit_30"]
    model,source_sha=TS.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if source_sha!=SOURCE_CHECKPOINT or model.hidden_size!=512:
        raise ValueError("native pretrained model source lock changed")
    model.eval().requires_grad_(False)
    if pilot:
        states,_=FIT.stage_one(model,fit_roster)
        fit=FIT.fit_only(states)
        asset=LOCK.save_exact(fit,fit_out)
    else:
        fit,asset=LOCK.load_exact(fit_in)
    physical=LOCK.checked_fingerprint(fit)
    if fit.fingerprint!=OLD_FIT_FINGERPRINT or physical!=OLD_PHYSICAL_FINGERPRINT:
        raise ValueError("prior LS-SAE02 calibration weights do not match exact source lock")
    frozen={
        "old_fit_roster_sha256":OLD.fingerprint(fit_roster),
        "original_checkpoint_file_sha256":source_sha,
        "old_fit_PCA_SAE_sha256":fit.fingerprint,
        "old_physical_dictionary_sha256":physical,
        "checkpoint_weights_frozen":True,
        "frozen_pilot_calibration_asset_sha256":asset["file_sha256"],
        "frozen_pilot_calibration_asset_bytes":asset["file_bytes"],
        "calibration_asset_transfer":"pilot_fitted" if pilot else "full_downloaded_sha_verified"
    }
    print("LS_SAE03_PRIOR_CALIBRATION_LOCKED",json.dumps(frozen,sort_keys=True),flush=True)
    # Only now begin generating the novel heldout source states.
    test_rosters,manifest=NEW.build_new(pilot=pilot)
    heldout={}
    for family,roster in test_rosters.items():
        zhs,anchors=FIT.stage_one(model,roster,retain_zl=True)
        trials=[]
        for i,row in enumerate(roster):
            pos=NEW.median_blank(row["x"])
            reduced=fit.encode(zhs[i,pos])
            trial=measure_case(model,anchors[i],reduced,fit,row,family,pos)
            trial.update({"puzzle_index":i,"puzzle_seed":row["seed"]})
            trial["sae_event"]=G.attenuation_event(trial["directions"]["sae"])
            trial["random_event"]=G.attenuation_event(trial["directions"]["random"])
            trials.append(trial)
            print("LS_SAE03_PUZZLE",family,i,trial["sae_event"],
                  trial["random_event"],flush=True)
        stats=G.family_score(trials,family)
        heldout[family]={"per_puzzle":trials,"statistics":stats}
        print("LS_SAE03_FAMILY",family,json.dumps(stats,sort_keys=True),flush=True)
    full=not pilot
    receipt={
      "schema":"LS-SAE03/full-native-autograd-JVP-512-v1",
      "preregistration_commit":PROTOCOL_COMMIT,
      "mode":"pilot" if pilot else "full",
      "source":frozen,"new_test_rosters":test_rosters,
      "new_test_roster_sha256":manifest,
      "depths":list(G.DEPTHS),
      "radius":G.RADIUS,
      "random_rank_one_observer_budget":G.N_RANK_NULL,
      "full_protocol":full,
      "heldout":heldout,
      "registered_success_both_families":bool(
        full and all(heldout[f]["statistics"]["registered_success"] for f in ("test_30","test_34"))),
      "interpretation_limit":"Full native forward-mode JVPs and finite effects on source-locked pretrained TRM; neither cross-checkpoint equivalence nor formal interval proof."
    }
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--fit-out",type=Path)
    ap.add_argument("--fit-in",type=Path)
    a=ap.parse_args()
    d=run(pilot=a.pilot,out=a.out,fit_out=a.fit_out,fit_in=a.fit_in)
    print(json.dumps({"output":str(a.out),
        "scientific_success":d["registered_success_both_families"],
        "family_summaries":{f:v["statistics"] for f,v in d["heldout"].items()}},indent=2))


if __name__=="__main__":
    main()
