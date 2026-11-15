"""PEF-12: preregistered probe-only linear-response forecast of frozen-model precision.

PROTOCOL.md committed independently at 600f686 before PEF-12 heldout noise scoring.
Only the original six PEF-07 checkpoint weights and sealed PEF-09/10/11
probe-ledgers are permitted. All 24 predictions freeze before heldout replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from research.pef07 import pef07 as p7
from research.pef08 import pef08 as p8
from research.pef09 import pef09 as p9
from research.pef10 import pef10 as p10
from research.pef11 import pef11 as p11

TANGENTS = ("random", "pair01")
ANGLES = (0.25, 4.0)
GAUSS = (0.002, 0.006, 0.02)
QUANT = (0.005, 0.02, 0.05)
REPS = 16
VISIT = 8
N = 768
GRAD_BATCH = 96
REPLAY_BATCH = 8
PRECISION_SHA = "f0bd80d8dc7b4af99dbc24be2723828bcf85be5b6f60f46452581e4a53661741"
PROTOCOL_COMMIT = "600f68663bbccf6be303d60008fa9da6255df3e2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(item):
    return (json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def noise_seed(seed, task, tangent_idx, angle_idx, level_idx, replica):
    if (tangent_idx not in range(len(TANGENTS)) or angle_idx not in range(len(ANGLES))
            or level_idx not in range(len(GAUSS)) or replica not in range(REPS)):
        raise ValueError("noise index outside preregistered grid")
    return (120120000 + 100000*seed + (50000 if task == "threshold" else 0)
            + 1000*tangent_idx + 300*angle_idx + 50*level_idx + replica)


def exact_linearized_gaussian(signed_margin, derivative, sigma):
    """Exact Gaussian correctness probability for a *linearized* scalar margin."""
    if sigma <= 0:
        raise ValueError("positive absolute Gaussian sigma expected")
    m = signed_margin.double()
    d = derivative.double().abs()
    p = torch.special.ndtr(m / (sigma*d.clamp_min(1e-100)))
    p = torch.where(d == 0, torch.where(m > 0, 1., torch.where(m < 0, 0., .5)), p)
    return p


def uniform_phase_forecast(signed_margin, derivative, step):
    """Uniform quantizer-phase *approximation*, not deterministic lattice forecast."""
    if step <= 0:
        raise ValueError("positive step expected")
    m = signed_margin.double()
    d = derivative.double().abs()
    p = (0.5 + m/(step*d.clamp_min(1e-100))).clamp(0, 1)
    p = torch.where(d == 0, torch.where(m > 0, 1., torch.where(m < 0, 0., .5)), p)
    return p


def exact_suffix_h12(model, bits, h8):
    """Differentiable replay; p9.replay_batched intentionally has no_grad."""
    h = h8
    for visit in range(VISIT, 12):
        h = model.step(h, model.bit_embedding(bits[:, visit]))
    return h


def probe_margin_jacobians(model, bits, starting, fit, directions):
    """Per-input Jacobian via autograd; nothing is trained or refitted."""
    margin_blocks = []
    grad_blocks = []
    with torch.enable_grad():
        for off in range(0, len(bits), GRAD_BATCH):
            part = starting[off:off+GRAD_BATCH].detach().clone().requires_grad_(True)
            state12 = exact_suffix_h12(model, bits[off:off+GRAD_BATCH], part)
            margin = p8.ridge_margin(state12, fit)
            (gradient,) = torch.autograd.grad(margin.sum(), part)
            margin_blocks.append(margin.detach().double())
            grad_blocks.append(gradient.detach().double())
    margins = torch.cat(margin_blocks)
    gradient = torch.cat(grad_blocks)
    projected = {name: gradient @ vector.double() for name, vector in directions.items()}
    if not all(torch.isfinite(x).all() for x in (margins,gradient,*projected.values())):
        raise AssertionError("nonfinite probe Jacobians")
    return margins, projected


def chosen_condition(selection, tangent, angle):
    matched = [c for c in selection["conditions"] if c["tangent"] == tangent
               and float(c["angle_deg"]) == angle and int(c["sign"]) == 1]
    if len(matched) != 1:
        raise AssertionError("expected one frozen PEF-11 condition")
    c = matched[0]
    if p9.tensor_digest(p11.v64(c["direction"])) != c["direction_sha256"]:
        raise AssertionError("PEF-10 frozen direction hash does not match")
    fit = p10.unmarshal_fit(c["ridge_h12_fit"])
    if p10.ridge_digest(fit) != c["ridge_h12_sha256"]:
        raise AssertionError("PEF-10 frozen ridge hash does not match")
    v,w,o = (p11.v64(c[name]) for name in
             ("direction", "retained_direction", "offplane_direction"))
    for u in (v,w,o):
        if not torch.isclose(u.norm(), torch.tensor(1.,dtype=torch.float64),atol=1e-8):
            raise AssertionError("frozen unit vector degraded")
    if abs(float(v@w))>1e-8 or abs(float(v@o))>1e-8 or abs(float(w@o))>1e-8:
        raise AssertionError("control direction not orthogonal to ablation plane")
    return c,fit,{"transverse":w,"offplane":o}


def validate_inputs(evidence, selection_ledger, angle_ledger, precision_ledger):
    if sha(precision_ledger) != PRECISION_SHA:
        raise AssertionError("PEF-11 precision selection ledger altered")
    catalog, recs, _ = p10.check_frozen_inputs(evidence, selection_ledger)
    if sha(angle_ledger) != p11.PEF10_LEDGER_HASH:
        raise AssertionError("PEF-10 fitted decoder ledger altered")
    precision = json.loads(Path(precision_ledger).read_text())
    angles = json.loads(Path(angle_ledger).read_text())
    if (precision.get("campaign") != "PEF-11" or
        not precision.get("all_conditions_and_noise_seeds_frozen_before_heldout") or
        precision["pef10_angle_ledger_sha256"] != p11.PEF10_LEDGER_HASH or
        angles["pef09_selection_ledger_sha256"] != p10.SELECTION_HASH or
        angles["pef07_training_ledger_sha256"] != p10.TRAINING_HASH):
        raise AssertionError("upstream frozen provenance incomplete")
    if len(recs)!=6 or len(precision["model_selections"])!=6:
        raise AssertionError("expected six sealed model rows")
    for rec in recs:
        sel = next((s for s in precision["model_selections"] if
                   (s["seed"],s["task"])==(rec["seed"],rec["task"])),None)
        if sel is None or sel["checkpoint_sha256"] != rec["checkpoint_sha256"]:
            raise AssertionError("missing model or wrong checkpoint")
        p8.checked_model(Path(evidence),rec)
    return catalog, recs, precision


def freeze_predictions(evidence, catalog, recs, precision, out):
    """Compute forecasts using *only* probe contexts, then seal their bytes."""
    records=[]
    for rec in recs:
        model=p8.checked_model(evidence,rec)
        row=next(s for s in precision["model_selections"] if
                 (s["seed"],s["task"])==(rec["seed"],rec["task"]))
        bits=catalog["probe"]
        truth=p8.labels(bits,rec["task"])
        with torch.no_grad():
            state8=p8.states_by_loop(model,bits)[8]
        center=p11.v64(row["center"])
        scale=float(row["pc_scale"])
        if not scale>0 or not torch.allclose(state8.double().mean(0),center,atol=2e-5,rtol=1e-5):
            raise AssertionError("probe-only center and scale changed")
        original_sha=p8.weights_digest(model)
        conditions=[]
        for tangent_idx,tangent in enumerate(TANGENTS):
            for angle_idx,angle in enumerate(ANGLES):
                selected,fit,directions=chosen_condition(row,tangent,angle)
                with torch.no_grad():
                    ablated=p9.projector(state8,center,p11.v64(selected["direction"]))
                margins, jac=probe_margin_jacobians(model,bits,ablated,fit,directions)
                signed=margins*(truth.double()*2-1)
                gaussian={site:[float(exact_linearized_gaussian(signed,jac[site],scale*rel).mean())
                                for rel in GAUSS] for site in directions}
                quant=[float(uniform_phase_forecast(signed,jac["transverse"],scale*rel).mean())
                       for rel in QUANT]
                conditions.append({
                    "tangent":tangent,"tangent_index":tangent_idx,
                    "angle_deg":angle,"angle_index":angle_idx,
                    "direction_sha256":selected["direction_sha256"],
                    "ridge_h12_sha256":selected["ridge_h12_sha256"],
                    "noise_seeds":[[noise_seed(rec["seed"],rec["task"],tangent_idx,angle_idx,l,r)
                                    for r in range(REPS)] for l in range(len(GAUSS))],
                    "probe_clean_correct":int(((margins>=0).long()==truth).sum()),
                    "probe_clean_fraction":float(((margins>=0).long()==truth).double().mean()),
                    "probe_margins":margins.tolist(),
                    "probe_signed_margins":signed.tolist(),
                    "probe_derivative_transverse":jac["transverse"].tolist(),
                    "probe_derivative_offplane":jac["offplane"].tolist(),
                    "probe_gaussian_forecast":gaussian,
                    "probe_quant_uniform_phase_forecast":quant,
                })
        if p8.weights_digest(model)!=original_sha:
            raise AssertionError("autograd modified frozen weights")
        records.append({"seed":rec["seed"],"task":rec["task"],
                        "checkpoint_sha256":rec["checkpoint_sha256"],
                        "probe_scale":scale,"conditions":conditions})
        print("PEF12_PROBE_FORECAST_READY",rec["seed"],rec["task"],flush=True)
    ledger={"campaign":"PEF-12","preregistration_commit":PROTOCOL_COMMIT,
            "pef11_precision_ledger_sha256":PRECISION_SHA,
            "pef10_angle_ledger_sha256":p11.PEF10_LEDGER_HASH,
            "pef09_selection_ledger_sha256":p10.SELECTION_HASH,
            "pef07_training_ledger_sha256":p10.TRAINING_HASH,
            "gaussian_relative":list(GAUSS),"quant_relative":list(QUANT),
            "gaussian_replicates":REPS,
            "all_probes_and_noise_seeds_frozen_before_heldout":True,
            "models":records}
    path=out/"sealed_probe_forecasts.json"
    path.write_bytes(canonical(ledger))
    print("PEF12_ALL_PROBE_FORECASTS_FROZEN",sha(path),flush=True)
    return ledger


def heldout_score(model,bits,state8,fit,truth):
    """No-grad natural suffix replay for one or several h8 conditions."""
    if state8.ndim==2:
        state8=state8[None,:,:]
    collected=[]
    with torch.no_grad():
        for start in range(0,len(state8),REPLAY_BATCH):
            states12,_=p9.replay_batched(model,bits,state8[start:start+REPLAY_BATCH],8)
            for x in states12:
                margin=p8.ridge_margin(x,fit)
                report=p9.class_breakdown(margin,truth)
                collected.append(report)
    return collected


def run_heldout(evidence,catalog,recs,precision,forecasts,out):
    results=[]
    for rec in recs:
        model=p8.checked_model(evidence,rec)
        digest=p8.weights_digest(model)
        row=next(s for s in precision["model_selections"] if
                 (s["seed"],s["task"])==(rec["seed"],rec["task"]))
        forecast=next(s for s in forecasts["models"] if
                      (s["seed"],s["task"])==(rec["seed"],rec["task"]))
        bits=catalog["heldout"]
        truth=p8.labels(bits,rec["task"])
        with torch.no_grad():
            trace=p8.states_by_loop(model,bits)
            if p9.class_breakdown(p9.native_score(model,trace[12]),truth)["correct"] != 768:
                raise AssertionError("PEF-07 natural native score no longer 768/768")
        center=p11.v64(row["center"])
        scale=float(row["pc_scale"])
        conditions=[]
        for c in forecast["conditions"]:
            frozen,fit,dirs=chosen_condition(row,c["tangent"],c["angle_deg"])
            if c["direction_sha256"]!=frozen["direction_sha256"] or c["ridge_h12_sha256"]!=frozen["ridge_h12_sha256"]:
                raise AssertionError("post-freeze direction or ridge mismatch")
            with torch.no_grad():
                base=p9.projector(trace[8],center,p11.v64(frozen["direction"]))
            clean=heldout_score(model,bits,base,fit,truth)[0]
            score_rows=[]
            for level_idx,rel in enumerate(GAUSS):
                states=[]
                metadata=[]
                for replica,seed in enumerate(c["noise_seeds"][level_idx]):
                    z=p11.gaussian_z(seed,len(bits))
                    for site in ("transverse","offplane"):
                        h8=p11.perturb_gaussian(base,dirs[site],z,scale*rel)
                        states.append(h8)
                        metadata.append({"kind":"gaussian","site":site,"relative_level":rel,
                                         "replica":replica,"seed":seed})
                for m,score in zip(metadata,heldout_score(model,bits,torch.stack(states),fit,truth),strict=True):
                    score_rows.append(m|{"score":score})
            for idx,rel in enumerate(QUANT):
                q=p11.scalar_quant(base,center,dirs["transverse"],scale*rel)
                score=heldout_score(model,bits,q,fit,truth)[0]
                score_rows.append({"kind":"retained_quant","site":"transverse",
                                   "relative_level":rel,"replica":None,"seed":None,"score":score})
            conditions.append({"tangent":c["tangent"],"angle_deg":c["angle_deg"],
                               "probe_clean_fraction":c["probe_clean_fraction"],
                               "heldout_clean":clean,"scores":score_rows})
            print("PEF12_HELDOUT_SCORED",rec["seed"],rec["task"],c["tangent"],c["angle_deg"],flush=True)
        if p8.weights_digest(model)!=digest or sha(evidence/rec.get("file",rec.get("checkpoint_file")))!=rec["checkpoint_sha256"]:
            raise AssertionError("frozen weights/checkpoint changed during noise evaluation")
        results.append({"seed":rec["seed"],"task":rec["task"],
                        "checkpoint_sha256":rec["checkpoint_sha256"],"conditions":conditions})
    return results


def summarize(forecasts,results):
    models=[]
    for fr in forecasts["models"]:
        rr=next(x for x in results if x["seed"]==fr["seed"] and x["task"]==fr["task"])
        conds=[]
        for f in fr["conditions"]:
            r=next(x for x in rr["conditions"] if x["tangent"]==f["tangent"]
                   and x["angle_deg"]==f["angle_deg"])
            for site in ("transverse","offplane"):
                for level_idx,rel in enumerate(GAUSS):
                    rows=[x for x in r["scores"] if x["kind"]=="gaussian" and
                          x["site"]==site and x["relative_level"]==rel]
                    if len(rows)!=REPS:
                        raise AssertionError("incomplete frozen Gaussian replicates")
                    empirical=sum(x["score"]["correct"] for x in rows)/(REPS*N)
                    pred=f["probe_gaussian_forecast"][site][level_idx]
                    conds.append({"tangent":f["tangent"],"angle_deg":f["angle_deg"],
                                  "site":site,"kind":"gaussian","relative_level":rel,
                                  "predicted":pred,"observed":empirical,
                                  "absolute_error":abs(pred-empirical),
                                  "clean_probe":f["probe_clean_fraction"]})
            for level_idx,rel in enumerate(QUANT):
                matches=[x for x in r["scores"] if x["kind"]=="retained_quant"
                         and x["relative_level"]==rel]
                if len(matches)!=1:
                    raise AssertionError("incomplete frozen quantization evaluation")
                pred=f["probe_quant_uniform_phase_forecast"][level_idx]
                empirical=matches[0]["score"]["correct"]/N
                conds.append({"tangent":f["tangent"],"angle_deg":f["angle_deg"],
                              "site":"transverse","kind":"uniform_phase_quant_approx",
                              "relative_level":rel,"predicted":pred,"observed":empirical,
                              "absolute_error":abs(pred-empirical),
                              "clean_probe":f["probe_clean_fraction"]})
        random_rows=[x for x in conds if x["tangent"]=="random" and
                     x["site"]=="transverse" and x["kind"]=="gaussian"]
        mae=sum(x["absolute_error"] for x in random_rows)/len(random_rows)
        clean_mae=sum(abs(x["clean_probe"]-x["observed"]) for x in random_rows)/len(random_rows)
        passed=mae<=.07 and (mae<=.75*clean_mae)
        models.append({"seed":fr["seed"],"task":fr["task"],
                       "random_gaussian_mae":mae,"random_clean_constant_mae":clean_mae,
                       "random_forecast_improvement_fraction":1-mae/max(clean_mae,1e-12),
                       "random_calibration_pass":passed,
                       "calibration_rows":conds})
    passed=sum(m["random_calibration_pass"] for m in models if m["task"]=="coalition")
    return {"coalition_seed_count":passed,
            "decision":"criterion_met" if passed>=2 else "criterion_not_met",
            "models":models}


def run(evidence,selection_ledger,angle_ledger,precision_ledger,out):
    os.environ.setdefault("DNNL_MAX_CPU_ISA","AVX2")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence,selection_ledger,angle_ledger,precision_ledger,out = map(Path,
        (evidence,selection_ledger,angle_ledger,precision_ledger,out))
    out.mkdir(parents=True,exist_ok=True)
    catalog,recs,precision=validate_inputs(evidence,selection_ledger,angle_ledger,precision_ledger)
    forecasts=freeze_predictions(evidence,catalog,recs,precision,out)
    seal=sha(out/"sealed_probe_forecasts.json")
    # Heldout states first become visible *after* all 24 probe predictions are on disk.
    results=run_heldout(evidence,catalog,recs,precision,forecasts,out)
    decision=summarize(forecasts,results)
    report={"campaign":"PEF-12","preregistration_commit":PROTOCOL_COMMIT,
            "prediction_ledger_sha256":seal,"all_forecasts_sealed_before_heldout":True,
            "source_contexts_heldout":48,"heldout_histories":N,"torch_version":str(torch.__version__),
            "decision":decision,"results":results}
    (out/"summary.json").write_bytes(canonical(report))
    with (out/"calibration.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=("seed","task","tangent","angle_deg",
                "site","kind","relative_level","predicted","observed","absolute_error","clean_probe"))
        writer.writeheader()
        for model in decision["models"]:
            for line in model["calibration_rows"]:
                writer.writerow({"seed":model["seed"],"task":model["task"]}|line)
    print(json.dumps({"campaign":"PEF-12","prediction_ledger_sha256":seal,
                      "decision":decision["decision"],"coalition_seeds_passed":decision["coalition_seed_count"],
                      "per_model":[{k:m[k] for k in ("seed","task","random_gaussian_mae",
                                           "random_clean_constant_mae","random_calibration_pass")}
                                   for m in decision["models"]]},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--selection-ledger",type=Path,required=True)
    parser.add_argument("--angle-ledger",type=Path,required=True)
    parser.add_argument("--precision-ledger",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.evidence,args.selection_ledger,args.angle_ledger,args.precision_ledger,args.out)


if __name__=="__main__":
    main()
