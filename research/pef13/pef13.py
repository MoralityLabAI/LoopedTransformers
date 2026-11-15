"""PEF-13: sealed probe-only lattice phase forecast, heldout replay afterwards.

A PEF-12 follow-up. First commit of protocol: 9483da079c3707711b071efa9dbd04fa1a286b40.
Weights, heads, source groups, and the 24 PEF-12 probe Jacobian arrays remain frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from research.pef08 import pef08 as p8
from research.pef09 import pef09 as p9
from research.pef10 import pef10 as p10
from research.pef11 import pef11 as p11
from research.pef12 import pef12 as p12

STEPS = (0.0075, 0.0125, 0.035, 0.08)
LEGACY_STEPS = (0.005, 0.02, 0.05)
EXPECTED_JACOBIAN_SHA = "fa9a3f3727b9551be721ca08e307082b81d1d2da2e359a34d88d26c9e7185b49"
PROTOCOL_COMMIT = "9483da079c3707711b071efa9dbd04fa1a286b40"
N = 768


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(record):
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def quant_residual(q, step):
    """FP64 error of PyTorch round-to-nearest-even at fixed absolute step."""
    if not step > 0 or not math.isfinite(step):
        raise ValueError("positive, finite quantization step required")
    return step * torch.round(q.double()/step) - q.double()


def phase_forecast(margins, derivative, labels, q, step):
    """Class fraction from exact probe phases and first-order downstream Jacobian."""
    if any(len(x) != len(q) for x in (margins, derivative, labels)):
        raise ValueError("probe array length mismatch")
    error = quant_residual(q, step)
    linear = margins.double() + derivative.double() * error
    pred = (linear >= 0).long()
    return float((pred == labels.long()).double().mean())


def phase_concentration(q, step):
    # Absolute first Fourier coefficient of lattice phase, in [0, 1].
    wrapped = q.double() / step
    angle = 2 * math.pi * (wrapped - torch.floor(wrapped))
    return float((angle.cos().mean().square() + angle.sin().mean().square()).sqrt())


def correlation(x, y):
    a, b = x.double()-x.double().mean(), y.double()-y.double().mean()
    denom = a.norm()*b.norm()
    return 0.0 if float(denom) == 0 else float((a@b)/denom)


def shuffled_seed(seed, task, tangent_idx, angle_idx, step_idx):
    if tangent_idx not in range(2) or angle_idx not in range(2) or step_idx not in range(4):
        raise ValueError("bad registered condition index")
    return 1313000 + seed*10000 + (1000 if task == "threshold" else 0) + 100*tangent_idx + 10*angle_idx + step_idx


def shuffled_forecast(margin, derivative, truth, error, seed):
    order = torch.randperm(len(error),generator=torch.Generator(device="cpu").manual_seed(seed))
    return float(((margin.double() + derivative.double()*error[order]) >= 0).long().eq(truth).double().mean())


def condition_frozen(precision_row, tangent, angle):
    selection, fit, directions = p12.chosen_condition(precision_row, tangent, angle)
    return selection, fit, directions['transverse']


def validate(evidence, selection_ledger, angle_ledger, precision_ledger, jacobian_ledger):
    if sha(jacobian_ledger) != EXPECTED_JACOBIAN_SHA:
        raise AssertionError("PEF-12 frozen probe/Jacobian ledger SHA256 changed")
    catalog, recs, precision = p12.validate_inputs(evidence, selection_ledger, angle_ledger, precision_ledger)
    jac = json.loads(Path(jacobian_ledger).read_text())
    if (jac.get("campaign") != "PEF-12" or
        not jac.get("all_probes_and_noise_seeds_frozen_before_heldout") or
        jac.get("pef11_precision_ledger_sha256") != p12.PRECISION_SHA or
        len(jac.get("models", [])) != 6):
        raise AssertionError("PEF-12 model/forecast provenance incomplete")
    for rec in recs:
        matched = [x for x in jac["models"] if (x["seed"],x["task"]) == (rec["seed"],rec["task"])]
        if len(matched) != 1 or matched[0]["checkpoint_sha256"] != rec["checkpoint_sha256"] or len(matched[0]["conditions"]) != 4:
            raise AssertionError("mismatched PEF-12 sealed probe records")
    return catalog, recs, precision, jac


def freeze_forecasts(evidence, catalog, recs, precision, jac, output):
    """Seal every model/condition/step prediction *before* heldout trace construction."""
    records = []
    for rec in recs:
        seed, task = rec["seed"], rec["task"]
        model = p8.checked_model(Path(evidence), rec)
        first_digest = p8.weights_digest(model)
        selected_row = next(r for r in precision["model_selections"] if (r["seed"],r["task"])==(seed,task))
        past = next(r for r in jac["models"] if (r["seed"],r["task"])==(seed,task))
        probe = catalog["probe"]
        truth = p8.labels(probe,task)
        if len(truth)!=N:
            raise AssertionError("expected 768 probe labels")
        with torch.no_grad():
            h8 = p8.states_by_loop(model,probe)[8]
        mu = p11.v64(selected_row["center"])
        scale = float(selected_row["pc_scale"])
        if not (scale > 0 and torch.allclose(h8.double().mean(0),mu,atol=2e-5,rtol=1e-5)):
            raise AssertionError("probe-only center/scale changed")
        conditions = []
        for t_idx,tangent in enumerate(p12.TANGENTS):
            for a_idx,angle in enumerate(p12.ANGLES):
                frozen,fit,w = condition_frozen(selected_row,tangent,angle)
                previous = next((x for x in past["conditions"] if x["tangent"]==tangent and x["angle_deg"]==angle), None)
                if (previous is None or previous["direction_sha256"]!=frozen["direction_sha256"] or
                    previous["ridge_h12_sha256"]!=frozen["ridge_h12_sha256"]):
                    raise AssertionError("different PEF-12 probe condition or readout")
                with torch.no_grad():
                    projected = p9.projector(h8, mu, p11.v64(frozen["direction"]))
                    q = (projected.double()-mu)@w
                    # Recompute *clean* margin from frozen weights to certify row alignment.
                    replayed12,_ = p9.replay_batched(model,probe,projected[None,:,:],8)
                    direct_margin = p8.ridge_margin(replayed12[0],fit)
                margin = p11.v64(previous["probe_margins"])
                signed = p11.v64(previous["probe_signed_margins"])
                deriv = p11.v64(previous["probe_derivative_transverse"])
                if not torch.allclose(direct_margin,margin,atol=1e-5,rtol=1e-5):
                    raise AssertionError("PEF-12 sealed margin arrays no longer align with probe input")
                if not torch.allclose(signed,margin*(truth.double()*2-1),atol=1e-7,rtol=1e-7):
                    raise AssertionError("probe signed margins and labels do not align")
                if not all(torch.isfinite(t).all() for t in (q,margin,deriv)):
                    raise AssertionError("nonfinite phase/margin/derivative")
                cond = {"tangent":tangent,"angle_deg":angle,"tangent_index":t_idx,"angle_index":a_idx,
                        "direction_sha256":frozen["direction_sha256"],
                        "ridge_h12_sha256":frozen["ridge_h12_sha256"],
                        "probe_clean_fraction":previous["probe_clean_fraction"],
                        "probe_q":q.tolist(), "probe_margin":margin.tolist(),
                        "probe_signed_margin":signed.tolist(),
                        "probe_derivative":deriv.tolist(), "probe_labels":truth.tolist(),
                        "steps":[]}
                for k, rel in enumerate(STEPS):
                    step = rel * scale
                    delta = quant_residual(q,step)
                    phase = phase_forecast(margin,deriv,truth,q,step)
                    uniform = float(p12.uniform_phase_forecast(signed,deriv,step).mean())
                    seed_shuf = shuffled_seed(seed,task,t_idx,a_idx,k)
                    shuffled = shuffled_forecast(margin,deriv,truth,delta,seed_shuf)
                    oriented = signed + (truth.double()*2-1)*deriv*delta
                    cond["steps"].append({"relative_step":rel,"absolute_step":step,
                        "phase_aware":phase,"uniform_phase":uniform,
                        "phase_shuffled":shuffled,"clean_constant":previous["probe_clean_fraction"],
                        "shuffled_seed":seed_shuf,
                        "phase_error_mean":float(delta.mean()),
                        "phase_error_sd":float(delta.std(unbiased=False)),
                        "phase_concentration":phase_concentration(q,step),
                        "signed_margin_error_correlation":correlation(signed,(truth.double()*2-1)*deriv*delta),
                        "linear_sign_robust_fraction":float((signed.abs() > (deriv*delta).abs()).double().mean()),
                        "linearized_wrong_count":int((oriented < 0).sum()),
                        "probe_delta":delta.tolist()})
                conditions.append(cond)
        if len(conditions)!=4 or p8.weights_digest(model)!=first_digest:
            raise AssertionError("missing probe conditions or modified model")
        records.append({"seed":seed,"task":task,"checkpoint_sha256":rec["checkpoint_sha256"],
                        "probe_pc_scale":scale,"conditions":conditions})
        print("PEF13_PROBE_READY",seed,task,flush=True)
    ledger = {"campaign":"PEF-13","registration_commit":PROTOCOL_COMMIT,
              "pef12_jacobian_ledger_sha256":EXPECTED_JACOBIAN_SHA,
              "pef11_precision_ledger_sha256":p12.PRECISION_SHA,
              "pef10_angle_ledger_sha256":p11.PEF10_LEDGER_HASH,
              "pef09_selection_ledger_sha256":p10.SELECTION_HASH,
              "pef07_training_ledger_sha256":p10.TRAINING_HASH,
              "new_relative_steps":list(STEPS),"probe_forecasts_sealed_before_heldout":True,
              "models":records}
    path=Path(output)/"sealed_phase_forecasts.json"
    path.write_bytes(canonical(ledger))
    print("PEF13_ALL_FORECASTS_FROZEN",sha(path),flush=True)
    return ledger


def heldout_replay(evidence, catalog, recs, precision, forecast):
    rows = []
    for rec in recs:
        seed,task = rec["seed"],rec["task"]
        model = p8.checked_model(Path(evidence),rec)
        checkpoint_digest = p8.weights_digest(model)
        bits = catalog["heldout"]
        truth = p8.labels(bits,task)
        selected_row = next(r for r in precision["model_selections"] if (r["seed"],r["task"])==(seed,task))
        probe_row = next(r for r in forecast["models"] if (r["seed"],r["task"])==(seed,task))
        with torch.no_grad():
            h8 = p8.states_by_loop(model,bits)[8]
        mu=p11.v64(selected_row["center"])
        conditions=[]
        for c in probe_row["conditions"]:
            selected,fit,w = condition_frozen(selected_row,c["tangent"],c["angle_deg"])
            if c["direction_sha256"]!=selected["direction_sha256"] or c["ridge_h12_sha256"] != selected["ridge_h12_sha256"]:
                raise AssertionError("PEF-13 frozen direction/ridge diverged after sealing")
            with torch.no_grad():
                base=p9.projector(h8,mu,p11.v64(selected["direction"]))
                state_stack = torch.stack([p11.scalar_quant(base,mu,w,s["absolute_step"]) for s in c["steps"]])
                scores=p12.heldout_score(model,bits,state_stack,fit,truth)
            if len(scores)!=4:
                raise AssertionError("four new heldout quantization scores expected")
            conditions.append({"tangent":c["tangent"],"angle_deg":c["angle_deg"],
                               "steps":[{"relative_step":s["relative_step"],"score":result} for s,result in zip(c["steps"],scores,strict=True)]})
            print("PEF13_HELDOUT_SCORED",seed,task,c["tangent"],c["angle_deg"],flush=True)
        ckpt = Path(evidence)/rec.get("file",rec.get("checkpoint_file"))
        if p8.weights_digest(model)!=checkpoint_digest or sha(ckpt)!=rec["checkpoint_sha256"]:
            raise AssertionError("frozen model changed during heldout evaluation")
        rows.append({"seed":seed,"task":task,"checkpoint_sha256":rec["checkpoint_sha256"],"conditions":conditions})
    return rows


def decision_table(forecast,heldout):
    models=[]
    for fr in forecast["models"]:
        hr=next(r for r in heldout if (r["seed"],r["task"])==(fr["seed"],fr["task"]))
        lines=[]
        for c in fr["conditions"]:
            oc=next(r for r in hr["conditions"] if r["tangent"]==c["tangent"] and r["angle_deg"]==c["angle_deg"])
            for s,obs in zip(c["steps"],oc["steps"],strict=True):
                if s["relative_step"]!=obs["relative_step"]:
                    raise AssertionError("misaligned frozen predictions and heldout results")
                actual=obs["score"]["correct"]/N
                line={"seed":fr["seed"],"task":fr["task"],"tangent":c["tangent"],
                      "angle_deg":c["angle_deg"],"relative_step":s["relative_step"],"observed":actual,
                      "correct":obs["score"]["correct"],
                      "phase_aware":s["phase_aware"],"uniform_phase":s["uniform_phase"],
                      "phase_shuffled":s["phase_shuffled"],"clean_constant":s["clean_constant"],
                      "phase_error":abs(actual-s["phase_aware"]),
                      "uniform_error":abs(actual-s["uniform_phase"]),
                      "shuffled_error":abs(actual-s["phase_shuffled"]),
                      "clean_error":abs(actual-s["clean_constant"])}
                lines.append(line)
        if len(lines)!=16:
            raise AssertionError("16 registered heldout forecast rows required per model")
        ma=lambda key:sum(row[key] for row in lines)/len(lines)
        phase,uniform=ma("phase_error"),ma("uniform_error")
        passed = phase<=.035 and phase<=.75*uniform
        models.append({"seed":fr["seed"],"task":fr["task"],
                       "phase_aware_mae":phase,"uniform_phase_mae":uniform,
                       "phase_shuffled_mae":ma("shuffled_error"),
                       "clean_constant_mae":ma("clean_error"),
                       "phase_improvement_fraction":1-phase/max(uniform,1e-12),
                       "criterion_met":passed,"conditions":lines})
    successes=sum(x["criterion_met"] for x in models if x["task"]=="coalition")
    return {"primary_decision":"criterion_met" if successes>=2 else "criterion_not_met",
            "coalition_seeds_met":successes,"models":models}


def run(evidence,selection_ledger,angle_ledger,precision_ledger,jacobian_ledger,out):
    os.environ.setdefault("DNNL_MAX_CPU_ISA","AVX2")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence,selection_ledger,angle_ledger,precision_ledger,jacobian_ledger,out=map(Path,
       (evidence,selection_ledger,angle_ledger,precision_ledger,jacobian_ledger,out))
    out.mkdir(parents=True,exist_ok=True)
    catalog,recs,precision,jac = validate(evidence,selection_ledger,angle_ledger,precision_ledger,jacobian_ledger)
    frozen=freeze_forecasts(evidence,catalog,recs,precision,jac,out)
    seal=sha(out/"sealed_phase_forecasts.json")
    observed=heldout_replay(evidence,catalog,recs,precision,frozen)
    decision=decision_table(frozen,observed)
    report={"campaign":"PEF-13","protocol_commit":PROTOCOL_COMMIT,
            "probe_forecast_ledger_sha256":seal,
            "all_predictions_frozen_before_heldout":True,
            "full_new_step_condition_count":96,
            "heldout_histories":N,"torch_version":str(torch.__version__),
            "decisions":decision,"heldout_results":observed}
    (out/"summary.json").write_bytes(canonical(report))
    with (out/"calibration.csv").open("w",newline="") as f:
        keys=("seed","task","tangent","angle_deg","relative_step","observed","correct",
              "phase_aware","uniform_phase","phase_shuffled","clean_constant",
              "phase_error","uniform_error","shuffled_error","clean_error")
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for row in decision["models"]:
            writer.writerows(row["conditions"])
    print(json.dumps({"campaign":"PEF-13","forecast_ledger":seal,
           "decision":decision["primary_decision"],"coalition_seeds_met":decision["coalition_seeds_met"],
           "models":[{k:x[k] for k in ("seed","task","phase_aware_mae","uniform_phase_mae",
                                        "phase_shuffled_mae","clean_constant_mae","criterion_met")}
                     for x in decision["models"]]},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("evidence","selection-ledger","angle-ledger","precision-ledger","jacobian-ledger","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    run(a.evidence,a.selection_ledger,a.angle_ledger,a.precision_ledger,a.jacobian_ledger,a.out)


if __name__=="__main__":
    main()