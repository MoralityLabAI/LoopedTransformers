"""PEF-11: preregistered frozen-head noise/quantization frontier.

Read-only reuse of PEF-07 checkpoints and PEF-09/10 probe selection ledgers.
All 20 directions/model, probe scales and random seeds are frozen in a
separate ledger before a single heldout natural memory state is generated.
See PROTOCOL.md (standalone commit 8d04153) for the decision rules.
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
from research.pef07 import pef07 as p7
from research.pef08 import pef08 as p8
from research.pef09 import pef09 as p9
from research.pef10 import pef10 as p10

TANGENTS = ("pair01", "pair23", "pca2", "random")
ANGLES = (0, .25, 1, 4, 16)
NOISE_REL = (0, .001, .003, .01, .03, .1)
QUANT_REL = (0, .003, .01, .03, .1)
REPLICATES = (0, 1, 2)
PEF10_LEDGER_HASH = "3ef6ce605540a5009f6896cf02a938c200bde23c7a5104d378aace7388430177"
PEF10_RESULTS_HASH = "84dbbb7b6b238edd157d81a240ebc4a2a152b22f1299c98d5d8ee53179086730"
PREREG_COMMIT = "8d04153c2c7328ec2998c4edb14f9ef96b4f250c"
BATCH = 16
EPS = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def v64(data):
    return torch.tensor(data, dtype=torch.float64)


def offplane(g, u, seed):
    """A frozen unit direction orthogonal to the full signal plane."""
    for extra in (0, 100000):
        raw = torch.randn(len(g), dtype=torch.float64,
                          generator=torch.Generator(device="cpu").manual_seed(seed + extra))
        x = raw - (raw @ g) * g - (raw @ u) * u
        if float(x.norm()) > 1e-10:
            o = x / x.norm()
            if abs(float(o @ g)) > 1e-9 or abs(float(o @ u)) > 1e-9:
                raise AssertionError("offplane direction is not orthogonal")
            return o
    raise AssertionError("both offplane generators were degenerate")


def orientation(g, u, theta):
    radians = math.radians(theta)
    v = math.cos(radians)*g + math.sin(radians)*u
    w = -math.sin(radians)*g + math.cos(radians)*u
    if not (abs(float(v@w)) < 1e-9 and
            abs(float(v@v)-1) < 1e-9 and abs(float(w@w)-1) < 1e-9):
        raise AssertionError("rotated basis not orthonormal")
    return v, w


def gaussian_z(seed, n=768):
    return torch.randn(n, dtype=torch.float64,
                       generator=torch.Generator(device="cpu").manual_seed(seed))


def scalar_quant(state, center, w, step):
    """Quantize the retained transverse coordinate, retain perpendicular residual."""
    if step == 0:
        return state.clone()
    if step <= 0:
        raise ValueError("quantization step must be nonnegative")
    x = state.double()
    center, w = center.double(), w.double()
    coordinate = (x-center)@w
    rounded = step * torch.round(coordinate / step)
    return (x + (rounded-coordinate)[:,None]*w[None,:]).to(dtype=state.dtype)


def channel_quant(state, center, per_channel_step):
    """Fixed per-channel midtread lattice, centered on probe mean."""
    if torch.count_nonzero(per_channel_step) == 0:
        return state.clone()
    if torch.any(per_channel_step <= 0):
        raise ValueError("positive quantization steps expected")
    x=state.double()
    center, step = center.double(), per_channel_step.double()
    return (center + step*torch.round((x-center)/step)).to(dtype=state.dtype)


def perturb_gaussian(state, direction, noise, absolute_sigma):
    if absolute_sigma == 0:
        return state.clone()
    return (state.double() + absolute_sigma*noise[:,None]*direction[None,:]).to(dtype=state.dtype)


def expected_noise_seed(seed, task, tangent_idx, angle_idx, replica):
    return (111100000 + 10000*seed + (1000 if task=="threshold" else 0) +
            100*tangent_idx + 10*angle_idx + replica)


def select_20(model_selection, probe_states):
    """Make every probe-only selector and verify all prior ridge & direction SHA256."""
    seed, task = model_selection["seed"], model_selection["task"]
    center=v64(model_selection["frozen_center"])
    g=v64(model_selection["global_pc"])
    all_u={name:v64(v) for name,v in model_selection["tangents"].items()}
    if not torch.allclose(probe_states.double().mean(0),center,atol=2e-5,rtol=1e-5):
        raise AssertionError("PEF-10 probe mean changed")
    pc_coeff = (probe_states.double()-center)@g
    pc_sd = float(pc_coeff.square().mean().sqrt())
    if not pc_sd > 0:
        raise AssertionError("zero principal-component scale")
    channel_std=probe_states.double().std(0,unbiased=False).clamp_min(pc_sd/1000)
    selected=[]
    for tidx,name in enumerate(TANGENTS):
        u=all_u[name]
        for aidx,angle in enumerate(ANGLES):
            sign=0 if angle==0 else 1
            matches=[r for r in model_selection["conditions"]
                     if r["tangent"]==name and r["source"]=="grid" and
                     r["angle_deg"]==angle and r["sign"]==sign]
            if len(matches)!=1:
                raise AssertionError(f"exactly one PEF-10 frozen grid row needed for {name}/{angle}")
            original=matches[0]
            v,w=orientation(g,u,angle)
            if not torch.allclose(v,v64(original["direction"]),atol=1e-10,rtol=1e-10):
                raise AssertionError("PEF-10 direction differs from frozen geometry")
            if p9.tensor_digest(v64(original["direction"]))!=original["direction_sha256"]:
                raise AssertionError("PEF-10 direction SHA mismatch")
            for horizon in ("h12","h14"):
                fit=p10.unmarshal_fit(original["frozen_ridge_"+horizon])
                if p10.ridge_digest(fit)!=original["ridge_"+horizon+"_sha256"]:
                    raise AssertionError("PEF-10 fitted ridge bytes differ")
            randseed=110110+seed+(10000 if task=="threshold" else 0)+100*tidx+aidx
            v=v64(original["direction"])  # preserve the *exact* PEF-10 serialized projector bytes
            o=offplane(g,u,randseed)
            seeds=[expected_noise_seed(seed,task,tidx,aidx,r) for r in REPLICATES]
            selected.append({
                "tangent":name,"tangent_index":tidx,"angle_deg":angle,
                "angle_index":aidx,"sign":sign,
                "direction":v.tolist(),"retained_direction":w.tolist(),
                "offplane_direction":o.tolist(),"offplane_seed":randseed,
                "ridge_h12_fit":original["frozen_ridge_h12"],
                "ridge_h14_fit":original["frozen_ridge_h14"],
                "ridge_h12_sha256":original["ridge_h12_sha256"],
                "ridge_h14_sha256":original["ridge_h14_sha256"],
                "direction_sha256":original["direction_sha256"],
                "noise_seeds":seeds,
            })
    if len(selected)!=len(TANGENTS)*len(ANGLES):
        raise AssertionError("20 frozen PEF-10 directions expected")
    return {"seed":seed,"task":task,"checkpoint_sha256":model_selection["checkpoint_sha256"],
            "center":center.tolist(),"pc_scale":pc_sd,
            "channel_std":channel_std.tolist(),"global_pc":g.tolist(),
            "tangents":{key:val.tolist() for key,val in all_u.items()},
            "conditions":selected}


def verify_and_select(evidence, source_selection, angle_ledger):
    if sha(angle_ledger)!=PEF10_LEDGER_HASH:
        raise AssertionError("frozen PEF-10 angle/ridge ledger SHA256 altered")
    catalog,recs,_=p10.check_frozen_inputs(evidence,source_selection)
    ledger=json.loads(Path(angle_ledger).read_text())
    if ledger.get("campaign")!="PEF-10" or not ledger.get("every_angle_and_ridge_frozen_before_any_heldout"):
        raise AssertionError("PEF-10 angle ledger lacks sealed provenance")
    if ledger["pef07_training_ledger_sha256"]!=p10.TRAINING_HASH or ledger["pef09_selection_ledger_sha256"]!=p10.SELECTION_HASH:
        raise AssertionError("frozen hierarchy SHA mismatch")
    if len(ledger["model_selections"])!=6:
        raise AssertionError("six original fitted PEF-10 models expected")
    all_select=[]
    for rec in recs:
        row=next((x for x in ledger["model_selections"]
                 if x["seed"]==rec["seed"] and x["task"]==rec["task"]),None)
        if row is None or row["checkpoint_sha256"]!=rec["checkpoint_sha256"]:
            raise AssertionError("checkpoint missing from PEF-10 ledger")
        model=p8.checked_model(evidence,rec)
        original=p8.weights_digest(model)
        with torch.no_grad():
            probe_h8=p8.states_by_loop(model,catalog["probe"])[8]
            selected=select_20(row,probe_h8)
        if p8.weights_digest(model)!=original:
            raise AssertionError("probe features modified checkpoint weights")
        all_select.append(selected)
    return catalog,recs,all_select


def scalar_report(margin,truth):
    return p9.class_breakdown(margin,truth)


def output_scores(model, original_h14_ridge, fitted_h12, fitted_h14, states12, states14,
                  truth, is_natural):
    original12=p9.native_score(model,states12)
    old14=p8.ridge_margin(states14,original_h14_ridge)
    row={"native_h12":scalar_report(original12,truth),"original_ridge_h14":scalar_report(old14,truth)}
    if not is_natural:
        row["frozen_pef10_ridge_h12"]=scalar_report(p8.ridge_margin(states12,fitted_h12),truth)
        row["frozen_pef10_ridge_h14"]=scalar_report(p8.ridge_margin(states14,fitted_h14),truth)
    return row


def score_condition(model,source_selection,condition,held_bits,trace,truth,center,pc_scale,channel_std):
    v=v64(condition["direction"])
    w=v64(condition["retained_direction"])
    o=v64(condition["offplane_direction"])
    base=p9.projector(trace[8],center,v)
    natural=trace[8]
    old14=p9.ridge_from_selection(source_selection)
    h12fit=p10.unmarshal_fit(condition["ridge_h12_fit"])
    h14fit=p10.unmarshal_fit(condition["ridge_h14_fit"])
    # Serializing every intervention's deterministic inputs makes scores independently auditable.
    branches=[{"kind":"baseline","site":"ablated","relative_level":0,
               "replicate":None,"noise_seed":None,"actual_l2":0.0,"state":base},
              {"kind":"baseline","site":"natural","relative_level":0,
               "replicate":None,"noise_seed":None,"actual_l2":0.0,"state":natural}]
    for rel in NOISE_REL[1:]:
        sigma=rel*pc_scale
        for rep,seed in enumerate(condition["noise_seeds"]):
            z=gaussian_z(seed,len(held_bits))
            for site,direction,starting in (("transverse",w,base),("offplane",o,base),
                                            ("natural_transverse",w,natural)):
                perturbed=perturb_gaussian(starting,direction,z,sigma)
                branches.append({"kind":"gaussian","site":site,"relative_level":rel,
                    "replicate":rep,"noise_seed":seed,
                    "actual_l2":float((perturbed-starting).double().norm(dim=1).mean()),
                    "state":perturbed})
    for rel in QUANT_REL[1:]:
        delta=rel*pc_scale
        for site,starting in (("ablated",base),("natural",natural)):
            state=scalar_quant(starting,center,w,delta)
            branches.append({"kind":"transverse_quant","site":site,"relative_level":rel,
                "replicate":None,"noise_seed":None,
                "actual_l2":float((state-starting).double().norm(dim=1).mean()),"state":state})
            state=channel_quant(starting,center,channel_std*rel)
            branches.append({"kind":"channel_quant","site":site,"relative_level":rel,
                "replicate":None,"noise_seed":None,
                "actual_l2":float((state-starting).double().norm(dim=1).mean()),"state":state})
    # All precision-zero baseline branches are exact no-op references to the same state.
    output=[]
    with torch.no_grad():
        for off in range(0,len(branches),BATCH):
            part=branches[off:off+BATCH]
            pack=torch.stack([r["state"] for r in part])
            states12,states14=p9.replay_batched(model,held_bits,pack,8)
            for i,row in enumerate(part):
                natural_branch=(row["site"]=="natural" or row["site"]=="natural_transverse")
                report=output_scores(model,old14,h12fit,h14fit,states12[i],states14[i],truth,natural_branch)
                output.append({k:v for k,v in row.items() if k!="state"} | {"scores":report})
    return output


def select_rows(m, tangent, angle, kind, site, level):
    cond=next(c for c in m["conditions"] if c["tangent"]==tangent and c["angle_deg"]==angle)
    return [r for r in cond["scores"] if r["kind"]==kind and r["site"]==site
            and r["relative_level"]==level]


def mean_correct(rows, key):
    if not rows:
        raise AssertionError("missing registered measurements")
    return sum(r["scores"][key]["correct"] for r in rows)/len(rows)


def judgement(results):
    lines=[]
    for model in results:
        if model["task"]!="coalition":
            continue
        small=select_rows(model,"random",.25,"baseline","ablated",0)
        noisy=select_rows(model,"random",.25,"gaussian","transverse",.01)
        natural=select_rows(model,"random",.25,"gaussian","natural_transverse",.01)
        qsmall=select_rows(model,"random",.25,"transverse_quant","ablated",.03)
        large=select_rows(model,"random",4,"gaussian","transverse",.01)
        qlarge=select_rows(model,"random",4,"transverse_quant","ablated",.03)
        off=select_rows(model,"random",.25,"gaussian","offplane",.01)
        clean=mean_correct(small,"frozen_pef10_ridge_h12")
        value={"seed":model["seed"],"clean_small":clean,
               "gaussian_small":mean_correct(noisy,"frozen_pef10_ridge_h12"),
               "offplane_small":mean_correct(off,"frozen_pef10_ridge_h12"),
               "natural_control_small":mean_correct(natural,"native_h12"),
               "quant_small":mean_correct(qsmall,"frozen_pef10_ridge_h12"),
               "gaussian_large":mean_correct(large,"frozen_pef10_ridge_h12"),
               "quant_large":mean_correct(qlarge,"frozen_pef10_ridge_h12")}
        value["fragility"]=(value["clean_small"]>=720 and
                            value["gaussian_small"] <= clean-96 and
                            value["natural_control_small"]>=720 and
                            value["quant_small"]<=clean-96)
        value["large_angle_robust"]=(value["gaussian_large"]>=720 and value["quant_large"]>=720)
        value["both_met"]=value["fragility"] and value["large_angle_robust"]
        value["offplane_minus_transverse_correct"]=(value["offplane_small"]-value["gaussian_small"])
        lines.append(value)
    passed=sum(v["both_met"] for v in lines)
    return {"coalition_per_seed":lines,"coalition_both_seed_count":passed,
            "criterion":"criterion_met" if passed>=2 else "criterion_not_met",
            "transverse_more_damaging_than_offplane_seed_count":sum(
                v["offplane_minus_transverse_correct"]>0 for v in lines)}


def frontier(model):
    rows=[]
    for c in model["conditions"]:
        baseline=mean_correct([r for r in c["scores"] if r["kind"]=="baseline" and
                               r["site"]=="ablated"],"frozen_pef10_ridge_h12")
        for mode in ("gaussian","transverse_quant","channel_quant"):
            site="transverse" if mode=="gaussian" else "ablated"
            levels=NOISE_REL if mode=="gaussian" else QUANT_REL
            detail=[]
            for rel in levels:
                matches=[r for r in c["scores"] if r["kind"]==mode and r["site"]==site
                         and r["relative_level"]==rel] if rel else [r for r in c["scores"]
                         if r["kind"]=="baseline" and r["site"]=="ablated"]
                scores=mean_correct(matches,"frozen_pef10_ridge_h12")
                detail.append({"relative_level":rel,"mean_correct":scores,">=720":scores>=720})
            rows.append({"tangent":c["tangent"],"angle_deg":c["angle_deg"],
                         "mode":mode,"clean_correct":baseline,"levels":detail,
                         "largest_passing_level":max((x["relative_level"] for x in detail
                                  if x[">=720"]),default=None)})
    return rows


def run(evidence, selection_ledger, angle_ledger, out, previous_results=None):
    os.environ.setdefault("DNNL_MAX_CPU_ISA","AVX2")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence,selection_ledger,angle_ledger,out=(Path(x) for x in
                                     (evidence,selection_ledger,angle_ledger,out))
    out.mkdir(parents=True,exist_ok=True)
    catalog,recs,all_select=verify_and_select(evidence,selection_ledger,angle_ledger)
    if previous_results is not None and sha(previous_results)!=PEF10_RESULTS_HASH:
        raise AssertionError("PEF-10 independent clean results SHA mismatch")
    previous=json.loads(Path(previous_results).read_text()) if previous_results else None
    seal={"campaign":"PEF-11","preregistration_commit":PREREG_COMMIT,
          "pef10_angle_ledger_sha256":PEF10_LEDGER_HASH,
          "pef09_selection_ledger_sha256":p10.SELECTION_HASH,
          "pef07_training_ledger_sha256":p10.TRAINING_HASH,
          "previous_pef10_results_sha256":PEF10_RESULTS_HASH if previous is not None else None,
          "noise_relative_levels":list(NOISE_REL),"quant_relative_levels":list(QUANT_REL),
          "replicates":list(REPLICATES),"angles_deg":list(ANGLES),"tangents":list(TANGENTS),
          "all_conditions_and_noise_seeds_frozen_before_heldout":True,
          "model_selections":all_select}
    (out/"frozen_precision_ledger.json").write_bytes(p9.canonical(seal))
    ledgerhash=sha(out/"frozen_precision_ledger.json")
    print("PEF11_ALL_SELECTIONS_FROZEN",ledgerhash,flush=True)
    results=[]
    for rec,sel in zip(recs,all_select,strict=True):
        model=p8.checked_model(evidence,rec)
        original=p8.weights_digest(model)
        with torch.no_grad():
            trace=p8.states_by_loop(model,catalog["heldout"])
            truth=p8.labels(catalog["heldout"],rec["task"])
            if p9.class_breakdown(p9.native_score(model,trace[12]),truth)["correct"]!=768:
                raise AssertionError("PEF-07 native h12 differs")
            source=json.loads(Path(selection_ledger).read_text())
            selection=next(s for s in source["selections"] if s["seed"]==rec["seed"]
                           and s["task"]==rec["task"])
            if p9.class_breakdown(p8.ridge_margin(trace[14],p9.ridge_from_selection(selection)),truth)["correct"]!=768:
                raise AssertionError("PEF-07 frozen h14 differs")
            conditions=[]
            for c in sel["conditions"]:
                scores=score_condition(model,selection,c,catalog["heldout"],trace,truth,
                    v64(sel["center"]),sel["pc_scale"],v64(sel["channel_std"]))
                clean=next(r for r in scores if r["kind"]=="baseline" and r["site"]=="ablated")
                nat=next(r for r in scores if r["kind"]=="baseline" and r["site"]=="natural")
                if nat["scores"]["native_h12"]["correct"]!=768:
                    raise AssertionError("natural-control zero noise differs from PEF-07")
                if previous is not None:
                    pm=next(m for m in previous["models"] if m["seed"]==rec["seed"] and m["task"]==rec["task"])
                    previous_row=next(r for r in pm["results"] if r["tangent"]==c["tangent"] and
                                      r["angle_deg"]==c["angle_deg"] and r["sign"]==c["sign"] and r["source"]=="grid")
                    for current_key, previous_key in (("native_h12","native_h12"),
                                   ("original_ridge_h14","old_ridge_h14"),
                                   ("frozen_pef10_ridge_h12","new_ridge_h12"),
                                   ("frozen_pef10_ridge_h14","new_ridge_h14")):
                        if clean["scores"][current_key]["correct"]!=previous_row[previous_key]["correct"]:
                            raise AssertionError("clean PEF-10 run did not reproduce exactly")
                conditions.append({"tangent":c["tangent"],"angle_deg":c["angle_deg"],
                                   "sign":c["sign"],"direction_sha256":c["direction_sha256"],
                                   "scores":scores})
        if p8.weights_digest(model)!=original or sha(evidence/rec.get("file",rec.get("checkpoint_file")))!=rec["checkpoint_sha256"]:
            raise AssertionError("frozen checkpoint altered during scoring")
        result={"seed":rec["seed"],"task":rec["task"],"checkpoint_sha256":rec["checkpoint_sha256"],
                "natural_native_h12_correct":768,"natural_original_h14_correct":768,
                "conditions":conditions,"frontier":None}
        result["frontier"]=frontier(result)
        results.append(result)
        print("PEF11_HELDOUT_SCORED",rec["seed"],rec["task"],len(conditions),flush=True)
    decision=judgement(results)
    report={"campaign":"PEF-11","preregistration_commit":PREREG_COMMIT,
            "frozen_precision_ledger_sha256":ledgerhash,"source_split_sha256":p9.SIGNATURE,
            "torch_version":str(torch.__version__),"heldout_count":768,
            "all_selections_frozen_before_heldout":True,"decision":decision,
            "models":results}
    (out/"summary.json").write_bytes(p9.canonical(report))
    with (out/"frontier.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=("seed","task","tangent","angle_deg",
                                        "mode","clean_correct","largest_passing_level"))
        writer.writeheader()
        for m in results:
            for row in m["frontier"]:
                writer.writerow({"seed":m["seed"],"task":m["task"],**{
                    key:row[key] for key in ("tangent","angle_deg","mode","clean_correct","largest_passing_level")}})
    print(json.dumps(decision,indent=2,sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--selection-ledger",type=Path,required=True)
    parser.add_argument("--angle-ledger",type=Path,required=True)
    parser.add_argument("--previous-results",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.evidence,args.selection_ledger,args.angle_ledger,args.out,args.previous_results)


if __name__=="__main__":
    main()
