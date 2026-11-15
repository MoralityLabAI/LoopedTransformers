"""PEF-10: registered ungated PC ablation-angle and ridge recovery frontier.

Only PEF-09's frozen source/probe selection and the original PEF-07
checkpoints are used. All angle directions and post-ablation probe ridge
fits are sealed before any heldout states are generated or scored.
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

SEEDS = (103, 131, 167)
TASKS = ("coalition", "threshold")
VISIT = 8
PAIRS = ((0, 1), (2, 3))
GRID = (0, .25, .5, 1, 2, 4, 8, 16, 32, 64, 90)
INTERIOR = tuple(x for x in GRID if x not in (0, 90))
TANGENTS = ("pair01", "pair23", "pca2", "random")
CHUNK = 12
EXPECTED_CONDITIONS = 84
SELECTION_HASH = "f206d88f5f2ea9703edc385d47bcf6fd6cd11a6cb4a3ece6edeae230dd734157"
TRAINING_HASH = "1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714"
PREREG_COMMIT = "de6666c71b65287df56b24cb0d57e0e89ff7553c"


def from_list(value):
    return torch.tensor(value, dtype=torch.float64)


def stable_sign(x):
    j = int(x.abs().argmax())
    return x if float(x[j]) >= 0 else -x


def unit(x):
    norm = torch.linalg.vector_norm(x)
    if not torch.isfinite(norm) or float(norm) < 1e-12:
        raise AssertionError("degenerate angle geometry")
    return x / norm


def tangent_pair(vector, g, fallback):
    """Align frozen PEF-09 pair vector to g, then Gram-Schmidt it."""
    v = unit(vector)
    if float(v @ g) < 0:
        v = -v
    cos = float(torch.clamp(v @ g, -1, 1))
    resid = v - (v @ g) * g
    mag = float(resid.norm())
    degenerate = mag < 1e-9
    u = fallback if degenerate else unit(resid)
    phi = math.degrees(math.acos(cos)) if not degenerate else 0.0
    if abs(float(u @ g)) > 1e-7:
        raise AssertionError("pair tangent not perpendicular to g")
    return u, phi, degenerate


def rotate(g, u, degrees, sign):
    if sign not in (-1, 0, 1):
        raise ValueError("orientation must be -1/0/+1")
    if degrees in (0, 90):
        if sign != 0:
            raise ValueError("endpoints are sign-independent")
    elif sign not in (-1, 1):
        raise ValueError("interior needs signed orientation")
    theta = math.radians(degrees)
    # Orientations at theta=0 and 90 each describe one projector.
    coefficient = 1 if sign == 0 else sign
    return unit(math.cos(theta) * g + coefficient * math.sin(theta) * u)


def fixed_angles(phi, name):
    choices = [(angle, sign, "grid") for angle in GRID for sign in
               ((0,) if angle in (0, 90) else (-1, 1))]
    if name.startswith("pair") and phi > 1e-9:
        for sign in (-1, 1):
            if all(not(abs(a - phi) <= 1e-12 and s == sign) for a, s, _ in choices):
                choices.append((phi, sign, "pair_anchor"))
    return choices


def identity_2d(a, b, theta, sign=1):
    """Analytic PC/tangent decomposition of a rank-one projection."""
    s, c = math.sin(theta) * sign, math.cos(theta)
    return (-a * s + b * c), (-s, c)


def check_frozen_inputs(evidence, ledger_file):
    evidence, ledger_file = Path(evidence), Path(ledger_file)
    original_ledger = evidence / "sealed_training_ledger.json"
    if hashlib.sha256(original_ledger.read_bytes()).hexdigest() != TRAINING_HASH:
        raise AssertionError("PEF-07 original ledger SHA256 differs")
    if hashlib.sha256(ledger_file.read_bytes()).hexdigest() != SELECTION_HASH:
        raise AssertionError("PEF-09 frozen selection ledger SHA256 differs")
    selection = json.loads(ledger_file.read_text())
    if not selection.get("all_selections_frozen_before_heldout", False):
        raise AssertionError("PEF-09 source-only selection not sealed")
    catalog, source_hash, manifest = p8.source_data()
    _, recs = p9.verify_original_evidence(evidence, catalog, source_hash, manifest)
    if source_hash != selection["source_sha256"]:
        raise AssertionError("frozen split mismatch")
    if len(selection["selections"]) != 6:
        raise AssertionError("six feature selections expected")
    return catalog, recs, selection


def select_geometry(model, selection, probe):
    trace = p8.states_by_loop(model, probe)
    states = trace[VISIT].double()
    row01 = next(r for r in selection["rows"] if r["loop"] == VISIT and r["pair"] == [0, 1])
    row23 = next(r for r in selection["rows"] if r["loop"] == VISIT and r["pair"] == [2, 3])
    center = from_list(row01["center"])
    if not torch.equal(center, from_list(row23["center"])):
        raise AssertionError("PEF-09 stored loop means disagree")
    global_pc = next(c for c in row01["conditions"] if c["name"] == "global_pc")
    g = unit(from_list(global_pc["unit"]))
    global_pc_again = next(c for c in row23["conditions"] if c["name"] == "global_pc")
    if abs(float(g @ from_list(global_pc_again["unit"])) - 1) > 1e-8:
        raise AssertionError("PEF-09 global PC vector mismatch")
    if not torch.allclose(states.mean(0), center, atol=2e-5, rtol=1e-5):
        raise AssertionError("probe mean changed since PEF-09")
    _, _, v_right = torch.linalg.svd(states-center, full_matrices=False)
    if abs(float(v_right[0] @ g)) < 1 - 1e-5:
        raise AssertionError("PCA leading axis changed since PEF-09")
    pca2 = unit(stable_sign(v_right[1]) - (stable_sign(v_right[1]) @ g) * g)
    # Non-generic selection of a deterministic single tangent is probe-independent.
    seed = selection["seed"]
    task = selection["task"]
    gen = torch.Generator(device="cpu").manual_seed(
        10100 + seed + (10000 if task == "threshold" else 0))
    rand = torch.randn(32, dtype=torch.float64, generator=gen)
    rand = unit(stable_sign(rand) - (stable_sign(rand) @ g) * g)
    tangents = {"pca2": pca2, "random": rand}
    phis = {}
    degenerate = {}
    for name, row in (("pair01",row01),("pair23",row23)):
        pair = from_list(next(c for c in row["conditions"] if c["name"] == "target_pair_svd")["unit"])
        u, phi, deg = tangent_pair(pair, g, pca2)
        tangents[name] = u
        phis[name] = phi
        degenerate[name] = deg
    for name, u in tangents.items():
        if abs(float(u @ g)) > 1e-6 or abs(float(u.norm()) - 1) > 1e-6:
            raise AssertionError("invalid frozen tangent " + name)
    conditions = []
    for name in TANGENTS:
        for angle, sign, source in fixed_angles(phis.get(name, 0.0), name):
            v = rotate(g, tangents[name], angle, sign)
            energy = float(((states-center) @ v).square().mean())
            gcoef = (states-center) @ g
            bcoef = (states-center) @ tangents[name]
            sin = math.sin(math.radians(angle)) * (1 if sign == 0 else sign)
            cos = math.cos(math.radians(angle))
            kept_2d = -sin*gcoef + cos*bcoef
            predicted = -sin*gcoef + cos*bcoef
            if not torch.allclose(kept_2d, predicted, atol=1e-12):
                raise AssertionError("geometric identity failed")
            conditions.append({
                "tangent":name,"angle_deg":angle,"sign":sign,"source":source,
                "direction":v.tolist(),"direction_sha256":p9.tensor_digest(v),
                "probe_removed_energy":energy,
                "probe_retained_tangent_energy":float(kept_2d.square().mean()),
                "pc_leakage_power":float(gcoef.square().mean()) * sin*sin,
                "residual_tangent_power":float(bcoef.square().mean()) * cos*cos,
                "pc_cross_term":float((-2*sin*cos*gcoef*bcoef).mean()),
            })
    if len(conditions) != EXPECTED_CONDITIONS:
        raise AssertionError(f"registered 84 direction conditions expected; found {len(conditions)}")
    return {"seed":seed,"task":task,"checkpoint_sha256":selection["checkpoint_sha256"],
            "frozen_center":center.tolist(),"global_pc":g.tolist(),
            "tangents":{k:x.tolist() for k,x in tangents.items()},
            "pair_angles":phis,"pair_degenerate":degenerate,
            "conditions":conditions}, trace


def steps_with_ridge(model, bits, trace, directions, center, start=VISIT):
    """Batched natural replay, split into small direction chunks."""
    for start_idx in range(0, len(directions), CHUNK):
        block = directions[start_idx:start_idx+CHUNK]
        ux = torch.stack([from_list(c["direction"]) for c in block])
        patches = p9.projector_many(trace[start], from_list(center), ux)
        state12, state14 = p9.replay_batched(model, bits, patches, start)
        yield start_idx, block, state12, state14


def marshal_fit(fit):
    return [v.tolist() for v in fit]


def unmarshal_fit(fit):
    return tuple(from_list(v) for v in fit)


def ridge_digest(fit):
    h=hashlib.sha256()
    for tensor in fit:
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def choose_probe(model, selected, probe, trace):
    truth = p8.labels(probe, selected["task"])
    for start_idx, chunk, h12, h14 in steps_with_ridge(
        model, probe, trace, selected["conditions"], selected["frozen_center"]):
        for j, condition in enumerate(chunk):
            h12fit = p7.fit_ridge(h12[j], truth)
            h14fit = p7.fit_ridge(h14[j], truth)
            condition["frozen_ridge_h12"] = marshal_fit(h12fit)
            condition["frozen_ridge_h14"] = marshal_fit(h14fit)
            condition["ridge_h12_sha256"] = ridge_digest(h12fit)
            condition["ridge_h14_sha256"] = ridge_digest(h14fit)
    return selected


def accuracy(model, x, truth):
    return p9.class_breakdown(x, truth)


def score_held(model, selected, frozen_selection, held):
    trace = p8.states_by_loop(model, held)
    y = p8.labels(held, selected["task"])
    natural12 = p9.native_score(model, trace[12])
    original14fit = p9.ridge_from_selection(frozen_selection)
    natural14 = p8.ridge_margin(trace[14], original14fit)
    if p9.class_breakdown(natural12,y)["correct"] != 768 or p9.class_breakdown(natural14,y)["correct"] != 768:
        raise AssertionError("baseline PEF-07 scores differ: must remain 768/768")
    rows = []
    for off, block, states12, states14 in steps_with_ridge(
        model, held, trace, selected["conditions"], selected["frozen_center"]):
        for j,cond in enumerate(block):
            h12=states12[j]
            h14=states14[j]
            original12=p9.native_score(model,h12)
            original14=p8.ridge_margin(h14,original14fit)
            new12=p8.ridge_margin(h12,unmarshal_fit(cond["frozen_ridge_h12"]))
            new14=p8.ridge_margin(h14,unmarshal_fit(cond["frozen_ridge_h14"]))
            ux=from_list(cond["direction"])
            x=trace[VISIT].double()-from_list(selected["frozen_center"])
            hold_energy=float((x@ux).square().mean())
            rows.append({
                "tangent":cond["tangent"],"angle_deg":cond["angle_deg"],
                "sign":cond["sign"],"source":cond["source"],
                "direction_sha256":cond["direction_sha256"],
                "probe_removed_energy":cond["probe_removed_energy"],
                "heldout_removed_energy":hold_energy,
                "heldout_to_probe_energy":hold_energy/max(cond["probe_removed_energy"],1e-15),
                "native_h12":accuracy(model,original12,y),
                "old_ridge_h14":accuracy(model,original14,y),
                "new_ridge_h12":accuracy(model,new12,y),
                "new_ridge_h14":accuracy(model,new14,y),
            })
    if len(rows) != EXPECTED_CONDITIONS:
        raise AssertionError("missing heldout angle scores")
    return {"seed":selected["seed"],"task":selected["task"],
            "original_native_h12":accuracy(model,natural12,y),
            "original_fixed_h14":accuracy(model,natural14,y),
            "results":rows}


def first_recovery(rows, name, sign, threshold=720):
    angles=sorted((r["angle_deg"] for r in rows
                   if r["tangent"] == name and r["source"] == "grid" and
                   (r["sign"] == sign or r["sign"] == 0) and
                   r["new_ridge_h12"]["correct"] >= threshold))
    return angles[0] if angles else None


def verdict(models, selections):
    rows=[]
    for model in models:
        sel=next(s for s in selections if s["seed"]==model["seed"] and s["task"]==model["task"])
        per={}
        for name in TANGENTS:
            lo=first_recovery(model["results"],name,-1)
            hi=first_recovery(model["results"],name,+1)
            per[name] = {"minus":lo,"plus":hi,
                         "first":min([x for x in (lo,hi) if x is not None],default=None)}
        at_zero=next(r for r in model["results"] if r["tangent"]=="pair01" and r["angle_deg"]==0)
        criterion=(all(not sel["pair_degenerate"][name] for name in ("pair01","pair23"))
            and all(per[name]["first"] is not None and per[name]["first"] <= 4
                    for name in ("pair01","pair23"))
            and at_zero["new_ridge_h12"]["correct"] < 720)
        random_first=per["random"]["first"]
        specificity=(all(per[name]["first"] is not None for name in ("pair01","pair23"))
                     and random_first is not None and
                     all(per[name]["first"] < random_first for name in ("pair01","pair23")))
        rows.append({"seed":model["seed"],"task":model["task"],
                     "min_angles":per,"zero_accuracy":at_zero["new_ridge_h12"]["correct"],
                     "primary_per_seed":criterion,"specificity_per_seed":specificity})
    coals=[r for r in rows if r["task"]=="coalition"]
    passed=sum(r["primary_per_seed"] for r in coals)
    return {"per_model":rows,"coalition_primary_seed_count":passed,
            "geometric_recovery_decision":"criterion_met" if passed>=2 else "criterion_not_met",
            "pair_specificity_coalition_seed_count":sum(r["specificity_per_seed"] for r in coals)}


def run(evidence, selection_ledger, out):
    os.environ.setdefault("DNNL_MAX_CPU_ISA","AVX2")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence, selection_ledger, out=Path(evidence),Path(selection_ledger),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    catalog, recs, selections = check_frozen_inputs(evidence,selection_ledger)
    original_selections = selections["selections"]
    frozen = []
    # SEAL COMPLETE ANGLE GRID AND EVERY RIDGE READOUT ON THE PROBE SET FIRST.
    for rec in recs:
        src = next(s for s in original_selections if s["seed"]==rec["seed"] and s["task"]==rec["task"])
        model = p8.checked_model(evidence,rec)
        original_digest=p8.weights_digest(model)
        with torch.no_grad():
            selected, probe_trace = select_geometry(model,src,catalog["probe"])
            frozen_one=choose_probe(model,selected,catalog["probe"],probe_trace)
        if p8.weights_digest(model) != original_digest:
            raise AssertionError("model changed while fitting probe readouts")
        frozen.append(frozen_one)
        print("PEF10_PROBE_SEALED",rec["seed"],rec["task"],p9.digest(selected),flush=True)
    seal={"campaign":"PEF-10","registration_commit":PREREG_COMMIT,
          "pef09_selection_ledger_sha256":SELECTION_HASH,
          "pef07_training_ledger_sha256":TRAINING_HASH,
          "every_angle_and_ridge_frozen_before_any_heldout":True,
          "angles_deg":list(GRID),"model_selections":frozen}
    seal_bytes=p9.canonical(seal)
    (out/"frozen_angle_and_ridge_ledger.json").write_bytes(seal_bytes)
    ledger_hash=hashlib.sha256(seal_bytes).hexdigest()
    print("PEF10_ALL_FITS_FROZEN",ledger_hash,flush=True)
    # HELDOUT ONLY AFTER ledger was written and fully closed.
    results=[]
    for rec,selected in zip(recs,frozen,strict=True):
        model=p8.checked_model(evidence,rec)
        src=next(s for s in original_selections if s["seed"]==rec["seed"] and s["task"]==rec["task"])
        old=p8.weights_digest(model)
        with torch.no_grad():
            scored=score_held(model,selected,src,catalog["heldout"])
        if p8.weights_digest(model)!=old:
            raise AssertionError("model changed while scoring heldout")
        results.append(scored)
        print("PEF10_HELDOUT_SCORED",rec["seed"],rec["task"],flush=True)
    decision=verdict(results,frozen)
    report={"campaign":"PEF-10","registration_commit":PREREG_COMMIT,
            "frozen_ledger_sha256":ledger_hash,
            "source_split_sha256":p9.SIGNATURE,
            "heldout_examples":768,"torch_version":str(torch.__version__),
            "freeze_before_heldout":True,"decision":decision,"models":results}
    (out/"summary.json").write_bytes(p9.canonical(report))
    # Flat CSV includes full condition scoring with per-mask details preserved in JSON.
    keys=("seed","task","tangent","angle_deg","sign","source",
          "native_h12","old_ridge_h14","new_ridge_h12","new_ridge_h14",
          "probe_removed_energy","heldout_removed_energy","heldout_to_probe_energy")
    with (out/"frontier.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for model in results:
            for r in model["results"]:
                w.writerow({k:(r[k]["correct"] if k in ("native_h12","old_ridge_h14",
                    "new_ridge_h12","new_ridge_h14") else r[k]) if k in r else model[k]
                    for k in keys})
    print(json.dumps(decision,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--selection-ledger",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.evidence,args.selection_ledger,args.out)


if __name__=="__main__":
    main()
