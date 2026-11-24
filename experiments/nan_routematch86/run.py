"""Audit physical coverage for a phase-1 learned-versus-static write trial."""

import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
import sys

from experiments.nan_lattice53.run import exact_infeasible, screen
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import digest, joined, write_new
from experiments.nan_recipe85.run import DOSE as DELAY_CONFIG, ROUTE, load_controllers, phase_edits
from experiments.nan_refit49.run import load_fits
from experiments.nan_route50.run import fit_packet, state_packet

from .geometry import ARMS, COSINE, DOSE, DOSE_TOLERANCE, plan, self_test


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/"experiments/nan_readpath37/registration.json"
MANIFEST=HERE/"SOURCE_MANIFEST.json"
BUDGET={"minimum_start_available_RAM_bytes":5*2**30,"minimum_available_RAM_bytes":4*2**30,
        "maximum_private_bytes":768*2**20,"maximum_seconds":600}


def seal():
    paths=[HERE/n for n in (".gitattributes","__init__.py","PROTOCOL.md","run.py","geometry.py")]
    paths += [BASE,DELAY_CONFIG,ROUTE,*[REPO/n for n in (
        "experiments/nan_recipe85/run.py","experiments/nan_recipe85/SOURCE_MANIFEST.json",
        "experiments/nan_route50/run.py","experiments/nan_refit49/run.py",
        "experiments/nan_radial76/geometry.py","experiments/nan_cidose74/geometry.py","experiments/nan_lattice53/run.py","experiments/nan_inverse65/lattice.py",
        "experiments/nan_delay39/instrument.py","experiments/nan_dose38/run.py",
        "experiments/nan_fit07/pointwise.py","experiments/nan_phase35/run.py",
        "experiments/nan_readpath37/run.py","experiments/nan_ram28/resources.py",
        "experiments/nan_ranksite13/resources.py",
    )]]
    return {"experiment":"NAN-ROUTEMATCH-86","resources":BUDGET,
            "files_sha256":{p.relative_to(REPO).as_posix():digest(p) for p in paths}}


def summarize(rows):
    pairs=defaultdict(list)
    for row in rows:
        pairs[row["root"]].append(row)
    if any({r["world"] for r in pair}!={"A","B"} or len(pair)!=2 for pair in pairs.values()):
        raise ValueError("A graph root lacks its native twin")
    good=[r for r in rows if r["status"]=="ok"]
    counts={"worlds":len(rows),"roots":len(pairs),"degenerate_worlds":len(rows)-len(good)}
    for name in ("continuous_radial_intersection","original_angle_impossible","raw_rays_match"):
        counts[name+"_worlds"]=sum(r[name] for r in good)
        counts[name+"_both_world_roots"]=sum(all(r.get(name,False) for r in pair) for pair in pairs.values())
    for mode in ("natural","tangent","locked_tangent","anchored_tangent"):
        candidates={r["id"] for r in good if r[mode]["status"]=="matched"}
        distinct={r["id"] for r in good if r[mode]["status"]=="matched" and
                  not r[mode]["same_learned_and_static_bytes"]}
        result={"matched_worlds":len(candidates),"distinct_worlds":len(distinct),
                "matched_both_world_roots":sum({r["id"] for r in pair}<=candidates for pair in pairs.values()),
                "distinct_both_world_roots":sum({r["id"] for r in pair}<=distinct for pair in pairs.values())}
        if distinct:
            qualified=[r[mode] for r in good if r["id"] in distinct]
            result.update(max_dose_gap=max(r["dose_gap_fraction"] for r in qualified),
                          max_radial_gap=max(r["radial_gap_fraction"] for r in qualified),
                          max_relative_norm_gap=max(r["relative_norm_gap"] for r in qualified),
                          median_min_original_cosine=median(min(r["measured"][a]["original_direction_cosine"]
                                                                for a in ARMS[:2]) for r in qualified))
        counts[mode]=result
    counts["exact_angular_failure_arms"]={arm:sum(r["exact_angle"][arm]["angular_infeasibility_certified"]
                                                for r in rows) for arm in ARMS[:2]}
    counts["any_exact_angular_failure_worlds"]=sum(any(x["angular_infeasibility_certified"]
                                                          for x in r["exact_angle"].values()) for r in rows)
    return counts



def exact_certs(native, directions, np, check):
    spec={"phase1_dose_fraction":DOSE,"dose_relative_tolerance":DOSE_TOLERANCE,
          "minimum_direction_cosine":COSINE}
    result={}
    for arm,d in zip(ARMS[:2],directions,strict=True):
        check()
        if np.linalg.norm(d)<=1e-12:
            result[arm]={"angular_infeasibility_certified":False,"status":"degenerate"}
            continue
        hint,scale=screen(native,d,spec,np)
        exact=hint<COSINE and exact_infeasible(native,d,scale,spec,np)
        result[arm]={"angular_infeasibility_certified":bool(exact),"screen_cosine_bound":hint}
    return result

def evaluate(backend, base, roster, banks, check, masks, np):
    controller,errors=load_controllers(backend,base,masks,np)
    _,t2,_,eps,_,_=fit_packet(backend,base,np)
    calibration={"mean_comp_acts":t2["mean_comp_acts"],"mean_write_rms":t2["mean_write_rms"],"eps":eps}
    fit_route=json.loads(ROUTE.read_bytes())["per_backend"][backend]["passes"]["1"]["static_prefit"]["selected_anchor"]
    mask=fit_route.removeprefix("binary:")
    if not fit_route.startswith("binary:") or len(mask)!=base["rank"] or set(mask)-{"0","1"}:
        raise ValueError("The registered CAL-static preference changed")
    anchor=np.array([int(x) for x in mask])
    initial,t1=banks[f"{backend}/T1"]
    shared,t2fas=banks[f"{backend}/T2"]
    if not all(np.array_equal(a,b) for a,b in zip(initial,shared,strict=True)):
        raise ValueError("Factor banks no longer share the SVD")
    rows,inputs,outputs=state_packet(backend,base,roster,check,np)
    focused=[i for i,row in enumerate(rows) if row["arm"]=="focused"]
    y=outputs[focused,0].astype(np.float64)
    directions,mass,_=phase_edits(inputs[focused,0].astype(np.float64),y,0,{0:initial,1:t1,2:t2fas},
                                  controller,calibration,anchor,masks,np)
    prior,dose=directions["T2_T2_sphere"],directions["T2_static_sphere"]
    private=[]
    output={}
    for j,i in enumerate(focused):
        check()
        source=rows[i]
        entries,diagnostic=plan(y[j],prior[j],dose[j],source["id"]+"|phase1",np)
        # The full vectors remain reconstructible from the pinned inputs; publish only reach geometry.
        if entries:
            for arm_set in entries.values():
                for outcome in arm_set.values():
                    assert outcome.shape==y[j].shape
        private.append(source | diagnostic | {"exact_angle":exact_certs(y[j],(prior[j],dose[j]),np,check)})
    for split,count in base["roots"].items():
        values=[r for r in private if r["split"]==split]
        assert len(values)==count*2
        output[split]=summarize(values)
    return {"coverage":output,"native_CI_max_abs_error":errors,"common_mass_max_error":mass},private


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze",action="store_true")
    cli.add_argument("--work",type=Path)
    args=cli.parse_args()
    registered=seal()
    if args.freeze:
        write_new(MANIFEST,registered)
        return
    if not __debug__ or not args.work or json.loads(MANIFEST.read_bytes())!=registered:
        cli.error("Require the frozen source, assertions and a fresh result directory")
    base=json.loads(BASE.read_bytes())
    with lease(BUDGET) as (check,samples):
        if version("numpy")!="2.5.5":
            raise ValueError("Pinned numerical runtime changed")
        import numpy as np
        from experiments.nan_fit07.pointwise import masks

        fixture=self_test(np)
        banks=load_fits(base,np)
        roster,_=joined(base)
        args.work.mkdir(parents=True,exist_ok=False)
        backend_scores={}
        for b in base["backends"]:
            result,private=evaluate(b,base,roster,banks,check,masks,np)
            backend_scores[b]=result
            write_new(args.work/f"{b}_private.json",private)
            print(b,result["coverage"],flush=True)
    assert registered==seal() and "torch" not in sys.modules
    write_new(args.work/"aggregate.json",{"experiment":registered["experiment"],"status":"passed",
               "source_manifest_sha256":digest(MANIFEST),"unit":"COMP14 graph root","roots":base["roots"],
               "verification":fixture,"per_backend":backend_scores,"pretrained_decoder_forwards":0,
               "formal_comparisons":0,"sealed_test_accessed":False,
               "resources":{"max_sampled_private_bytes":max(s["private_bytes"] for s in samples),
               "max_sampled_working_set_bytes":max(s["working_set_bytes"] for s in samples),
               "min_sampled_available_RAM_bytes":min(s["available_physical_bytes"] for s in samples)}})


if __name__=="__main__":
    main()
