"""Offline geometry of the saved native pass-boundary and final-query states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
import numpy as np
from activation_io import ARMS, SOURCE, read_run


def norm(x):
    return float(np.linalg.norm(x.astype(np.float64)))


def measure(data):
    r=data["recipient.final_query"]
    d=data["effect_donor.final_query"]-r
    s=data["sham_donor.final_query"]-r
    den=float(d.astype(np.float64)@d.astype(np.float64))
    arms={}
    for a in ARMS:
        delta=data["arm."+a+".final_query"]-r
        size=norm(delta)
        alignment=float(delta.astype(np.float64)@d.astype(np.float64))
        arms[a]={"final_query_l2":size,"effect_donor_projection":alignment/den if den else None,
                 "effect_donor_cosine":alignment/(size*den**.5) if size and den else None}
    pref=data["recipient.pass1_boundary"][0,:-1,:]
    return {"identity_query_exact":bool(np.array_equal(r,data["arm.identity.final_query"])),
            "effect_clean_final_l2":norm(d),"sham_clean_final_l2":norm(s),
            "effect_clean_prefix_l2":norm(data["effect_donor.pass1_boundary"][0,:-1,:]-pref),
            "sham_clean_prefix_l2":norm(data["sham_donor.pass1_boundary"][0,:-1,:]-pref),
            "arms":arms}


def report(manifest,loaded):
    per=[]
    for c,rec,data in loaded:
        result=measure(data)
        result.update(case_id=c["case_id"],root_id=c["root_id"])
        per.append(result)
    clusters=defaultdict(list)
    planned=defaultdict(int)
    for c in manifest["cases"]:
        planned[c["root_id"]]+=1
    for row in per:
        clusters[row["root_id"]].append(row)
    means={}
    for root,rows in clusters.items():
        if len(rows)==planned[root] and all(x["identity_query_exact"] and
                all(x["arms"][a]["effect_donor_projection"] is not None for a in ARMS) for x in rows):
            means[str(root)]={a:fmean(x["arms"][a]["effect_donor_projection"] for x in rows) for a in ARMS}
    return {"source":SOURCE,"verdict":"PILOT_ONLY","n_planned_cases":len(manifest["cases"]),
            "n_saved_complete_cases":len(per),"n_planned_roots":len(planned),
            "n_complete_identity_checked_roots":len(means),"root_projection_means":means,"per_case":per,
            "scope":"saved-vector geometry only; arm first-pass query and delivered prefix edits are not saved and cannot be independently reconstructed from these vectors; no factor mediation or statistical inference"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    a=p.parse_args()
    m,loaded=read_run(a.run)
    print(json.dumps(report(m,loaded),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
