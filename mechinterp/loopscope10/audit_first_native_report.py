"""Documentation-only audit of the FIRST LGA-07B original 31060 development.

Run against archived source_lock.json, full.json, audit.json from Actions
44277352248. Does not load Torch, rerun models, or generate source examples.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path

ORIGINAL={
    "lock":"b6783a4b49443f11ed176affd2a31ed12816868517b1bd15e1df976b13e71e35",
    "full":"1b2c53f5175dd95fef8aed06a38aecc36950ebbf946c4a0819706f5b51171c64",
    "audit":"993a573efc86a68cc74330e102e91fca432a01b468874f0d18ac79fefc833148",
}
NATIVE_TARGET={
    "103":{"effect":.00390625,"ci":[0.,.009765625],
           "clean_competent":495,"natural_correct":336,
           "selected_rate":.009765625,"sham":.005859375,
           "k6_effect":0., "k6_selected":.123046875},
    "211":{"effect":0.,"ci":[0.,0.],
           "clean_competent":489,"natural_correct":320,
           "selected_rate":0.,"sham":0.,"k6_effect":.001953125,
           "k6_selected":.130859375},
    "307":{"effect":.005859375,"ci":[0.,.013671875],
           "clean_competent":501,"natural_correct":320,
           "selected_rate":.0078125,"sham":.001953125,
           "k6_effect":-.001953125,"k6_selected":.125},
}
DOSE_MEDIAN={"103":7.51,"211":7.39,"307":7.22}
PROJECT_FRAC_MEDIAN={"103":.329,"211":.345,"307":.337}
SECONDARY_CE_DELTA={"103":-.24782,"211":-.25163,"307":-.28129}
BOOT=2000


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(x,y,label,tol=1e-9):
    if not math.isfinite(float(x)) or abs(float(x)-float(y))>tol:
        raise AssertionError(f"original source-only report changed: {label} {x} {y}")


def verify(lock_path,full_path,audit_path):
    for label,path in [("lock",lock_path),("full",full_path),("audit",audit_path)]:
        if file_sha(path)!=ORIGINAL[label]:
            raise AssertionError("first scientific artifact SHA mismatch: "+label)
    lock=json.loads(Path(lock_path).read_bytes())
    full=json.loads(Path(full_path).read_bytes())
    audit=json.loads(Path(audit_path).read_bytes())
    if (lock["test_seed_reserved"]!=31060 or lock["new_test_generated"] is not False
        or full["test_source"]["seed"]!=31060 or
        full["source_lock_sha256"]!=ORIGINAL["lock"] or
        audit["audit_status"]!="PASS" or
        audit["verified_original_test_sha256"]!=ORIGINAL["full"] or
        full["registered_development_verdict"]!="DEVELOPMENT_NULL_OR_INELIGIBLE" or
        audit["registered_development_verdict"]!=full["registered_development_verdict"]):
        raise AssertionError("archived original source chronology or negative decision changed")
    if full["teacher_seeds"]!=[103,211,307] or full["n_independent_base_sources"]!=64:
        raise AssertionError("teacher/source roster changed")
    out={}
    for sid,expected in NATIVE_TARGET.items():
        root=full["teachers"][sid]
        k2=root["2"];k6=root["6"];s=k2["summary"];stress=k6["summary"]
        for field,key in [("effect","ITT_paired_effect"),
                          ("selected_rate","selected_ITT_rate"),("sham","sham_median_ITT_rate")]:
            close(s[key],expected[field],f"{sid}/{key}")
        for i,entry in enumerate(expected["ci"]):
            close(s["whole_base_bootstrap95"][i],entry,f"{sid}/bootstrap[{i}]")
        if (s["n_original_goals"]!=512 or s["n_admitted"]!=512 or
            s["n_jointly_competent"]!=expected["clean_competent"] or
            s["teacher_engineering_gate"] is not False):
            raise AssertionError("source admission, competence, or engineering gate changed")
        close(s["full_donor_token_correct_among_jointly_competent"],
              expected["natural_correct"]/expected["clean_competent"],
              f"{sid}/natural donor success")
        close(stress["ITT_paired_effect"],expected["k6_effect"],f"{sid}/k6 effect")
        close(stress["selected_ITT_rate"],expected["k6_selected"],f"{sid}/k6 selected")
        rows=k2["cases"]
        if len(rows)!=512 or any(x["fiber_dimension"]!=37 for x in rows):
            raise AssertionError("physical K dimension/source cases changed")
        fractions=[];ratios=[];ce_delta=[]
        for x in rows:
            candidate=x["arms"]["projected_donor"]["edit_vector_float32"]
            natural=x["arms"]["full_donor_token"]["edit_vector_float32"]
            dose=x["radius"]
            cu=sum(a*b for a,b in zip(candidate,natural))
            nv=math.sqrt(sum(a*a for a in natural))
            nc=math.sqrt(sum(a*a for a in candidate))
            fractions.append(cu/(nc*nv))
            ratios.append(nv/dose)
            sham=statistics.median(x["arms"][f"same_k_sham_{i}"]["donor_goal_CE"]
                                   for i in range(8))
            ce_delta.append(x["arms"]["projected_donor"]["donor_goal_CE"]-sham)
        close(statistics.median(fractions),PROJECT_FRAC_MEDIAN[sid],
              f"{sid}/projection fraction",tol=.0006)
        close(statistics.median(ratios),DOSE_MEDIAN[sid],f"{sid}/dose ratio",tol=.01)
        close(statistics.mean(ce_delta),SECONDARY_CE_DELTA[sid],
              f"{sid}/secondary CE difference",tol=.0001)
        out[sid]={"paired_effect":s["ITT_paired_effect"],
                  "n_competent":s["n_jointly_competent"],
                  "median_dose_ratio":statistics.median(ratios),
                  "median_projected_fraction":statistics.median(fractions),
                  "secondary_CE_delta":statistics.mean(ce_delta)}
    print("LGA07B_FIRST_ORIGINAL_NEGATIVE_REPORT_AUDIT_PASS",out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("lock","full","audit"):
        p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    verify(args.lock,args.full,args.audit)
