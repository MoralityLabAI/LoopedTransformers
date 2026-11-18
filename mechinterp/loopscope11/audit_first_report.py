"""Post-result documentation QA against immutable FIRST LGA-08 artifacts.

Python stdlib only, no model inference or 31062 generation. This report
validation is separate from the prospectively registered independent auditor,
which already passed on the exact original Actions 44308907706 run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ORIGINAL={
    "lock":"b195c62aff99aa9a98f537910b5c43e581d792b1eb9277d8afea3cdf24ce28ea",
    "full":"4470544993ca4cad3aacbf67d0871a81c57bf3fe7366ba82ea585b675378d418",
    "edits":"6c9b7b92a1ad206b4de235d2ac522049f672a6b5389bf2b833e1c10f76669cfa",
    "source":"19b6d146bfee2868ef8e4ef8aa934475aeed84377ba40988fa2a8fee5e81bd89",
    "audit":"4e49d91b96482401fd09f2c5b173bca9d5ccd8951f357699e956fc3aef4de3ec",
}
EXPECTED={
    "103":{"selected":180,"strict":.2197265625,"strict_ci":(.1875,.251953125),
           "random":.349609375,"random_ci":(.314453125,.384765625),
           "joint":499,"natural_pair":501,"full":506,"head_only":185,
           "grad_linear_match_max":4.0573589465964233e-7,
           "ce_delta":-1.6238869206719144,"mean_energy":15.113008924643509},
    "211":{"selected":229,"strict":.2705078125,"strict_ci":(.234375,.3056640625),
           "random":.447265625,"random_ci":(.41015625,.484375),
           "joint":498,"natural_pair":494,"full":502,"head_only":242,
           "grad_linear_match_max":5.411389452092408e-7,
           "ce_delta":-1.645486510522082,"mean_energy":12.41644101444399},
    "307":{"selected":211,"strict":.26953125,"strict_ci":(.232421875,.3046875),
           "random":.412109375,"random_ci":(.373046875,.45703125),
           "joint":494,"natural_pair":489,"full":500,"head_only":225,
           "grad_linear_match_max":4.346842485958291e-7,
           "ce_delta":-1.8711354737249206,"mean_energy":15.984172682161443},
}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(actual,registered,name,eps=1e-8):
    if abs(float(actual)-float(registered))>eps:
        raise ValueError("first original scientific paper statistic changed: "+name)


def audit(lock,full,edits,source,original_audit,out):
    paths={"lock":lock,"full":full,"edits":edits,"source":source,"audit":original_audit}
    for k,path in paths.items():
        if sha_file(path)!=ORIGINAL[k]:
            raise ValueError("FIRST original data/archive SHA drift: "+k)
    old=json.loads(Path(lock).read_bytes())
    paper=json.loads(Path(full).read_bytes())
    raw_audit=json.loads(Path(original_audit).read_bytes())
    if (old["reserved_seed"]!=31062 or old["new_test_generated"] is not False
        or paper["source"]["seed"]!=31062
        or paper["source"]["x_sha256"]!="55aff9ac35186800607d0c23bb4e35935583dc01286786a116b5a107d3f9d6ec"
        or paper["first_pretest_lock_sha256"]!=ORIGINAL["lock"]
        or paper["first_registered_development_verdict"]!="DEVELOPMENT_STRONG_INCREMENTAL_PASS"
        or raw_audit["audit_status"]!="PASS"
        or raw_audit["registered_first_development_verdict"]!=paper["first_registered_development_verdict"]
        or raw_audit["original_full_receipt_sha256"]!=ORIGINAL["full"]
        or raw_audit["physical_pair_edits"]!=35328
        or raw_audit["source_vectors"]!=1536
        or raw_audit["n_passing_old_teachers"]!=3):
        raise ValueError("original source chronology, verdict or first no-Torch audit diverged")
    if len(paper["arms"])!=23:
        raise ValueError("frozen 23-arm protocol changed")
    evidence={}
    for sid,expected in EXPECTED.items():
        data=paper["teachers"][sid]
        cases=data["cases"];s=data["summary"];by=s["contrasts"]
        if (len(cases)!=512 or s["n_admitted"]!=512 or
            s["n_jointly_competent"]!=expected["joint"] or
            s["arm_correct_counts"]["pair_energy_tangent"]!=expected["selected"] or
            s["arm_correct_counts"]["pair_natural"]!=expected["natural_pair"] or
            s["arm_correct_counts"]["full_donor_state"]!=expected["full"] or
            s["arm_correct_counts"]["pair_head_only"]!=expected["head_only"] or
            not s["registered_strong_teacher_gate"]):
            raise ValueError("teacher admission, positives or registered gate changed")
        for key,label in (("gradient","strict"),("random","random")):
            close(by[key]["paired_ITT_effect"],expected[label],sid+"/"+label)
            for x,y in zip(by[key]["whole_base_bootstrap95"],expected[label+"_ci"]):
                close(x,y,sid+"/"+label+"_ci")
        ce_diff=statistics.mean(
            c["arms"]["pair_energy_tangent"]["donor_goal_CE"]-
            statistics.median(c["arms"][f"pair_gradient_match_{i}"]["donor_goal_CE"]
                              for i in range(8)) for c in cases)
        e_mean=statistics.mean(c["arms"]["pair_energy_tangent"]["finite_shared_E_change"]
                               for c in cases)
        maxgrad=max(abs(c["arms"]["pair_energy_tangent"]["signed_future_margin_derivative"]-
                        c["arms"][f"pair_gradient_match_{i}"]["signed_future_margin_derivative"])
                    for c in cases for i in range(8))
        close(ce_diff,expected["ce_delta"],sid+"/descriptive CE",eps=1e-7)
        close(e_mean,expected["mean_energy"],sid+"/mean finite E",eps=1e-7)
        close(maxgrad,expected["grad_linear_match_max"],sid+"/max future first-order",eps=1e-9)
        if any(c["fiber_dimension"]!=75 or not c["energy_row_active"] or
               not c["admitted"] or
               c["arms"]["pair_energy_tangent"]["finite_shared_E_change"]<=0
               for c in cases):
            raise ValueError("pair 75D source rank or observed finite E sign differs")
        evidence[sid]={
            "selected_target_correct":expected["selected"],
            "strict_gradient_matched_effect":expected["strict"],
            "strict_95pct_base_interval":expected["strict_ci"],
            "random_K_effect":expected["random"],
            "natural_pair_target_correct":expected["natural_pair"],
            "mean_selected_minus_median_matched_goal_CE":ce_diff,
            "mean_finite_external_E_change":e_mean,
            "max_signed_derivative_difference":maxgrad,
        }
    result={
        "schema":"LGA08/post-result-first-science-paper-qa-v1",
        "qa_status":"PASS","evidence_class":"POST_RESULT_DOCUMENTATION_ONLY",
        "original_native_receipt_sha256":ORIGINAL["full"],
        "original_stdlib_audit_sha256":ORIGINAL["audit"],
        "n_original_physical_pair_edits":35328,
        "n_original_sources":1536,"n_original_teacher_gates":3,
        "first_preregistered_verdict":"DEVELOPMENT_STRONG_INCREMENTAL_PASS",
        "teachers":evidence,
    }
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("LGA08_FIRST_NATIVE_PAPER_QA_PASS",result["first_preregistered_verdict"],flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("lock","full","edits","source","original-audit","out"):
        p.add_argument("--"+n,required=True,type=Path)
    a=p.parse_args()
    audit(a.lock,a.full,a.edits,a.source,a.original_audit,a.out)
