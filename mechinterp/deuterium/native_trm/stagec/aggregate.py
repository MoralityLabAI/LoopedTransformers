"""DEUTERIUM-NATIVE-01 prespecified three-old-teacher development verdict.

Pure standard library. Requires three independent no-Torch audits of first
native heldout logits before computing the across-root Student t interval.
The three original teacher weights have been examined in prior LOOPSCOPE work.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

TEACHERS=(103,211,307)
T95_DF2=4.302652729911275


def aggregate(paths,out):
    results={}
    for seed,path in zip(TEACHERS,paths):
        r=json.loads(Path(path).read_text())
        if (r.get("schema")!="DEUTERIUM-NATIVE-01/first-stdlib-heldout-audit-v1"
            or r.get("teacher_seed")!=seed or r.get("torch_imported") is not False):
            raise ValueError("original independent audit teacher lineage changed")
        results[str(seed)]=r
    complete=all(r.get("status")=="PASS" and
                 r.get("comparison_completed") is True and
                 r.get("clean_task_gate_pass") is True
                 for r in results.values())
    report={
        "schema":"DEUTERIUM-NATIVE-01/first-three-teacher-development-verdict-v1",
        "status":"BLOCKED_AUDIT_OR_CLEAN_SOURCE",
        "comparison_completed":complete,
        "teacher_seeds":list(TEACHERS),
        "prior_teacher_roots_examined":True,
        "evidence_class":"DEVELOPMENT_ONLY_OLD_TEACHER_ROOTS",
        "registered_primary":"random_minus_vpd_goal_nll",
        "score_units":"256 independent source strings per teacher, same cohort reused across teachers",
        "across_root_interval":"Student t df=2; descriptive over previously examined trained roots",
        "per_teacher_audits":results,
    }
    if complete:
        values=[results[str(s)]["random_minus_vpd_goal_nll"] for s in TEACHERS]
        avg=math.fsum(values)/3
        sem=math.sqrt(math.fsum((v-avg)**2 for v in values)/2)/math.sqrt(3)
        ci=[avg-T95_DF2*sem,avg+T95_DF2*sem]
        collateral_ok=all(results[str(s)]["vpd_offgoal_accuracy_minus_clean"]>=-.02
                          for s in TEACHERS)
        positive=all(v>0 for v in values) and ci[0]>0 and collateral_ok
        report.update({
            "status":"DEVELOPMENT_POSITIVE" if positive else
                     "DEVELOPMENT_NO_REGISTERED_ADVANTAGE",
            "primary_by_old_teacher":dict(zip(map(str,TEACHERS),values)),
            "primary_mean":avg,
            "primary_ci95":ci,
            "all_seed_signs_positive":all(v>0 for v in values),
            "collateral_gate_pass":collateral_ok,
            "registered_positive_rule_pass":positive,
        })
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("DEUTERIUM_NATIVE_FIRST_HELDOUT_REGISTERED",report["status"],flush=True)
    if not complete:
        raise RuntimeError("native test audit/root prerequisite blocked scientific verdict")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audit-103",type=Path,required=True)
    p.add_argument("--audit-211",type=Path,required=True)
    p.add_argument("--audit-307",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    aggregate([a.audit_103,a.audit_211,a.audit_307],a.out)


if __name__=="__main__":
    main()
