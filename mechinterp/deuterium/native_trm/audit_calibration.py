"""Independent no-Torch source and search arithmetic audit of Stage B receipts.

Rereads all source rows and scored candidate per-sequence losses; validates
source/target byte hashes, goal labels, full search budgets/tie breaking and
selected matrix artifacts. Model computations stay inside the separately
executed trained-source fit, never inside this independent auditor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

SCHEMA = "DEUTERIUM-NATIVE-01/first-calibration-and-selection-v1"
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
N = 256
COMPONENTS = 16
MAX_DOSE_ERROR = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(rows):
    flat = [int(v) for row in rows for v in row]
    return hashlib.sha256(struct.pack("<" + "q"*len(flat), *flat)).hexdigest()


def audit(root, out):
    root = Path(root)
    manifest = json.loads((root/"seed_result.json").read_text())
    searches = json.loads((root/"calibration_searches.json").read_text())
    sources = json.loads((root/"calibration_sources.json").read_text())
    exposed = json.loads((root/"calibration_exposure.json").read_text())
    errors = []
    def require(value, name):
        if not value:
            errors.append(name)

    require(manifest.get("schema")==SCHEMA and searches.get("schema")==SCHEMA,
            "schema")
    require(manifest.get("status")=="CALIBRATION_SELECTION_FROZEN",
            "calibration_completion")
    require(manifest.get("selected_before_heldout") is True and
            manifest.get("heldout_generated") is False and
            manifest.get("heldout_labels_read") is False and
            manifest.get("scoring_completed") is False, "reserved_heldout_closed")
    require(manifest.get("new_calibration_exposed") is True and
            exposed.get("status")=="GENERATION_STARTED", "first_calibration_exposure")
    require(manifest.get("heldout_seed")==731057 and
            sources.get("seed")==731055 and
            exposed.get("seed")==731055, "source_roster")
    require(exposed.get("candidate_bank_sha256")==manifest.get("candidate_bank_sha256") and
            sha(root/"candidate_bank.pt")==manifest.get("candidate_bank_sha256"),
            "frozen_candidate_bank_bytes")
    require(sha(root/"selected_edits.pt")==manifest.get("selected_edits_sha256") and
            searches.get("selected_edits_sha256")==manifest.get("selected_edits_sha256"),
            "frozen_selected_edit_bytes")
    require(manifest.get("teacher_bundle_sha256") ==
            "d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
            "original_teacher_hash")
    seed = manifest.get("teacher_seed")
    require(seed in (103,211,307) and searches.get("teacher_seed")==seed,
            "teacher_seed")
    x,y=sources.get("input",[]),sources.get("target",[])
    require(len(x)==N and len(y)==N, "cohort_size")
    if len(x)==N and len(y)==N:
        require(all(len(row)==81 and all(type(v) is int and 1<=v<=9 for v in row)
                    for row in x), "legal_source_digits")
        require(all(len(row)==81 for row in y), "legal_target_shape")
        require(all(y[i][j]==x[i][(j-1)%81]
                    for i in range(N) for j in range(81)), "cyclic_copy_labels")
        require(len({tuple(row) for row in x})==N, "duplicate_new_source")
        require(tensor_sha(x)==sources.get("input_sha256")==
                manifest.get("calibration",{}).get("input_sha256")==
                searches.get("calibration_source_sha256"), "input_sha256")
        require(tensor_sha(y)==sources.get("target_sha256")==
                manifest.get("calibration",{}).get("target_sha256"), "target_sha256")
    require(len(sources.get("old_source_roster",{}))>=18 and
            all(k not in sources.get("old_source_roster",{})
                for k in ("31059","31064","31065")), "old_source_exclusions")
    require(len(manifest.get("vpd",{}).get("checkpoint_sha256",""))==64 and
            manifest.get("vpd",{}).get("upstream_commit") ==
            "2df0092d5be4d8e4748b3c66ef8325dedef17549",
            "pinned_stock_optimize")
    replay=manifest.get("vpd",{}).get("full_residual_replay_max_logit_error")
    require(isinstance(replay,(int,float)) and math.isfinite(replay)
            and 0<=replay<=1e-4, "residual_plus_components_native_replay")
    n_checks=0
    winners={}
    for arm in ("vpd","random"):
        record=searches.get("searches",{}).get(arm,{})
        rows=record.get("all_candidates",[])
        require(record.get("budget")==32 and len(rows)==32 and
                record.get("sign_order")==[1,-1], arm+":equal_budget")
        means=[]
        for idx,row in enumerate(rows):
            losses=row.get("per_sequence_goal_nll",[])
            require(row.get("candidate_index")==idx//2 and
                    row.get("sign")== (1 if idx%2==0 else -1),
                    arm+f":ordering_{idx}")
            require(len(losses)==N and all(isinstance(t,(int,float)) and
                    math.isfinite(t) and t>=0 for t in losses),
                    arm+f":finite_nll_{idx}")
            if len(losses)==N:
                m=math.fsum(losses)/N
                means.append(m)
                reported=row.get("mean_goal_nll")
                require(isinstance(reported,(int,float)) and math.isfinite(reported) and
                        abs(m-reported) <=1e-6,arm+f":loss_mean_{idx}")
            applied=row.get("applied",{})
            norm=applied.get("actual_applied_frobenius")
            rel=applied.get("actual_relative_dose_error")
            dose=manifest.get("dose")
            require(isinstance(dose,(int,float)) and dose>0 and
                    isinstance(norm,(int,float)) and math.isfinite(norm) and
                    abs(norm-dose)/dose<=MAX_DOSE_ERROR and
                    isinstance(rel,(int,float)) and rel<=MAX_DOSE_ERROR,
                    arm+f":actual_dose_{idx}")
            require(len(row.get("candidate_matrix_sha256",""))==64,
                    arm+f":matrix_sha_{idx}")
            require(applied.get("max_native_manual_replay_error",float("inf"))<=1e-4 and
                    applied.get("weight_restore_max_logit_error",float("inf"))<=1e-4,
                    arm+f":native_closure_{idx}")
            n_checks+=1
        if len(means)==32:
            best=min(range(32), key=lambda ix:(means[ix],ix))
            require(best==record.get("best_index") and
                    rows[best]["candidate_index"]==record.get("best_candidate_index") and
                    rows[best]["sign"]==record.get("best_sign"),
                    arm+":winner_consistency")
            require(abs(means[best]-record.get("best_calibration_goal_nll",float("inf")))<1e-6,
                    arm+":winning_nll")
            brief=manifest.get("searches",{}).get(arm,{})
            require(brief.get("best_index")==best and brief.get("budget")==32,
                    arm+":manifest_winner")
            winners[arm]={"index":best,"mean_goal_nll":means[best]}
    report={
        "schema":"DEUTERIUM-NATIVE-01/independent-calibration-receipt-audit-v1",
        "status":"PASS" if not errors else "FAIL",
        "errors":errors,
        "torch_imported":"torch" in sys.modules,
        "teacher_seed":seed,
        "source_rows_rechecked": len(x),
        "signed_candidate_scores_rechecked":n_checks,
        "winners": winners,
        "heldout_generated":False,
        "test_seed_731057_never_opened":True,
        "study_verdict":"CALIBRATION_ONLY_NOT_SCORED",
    }
    if report["torch_imported"]:
        report["status"]="FAIL"
        report["errors"].append("torch_imported")
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    if report["status"]!="PASS":
        raise ValueError("independent Stage B audit failed: "+", ".join(errors))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=audit(a.root,a.out)
    print("DEUTERIUM_NATIVE_STAGEB_STDLIB_AUDIT",report["status"],flush=True)


if __name__=="__main__":
    main()
