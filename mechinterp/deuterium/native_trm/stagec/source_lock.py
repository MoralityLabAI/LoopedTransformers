"""DEUTERIUM-NATIVE-01 Stage C first heldout source lock.

Standard-library-only. Verifies all first Stage B model, source, selection
and independent audit artifacts BEFORE any release of heldout seed 731057.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
SCHEMA="DEUTERIUM-NATIVE-01/first-heldout-source-lock-v1"
STAGEB_RUN=44463652662
TEACHERS=(103,211,307)
MODEL_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
CODE_FILES=(
    "mechinterp/deuterium/native_trm/PROTOCOL.md",
    "mechinterp/deuterium/native_trm/stagec/source_lock.py",
    "mechinterp/deuterium/native_trm/stagec/score_native.py",
    "mechinterp/deuterium/native_trm/stagec/audit_native.py",
    "mechinterp/deuterium/native_trm/stagec/aggregate.py",
    "mechinterp/deuterium/native_trm/stagec/tests/test_stagec.py",
    ".github/workflows/deuterium-native-stagec.yml",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lineage(paths):
    return {p:sha(REPO / p) for p in paths}


def verify_stageb(seed, stageb_root, audit_root):
    model=Path(stageb_root) / f"seed_{seed}"
    apath=Path(audit_root) / f"seed_{seed}.json"
    manifest=json.loads((model/"seed_result.json").read_text())
    audit=json.loads(apath.read_text())
    searches=json.loads((model/"calibration_searches.json").read_text())
    source=json.loads((model/"calibration_sources.json").read_text())
    exposed=json.loads((model/"calibration_exposure.json").read_text())
    if (manifest.get("status")!="CALIBRATION_SELECTION_FROZEN"
        or manifest.get("teacher_seed")!=seed
        or manifest.get("heldout_generated") is not False
        or manifest.get("heldout_labels_read") is not False
        or manifest.get("selected_before_heldout") is not True
        or manifest.get("new_calibration_exposed") is not True):
        raise RuntimeError(f"teacher {seed} has incomplete first calibration")
    if (audit.get("status")!="PASS" or audit.get("torch_imported") is not False
        or audit.get("teacher_seed")!=seed
        or audit.get("signed_candidate_scores_rechecked")!=64
        or audit.get("source_rows_rechecked")!=256):
        raise RuntimeError(f"teacher {seed} has no independent first Stage B audit")
    if (exposed.get("status")!="GENERATION_STARTED"
        or exposed.get("seed")!=731055 or source.get("seed")!=731055
        or manifest.get("heldout_seed")!=731057
        or searches.get("teacher_seed")!=seed):
        raise RuntimeError(f"teacher {seed} original source cohort provenance drift")
    if (manifest.get("teacher_bundle_sha256")!=MODEL_SHA
        or manifest.get("teacher_meta_sha256")!=META_SHA
        or manifest.get("vpd",{}).get("upstream_commit")!="2df0092d5be4d8e4748b3c66ef8325dedef17549"):
        raise RuntimeError(f"teacher {seed} original optimized target provenance drift")
    if (sha(model/"selected_edits.pt")!=manifest.get("selected_edits_sha256")
        or sha(model/"candidate_bank.pt")!=manifest.get("candidate_bank_sha256")
        or searches.get("selected_edits_sha256")!=manifest.get("selected_edits_sha256")):
        raise RuntimeError(f"teacher {seed} selected edit bytes changed")
    if (source.get("input_sha256")!=manifest.get("calibration",{}).get("input_sha256")
        or source.get("input_sha256")!=searches.get("calibration_source_sha256")):
        raise RuntimeError(f"teacher {seed} calibration cohort SHA drift")
    if audit.get("heldout_generated") is not False:
        raise RuntimeError("first test already exposed")
    return {
        "teacher_seed":seed,
        "manifest_sha256":sha(model/"seed_result.json"),
        "calibration_search_sha256":sha(model/"calibration_searches.json"),
        "calibration_source_sha256":sha(model/"calibration_sources.json"),
        "calibration_source_tensor_sha256":source["input_sha256"],
        "selected_edits_sha256":sha(model/"selected_edits.pt"),
        "candidate_bank_sha256":sha(model/"candidate_bank.pt"),
        "independent_audit_sha256":sha(apath),
        "calibration_clean_full_accuracy":manifest["calibration_clean"]["mean_full_accuracy"],
        "stageb_original_target_sha256":manifest["vpd"]["original_target_tensor_sha256"],
    }


def build(original_bundle,original_meta,stageb_root,audit_root,stageb_lock,out):
    if sha(original_bundle)!=MODEL_SHA or sha(original_meta)!=META_SHA:
        raise RuntimeError("original LGA-02 teacher artifact bytes changed")
    old=json.loads(Path(stageb_lock).read_text())
    if (old.get("status")!="SOURCE_ONLY_LOCK_PASS"
        or old.get("calibration_generated") is not False
        or old.get("heldout_generated") is not False
        or old.get("upstream_param_decomp_commit")!="2df0092d5be4d8e4748b3c66ef8325dedef17549"
        or old.get("teacher_seeds")!=list(TEACHERS)):
        raise RuntimeError("first Stage B prospective source lock missing")
    if any(sha(REPO/p)!=h for p,h in old.get("source_files_sha256",{}).items()):
        raise RuntimeError("stage B scientific source no longer matches its original lock")
    per={}
    for seed in TEACHERS:
        per[str(seed)]=verify_stageb(seed,stageb_root,audit_root)
    if len({x["calibration_source_tensor_sha256"] for x in per.values()})!=1:
        raise RuntimeError("stage B calibration teacher cohort mismatches")
    record={
        "schema":SCHEMA,
        "status":"READY_FOR_FIRST_HELDOUT_RELEASE",
        "original_bundle_sha256":MODEL_SHA,
        "original_meta_sha256":META_SHA,
        "first_stageb_workflow_run":STAGEB_RUN,
        "original_stageb_source_lock_sha256":sha(stageb_lock),
        "stagec_code_sha256":lineage(CODE_FILES),
        "teacher_seed_roster":list(TEACHERS),
        "first_calibration_seed":731055,
        "first_scored_seed_reserved":731057,
        "first_scored_source_count":256,
        "goal_token_indices":[0,10,20,30,40,50,60,70],
        "per_teacher":per,
        "new_test_generated":False,
        "new_test_labels_read":False,
        "first_test_scored":False,
    }
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    return record


def verify_lock(lock,original_bundle,original_meta,stageb_root,audit_root,stageb_lock):
    data=json.loads(Path(lock).read_text())
    if (data.get("schema")!=SCHEMA or
        data.get("status")!="READY_FOR_FIRST_HELDOUT_RELEASE" or
        data.get("new_test_generated") is not False or
        data.get("new_test_labels_read") is not False or
        data.get("first_test_scored") is not False or
        data.get("first_scored_seed_reserved")!=731057 or
        data.get("stagec_code_sha256")!=lineage(CODE_FILES) or
        data.get("original_stageb_source_lock_sha256")!=sha(stageb_lock)):
        raise RuntimeError("first-test code/source lock changed after release")
    if sha(original_bundle)!=MODEL_SHA or sha(original_meta)!=META_SHA:
        raise RuntimeError("original teacher weights changed after first-test lock")
    for seed in TEACHERS:
        if data.get("per_teacher",{}).get(str(seed))!=verify_stageb(
                seed,stageb_root,audit_root):
            raise RuntimeError("calibration receipts/selected directions changed")
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher-bundle",type=Path,required=True)
    p.add_argument("--teacher-meta",type=Path,required=True)
    p.add_argument("--stageb-root",type=Path,required=True)
    p.add_argument("--audit-root",type=Path,required=True)
    p.add_argument("--stageb-lock",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=build(a.teacher_bundle,a.teacher_meta,a.stageb_root,a.audit_root,
            a.stageb_lock,a.out)
    print("DEUTERIUM_NATIVE_FIRST_TEST_SOURCE_LOCK",r["status"],flush=True)


if __name__=="__main__":
    main()
