"""Hash-locked Stage B boundary: no new calibration source generated here.

The lock authenticates complete Stage B science source and workflow, previous
hosted original-model preflight receipts, historical trained teacher artifacts,
original model implementation and the pinned param-decomp commit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from mechinterp.deuterium.native_trm.preflight import (
    MODEL_SHA, META_SHA, SOURCE_BLOBS, check_source_bytes, sha256
)
from mechinterp.loopscope05.fit_and_freeze import load_frozen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHEMA = "DEUTERIUM-NATIVE-01/first-calibration-source-lock-v1"
UPSTREAM_SHA = "2df0092d5be4d8e4748b3c66ef8325dedef17549"
STAGE_A_RUN = 44459629978
PROTOCOL_REGISTRATION = "d808db9122a60131c395804d7780bb3adec80fa7"
ROSTER = (103, 211, 307)
SOURCE_FILES = (
    "mechinterp/deuterium/native_trm/PROTOCOL.md",
    "mechinterp/deuterium/native_trm/README.md",
    "mechinterp/deuterium/native_trm/stageb_lock.py",
    "mechinterp/deuterium/native_trm/fit_and_select.py",
    "mechinterp/deuterium/native_trm/audit_calibration.py",
    "mechinterp/deuterium/native_trm/tests/test_stageb.py",
    ".github/workflows/deuterium-native-stageb.yml",
)


def source_contract():
    return {p: sha256(REPO / p) for p in SOURCE_FILES}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def build_lock(original_bundle, original_meta, previous, previous_audit,
               upstream, output):
    if sha256(original_bundle) != MODEL_SHA or sha256(original_meta) != META_SHA:
        raise ValueError("original LGA-02 checkpoint source SHA mismatch")
    if check_source_bytes() != SOURCE_BLOBS:
        raise ValueError("original four-pass source implementation changed")
    if git(upstream, "rev-parse", "HEAD") != UPSTREAM_SHA:
        raise ValueError("upstream param_decomp optimize code commit drift")
    p = json.loads(Path(previous).read_text())
    a = json.loads(Path(previous_audit).read_text())
    if (p.get("schema") != "DEUTERIUM-NATIVE-01/native-source-only-preflight-v1"
        or p.get("status") != "PREFLIGHT_PASS"
        or p.get("new_heldout_generated") is not False
        or p.get("new_calibration_generated") is not False
        or a.get("audit_status") != "PASS" or a.get("torch_imported") is not False):
        raise ValueError("first hosted canonical original-model source preflight not verified")
    meta, _bundle = load_frozen(original_bundle, original_meta)
    if meta.get("mode") != "full" or meta.get("teacher_init_seeds") != list(ROSTER):
        raise ValueError("original trained teacher root roster changed")
    if git(REPO, "cat-file", "-e", PROTOCOL_REGISTRATION + ":mechinterp/deuterium/native_trm/PROTOCOL.md") != "":
        raise RuntimeError("prospective registration is not immutable in Git")
    record = {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_LOCK_PASS",
        "protocol_registration_commit": PROTOCOL_REGISTRATION,
        "source_files_sha256": source_contract(),
        "original_teacher_bundle_sha256": MODEL_SHA,
        "original_teacher_meta_sha256": META_SHA,
        "original_source_blobs": SOURCE_BLOBS,
        "upstream_param_decomp_commit": UPSTREAM_SHA,
        "hosted_stage_a_run_id": STAGE_A_RUN,
        "original_stage_a_preflight_receipt_sha256": sha256(previous),
        "original_stage_a_independent_audit_sha256": sha256(previous_audit),
        "teacher_seeds": list(ROSTER),
        "training_seed": 31041,
        "calibration_seed_reserved": 731055,
        "heldout_seed_reserved": 731057,
        "calibration_generated": False,
        "heldout_generated": False,
        "heldout_labels_read": False,
        "scoring_completed": False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


def verify_lock(lock, original_bundle, original_meta, upstream):
    r = json.loads(Path(lock).read_text())
    checks = {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_LOCK_PASS",
        "protocol_registration_commit": PROTOCOL_REGISTRATION,
        "source_files_sha256": source_contract(),
        "original_teacher_bundle_sha256": MODEL_SHA,
        "original_teacher_meta_sha256": META_SHA,
        "original_source_blobs": SOURCE_BLOBS,
        "upstream_param_decomp_commit": UPSTREAM_SHA,
        "hosted_stage_a_run_id": STAGE_A_RUN,
        "teacher_seeds": list(ROSTER),
        "training_seed": 31041,
        "calibration_seed_reserved": 731055,
        "heldout_seed_reserved": 731057,
        "calibration_generated": False,
        "heldout_generated": False,
        "heldout_labels_read": False,
        "scoring_completed": False,
    }
    if any(r.get(k) != v for k, v in checks.items()):
        raise ValueError("Stage B protocol / source / release lock changed")
    if len(r.get("original_stage_a_preflight_receipt_sha256", "")) != 64:
        raise ValueError("missing hosted Stage A evidence")
    if len(r.get("original_stage_a_independent_audit_sha256", "")) != 64:
        raise ValueError("missing independent native parity audit receipt")
    if (sha256(original_bundle) != MODEL_SHA or sha256(original_meta) != META_SHA or
            git(upstream, "rev-parse", "HEAD") != UPSTREAM_SHA):
        raise ValueError("trained teacher or pinned param_decomp changed before source opening")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-bundle", type=Path, required=True)
    p.add_argument("--teacher-meta", type=Path, required=True)
    p.add_argument("--prior-preflight", type=Path, required=True)
    p.add_argument("--prior-audit", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build_lock(a.teacher_bundle, a.teacher_meta, a.prior_preflight,
                        a.prior_audit, a.upstream, a.out)
    print("DEUTERIUM_NATIVE_STAGEB_SOURCE_LOCK", result["status"], flush=True)


if __name__ == "__main__":
    main()
