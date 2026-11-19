"""Post-result, read-only recovery of DEUTERIUM-NATIVE-01 first scoring receipts.

The original 731057 experiment is closed and consumed. Read ONLY byte-pinned
GitHub Actions artifacts from run 44465490737 (and source-frozen Stage B
run 44463652662). No model is loaded or trained; no first-test generator
is imported. The original buggy auditor remains unchanged at its locked SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from mechinterp.deuterium.native_trm.stagec.recovery.audit_recovered import audit
from mechinterp.deuterium.native_trm.stagec.aggregate import aggregate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TEACHERS = (103, 211, 307)
SCORED_RUN_ID = 44465490737
SCORED_MERGE_SHA = "67b248fdbcc2ac1f73ed58fce3f171db3e0db8d0"
STAGEB_RUN_ID = 44463652662
FIRST_LOCK_SHA = "4f39f9e9b33ac538aab8eeb2b977a243b8b3b6d83e7dd800677e3a1678daffd1"
FIRST_SCIENCE_FILES = (
    "mechinterp/deuterium/native_trm/PROTOCOL.md",
    "mechinterp/deuterium/native_trm/stagec/score_native.py",
    "mechinterp/deuterium/native_trm/stagec/audit_native.py",
    "mechinterp/deuterium/native_trm/stagec/aggregate.py",
    "mechinterp/deuterium/native_trm/stagec/source_lock.py",
    "mechinterp/deuterium/native_trm/stagec/tests/test_stagec.py",
    ".github/workflows/deuterium-native-stagec.yml",
)
FIRST_FILES_SHA = {
    103: {
        "seed_result.json": "c2e3651b9998b716071164ef532ccb900633c01d276bdb914251186b79c7080d",
        "all_arm_logits_f32.gz": "758faf051016993b68d616103b799081955f94343783cb670e5231a8871fe0f1",
        "heldout_sources.json": "f7adf262dfec31ce5b6dd91b5f59a7c650bc1d860fc5f73b7d073f0f72f758c3",
        "heldout_exposure.json": "873c889213355c000fb03213ea649b82dda9d2883c56c18e9c04741e01cce3c2",
        "scored_metrics.json": "010ff5c5ed1dfcd2ea9a4efedc98ae8e34b69fda0ce37959fc858da7ec9ebbf3",
    },
    211: {
        "seed_result.json": "9fd88599a3edc63b4f511365fdc643ad98b9fcbb1684fae471cb06652505701a",
        "all_arm_logits_f32.gz": "09838b3d1c4500c716c3420980f7530dacce761d04727d7c747c0705c0dcab1b",
        "heldout_sources.json": "f7adf262dfec31ce5b6dd91b5f59a7c650bc1d860fc5f73b7d073f0f72f758c3",
        "heldout_exposure.json": "6c403277ffad0fc00cc6bc6ac8815eaced085e1776a2ab410886a8f7cf9dad28",
        "scored_metrics.json": "116aa5f15f41c8841cfb91f42e5b17ef078fd51e745141bdc203fc737b5e34f1",
    },
    307: {
        "seed_result.json": "2da04ac7fe8ab26c1536b389633ce0aa8589660a2c9548912ff199ad9bce0cd0",
        "all_arm_logits_f32.gz": "4096f5149b839ca65a1e17e04a7b319aa8c885536dae48a98ee6b205ebfd20ba",
        "heldout_sources.json": "f7adf262dfec31ce5b6dd91b5f59a7c650bc1d860fc5f73b7d073f0f72f758c3",
        "heldout_exposure.json": "086542eb6a52a2428b84b5010e425f6400318fedc57c0818d696e49a955e400a",
        "scored_metrics.json": "1240a19349700ea570ef3584acee9187fca00135a397ec9e8e20ab796d994e9a",
    }
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n")


def replay(source_lock: Path, score_root: Path, stageb_root: Path, out: Path) -> dict:
    if "torch" in sys.modules:
        raise RuntimeError("recovery may not import torch")
    if sha(source_lock) != FIRST_LOCK_SHA:
        raise RuntimeError("first pre-test source lock bytes changed")
    lock = json.loads(source_lock.read_text())
    if lock.get("status") != "READY_FOR_FIRST_HELDOUT_RELEASE":
        raise RuntimeError("original source lock not verified")
    original = lock["stagec_code_sha256"]
    for name in FIRST_SCIENCE_FILES:
        if original.get(name) != sha(REPO / name):
            raise RuntimeError("original first-score implementation changed: "+name)
    if lock.get("first_stageb_workflow_run") != STAGEB_RUN_ID:
        raise RuntimeError("source-provenance Stage B Actions run changed")

    out.mkdir(parents=True, exist_ok=True)
    audited = []
    source_rosters = []
    for seed in TEACHERS:
        first = score_root / f"seed_{seed}"
        stageb = stageb_root / f"seed_{seed}"
        if not first.is_dir() or not stageb.is_dir():
            raise RuntimeError("one of the original first-science artifacts is missing")
        for fname, expected in FIRST_FILES_SHA[seed].items():
            if sha(first / fname) != expected:
                raise RuntimeError(f"first-science bytes drift: seed {seed} / {fname}")
        saved = json.loads((first/"seed_result.json").read_text())
        if (saved.get("teacher_seed") != seed or
            saved.get("github_sha") != SCORED_MERGE_SHA or
            saved.get("test_lock_sha256") != FIRST_LOCK_SHA):
            raise RuntimeError("first-science GitHub run/teacher/source lineage mismatched")
        outfile = out / f"seed_{seed}.json"
        audit(first, stageb, source_lock, outfile)
        record = json.loads(outfile.read_text())
        if record["status"] != "PASS" or record["teacher_seed"] != seed:
            raise RuntimeError(f"independent recovered score audit failed for {seed}")
        source_rosters.append(json.loads((first/"heldout_sources.json").read_text())["input_sha256"])
        audited.append(outfile)

    if len(set(source_rosters)) != 1:
        raise RuntimeError("first source cohort changed across trained teachers")
    summary_path = out/"registered_verdict.json"
    summary = aggregate(audited, summary_path)
    manifest = {
        "schema": "DEUTERIUM-NATIVE-01/first-postresult-readonly-audit-recovery-v1",
        "status": "REPLAY_PASS",
        "first_science_actions_run_id": SCORED_RUN_ID,
        "first_science_commit": SCORED_MERGE_SHA,
        "first_source_lock_sha256": FIRST_LOCK_SHA,
        "original_buggy_auditor_kept_unchanged": True,
        "first_audit_failure": "NameError: undefined row at original audit_native.py line 133",
        "patch_scope": "A separate post-result audit copy binds the source target-row length to len(y[i]).",
        "heldout_seed": 731057,
        "test_cohort_exposed": True,
        "first_science_new_inference_performed": False,
        "new_test_source_generated": False,
        "independent_auditor_torch_imported": False,
        "per_teacher_audit_sha256": {str(seed): sha(p) for seed,p in zip(TEACHERS,audited)},
        "original_first_science_artifacts_sha256": {
            str(seed): FIRST_FILES_SHA[seed] for seed in TEACHERS
        },
        "three_teacher_verdict_sha256": sha(summary_path),
        "registered_status": summary["status"],
        "primary_effect_mean_nats": summary["primary_mean"],
        "primary_ci95_nats": summary["primary_ci95"],
        "registered_positive_rule_pass": summary["registered_positive_rule_pass"],
    }
    write(out/"recovery_manifest.json",manifest)
    if "torch" in sys.modules:
        raise RuntimeError("torch loaded during supposed independent audit")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first-lock",type=Path,required=True)
    p.add_argument("--score-root",type=Path,required=True)
    p.add_argument("--stageb-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=replay(a.first_lock,a.score_root,a.stageb_root,a.out)
    print("DEUTERIUM_NATIVE_ORIGINAL_FIRST_SCORE_READONLY_AUDIT",result["status"],flush=True)
    print("DEUTERIUM_NATIVE_ORIGINAL_VERDICT",result["registered_status"],flush=True)


if __name__=="__main__":
    main()
