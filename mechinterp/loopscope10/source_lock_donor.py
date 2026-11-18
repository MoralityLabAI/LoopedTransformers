"""LGA-07B first native SOURCE-ONLY lock. Never generates reserved 31060.

The original DONOR_PROTOCOL and FIRST_NATIVE_RELEASE files were committed
before this implementation. All original checkpoint tensors remain frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mechinterp.loopscope05.fit_and_freeze import digest, load_frozen
from mechinterp.loopscope08.fit_potential import load_potentials

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = "LGA07B/first-native-source-lock-v1"
TEACHERS = (103, 211, 307)
SEED = 31060
CHECKSUMS = {
    "teacher_bundle_sha256": "d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
    "teacher_meta_sha256": "7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
    "potential_bundle_sha256": "24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
    "potential_meta_sha256": "69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
PROTOCOL_FILES = ("DONOR_PROTOCOL.md", "LGA07B_FIRST_NATIVE_RELEASE.md",
                  "POWER_PREFLIGHT.md")
IMPLEMENTATION_FILES = ("source_lock_donor.py", "native_donor_eval.py",
                        "audit_donor_receipt.py", "donor_source.py",
                        "energy_fiber.py")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_contract():
    return {
        "protocol_sha256": {p: file_sha(HERE/p) for p in PROTOCOL_FILES},
        "implementation_sha256": {p: file_sha(HERE/p) for p in IMPLEMENTATION_FILES},
        "test_seed_reserved": SEED,
        "n_independent_base_sequences": 64,
        "goals": [0, 10, 20, 30, 40, 50, 60, 70],
        "teacher_seeds": list(TEACHERS),
        "depths": [2, 6],
        "primary_depth": 2,
        "new_test_generated": False,
        "test_labels_read": False,
    }


def build_source_lock(teacher_bundle, teacher_meta, potential_bundle, potential_meta, out):
    files = {
        "teacher_bundle_sha256": teacher_bundle,
        "teacher_meta_sha256": teacher_meta,
        "potential_bundle_sha256": potential_bundle,
        "potential_meta_sha256": potential_meta,
    }
    found = {k: file_sha(v) for k, v in files.items()}
    if found != CHECKSUMS:
        raise ValueError("frozen first-science teacher/potential artifact SHA differs")
    tmeta, _ = load_frozen(teacher_bundle, teacher_meta)
    if (tmeta["teacher_init_seeds"] != list(TEACHERS)
            or tmeta["mode"] != "full"):
        raise ValueError("teacher weight and SAE roster changed")
    for seed in TEACHERS:
        trained, _, mu, std, _ = load_potentials(potential_bundle,
                                                  potential_meta, seed)
        if (tuple(trained) != ("shared", "conditioned")
                or not bool(torch.isfinite(mu).all() and torch.isfinite(std).all())):
            raise ValueError("old learned potential tensor/normalizer invalid")
    record = {
        "schema": SCHEMA,
        "original_artifact_shas": found,
        **source_contract(),
        "verified_tensor_loaders": True,
        "source_generation_complete": False,
        "experiment_scored": False,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("LGA07B_PRETEST_SOURCE_LOCK", file_sha(out), flush=True)
    return record


def verify_lock(lock, teacher_bundle, teacher_meta, potential_bundle, potential_meta):
    data = json.loads(Path(lock).read_text())
    if data.get("schema") != SCHEMA or data.get("original_artifact_shas") != CHECKSUMS:
        raise ValueError("registered source-lock schema/byte SHA changed")
    if any(data.get(k) != v for k, v in source_contract().items()):
        raise ValueError("registered protocol, implementation or new test roster changed")
    if data.get("source_generation_complete") is not False or data.get("experiment_scored") is not False:
        raise ValueError("source lock constructed after data was inspected")
    if data.get("verified_tensor_loaders") is not True:
        raise ValueError("model tensor verification absent")
    found = {k: file_sha(v) for k, v in {
        "teacher_bundle_sha256": teacher_bundle,
        "teacher_meta_sha256": teacher_meta,
        "potential_bundle_sha256": potential_bundle,
        "potential_meta_sha256": potential_meta,
    }.items()}
    if found != CHECKSUMS:
        raise ValueError("native run did not reuse exact verified original source")
    return data


def main():
    p = argparse.ArgumentParser()
    for name in ("teacher-bundle", "teacher-meta", "potential-bundle",
                 "potential-meta", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    build_source_lock(a.teacher_bundle, a.teacher_meta,
                      a.potential_bundle, a.potential_meta, a.out)


if __name__ == "__main__":
    main()
