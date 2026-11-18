"""LGA-07C Stage A: canonical pretrained byte/tensor lock BEFORE test seed 31061.

No source generation and no new evaluation here. Original LGA07C protocol was
committed first. Every native/auditor/protocol/workflow byte is frozen before
the separate job may open the reserved fresh donor roster.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

from mechinterp.loopscope05.fit_and_freeze import load_frozen
from mechinterp.loopscope08.fit_potential import load_potentials

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SCHEMA="LGA07C/pretest-frozen-source-lock-v1"
TEACHERS=(103,211,307)
ORIGINAL_SHAS={
    "teacher_bundle_sha256":"d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
    "teacher_meta_sha256":"7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
    "potential_bundle_sha256":"24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
    "potential_meta_sha256":"69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
PROTOCOLS=("LGA07C_GLOBAL_FIBER_PROTOCOL.md",)
CODES=("global_energy_fiber.py","global_donor_eval.py",
       "audit_global_receipt.py","source_lock_global.py",
       "donor_source.py","energy_fiber.py")
WORKFLOW=".github/workflows/loopscope10c-first-global-donor.yml"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_contract():
    codes={p:sha_file(HERE/p) for p in CODES}
    codes[WORKFLOW]=sha_file(REPO/WORKFLOW)
    return {
        "protocol_shas":{p:sha_file(HERE/p) for p in PROTOCOLS},
        "code_shas":codes,
        "test_seed_reserved":31061,
        "n_base":64,
        "goals":[0,10,20,30,40,50,60,70],
        "teacher_seeds":list(TEACHERS),
        "new_test_generated":False,
        "test_labels_read":False,
    }


def build_source_lock(teacher_bundle,teacher_meta,potential_bundle,potential_meta,out):
    inputs={
        "teacher_bundle_sha256":teacher_bundle,
        "teacher_meta_sha256":teacher_meta,
        "potential_bundle_sha256":potential_bundle,
        "potential_meta_sha256":potential_meta,
    }
    measured={k:sha_file(f) for k,f in inputs.items()}
    if measured!=ORIGINAL_SHAS:
        raise ValueError("original LGA-02/LGA-05 first artifact SHA differs")
    metadata,teacher=load_frozen(teacher_bundle,teacher_meta)
    if metadata["teacher_init_seeds"]!=list(TEACHERS) or metadata["mode"]!="full":
        raise ValueError("original teacher/SAE roster differs")
    for seed in TEACHERS:
        models,_,mu,std,rms=load_potentials(potential_bundle,potential_meta,seed)
        if (tuple(models)!=("shared","conditioned")
            or not bool(torch.isfinite(mu).all() and torch.isfinite(std).all())
            or not bool(torch.isfinite(rms))):
            raise ValueError("original E normalization/tensor fingerprint invalid")
    record={
        "schema":SCHEMA,
        "original_shas":measured,
        **code_contract(),
        "verified_canonical_loaders":True,
        "native_test_performed":False,
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print("LGA07C_PRETEST_FROZEN_SOURCE_LOCK",sha_file(out),flush=True)
    return record


def verify_source_lock(lock_path,teacher_bundle,teacher_meta,
                       potential_bundle,potential_meta):
    lock=json.loads(Path(lock_path).read_bytes())
    if (lock.get("schema")!=SCHEMA
        or lock.get("original_shas")!=ORIGINAL_SHAS
        or lock.get("native_test_performed") is not False
        or lock.get("verified_canonical_loaders") is not True):
        raise ValueError("original registered pretest source lock invalid")
    if any(lock.get(k)!=v for k,v in code_contract().items()):
        raise ValueError("any prospective protocol or complete implementation source byte drift")
    current={
        "teacher_bundle_sha256":sha_file(teacher_bundle),
        "teacher_meta_sha256":sha_file(teacher_meta),
        "potential_bundle_sha256":sha_file(potential_bundle),
        "potential_meta_sha256":sha_file(potential_meta),
    }
    if current!=ORIGINAL_SHAS:
        raise ValueError("model or energy bytes differ after source-only gate")
    return lock


def main():
    p=argparse.ArgumentParser()
    for a in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta","out"):
        p.add_argument("--"+a,required=True,type=Path)
    z=p.parse_args()
    build_source_lock(z.teacher_bundle,z.teacher_meta,
                      z.potential_bundle,z.potential_meta,z.out)


if __name__=="__main__":
    main()
