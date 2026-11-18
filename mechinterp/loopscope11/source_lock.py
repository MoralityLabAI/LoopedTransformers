"""LGA-08 source-only pretest lock: verify frozen models, protocol, code.

Runs before first reserved 31062 is generated; no new task data allowed here.
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
SCHEMA="LGA08/first-source-pretest-frozen-lock-v1"
TEACHERS=(103,211,307)
ORIGINAL={
    "teacher_bundle_sha256":"d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
    "teacher_meta_sha256":"7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
    "potential_bundle_sha256":"24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
    "potential_meta_sha256":"69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
PROTO=("PROTOCOL.md","POWER_PREFLIGHT.md")
CODES=("source_lock.py","pair_fiber.py","native_pair.py","audit_pair.py",
       "tests/test_pair_fiber.py","tests/test_native_pair.py")
PARENTS=(
    "mechinterp/loopscope10/donor_source.py",
    "mechinterp/loopscope10/global_energy_fiber.py",
    "mechinterp/loopscope10/energy_fiber.py",
    "mechinterp/loopscope04/gradient_flow.py",
    "mechinterp/loopscope05/fit_and_freeze.py",
    "mechinterp/loopscope08/fit_potential.py",
)
WORKFLOWS=(
    ".github/workflows/loopscope11-pair-fiber.yml",
    ".github/workflows/loopscope11-first-native.yml",
)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contract():
    return {
      "protocol_sha256":{p:sha_file(HERE/p) for p in PROTO},
      "implementation_sha256":{
        **{p:sha_file(HERE/p) for p in CODES},
        **{p:sha_file(REPO/p) for p in (*PARENTS,*WORKFLOWS)}
      },
      "reserved_seed":31062,
      "excluded_smoke_seed":61062,
      "teacher_seeds":list(TEACHERS),
      "n_independent_bases":64,
      "goals":[0,10,20,30,40,50,60,70],
      "native_depth":2,
      "native_endpoint":4,
      "new_test_generated":False,
      "new_test_labels_read":False,
    }


def build_lock(teacher_bundle,teacher_meta,potential_bundle,potential_meta,out):
    paths={
      "teacher_bundle_sha256":teacher_bundle,
      "teacher_meta_sha256":teacher_meta,
      "potential_bundle_sha256":potential_bundle,
      "potential_meta_sha256":potential_meta,
    }
    found={name:sha_file(p) for name,p in paths.items()}
    if found!=ORIGINAL:
        raise ValueError("original LGA-02/LGA-05 pretrained checkpoint SHA mismatch")
    tmeta,_=load_frozen(teacher_bundle,teacher_meta)
    if tmeta.get("teacher_init_seeds")!=list(TEACHERS) or tmeta.get("mode")!="full":
        raise ValueError("original source teacher roster or weights drift")
    for sid in TEACHERS:
        models,_,mu,std,rms=load_potentials(potential_bundle,potential_meta,sid)
        if (tuple(models)!=("shared","conditioned") or
            not bool(torch.isfinite(mu).all() and torch.isfinite(std).all()
                     and torch.isfinite(rms))):
            raise ValueError("original LGA-05 frozen E and train-normalizer drift")
    record={
      "schema":SCHEMA,
      "original_sha256":found,
      **contract(),
      "canonical_original_tensor_loaders":"PASS",
      "prospective_first_test_scored":False,
    }
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print("LGA08_ORIGINAL_SOURCE_ONLY_PRETEST_LOCK",sha_file(out),flush=True)
    return record


def verify_lock(lock,teacher_bundle,teacher_meta,potential_bundle,potential_meta):
    r=json.loads(Path(lock).read_bytes())
    if (r.get("schema")!=SCHEMA or r.get("original_sha256")!=ORIGINAL or
        r.get("canonical_original_tensor_loaders")!="PASS" or
        r.get("prospective_first_test_scored") is not False or
        any(r.get(key)!=value for key,value in contract().items())):
        raise ValueError("prospective pretest protocol/code/roster byte lock invalid")
    found={
      "teacher_bundle_sha256":sha_file(teacher_bundle),
      "teacher_meta_sha256":sha_file(teacher_meta),
      "potential_bundle_sha256":sha_file(potential_bundle),
      "potential_meta_sha256":sha_file(potential_meta),
    }
    if found!=ORIGINAL:
        raise ValueError("native inference did not use frozen first model/energy bytes")
    return r


def main():
    p=argparse.ArgumentParser()
    for x in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta","out"):
        p.add_argument("--"+x,type=Path,required=True)
    a=p.parse_args()
    build_lock(a.teacher_bundle,a.teacher_meta,
               a.potential_bundle,a.potential_meta,a.out)


if __name__=="__main__":
    main()
