"""LGA-09 original frozen source/code lock before any 31063 generation."""
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
SCHEMA="LGA09/pretest-source-and-all-ten-logit-code-lock-v1"
TEACHERS=(103,211,307)
ORIGINAL={
  "teacher_bundle_sha256":"d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266",
  "teacher_meta_sha256":"7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d",
  "potential_bundle_sha256":"24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53",
  "potential_meta_sha256":"69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d",
}
PROTO=("PROTOCOL.md","POWER_PREFLIGHT.md")
CODES=("source_lock.py","full_jacobian.py","native_full_j.py",
       "audit_full_j.py","tests/test_full_jacobian.py",
       "tests/test_native_full_j.py")
PARENTS=(
 "mechinterp/loopscope11/pair_fiber.py",
 "mechinterp/loopscope10/donor_source.py",
 "mechinterp/loopscope10/global_energy_fiber.py",
 "mechinterp/loopscope10/energy_fiber.py",
 "mechinterp/loopscope04/gradient_flow.py",
 "mechinterp/loopscope05/fit_and_freeze.py",
 "mechinterp/loopscope08/fit_potential.py",
)
WORKFLOWS=(".github/workflows/loopscope12-source-only.yml",
           ".github/workflows/loopscope12-first-native.yml")


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_contract():
    return {
      "protocol_sha256":{p:file_sha(HERE/p) for p in PROTO},
      "implementation_sha256":{
        **{p:file_sha(HERE/p) for p in CODES},
        **{p:file_sha(REPO/p) for p in (*PARENTS,*WORKFLOWS)}
      },
      "reserved_new_seed":31063,
      "excluded_smoke_seed":61063,
      "teacher_seeds":list(TEACHERS),
      "n_independent_bases":64,
      "goals":[0,10,20,30,40,50,60,70],
      "native_intervention_depth":2,
      "native_final_depth":4,
      "new_test_generated":False,
      "new_test_labels_read":False,
    }


def freeze(teacher_bundle,teacher_meta,potential_bundle,potential_meta,out):
    paths={
       "teacher_bundle_sha256":teacher_bundle,
       "teacher_meta_sha256":teacher_meta,
       "potential_bundle_sha256":potential_bundle,
       "potential_meta_sha256":potential_meta
    }
    found={name:file_sha(path) for name,path in paths.items()}
    if found!=ORIGINAL:
        raise ValueError("original pretrained LGA02/LGA05 source artifact SHA drift")
    m,_=load_frozen(teacher_bundle,teacher_meta)
    if m["mode"]!="full" or m["teacher_init_seeds"]!=list(TEACHERS):
        raise ValueError("original teacher source/SAE canonical tensor roster drift")
    for sid in TEACHERS:
        models,_,mu,std,rms=load_potentials(potential_bundle,potential_meta,sid)
        if (tuple(models)!=("shared","conditioned") or
            not bool(torch.isfinite(mu).all() and torch.isfinite(std).all()
                     and torch.isfinite(rms))):
            raise ValueError("original frozen potential/source fit invalid")
    receipt={
       "schema":SCHEMA,"original_sha256":found,
       **source_contract(),
       "canonical_original_tensor_loader_status":"PASS",
       "first_native_scored":False
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("LGA09_ORIGINAL_SOURCE_FROZEN",file_sha(out),flush=True)
    return receipt


def verify(lock,teacher_bundle,teacher_meta,potential_bundle,potential_meta):
    source=json.loads(Path(lock).read_bytes())
    if (source.get("schema")!=SCHEMA or source.get("original_sha256")!=ORIGINAL
        or source.get("first_native_scored") is not False
        or source.get("canonical_original_tensor_loader_status")!="PASS"
        or any(source.get(k)!=v for k,v in source_contract().items())):
        raise ValueError("original prospective LGA09 protocol/code/pretest lock changed")
    paths=(teacher_bundle,teacher_meta,potential_bundle,potential_meta)
    if dict(zip(ORIGINAL,(file_sha(p) for p in paths)))!=ORIGINAL:
        raise ValueError("unlocked native teacher or scalar potential source bytes")
    return source


def main():
    p=argparse.ArgumentParser()
    for name in ("teacher-bundle","teacher-meta","potential-bundle",
                 "potential-meta","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    freeze(a.teacher_bundle,a.teacher_meta,a.potential_bundle,a.potential_meta,a.out)


if __name__=="__main__":
    main()
