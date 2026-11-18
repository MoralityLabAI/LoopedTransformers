"""LGA-06 pretest source-lock receipt. Opens no new test 31058 cases."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen
from mechinterp.loopscope08.fit_potential import load_potentials
from mechinterp.loopscope09.eval_shift import protocol_sha,TEACHERS,ARMS,N_LOOPS,TEST_SEED
SOURCE_SHA="d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
SOURCE_META_SHA="7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
ENERGY_SHA="24bb276aafbfd79301694c49d783aa13f8fe5e311155a80c4900529d79cb5a53"
ENERGY_META_SHA="69fdace6eb58f36e96ff20d32c03d7c337989ec0882e57e95c0834449b06c19d"
SCHEMA="LGA-06/frozen-pretest-source-and-hypothesis-v1"


def lock(source_bundle,source_meta,energy_bundle,energy_meta,out):
    a,b,c,d=[Path(p).read_bytes() for p in
             (source_bundle,source_meta,energy_bundle,energy_meta)]
    found=[digest(v) for v in (a,b,c,d)]
    if found!=[SOURCE_SHA,SOURCE_META_SHA,ENERGY_SHA,ENERGY_META_SHA]:
        raise ValueError("original frozen LGA02/LGA05 source artifact byte mismatch")
    sm,bundle=load_frozen(source_bundle,source_meta)
    if sm["teacher_init_seeds"]!=list(TEACHERS) or sm["mode"]!="full":
        raise ValueError("original trained teacher roster changed")
    for sid in TEACHERS:
        models,initial,mu,std,rms=load_potentials(energy_bundle,energy_meta,sid)
        if tuple(models)!=( "shared","conditioned") or not torch.isfinite(std).all():
            raise ValueError("original frozen trained potential or normalizer invalid")
    record={
        "schema":SCHEMA,
        "protocol_sha256":protocol_sha(),
        "teacher_sha256":found[0],"teacher_manifest_sha256":found[1],
        "potential_sha256":found[2],"potential_manifest_sha256":found[3],
        "teachers":list(TEACHERS),"test_seed_reserved":TEST_SEED,
        "n_native_extra_loops":N_LOOPS,"test_arms":list(ARMS),
        "new_test_generated":False,"test_labels_read":False,
        "source_verification":"all teacher, SAE, potential and source normalizer tensor hashes verified by previous registered loaders",
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("LGA06_SOURCES_LOCKED",record["protocol_sha256"],flush=True)
    return record


def main():
    p=argparse.ArgumentParser()
    for arg in ("source-bundle","source-meta","energy-bundle","energy-meta","out"):
        p.add_argument("--"+arg,required=True,type=Path)
    a=p.parse_args()
    lock(a.source_bundle,a.source_meta,a.energy_bundle,a.energy_meta,a.out)


if __name__=="__main__":
    main()
