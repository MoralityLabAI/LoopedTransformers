"""Synthetic label-free NEC-03V geometry fixtures for numerical regression only.

These 128-dimensional code dictionaries are artificial. Their test outcomes
are NEVER evidence about the original source-trained neurons or SAE geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .nec03v_geometry import (
    ACTIVE, ARM_SCHEMA, ARMS, DIM, FEATURES, GROUPS, MASKS, SCHEMA,
    ORIGINAL_SOURCE_FIT_ROWS, canonical, file_hash, original_lock_digest,
)


def _decoder(feature, visit, arm):
    # Shared positive and gently feature-varying directions create a
    # nontrivial structural null with many admissible disjoint candidates.
    offset = 0.013 if arm == "composed" else -0.009
    return [
        round((0.4 + 0.18*math.sin((feature % 16)*.12 + dim*.27 + visit*.1)
               + 0.075*math.cos(feature*.035 + dim*.11))*(1+offset), 7)
        for dim in range(DIM)
    ]


def _features(index):
    start = (index % 15)*8
    return sorted((start + j) % FEATURES for j in range(ACTIVE))


def make_fixture(directory: Path, world_index: int = 0):
    if directory.exists() or type(world_index) is not int or world_index not in range(8):
        raise ValueError("fresh fixture path and original 0..7 world index required")
    directory.mkdir(parents=True,exist_ok=False)
    shards={}
    for arm in ARMS:
        visits={}
        # Synthetic root 3/5's additive teacher imitates an ineligible
        # source-instrument arm with no registered intervenable site.
        selected_visits = () if world_index in (3, 5) and arm == "additive" else (3, 7)
        for visit in selected_visits:
            decoder=[_decoder(feature,visit,arm) for feature in range(FEATURES)]
            # On visit 7, features 112..127 have *zero* physical projected
            # decoder columns. Select a case with those codes to exercise
            # paired invalid-all-visits zeroing despite positive TopK features.
            if visit==7:
                for i in range(112,128):
                    decoder[i]=[0.0]*DIM
            fits=[_features(i) for i in range(ORIGINAL_SOURCE_FIT_ROWS)]
            visits[str(visit)]={"decoder32":decoder,"source_fit_active":fits}
        cases=[]
        for n in range(len(GROUPS)*len(MASKS)):
            case_visits={}
            for visit in selected_visits:
                group=_features(n)
                if visit==7 and n in (0,17):
                    group=list(range(112,128))
                case_visits[str(visit)]=[
                    [idx,round(0.11+((n+idx+visit)%9)*.018,7)]
                    for idx in group
                ]
            cases.append({"group":GROUPS[n//len(MASKS)],"mask":n%len(MASKS),
                          "visits":case_visits})
        path=directory/f"{arm}.json"
        path.write_text(canonical({"schema":ARM_SCHEMA,"arm":arm,
                                   "world_index":world_index,"visits":visits,
                                   "cases":cases})+"\n",encoding="utf-8")
        shards[arm]={"file":path.name,"sha256":file_hash(path)}
    fake_checkpoint_receipts={}
    for arm in ARMS:
        fake_checkpoint_receipts[arm]={
            tag:hashlib.sha256(f"SYNTHETIC_ONLY|{arm}|{world_index}|{tag}".encode()).hexdigest()
            for tag in ("native_sha256","sae_sha256","source_sites_sha256",
                        "origin_receipt_sha256")
        }
    manifest={
        "schema":SCHEMA,"world_index":world_index,
        "world_id":f"dev5-{world_index:03d}",
        "source_panel":list(GROUPS),
        "original_lock_sha256":original_lock_digest(),
        "evidence":"synthetic_geometry_fixture",
        "source_checkpoint_receipts":fake_checkpoint_receipts,
        "arm_shards":shards,
    }
    (directory/"manifest.json").write_text(canonical(manifest)+"\n",encoding="utf-8")
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--world-index",type=int,default=0)
    args=p.parse_args(argv)
    result=make_fixture(args.out,args.world_index)
    print(canonical({"schema":result["schema"],"evidence":result["evidence"],
                     "world_id":result["world_id"]}))
    return result


if __name__=="__main__":
    main()
