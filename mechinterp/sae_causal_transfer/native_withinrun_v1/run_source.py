"""LS-SAE11 prospective source-only registry; always runs before any native test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as A8
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as F

SCHEMA="LS-SAE11/source-only-Champagne-20-v1"
PROTOCOL="e94a4439db78eacdb4988ecd167bcfb62a91230f"
ORIGINAL_SHA="69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"


def run(original_file,out):
    data=Path(original_file).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if digest!=ORIGINAL_SHA:raise ValueError("original LS-SAE08 reference bytes differ")
    reference=json.loads(data)
    A8.audit(reference)
    rosters,fingerprints=F.build_rosters()
    old_x={tuple(x["x"]) for f in reference["families"].values() for x in f["source_roster"]}
    old_target={tuple(x["target"]) for f in reference["families"].values() for x in f["source_roster"]}
    for group,rows in rosters.items():
        for source in rows:
            if tuple(source["x"]) in old_x or tuple(source["target"]) in old_target:
                raise ValueError("external hard source duplicates original LS-SAE08 reference source")
    report={
        "schema":SCHEMA,"registration_commit":PROTOCOL,
        "external_repo":F.REPO,"external_repo_commit":F.COMMIT,
        "source_file_hashes":[list(x) for x in F.SOURCE_FILES],
        "source_original_reference_sha256":digest,
        "rosters":rosters,"fingerprints":fingerprints,
        "pilot_cases":4,"full_cases":16,
        "model_inference_executed":False,
        "limits":"Source-disjoint from LS-SAE08 experimental rosters, not certified disjoint from public model pretraining or all Sudoku automorphisms."
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    record=run(a.original,a.out)
    print(json.dumps({"out":str(a.out),
       "original_source_SHA":record["source_original_reference_sha256"],
       "fingerprints":record["fingerprints"],
       "source_groups":{k:len(v) for k,v in record["rosters"].items()},
       "all_source_unique_and_single_incomplete":True},indent=2))


if __name__=="__main__":
    main()
