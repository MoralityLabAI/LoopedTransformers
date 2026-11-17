"""LS-SAE08 source-only prospective admission manifest; no model inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as FAMILY


def run(out):
    manifests=FAMILY.full_manifest()
    report={
      "schema":"LS-SAE08/model-blind-source-admission-v1",
      "protocol_commit":"6b38c3dcf65dbc3a812e0f7c63b79dc14c6893fe",
      "model_access":False,
      "fixed_candidate_budget_per_slot":FAMILY.MAX_CANDIDATES,
      "registered_holes":FAMILY.HOLES,
      "registered_family_sizes":FAMILY.SIZES,
      "canonical_roster_sha256":manifests,
      "families":{name:FAMILY.roster(name) for name in FAMILY.BASES}
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"source_model_access":False,
         "sha256":manifests,
         "attempts":{family:[rec["candidate_attempt_index"] for rec in group]
                     for family,group in report["families"].items()}},indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run(args.out)


if __name__=="__main__":
    main()
