"""Prepare LS-SAE12 immutable Actions evidence for cross-platform cloud audits.

All outputs are copies of verified original bytes. Checks the pinned evidence
branch SHA through the workflow checkout and every archived compressed hash.
No neural checkpoint execution, scientific rescore, or threshold edits here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

EVIDENCE_HEAD = "70c765cacc86504eba63d3de1e7491fc0defc1f8"
REGISTERED_RUN = 44084948890
REGISTERED_COMMIT = "74b8882d727ac8da1fae8f735c800656afda4aab"
SOURCE_REPO_FILE = Path("mechinterp/sae_causal_transfer/results/ls_sae12/source_48.json")
ANALYSIS_FILE = Path("mechinterp/sae_causal_transfer/results/ls_sae12/ls_sae12_analysis_summary.json")

EXPECTED_GZIP = {
    "ls-sae12-engineering-pilot/pilot.json.gz": "2dcb0672134edb97e8b9dfbeb1b9a02c69744fe456535b290eb10a06e032db0b",
    "ls-sae12-registered-screen-families-and-audit/active_a-audit.json.gz": "b688e1c79ab2f901e07f2bce439173f1d357ac9a3a25317e54b7b8951798b467",
    "ls-sae12-registered-screen-families-and-audit/active_a.json.gz": "52c7e95caa2048a45f9be18d8b3fdf44c47c6435257d9512012a733f54b511f6",
    "ls-sae12-registered-screen-families-and-audit/active_b-audit.json.gz": "3186add81528c88233f8c6fbd06b58b411f7041a15d5021b18c476b234a80476",
    "ls-sae12-registered-screen-families-and-audit/active_b.json.gz": "432b5acfd11314cd64281ce7e9a13475f45754b5fad46d746645e0c8b67c05fc",
    "ls-sae12-registered-screen-families-and-audit/screen.json.gz": "78b26c8cb7f660b41f898bc10ddba3bd53b15cde5cfcaccfa44a757f14460197",
    "ls-sae12-source-only-48-pool/source-audit.json.gz": "f74601cb490260a6773e1415f91dc0b94c55ed3e7538ff3e3879bfef708d868c",
    "ls-sae12-source-only-48-pool/source_48.json.gz": "353ef01a1313143e624c948b8258640365a4b6bdbf482768d380fc97afc1d217",
}

def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def stage(evidence:Path,out:Path):
    run=json.loads((evidence/"RUN.json").read_bytes())
    if (run.get("run_id")!=REGISTERED_RUN
        or run.get("run_attempt")!=1
        or run.get("workflow_commit")!=REGISTERED_COMMIT
        or run.get("job_results")!={"source":"success","pilot":"success","registered":"success"}):
        raise ValueError("original registered workflow provenance changed")
    expected_manifest=set(EXPECTED_GZIP)
    parsed={}
    for line in (evidence/"SHA256SUMS").read_text().splitlines():
        sha,rel=line.split(maxsplit=1)
        if not rel.startswith("./") or rel[2:] in parsed:
            raise ValueError("duplicate or noncanonical original artifact manifest")
        parsed[rel[2:]]=sha
    if parsed!=EXPECTED_GZIP:
        raise ValueError("original SHA256SUMS changed")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for rel,sha in sorted(EXPECTED_GZIP.items()):
        archive=evidence/rel
        b=archive.read_bytes()
        if digest(b)!=sha:
            raise ValueError(f"compressed GitHub artifact mismatch: {rel}")
        uncompressed=gzip.decompress(b)
        name=Path(rel).name.removesuffix(".gz")
        if rel.startswith("ls-sae12-registered-"):
            path=out/"registered"/name
        elif rel.startswith("ls-sae12-source-only-"):
            path=out/name
        else:
            path=out/"pilot"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(uncompressed)
        rows.append({"artifact":rel,"compressed_sha256":sha,
                     "raw_sha256":digest(uncompressed),"raw_bytes":len(uncompressed)})
    source=out/"source_48.json"
    if source.read_bytes()!=SOURCE_REPO_FILE.read_bytes():
        raise ValueError("registered 48-puzzle source differs from committed frozen file")
    analysis=json.loads(ANALYSIS_FILE.read_text())
    for name,sha in analysis["inputs"].items():
        if digest((out/"registered"/name).read_bytes())!=sha:
            raise ValueError(f"registered original raw numeric evidence differs: {name}")
    if (analysis["registered_outcome"]!="FAIL" or analysis["eligible"]!=36 or
        {family:analysis["family_results"][family]["primary"]["events_positive"]
         for family in ("active_a","active_b")}!={"active_a":8,"active_b":8}):
        raise ValueError("registered FAIL or original denominator altered")
    receipt={"schema":"LS-SAE12/registered-cloud-evidence-manifest-v1",
             "evidence_branch_sha":EVIDENCE_HEAD,"registered_run":REGISTERED_RUN,
             "registered_workflow_commit":REGISTERED_COMMIT,
             "compressed_artifacts_verified":rows,
             "source_sha256":digest(source.read_bytes()),
             "registered_outcome":"FAIL","eligible":36,
             "gate_positive_state_counts":{"active_a":8,"active_b":8}}
    (out/"archive_verified.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="compressed_artifacts_verified"},sort_keys=True))
    return receipt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    stage(a.evidence,a.out)

if __name__=="__main__":
    main()
