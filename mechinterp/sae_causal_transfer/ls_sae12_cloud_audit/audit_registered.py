"""Cross-platform cloud audit of immutable LS-SAE12 registered evidence.

Runs the original exact standard-library auditor; on platform libm mismatch,
requires the separately committed 16-ulp AND bit-identical float32 edit check
before the exact original auditor is used against verified saved vectors.

No scientific thresholds, cohorts, outcomes or source receipts are altered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

FAMILIES=("active_a","active_b")
ROOT=Path("mechinterp/sae_causal_transfer")
ORIGINAL_RESULTS=ROOT/"results"/"ls_sae12"
EXACT_MODULE="mechinterp.sae_causal_transfer.active_reasoning_v1.audit"
PORTABLE_MODULE="mechinterp.sae_causal_transfer.ls_sae12_analysis.replay_audit"

def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(argv, logpath):
    p=subprocess.run([sys.executable,*argv],capture_output=True,text=True)
    logpath.parent.mkdir(parents=True,exist_ok=True)
    logpath.write_text("COMMAND: "+repr(argv)+"\nRETURN CODE: "+str(p.returncode)
                       +"\nSTDOUT:\n"+p.stdout+"\nSTDERR:\n"+p.stderr,encoding="utf-8")
    return p

def inspect(receipts:Path,output:Path,ls11:Path,ls08:Path):
    output.mkdir(parents=True,exist_ok=True)
    original=receipts/"registered"
    source=receipts/"source_48.json"
    source_out=output/"source-audit.json"
    args=["-m",EXACT_MODULE,"--source",str(source),
          "--ls-sae11-source",str(ls11),"--ls-sae08-full",str(ls08),
          "--out",str(source_out)]
    r=execute(args,output/"source-audit.log")
    if r.returncode:
        raise RuntimeError("original independent source/uniqueness/disjointness auditor failed")
    if json.loads(source_out.read_text()) != json.loads((receipts/"source-audit.json").read_text()):
        raise ValueError("source audit is not the original published source audit")
    summary={"schema":"LS-SAE12/cross-platform-registered-evidence-audit-v1",
             "original_run":44084948890,"platform":platform.platform(),
             "python":sys.version.split()[0],"source_independent_audit":"PASS",
             "family_audits":{}}
    for family in FAMILIES:
        result=output/(family+"-audit.json")
        common=["--source",str(source),"--screen",str(original/"screen.json"),
                "--input",str(original/(family+".json")),"--out",str(result)]
        strict=execute(["-m",EXACT_MODULE,*common],output/(family+"-strict.log"))
        method="registered-exact"
        iso_check=None
        if strict.returncode:
            if "isotropic vectors do not regenerate from their seeds" not in (
                strict.stdout+"\n"+strict.stderr):
                raise RuntimeError(f"original registered auditor failed for {family} for a reason other than platform Gaussian rounding")
            portable=execute(["-m",PORTABLE_MODULE,*common],
                             output/(family+"-portable.log"))
            if portable.returncode:
                raise RuntimeError(f"float32-edit-preserving original-auditor replay failed: {family}")
            checkpath=result.with_name(result.stem+".isotropic-check.json")
            iso_check=json.loads(checkpath.read_text())
            if iso_check.get("max_ulp",17)>16 or not iso_check.get(
                    "float32_edit_components_checked_bit_identical",0):
                raise ValueError("Gaussian float32 edit parity evidence incomplete")
            method="original-auditor-after-verified-f32-preserving-Gaussian-replay"
        produced=json.loads(result.read_text())
        original_audit=json.loads((original/(family+"-audit.json")).read_text())
        if produced!=original_audit:
            raise ValueError(f"{family} independently replayed registered audit differs from published result")
        primary=produced["primary"]
        if (not produced["screen_pass"] or produced["states"]!=12
            or primary["events_positive"]!=8
            or primary["registered_success"] is not False
            or produced["registered_family_success"] is not False):
            raise ValueError(f"{family} registered source/counter or FAIL outcome changed")
        summary["family_audits"][family]={
            "method":method,"independent_report_matches_published":True,
            "archived_sha256":sha(original/(family+"-audit.json")),
            "replay_sha256":sha(result),"positive_states":primary["events_positive"],
            "mean":sum([x["selected"]["summary"]["0.02"]["D_structural_minus_isotropic"]
                        for x in json.loads((original/(family+".json")).read_text())["states"]])/12,
            "bootstrap95":primary["bootstrap95"],
            "registered_success":False,"Gaussian_replay_validation":iso_check}
    analysis=output/"recomputed"
    r=execute(["-m","mechinterp.sae_causal_transfer.ls_sae12_analysis.analyze",
               "--run-dir",str(original),"--source",str(source),"--out",str(analysis)],
              output/"analysis.log")
    if r.returncode:
        raise RuntimeError("independent post-hoc table script failed on immutable evidence")
    recomputed=json.loads((analysis/"ls_sae12_analysis_summary.json").read_text())
    published=json.loads((ORIGINAL_RESULTS/"ls_sae12_analysis_summary.json").read_text())
    if recomputed!=published:
        raise ValueError("registered per-state derived tables/summary disagree with pinned published results")
    for name,count in published["tables"].items():
        if sum(1 for _ in (analysis/name).open(encoding="utf-8"))-1!=count:
            raise ValueError("original table coverage differs: "+name)
    summary.update({"analysis_matches_original_published":True,
                    "derived_rows":published["tables"],"registered_scientific_gate":"FAIL",
                    "limitations":"Audit recomputes from saved source, logits and direction receipts. Neural state/Jacobian execution is not repeated by the audit."})
    (output/"audit_verdict.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"source":summary["source_independent_audit"],
                      "family_a":summary["family_audits"]["active_a"]["method"],
                      "family_b":summary["family_audits"]["active_b"]["method"],
                      "result":"registered FAIL independently reproduced"},
                     indent=2))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ls11-source",type=Path,required=True)
    p.add_argument("--ls08-full",type=Path,required=True)
    a=p.parse_args()
    inspect(a.receipts,a.out,a.ls11_source,a.ls08_full)

if __name__=="__main__":
    main()
