"""Rebuild RAM-07's strongest symbolic and loop64 original Lean witnesses.

This is a postscore independent reproducibility check. Original theorem
source/commit, registered tactic scripts, certificate SHA256 and allowed
axioms are immutable. It neither retrains nor rescans the holdout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import ram07_core as c

SOURCE_COMMIT="fca46bfb3430ac85ffca16f6a6a83bb0b0231e41"
LEAN_TOOLCHAIN="leanprover/lean4:v4.35.0"
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/"certificates/tactic_fibers_ordering_witnesses.json"

def sha_bytes(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

def run(mathcan:Path,lean:str,out:Path):
    commit=subprocess.check_output(
        ["git","-C",str(mathcan),"rev-parse","HEAD"],text=True).strip()
    if commit!=SOURCE_COMMIT:
        raise AssertionError("original MathCan source commit differs")
    if (mathcan/"lean-toolchain").read_text(encoding="utf8").strip()!=LEAN_TOOLCHAIN:
        raise AssertionError("original Lean toolchain differs")
    manifest=json.loads(MANIFEST.read_text(encoding="utf8"))
    assert manifest["original_new_mathcan_commit"]==SOURCE_COMMIT
    assert manifest["toolchain"]==LEAN_TOOLCHAIN
    assert manifest["original_theorem"]=="exactly_two_matching_orbit"
    assert len(manifest["evidence"])==2
    original_path=mathcan/manifest["original_source_file"]
    original=original_path.read_text(encoding="utf8")
    assert c.digest(original_path)==manifest["original_source_sha256"]
    out.mkdir(parents=True,exist_ok=False)
    checks=[]
    for receipt in manifest["evidence"]:
        actions=receipt["actions"]
        if not actions or any(not c.valid_script(a) for a in actions):
            raise AssertionError("unregistered natural-source proof tactic")
        if receipt["policy"] not in {"loop64","frequency"}:
            raise AssertionError("unexpected policy witness")
        source=c.make_source(original,manifest["original_theorem"],actions,server=False)
        raw=source.encode("utf8")
        if sha_bytes(raw)!=receipt["full_sorry_free_original_source_sha256"]:
            raise AssertionError("saved original source hash differs for "+receipt["policy"])
        result_file=out/(receipt["policy"]+".lean")
        result_file.write_bytes(raw)
        assert sha_bytes(result_file.read_bytes())==receipt["full_sorry_free_original_source_sha256"]
        compiler=subprocess.run([lean,str(result_file.resolve())],
            capture_output=True,text=True,timeout=90)
        output=compiler.stdout+compiler.stderr
        logfile=out/(receipt["policy"]+".log")
        logfile.write_text(output,encoding="utf8")
        gate,axioms=c.axiom_audit(output)
        if (compiler.returncode!=0 or not gate or
            set(axioms)!=set(receipt["axioms"]) or "error:" in output or
            "sorryAx" in output):
            raise AssertionError("original fully compiled theorem/axiom gate failed: "+receipt["policy"])
        row=dict(policy=receipt["policy"],
            theorem=manifest["original_theorem"],
            original_source_file=manifest["original_source_file"],
            original_source_sha256=manifest["original_source_sha256"],
            generated_complete_theorem_sha256=sha_bytes(raw),
            independent_compiler_log_sha256=c.digest(logfile),
            compiled=True,valid_axiom_gate=gate,axioms=axioms,
            original_gold_prefix_used=False,
            previously_known_mathcan_theorem=True)
        checks.append(row)
        print("RAM07_REBUILT_WITNESS",json.dumps(row,sort_keys=True),flush=True)
    report=dict(experiment="RAM07_postscore_original_pinned_witness_rebuild",
        source_commit=commit,
        original_scorer_report_sha256=manifest["original_unchanged_scored_report_sha256"],
        independent_theorems=1,
        complete_source_policy_instances=len(checks),
        false_positive_certificates=0,
        policy_results=checks)
    c.write_json(out/"witness_rebuild_receipt.json",report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.mathcan,args.lean,args.out)
if __name__=="__main__":
    main()
