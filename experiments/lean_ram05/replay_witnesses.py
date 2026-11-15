"""Reconstruct two RAM-05 original Lean proofs that fixed native order missed.

Reads original pinned MathCan theorem declarations and preserves the complete
source hash. Uses RAM-05's registered grammar and exact independent compiler/
axiom gate. Does not retrain, read gold proof commands, or alter prior scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import ram05_core as c

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/"certificates/bounded_ordering_witnesses.json"
SOURCE_COMMIT="c84e2025334e9a967c1297425a490f98d5b3d189"
TOOLCHAIN="leanprover/lean4:v4.35.0"

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def run(mathcan:Path,lean:str,out:Path)->dict:
    if subprocess.check_output(["git","-C",str(mathcan),"rev-parse","HEAD"],
                               text=True).strip()!=SOURCE_COMMIT:
        raise ValueError("original MathCan SHA differs")
    if (mathcan/"lean-toolchain").read_text(encoding="utf8").strip()!=TOOLCHAIN:
        raise ValueError("Lean toolchain differs")
    manifest=json.loads(MANIFEST.read_text(encoding="utf8"))
    if manifest["original_mathcan_commit"]!=SOURCE_COMMIT or len(manifest["proof_scripts"])!=2:
        raise ValueError("changed published proof manifest")
    out.mkdir(parents=True,exist_ok=False)
    checked=[]
    for row in manifest["proof_scripts"]:
        pristine=(mathcan/row["source_file"]).read_text(encoding="utf8")
        if sha(pristine.encode("utf8"))!=row["source_pristine_sha256"]:
            raise AssertionError("original MathCan source bytes changed")
        actions=row["selected_tactics"]
        if any(not c.valid_script(action) for action in actions):
            raise ValueError("candidate script is outside RAM05 grammar")
        proof=c.make_source(pristine,row["theorem"],actions,server=False)
        if sha(proof.encode("utf8"))!=row["certified_complete_source_sha256"]:
            raise AssertionError("exact original complete Lean source SHA mismatch")
        target=out/(row["theorem"]+".lean")
        log_file=out/(row["theorem"]+".log")
        target.write_text(proof,encoding="utf8",newline="\n")
        if sha(target.read_bytes())!=row["certified_complete_source_sha256"]:
            raise AssertionError("source materialization altered proof bytes")
        cp=subprocess.run([lean,str(target.resolve())],capture_output=True,
                          text=True,timeout=60)
        stdout=cp.stdout+cp.stderr
        log_file.write_text(stdout,encoding="utf8")
        allowed,axioms=c.axiom_audit(stdout)
        success=(cp.returncode==0 and allowed and
                 "sorryAx" not in stdout and "error:" not in stdout)
        entry=dict(source_file=row["source_file"],theorem=row["theorem"],
                   certified=success,axioms=axioms,
                   source_sha256=sha(target.read_bytes()),
                   log_sha256=c.digest(log_file))
        checked.append(entry)
        print("RAM05_WITNESS",json.dumps(entry,sort_keys=True),flush=True)
        if not success:
            raise AssertionError("original Lean proof or axiom gate failed")
    receipt=dict(experiment="RAM05_postscore_original_witness_replay",
                 source_commit=SOURCE_COMMIT,
                 original_scorer_report_sha256=manifest["original_score_report_sha256"],
                 independently_compiled_witnesses=len(checked),
                 rejected=0,certificates=checked)
    c.write_json(out/"witness_receipt.json",receipt)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.mathcan,args.lean,args.out)

if __name__=="__main__":main()
