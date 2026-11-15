"""RAM-08 source-only smoke: real original MathCan exact from distinct roots.

Each proof originates in a legitimate unmodified original MathCan theorem.
Actions and contexts are source-visible, independently compiled and axiom-gated.
This diagnostic does not add rows to the preregistered training collector.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import ram08_core as c
from runtime import ResourceGuard, Session, certify, compile_source

SOURCE_COMMIT="c84e2025334e9a967c1297425a490f98d5b3d189"
CASES=(
    ("MathCan/Intervention.lean","obsEqOn_trans",
      ("intro ram07_arg0","intro ram07_arg1"),
      "exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)",
      ()),
    ("MathCan/TOEMixedPrincipal.lean","independent_kinetic_and_spectral_gates",
      ("constructor",),
      "exact hk",
      ("omega",)),
    ("MathCan/TOENonAbelian.lean","zero_mixed_nonzero_source",
      ("constructor","rfl"),
      "exact h",
      ()),
)

def run(mathcan:Path,lean:str,out:Path):
    commit=subprocess.check_output(
        ["git","-C",str(mathcan),"rev-parse","HEAD"],text=True).strip()
    if commit!=SOURCE_COMMIT:
        raise AssertionError("RAM08 original train source SHA differs")
    out.mkdir(parents=True,exist_ok=False)
    found=[]
    with ResourceGuard(out/"resources.jsonl") as guard:
        for index,(filename,theorem,prefix,exact,postfix) in enumerate(CASES):
            original=(mathcan/filename).read_text(encoding="utf8")
            path=out/f"{index:02d}_{theorem}"
            for tactic in (*prefix,exact,*postfix):
                if not c.valid_script(tactic):
                    raise AssertionError("RAM08 unregistered original tactic "+tactic)
            with Session(lean,original,theorem,path/"observations",guard) as server:
                before=server.query(list(prefix),3.)
                if not before["accepted"] or not before["goals"]:
                    raise AssertionError(("original visible theorem prefix failed",theorem,before))
                if ("exact_typed",exact) not in c.candidates(before) and (
                    "exact_local",exact) not in c.candidates(before):
                    raise AssertionError(("original compiler-visible exact not generated",theorem,exact))
                progressed=server.query(list(prefix)+[exact],3.)
                if not progressed["accepted"] or len(progressed["goals"])>=len(before["goals"]):
                    # A closed theorem is success: goal count 1 -> 0.
                    raise AssertionError(("natural exact did not advance",theorem,progressed))
                terminal=server.query(list(prefix)+[exact]+list(postfix),3.)
                if not terminal["accepted"] or terminal["goals"]:
                    raise AssertionError(("natural original theorem remained open",theorem,terminal))
            actions=list(prefix)+[exact]+list(postfix)
            certificate=certify(lean,original,theorem,actions,path/"full_original",guard,15.)
            if not certificate["accepted"]:
                raise AssertionError(("original complete proof failed",theorem,certificate))
            source=Path(certificate["source_file"]).read_text(encoding="utf8")
            replay=compile_source(lean,source,path/"fresh_original",guard,15.)
            if not replay["accepted"] or replay["source_sha256"]!=certificate["source_sha256"]:
                raise AssertionError(("third source proof mismatch",theorem,replay))
            rec=dict(source_file=filename,theorem=theorem,source_commit=commit,
                actual_original_mathcan_goal=True,teacher_prefix_used=False,
                reference_statement_oracle=False,registered_actions=actions,
                positive_exact_action=exact,
                complete_original_source_sha256=certificate["source_sha256"],
                accepted=True,independent_full_source_replay=True,
                axiom_inventory=certificate["axioms"])
            found.append(rec)
            c.append_json(out/"natural_original_exact_smoke.jsonl",rec)
            print("RAM08_DISTINCT_NATURAL_EXACT",json.dumps(rec,sort_keys=True),flush=True)
    if len({(z["source_file"],z["theorem"]) for z in found})!=3:
        raise AssertionError("expected three distinct natural source theorem roots")
    report=dict(status="passed",original_training_sha=commit,source_roots=3,
        real_original_mathematical_theorems=True,synthetic_auxiliary=0,
        gold_reference_prefix_used=False,certified_source_actions=found)
    c.write_json(out/"smoke_receipt.json",report)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--lean",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.mathcan,a.lean,a.out)
