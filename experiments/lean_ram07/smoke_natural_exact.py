"""Training-only natural exact smoke for original MathCan Intervention proof roots.

No reference proof commands, no artificial assumptions. The original
Lean theorem goal is transformed by the registered named-intro tactic
family and the exact application is generated from the live context.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import ram07_core as c
from runtime import ResourceGuard, Session, certify, compile_source

SOURCE_COMMIT="c84e2025334e9a967c1297425a490f98d5b3d189"
CASES={
    "obsEqOn_trans": "exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)",
    "obsEqOn_symm": "exact (h ram07_arg0 ram07_arg1).symm",
}

def run(mathcan:Path,lean:str,out:Path):
    sha=subprocess.check_output(["git","-C",str(mathcan),"rev-parse","HEAD"],text=True).strip()
    if sha!=SOURCE_COMMIT:raise AssertionError("train source commit changed: "+sha)
    original=(mathcan/"MathCan/Intervention.lean").read_text(encoding="utf8")
    out.mkdir(parents=True,exist_ok=False)
    result=[]
    with ResourceGuard(out/"resources.jsonl") as guard:
        for i,(theorem,script) in enumerate(CASES.items()):
            root=out/f"{i}_{theorem}"
            with Session(lean,original,theorem,root/"server",guard) as server:
                prefixes=[["intro ram07_arg0"],
                         ["intro ram07_arg0","intro ram07_arg1"]]
                for actions in prefixes:
                    state=server.query(actions,3.)
                    if not state["accepted"] or len(state["goals"])!=1:
                        raise AssertionError(("named original introduction rejected",theorem,actions,state))
                typed=c.candidates(state)
                if ("exact_typed",script) not in typed:
                    raise AssertionError(("live source-derived exact template absent",theorem,script,typed))
                selected=prefixes[-1]+[script]
                closed=server.query(selected,3.)
                if not closed["accepted"] or closed["goals"]:
                    raise AssertionError(("original Lean proof not closed by exact",theorem,selected,closed))
            certificate=certify(lean,original,theorem,selected,root/"complete_original",guard,10.)
            if not certificate["accepted"]:
                raise AssertionError(("original theorem full certificate failed",theorem,certificate))
            source=Path(certificate["source_file"]).read_text(encoding="utf8")
            replay=compile_source(lean,source,root/"independent_original",guard,10.)
            if not replay["accepted"] or replay["source_sha256"]!=certificate["source_sha256"]:
                raise AssertionError(("independent exact full source replay failed",theorem,replay))
            row=dict(theorem=theorem,source_file="MathCan/Intervention.lean",
                     provenance="genuine_original_mathcan_training_or_development_root",
                     actions=selected,
                     from_actual_named_intro_goals=True,
                     source_sha256=certificate["source_sha256"],
                     allowed_axioms=certificate["axioms"],
                     full_original_theorem_compiled=True,
                     independent_byte_identical_replay=True,
                     original_reference_prefix_used=False)
            c.append_json(out/"natural_exact_training_smoke.jsonl",row)
            result.append(row)
    rec=dict(status="passed",source_commit=sha,training_roots=1,
             development_roots=1,verified_natural_exact_transitions=2,
             original_gold_proof_steps_used=False,rows=result)
    c.write_json(out/"natural_exact_smoke.json",rec)
    print("RAM07_NATURAL_EXACT_SMOKE",json.dumps(rec,sort_keys=True),flush=True)
    return rec

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mathcan",type=Path,required=True)
    parser.add_argument("--lean",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.mathcan,args.lean,args.out)
if __name__=="__main__":main()
