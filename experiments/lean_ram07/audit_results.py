"""Independent RAM-06 post-score verification of original Lean proof scripts.

The scorer already performs two full original-source checks inside each
per-theorem budget. This independent extra recompile is unscored and cannot
alter frozen proof coverage. It protects against artifact tampering,
incorrect case accounting and acceptance of disallowed axioms.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ram07_core as c
sys.path.insert(0,str(HERE))
from evaluate import EXACT_CASES

PHASE_POLICIES={"primary":["native","frequency","shape_frequency","loop32","loop64","flat64"],
                "secondary":["native","frequency","shape_frequency","loop64"]}

def run(root:Path,lean:str,output:Path)->dict:
    report=json.loads((root/"evaluation_report.json").read_text())
    assert report["status"]=="complete"
    assert report["original_oracles"]=={"prospective_ram07":12}
    assert report["experiment"]=="RAM07"
    all_names=[(f,n) for f,names in EXACT_CASES.items() for n in names]
    confirmations=[]
    total=0
    for phase,names in PHASE_POLICIES.items():
        by_policy={}
        for policy in names:
            path=root/f"{phase}__prospective_ram07__{policy}.jsonl"
            rows=[json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
            assert len(rows)==12
            assert [(r["source_file"],r["theorem"]) for r in rows]==all_names
            assert all(r["budget"]=={"seconds":30 if phase=="primary" else 60,
                                    "evaluations":48 if phase=="primary" else 128,
                                    "transitions":8 if phase=="primary" else 16,
                                    "request_seconds":3,"max_goals":8}
                       for r in rows)
            by_policy[policy]=rows
            for index,row in enumerate(rows):
                assert row["logical_evaluations"]<=row["budget"]["evaluations"]
                assert row["max_path_transitions"]<=row["budget"]["transitions"]+1
                if not row["accepted"]:
                    continue
                assert row["reason"]=="independently_certified"
                assert row["original_source_sha256"]==report["source_hashes"][row["source_file"]]["normalized_sha256"]
                cert=row["certificate"];replay=row["replay"]
                assert cert and replay and cert["accepted"] and replay["accepted"]
                base=root/"evidence"/phase/"prospective_ram07"/f"{index:02d}"/policy
                a=base/"certificate.lean"
                b=base/"independent_replay.lean"
                assert a.is_file() and b.is_file()
                raw=a.read_bytes()
                assert raw==b.read_bytes()
                digest=hashlib.sha256(raw).hexdigest()
                assert digest==cert["source_sha256"]==replay["source_sha256"]
                text=raw.decode("utf8")
                # The original MathCan source legitimately has comments such as
                # "admits two ...". String searching the entire source for
                # "admit" rejects correctly compiled, independently replayed
                # proofs. Inspect the *selected tactic lines* instead; a raw
                # original comment cannot become an executable candidate.
                assert isinstance(row["actions"],list) and row["actions"]
                assert all(c.valid_script(action) for action in row["actions"])
                assert not any(re.search(r"\\b(?:sorry|sorryAx|admit)\\b", action)
                               for action in row["actions"])
                assert "ram03_probe_server" not in text
                assert text.count("#print axioms ")==1
                assert "#print axioms "+row["theorem"] in text
                # A third independent exact-source Lean compile occurs
                # outside the scoring deadline; its verdict is never used
                # to add or delete a scored theorem.
                result=subprocess.run([lean,str(a.resolve())],
                    capture_output=True,text=True,timeout=60)
                stdout=result.stdout+result.stderr
                gate,axioms=c.axiom_audit(stdout)
                if (result.returncode!=0 or not gate or
                    set(axioms)!=set(cert["axioms"]) or
                    set(axioms)!=set(replay["axioms"]) or
                    "error:" in stdout or "sorryAx" in stdout):
                    raise AssertionError(("invalid independent theorem certificate",phase,
                        policy,row["theorem"],result.returncode,stdout[-450:]))
                confirmation=dict(phase=phase,policy=policy,theorem=row["theorem"],
                    source_file=row["source_file"],source_sha256=digest,
                    recompiled=True,axioms=axioms)
                confirmations.append(confirmation)
        baseline={f"{r['source_file']}:{r['theorem']}" for r in by_policy["native"] if r["accepted"]}
        for policy,rows in by_policy.items():
            got={f"{r['source_file']}:{r['theorem']}" for r in rows if r["accepted"]}
            summary=next(x for x in report["summaries"] if x["experiment"]==phase and x["suite"]=="prospective_ram07" and x["model"]==policy)
            assert summary["certified"]==len(got)
            assert set(summary["solved"])==got
            assert set(summary["model_only_vs_native"])==got-baseline
            assert set(summary["native_only_vs_model"])==baseline-got
            total+=len(rows)
    assert total==120
    output.mkdir(parents=True,exist_ok=False)
    manifest="".join(json.dumps(r,sort_keys=True)+"\n" for r in confirmations)
    (output/"independent_verified_sources.jsonl").write_text(manifest)
    digest=hashlib.sha256(manifest.encode()).hexdigest()
    receipt=dict(experiment="RAM07_independent_third_Lean435_replay",
      scored_cases=total,independently_recompiled_successes=len(confirmations),
      source_manifest_sha256=digest,
      original_scored_report_sha256=c.digest(root/"evaluation_report.json"),
      false_positive_certificates=0,all_scores_unchanged=True,
      per_policy={
          f"{phase}/{policy}":sum(r["phase"]==phase and r["policy"]==policy for r in confirmations)
          for phase,names in PHASE_POLICIES.items() for policy in names
      })
    c.write_json(output/"RAM07_independent_audit.json",receipt)
    print("RAM07_AUDITED",json.dumps(receipt,sort_keys=True),flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.results,a.lean,a.out)
if __name__=="__main__":main()
