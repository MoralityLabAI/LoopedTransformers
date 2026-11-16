"""RAM-17 prospective full-matrix analysis after independent Lean third replay.

Strictly refuse incomplete 16-policy x 10-case matrices, mixed budgets,
duplicate receipts, invalid accepted proofs, and retrospective phase results.
The registered recurrent-specific claim uses matched *seed-wise* outcomes.
"""
from __future__ import annotations
from collections import defaultdict
import argparse,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text())
AXIOMS=frozenset(("propext","Classical.choice","Quot.sound"))

def _locked_roots(manifest:dict,config:dict)->set[tuple[str,str]]:
    if manifest["test_source_commit"]!=config["test_source_commit"]:
        raise ValueError("source TEST commit differs from preregistration")
    if manifest["selected_before_goal_state_access"] is not True:
        raise ValueError("TEST theorem root list was not frozen before proof search")
    rows=manifest["cases"]
    if len(rows)!=len(config["source_modules"]["test"]):
        raise ValueError("missing original TEST modules")
    roots=set()
    for row in rows:
        mod=row["module"]
        if mod not in config["source_modules"]["test"] or row["git_blob_sha1"]!=config["source_modules"]["test"][mod]:
            raise ValueError("TEST module/blob not registered")
        if len(row["theorems"])!=config["selection"]["test_first"]:
            raise ValueError("TEST declaration count outside deterministic first-two selector")
        for thm in row["theorems"]:
            key=(mod,thm)
            if key in roots:
                raise ValueError("duplicate original theorem root")
            roots.add(key)
    if len(roots)!=config["anticipated_test_count_max"]:
        raise ValueError("TEST theorem root matrix cardinality changed")
    return roots

def analyse(rows:list[dict],manifest:dict,config:dict=CONFIG)->dict:
    roots=_locked_roots(manifest,config)
    controls=tuple(config["primary"]["policies"])
    if len(set(controls))!=16 or len(roots)!=10:
        raise ValueError("unregistered primary policy/test roster")
    expected={(mod,thm,p) for mod,thm in roots for p in controls}
    seen={}
    for row in rows:
        if row.get("phase")!="primary":
            raise ValueError("nonprimary or repaired receipt in first prospective primary")
        key=(row.get("module"),row.get("theorem"),row.get("policy"))
        if key not in expected or key in seen:
            raise ValueError("unexpected or repeated TEST policy case")
        calls=row.get("logical_evaluations")
        seconds=row.get("elapsed_seconds")
        if not isinstance(calls,int) or not 0<=calls<=config["primary"]["evaluations"]:
            raise ValueError("invalid Lean action request accounting")
        if not isinstance(seconds,(float,int)) or seconds<0:
            raise ValueError("invalid policy wall accounting")
        proof=row.get("accepted")
        if not isinstance(proof,bool):
            raise ValueError("policy receipt lacks a Boolean result")
        if proof:
            checks=[row.get("first_source_sha256"),row.get("second_source_sha256"),
                    row.get("third_source_sha256")]
            if any(not isinstance(s,str) or len(s)!=64 for s in checks) or len(set(checks))!=1:
                raise ValueError("accepted theorem lacks identical three original full-source SHA receipts")
            if row.get("independent_third_valid") is not True:
                raise ValueError("accepted theorem source lacks independent third Lean replay")
            if not set(row.get("allowed_axioms",[])).issubset(AXIOMS):
                raise ValueError("unexpected transitive theorem axiom")
            if seconds>config["primary"]["seconds"]:
                raise ValueError("postdeadline theorem counted as certified")
        seen[key]=row
    if set(seen)!=expected:
        missing=expected-set(seen)
        raise ValueError(f"incomplete prospective matrix: {len(missing)} missing policy cases")
    wins={p:set() for p in controls}
    for (mod,thm,p),row in seen.items():
        if row["accepted"]:
            wins[p].add((mod,thm))
    per_seed={}
    for seed in config["learning"]["train_seeds"]:
        paired={}
        for arch in ("loop64","flat71"):
            task=f"{arch}_task_only_s{seed}"
            structural=f"{arch}_structural_s{seed}"
            t=wins[task];s=wins[structural]
            paired[arch]=dict(task_only=len(t),structural=len(s),
                difference=len(s)-len(t),
                newly_proved=sorted([list(x) for x in s-t]),
                lost_proofs=sorted([list(x) for x in t-s]))
        per_seed[str(seed)]=paired
    witnesses=[]
    for root in sorted(roots):
        if any(root in wins[p] for p in ("native","frequency","shape_frequency")):
            continue
        seeds=[]
        for seed in config["learning"]["train_seeds"]:
            if (root in wins[f"loop64_structural_s{seed}"] and
                root not in wins[f"loop64_task_only_s{seed}"] and
                root not in wins[f"flat71_structural_s{seed}"]):
                seeds.append(seed)
        if len(seeds)>=2:
            witnesses.append(dict(module=root[0],theorem=root[1],paired_recurrent_witness_seeds=seeds))
    means={arch:sum(per_seed[str(s)][arch]["difference"] for s in config["learning"]["train_seeds"])/3
           for arch in ("loop64","flat71")}
    return dict(schema="RAM17-certified-primary-full-matrix",
                status="completed_independent_triple_source_replay",
                source_test_commit=config["test_source_commit"],
                original_theorems=len(roots),policies=len(controls),cases=len(seen),
                coverage_by_policy={p:len(wins[p]) for p in controls},
                per_seed_paired_coverage=per_seed,mean_paired_coverage_difference=means,
                recurrent_specific_witnesses=witnesses,
                registered_recurrent_specific_criterion_passed=bool(witnesses),
                false_certificates=0,
                original_first_primary_only=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cases",type=Path,required=True,help="independently third-audited normalized original primary JSONL")
    p.add_argument("--manifest",type=Path,required=True,help="committed pre-score original TEST theorem manifest")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("preserve original result; choose a new path")
    rows=[json.loads(s) for s in a.cases.read_text().splitlines() if s.strip()]
    result=analyse(rows,json.loads(a.manifest.read_text()))
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("RAM17_REGISTERED_PRIMARY",result["registered_recurrent_specific_criterion_passed"],
          result["cases"],"certificates",result["false_certificates"])
