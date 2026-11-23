"""LSA-02K native Lean equality benchmark preflight.

No model training. Freeze whole-equation-system-disjoint splits, certify 64
training reference paths, then evaluate bounded public symbolic planning and
live random/local DFS on 32 independent development roots. Every reference
and every claimed closure must pass actual Lean kernel verification and a
second standalone axiom-free compile. Ordinary/OOD contents stay sealed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import subprocess
import time

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .equality_suite import EqCase, at_arg, load_split, neighbors
from .equality_search import (
    BUDGET, CAPS, EXACT_CAP, RANDOM_SEEDS, plan, path_moves, run_live, replay,
)
from .live_lean import fingerprint

SCHEMA = "lsa02k.native_equality_preflight.v1"


def _write(path,obj):
    path.write_text(canonical(obj)+"\n",encoding="utf-8")


def _rows(path,rows):
    path.write_text("".join(canonical(r)+"\n" for r in rows),encoding="utf-8")


def _check_final_script(case,script):
    """Whitelist by reconstruction, rather than accepting arbitrary Lean text."""
    w=case.target
    if not script or script[-1]!="exact w" or len(script)>BUDGET:
        raise ValueError("standalone script is missing a terminal witness or violates budget")
    for tactic in script[:-1]:
        candidates=[m for m in neighbors(w,case.equations) if m.tactic(w)==tactic]
        if len(candidates)!=1:
            raise ValueError("standalone proof text is not a unique public grammar move")
        w=candidates[0].next_word
    if w!=case.source:
        raise ValueError("standalone proof did not reach the witness index")


def compile_scripts(items, out:Path, timeout:int=240):
    """Lean independently elaborates exact frozen statements and whitelisted scripts."""
    if not items:
        raise ValueError("no independent proof certificates requested")
    lines=["import Lean","namespace EqTransportCert",""]
    names=[]
    for case,label,script in items:
        case.validate()
        _check_final_script(case,script)
        if not re.fullmatch(r"[a-z0-9_]+",label):
            raise ValueError("invalid registered proof certificate label")
        name=f"t_{case.root}_{label}"
        if name in names:
            raise ValueError("duplicate Lean certificate declaration")
        names.append(name)
        lines.append(f"theorem {name} {case.binders()} : P {at_arg(case.target)} := by")
        lines.extend("  "+t for t in script)
        lines.append("")
    lines.append("end EqTransportCert")
    lines.extend(f"#print axioms EqTransportCert.{name}" for name in names)
    path=out/"standalone.lean"
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    p=subprocess.run([str(config.LEAN_EXE),str(path.resolve())],
                     capture_output=True,text=True,encoding="utf-8",
                     timeout=timeout,env=config.lean_env())
    output=p.stdout+p.stderr
    (out/"compiler.log").write_text(output,encoding="utf-8")
    passed=set(re.findall(
        r"'EqTransportCert\.([a-zA-Z0-9_]+)' does not depend on any axioms",
        output,
    ))
    if p.returncode!=0 or passed!=set(names):
        raise ValueError("standalone Lean equality proof/axiom audit failed:\n"+output[-3500:])
    receipt={"returncode":p.returncode,"declarations":len(names),
             "passed":len(passed),"no_extra_axioms":True,
             "source_sha256":sha256_file(path),
             "compiler_log_sha256":sha256_file(out/"compiler.log"),
             "names":names}
    _write(out/"compiler_receipt.json",receipt)
    return receipt


def _native_backend(repl):
    if str(repl.version)!="4.19.0" or not isinstance(repl,LeanRepl):
        raise ValueError("actual pinned Lean 4.19 proof server required")
    return fingerprint(config.REPL_EXE,config.LEAN_EXE,str(repl.version))


def _plan_reference(repl,case):
    result=plan(case,cap=EXACT_CAP)
    if not result["found"] or result["rewrite_distance"]+1>BUDGET:
        raise ValueError("frozen equality root lacks reachable reference within tactic budget")
    ep=run_live(repl,case,"bidirectional_bfs",words=result["words"])
    if not ep["live_kernel"] or not replay(repl,case,ep):
        raise ValueError("native equality reference path failed kernel or replay")
    result["certification_status"]="native_kernel_and_replay"
    return result,ep


def _verify_bounded_plans(case,ref):
    """Actually rerun each public cap, without oracle data as input."""
    rows={}
    for cap in CAPS:
        result=plan(case,cap=cap)
        success=result["found"]
        if success and result["words"]!=ref["words"]:
            raise ValueError("bounded public planner changed its deterministic shortest path")
        if success != (ref["expansions"]<=cap):
            raise ValueError("bounded planner failed deterministic prefix/cap identity")
        rows[str(cap)]={"found":success,"expansions":result["expansions"],
                        "candidate_edges_enumerated":result["candidate_edges_enumerated"],
                        "planning_seconds":result["planning_seconds"],
                        "joint_work":(result["expansions"]+ref["rewrite_distance"]+1
                                      if success else None)}
    return rows


def training(repl,suite:Path,out:Path):
    if out.exists():
        raise ValueError("fresh immutable native train proof output required")
    backend=_native_backend(repl)
    cases,meta=load_split(suite,"train")
    out.mkdir(parents=True)
    episodes=[]
    items=[]
    receipts=[]
    for case in cases:
        ref,ep=_plan_reference(repl,case)
        items.append((case,"reference",ep["script"]))
        receipts.append({"root":case.root,"lineage":case.lineage,
                         "minimum_public_rewrite_count":ref["rewrite_distance"],
                         "native_tactic_count":ep["effort"],
                         "symbolic_expansions":ref["expansions"]})
        episodes.append(ep)
    compiler=compile_scripts(items,out)
    for ep in episodes:
        ep["standalone_verified"]=True
    _rows(out/"reference_episodes.jsonl",episodes)
    _rows(out/"root_census.jsonl",receipts)
    # Construct only teacher-selected positives. Other generated candidates
    # are unlabelled proposals, not native verified negative examples.
    frames=[]
    for case,ep in zip(cases,episodes):
        for event in ep["events"]:
            word=event["word_before"]
            moves=[m.tactic(word) for m in neighbors(word,case.equations)]
            if word==case.source:
                moves.append("exact w")
            if event["tactic"] not in moves:
                raise ValueError("reference action absent from model-free proposal menu")
            frames.append({"root":case.root,"lineage":case.lineage,"split":"train",
                           "equations":case.equations,"witness_index":case.source,
                           "current_index":word,"candidate_tactics":moves,
                           "selected_native_reference_action":event["tactic"],
                           "remaining_effort":BUDGET-event["unit"]+1,
                           "label_status":"selected_native_certified_path_only",
                           "public_state_sha256":event["state_before_sha256"]})
    _rows(out/"train_reference_frames.jsonl",frames)
    report={"schema":"lsa02k.native_equality_training_source.v1",
            "suite_manifest_sha256":sha256_file(suite/"manifest.json"),
            "backend":backend,"training_roots":len(cases),
            "reference_proofs_compiled":compiler["passed"],"reference_frames":len(frames),
            "actual_native_replays":len(episodes),
            "train_only":True,"no_candidate_negative_labels_claimed":True,
            "model_training_updates":0,"heldout_labels_opened":False,
            "artifacts":{name:sha256_file(out/name) for name in (
                "reference_episodes.jsonl","root_census.jsonl",
                "train_reference_frames.jsonl","standalone.lean","compiler_receipt.json",
                "compiler.log")}}
    _write(out/"manifest.json",report)
    return report


def _bootstrap(values):
    rng=random.Random(20261204)
    n=len(values)
    vals=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [vals[50],vals[1949]]


def calibration(repl,suite:Path,teacher:Path,out:Path):
    if out.exists():
        raise ValueError("fresh native equality development output required")
    backend=_native_backend(repl)
    cases,meta=load_split(suite,"calibration")
    train=json.loads((teacher/"manifest.json").read_text(encoding="utf-8"))
    if not (train.get("schema")=="lsa02k.native_equality_training_source.v1"
        and train.get("suite_manifest_sha256")==sha256_file(suite/"manifest.json")
        and train.get("training_roots")==64 and train.get("reference_proofs_compiled")==64
        and train.get("train_only") is True and train.get("heldout_labels_opened") is False):
        raise ValueError("native equality train reference custody missing")
    for name,sha in train["artifacts"].items():
        if name!=Path(name).name or sha256_file(teacher/name)!=sha:
            raise ValueError("native equality train artifact changed")
    out.mkdir(parents=True)
    records=[]
    all_episodes=[]
    items=[]
    # Public local/random live search happens before reference planning.
    for case in cases:
        arms=[run_live(repl,case,"hamming_dfs")]
        arms.extend(run_live(repl,case,"random_dfs",seed=s) for s in RANDOM_SEEDS)
        for ep in arms:
            if not replay(repl,case,ep):
                raise ValueError("independent native equality policy replay failed")
            if ep["live_kernel"]:
                label=ep["policy"] if ep["seed"] is None else f"random_{ep['seed']}"
                items.append((case,label,ep["script"]))
        all_episodes.extend(arms)
        records.append({"case":case,"arms":arms})
    # The public exact planner is a separate strong comparator. Its actual
    # expansion/edge cost is reported independently of Lean tactic attempts.
    for rec in records:
        case=rec["case"]
        ref,ep=_plan_reference(repl,case)
        rec["reference"]=ref
        rec["reference_episode"]=ep
        rec["capped_planners"]=_verify_bounded_plans(case,ref)
        items.append((case,"bidirectional",ep["script"]))
        all_episodes.append(ep)
    receipt=compile_scripts(items,out)
    for ep in all_episodes:
        ep["standalone_verified"]=ep["live_kernel"]
    root_rows=[]
    for rec in records:
        case,ref,ep=rec["case"],rec["reference"],rec["reference_episode"]
        arms=rec["arms"]
        hamming=next(a for a in arms if a["policy"]=="hamming_dfs")
        uniform=[a for a in arms if a["policy"]=="random_dfs"]
        root_rows.append({"root":case.root,"lineage":case.lineage,
                          "word_length":len(case.source),
                          "reference_rewrite_count":ref["rewrite_distance"],
                          "reference_native_tactics":ep["effort"],
                          "reference_certified":ep["standalone_verified"],
                          "reference_expansions":ref["expansions"],
                          "reference_edges":ref["candidate_edges_enumerated"],
                          "reference_planning_seconds":ref["planning_seconds"],
                          "hamming_closed":hamming["standalone_verified"],
                          "hamming_effort":hamming["effort"],
                          "hamming_backtracks":hamming["backtracks"],
                          "uniform_closures":[int(a["standalone_verified"]) for a in uniform],
                          "uniform_mean":sum(a["standalone_verified"] for a in uniform)/len(uniform),
                          "uniform_effort":sum(a["effort"] for a in uniform),
                          "uniform_backtracks":sum(a["backtracks"] for a in uniform),
                          "capped_planners":rec["capped_planners"]})
    n=len(root_rows)
    caps={}
    for cap in CAPS:
        caps[str(cap)]={"certified_closures_at_planning_cap":sum(
            r["capped_planners"][str(cap)]["found"] and r["reference_certified"]
            for r in root_rows),
            "certified_closures_at_joint_cap":sum(
                r["reference_certified"] and r["capped_planners"][str(cap)]["found"]
                and r["capped_planners"][str(cap)]["joint_work"]<=cap
                for r in root_rows),
            "total_expanded_states":sum(r["capped_planners"][str(cap)]["expansions"] for r in root_rows),
            "total_enumerated_candidate_edges":sum(r["capped_planners"][str(cap)]["candidate_edges_enumerated"] for r in root_rows)}
    byband={}
    for a,b in ((4,5),(7,8),(10,11)):
        band=[r for r in root_rows if a<=r["reference_rewrite_count"]<=b]
        byband[f"{a}-{b}"]={"roots":len(band),"hamming_closures":sum(r["hamming_closed"] for r in band),
                           "uniform_mean":sum(r["uniform_mean"] for r in band)/max(1,len(band)),
                           "exact_bfs_closures":sum(r["reference_certified"] for r in band),
                           "cap64_closures":sum(r["capped_planners"]["64"]["found"] for r in band),
                           "mean_reference_expansions":sum(r["reference_expansions"] for r in band)/max(1,len(band))}
    _rows(out/"native_episodes.jsonl",all_episodes)
    _rows(out/"root_results.jsonl",root_rows)
    report={"schema":SCHEMA,"suite_manifest_sha256":sha256_file(suite/"manifest.json"),
            "train_source_sha256":sha256_file(teacher/"manifest.json"),
            "backend":backend,"independent_calibration_roots":n,"work_budget":BUDGET,
            "native_cost_definition":"one unit per rw, exact witness, or charged rollback",
            "planner_cost_definition":"one expansion per public word neighbor enumeration; edges separate",
            "symbolic_planning_cost_not_equated_to_native_runtime":True,
            "reference_proofs":sum(r["reference_certified"] for r in root_rows),
            "reference_rewrite_counts":[r["reference_rewrite_count"] for r in root_rows],
            "reference_mean_planner_expansions":sum(r["reference_expansions"] for r in root_rows)/n,
            "reference_max_planner_expansions":max(r["reference_expansions"] for r in root_rows),
            "hamming_dfs_closures":sum(r["hamming_closed"] for r in root_rows),
            "hamming_dfs_effort":sum(r["hamming_effort"] for r in root_rows),
            "uniform_dfs_closures":sum(sum(r["uniform_closures"]) for r in root_rows),
            "uniform_dfs_episodes":n*len(RANDOM_SEEDS),"uniform_seeds":list(RANDOM_SEEDS),
            "uniform_dfs_effort":sum(r["uniform_effort"] for r in root_rows),
            "uniform_dfs_backtracks":sum(r["uniform_backtracks"] for r in root_rows),
            "hamming_minus_uniform_root_bootstrap95_descriptive":_bootstrap(
                [int(r["hamming_closed"])-r["uniform_mean"] for r in root_rows]),
            "public_bidirectional_caps":caps,"distance_bands":byband,
            "headroom_at_joint_64":caps["64"]["certified_closures_at_joint_cap"]<=n*0.8,
            "exact_classical_solver_saturates":sum(r["reference_certified"] for r in root_rows)==n,
            "standalone_compiled_proofs":receipt["passed"],
            "all_episodes_replayed":len(all_episodes),
            "model_training_updates":0,"native_model_calls":0,
            "locked_ordinary_ood_contents_parsed":False,"scientific_eligible":False,
            "architecture_promotion_eligible":False,
            "artifacts":{name:sha256_file(out/name) for name in (
                "native_episodes.jsonl","root_results.jsonl","standalone.lean",
                "compiler_receipt.json","compiler.log")}}
    _write(out/"manifest.json",report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("train","calibration"))
    p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--teacher",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    with LeanRepl(imports=("Lean",),timeout=60,exe=config.REPL_EXE,
                  cwd=config.LEAN_DIR,env=config.lean_env()) as repl:
        if a.command=="train":
            r=training(repl,a.suite,a.out)
        else:
            if a.teacher is None:
                raise ValueError("calibration needs immutable native train source")
            r=calibration(repl,a.suite,a.teacher,a.out)
    print(canonical(r))
    return r


if __name__=="__main__":
    main()
