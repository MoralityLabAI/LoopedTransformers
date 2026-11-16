"""Third compiler audit of the original frozen RAM-16 source-family proof matrix.

Reads only original first scored cases. Reconstruct every accepted source
byte-for-byte, independently rerun Lean4.35 and its transitive axiom gate.
No original scores or checkpoints are changed in this postscore audit.
"""
from __future__ import annotations
import collections,json,subprocess,time
from pathlib import Path
import harness
from native_replay import render_source,axiom_inventory

PHASES={
 "primary":harness.POLICIES,
 "secondary":harness.SECOND,
 "ablation":harness.ABLATED}
BUDGETS={
 "primary":harness.PRIMARY,
 "secondary":harness.SECONDARY,
 "ablation":harness.PRIMARY}

def audit():
    original_sources=harness.module_sources()
    cases=[(n,t) for n,(_,roots) in harness.LOCKED.items() for t in roots]
    if len(cases)!=10:raise AssertionError("source-registered test theorem list changed")
    if subprocess.check_output(["git","-C",str(harness.SRC),"rev-parse","HEAD"],
         text=True).strip()!=harness.CONFIG["test_source_commit"]:
        raise AssertionError("independent source checkout differs from original TEST revision")
    dest=harness.OUT/"independent_third"
    if dest.exists():raise FileExistsError("third-source audit output already exists")
    dest.mkdir(parents=True)
    all_rows=[]
    start=time.perf_counter()
    for phase,policies in PHASES.items():
        for index,(module,theorem) in enumerate(cases):
            for policy in policies:
                src=harness.OUT/phase/f"{index:02d}"/policy
                case_file=src/"case.json"
                if not case_file.is_file():
                    raise AssertionError("missing original first scored case "+str(case_file))
                result=json.loads(case_file.read_text())
                budget=BUDGETS[phase]
                if result["policy"]!=policy or result["theorem"]!=theorem or result["budget"]!=budget:
                    raise AssertionError("scorer metadata changed")
                if result["logical_evaluations"]>budget["evaluations"]:
                    raise AssertionError("test over registered tactic budget")
                if result["original_source_sha256"]!=harness.sha(original_sources[module].encode()):
                    raise AssertionError("source SHA mismatch")
                if result["accepted"] and (result["reason"]!="independently_certified" or
                                           result["elapsed_seconds"]>budget["seconds"]):
                    raise AssertionError("noncertified or late proof received credit")
                assert result["grammar_receipt"]["theorem"]==theorem
                assert result["grammar_receipt"]["ablation_original_20"]==(phase=="ablation")
                record=dict(phase=phase,index=index,module=module,theorem=theorem,
                    policy=policy,accepted=result["accepted"],reason=result["reason"],
                    calls=result["logical_evaluations"],wall_seconds=result["elapsed_seconds"],
                    original_actions=result["actions"],source_sha256=result["original_source_sha256"],
                    first_certified_source_sha256=(result["certificate"] or {}).get("source_sha256"))
                if result["accepted"]:
                    harness.grammar.set_context(original_sources[module],theorem,
                                                 ablation=(phase=="ablation"))
                    if not result["actions"] or len(result["actions"])>budget["transitions"]:
                        raise AssertionError("complete proof missing legal action sequence")
                    if not all(harness.grammar.valid_script(s) for s in result["actions"]):
                        raise AssertionError("complete proof used an unregistered action")
                    certificate=result["certificate"];replay=result["replay"]
                    if not (certificate and replay and certificate["accepted"] and replay["accepted"]):
                        raise AssertionError("paired in-budget source proof data absent")
                    raw=(src/"certificate.lean").read_bytes()
                    second=(src/"independent_replay.lean").read_bytes()
                    reconstructed=render_source(original_sources[module],theorem,result["actions"]).encode()
                    if raw!=second or raw!=reconstructed:
                        raise AssertionError("certified original target source bytes changed")
                    h=harness.sha(raw)
                    if h!=certificate["source_sha256"] or h!=replay["source_sha256"]:
                        raise AssertionError("first two original compiler SHA disagree")
                    out=dest/phase/f"{index:02d}"/policy
                    out.mkdir(parents=True)
                    source_file=out/"third.lean";source_file.write_bytes(raw)
                    process=subprocess.run([str(harness.LEAN),str(source_file.resolve())],
                        cwd=harness.SRC,env=harness.make_env(),
                        capture_output=True,text=True,timeout=30)
                    log=process.stdout+process.stderr
                    (out/"third.log").write_text(log,encoding="utf8")
                    gate,axioms,reason=axiom_inventory(log)
                    if process.returncode!=0 or not gate or "error:" in log or "sorryAx" in log:
                        raise AssertionError("third original Lean source rejected: "+reason+log[-200:])
                    if axioms!=certificate["axioms"] or axioms!=replay["axioms"]:
                        raise AssertionError("transitive axiom inventory changed between proof compilations")
                    record.update(third_source_accepted=True,complete_source_sha256=h,
                        third_compiler_log_sha256=harness.sha(log.encode()),allowed_axioms=axioms,
                        source_visible_structural_macro_used=any(
                            not harness.grammar.OLD.valid_script(s) for s in result["actions"]))
                    (out/"third_receipt.json").write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
                all_rows.append(record)
                print("RAM16_THIRD",phase,index,policy,int(result["accepted"]),flush=True)
    expected=10*sum(len(p) for p in PHASES.values())
    if len(all_rows)!=expected:raise AssertionError("original whole case matrix incomplete")
    scored={}
    for phase,policies in PHASES.items():
        scored[phase]={name:sum(int(x["accepted"]) for x in all_rows
                  if x["phase"]==phase and x["policy"]==name) for name in policies}
    success={}
    for row in all_rows:
        if row["phase"]!="primary" or not row["accepted"]:continue
        key=f'{row["module"]}:{row["theorem"]}'
        success.setdefault(key,set()).add(row["policy"])
    symbols={"native","frequency","shape_frequency"}
    gains=[];recurrence_specific=[]
    for case,policies in success.items():
        if "loop64_structural" in policies and not policies&symbols and "loop64_task_only" not in policies:
            gains.append(case)
            if "flat71_structural" not in policies:
                recurrence_specific.append(case)
    by_case={}
    for row in all_rows:
        if row["phase"] not in ("primary","ablation"):continue
        by_case.setdefault((row["module"],row["theorem"],row["policy"]),{})[row["phase"]]=row["accepted"]
    grammar_gains=[]
    for (module,theorem,policy),result in by_case.items():
        if "ablation" in result and result.get("primary") and not result["ablation"]:
            grammar_gains.append(f"{module}:{theorem}:{policy}")
    totals=collections.Counter((row["phase"],row["policy"]) for row in all_rows if row["accepted"])
    receipt=dict(schema="RAM16-first-prospective-23family-transfer",
        original_TEST_source_commit=harness.CONFIG["test_source_commit"],
        original_TRAIN_source_commit=harness.CONFIG["train_source_commit"],
        source_manifest_lock_sha=harness.LOCKED_SOURCE["selected_before_goal_state_access"],
        first_selected_checkpoint_lock=harness.FROZEN["verified_freeze_actions_run"],
        original_source_count=len(cases),
        total_policy_theorem_cases=expected,
        phase_case_counts={phase:len(cases)*len(policies) for phase,policies in PHASES.items()},
        accepted_first_original_source_cases=sum(x["accepted"] for x in all_rows),
        independent_third_compiler_certificates=sum(x.get("third_source_accepted",False) for x in all_rows),
        full_original_source_lean_compilations=3*sum(x["accepted"] for x in all_rows),
        false_certificates=0,
        certified_by_policy=scored,
        registered_structural_recurrent_only_theorems=sorted(gains),
        recurrence_specific_theorems=sorted(recurrence_specific),
        original_grammar_ablation_new_theorems=sorted(grammar_gains),
        original_certificates_using_structural_scripts=sum(x.get("source_visible_structural_macro_used",False)
                                                          for x in all_rows),
        registered_primary_criterion_passed=bool(gains),
        old_RAM08_PEF_cases_scored=0,original_RAM10_new_cases_scored=0,
        test_source_public_and_procedurally_heldout=True,
        elapsed_audit_seconds=round(time.perf_counter()-start,3),
        scientific_note="new module-disjoint theorem-family test, one time-new MathCan module; same concrete shared structural grammar and matched recurrent/flat budgets")
    (harness.OUT/"RAM16_CASES.jsonl").write_text(
        "".join(json.dumps(x,sort_keys=True)+"\n" for x in all_rows))
    (harness.OUT/"RAM16_RESULT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("RAM16_FINAL",json.dumps(receipt,sort_keys=True),flush=True)

if __name__=="__main__":
    audit()
