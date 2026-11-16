"""RAM-17 independent original-source third Lean replay for primary matrix.

Every accepted case is reconstructed from its original TEST source and action
script and recompiled in a fresh Lean 4.35 process. A partial scorer receipt
never receives proof credit. Invalid/timeout source cases remain in the record.
The registered claim is computed only from the complete 160-case primary matrix.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import harness
import analysis
from native_replay import render_source,axiom_inventory

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def audit(out:Path):
    if out.exists():raise FileExistsError("independent original-source audit requires a new output")
    if subprocess.check_output(["git","-C",str(harness.SRC),"rev-parse","HEAD"],
                               text=True).strip()!=harness.CONFIG["test_source_commit"]:
        raise ValueError("third Lean TEST checkout differs from frozen source revision")
    if not harness.LEAN.is_file():
        raise FileNotFoundError("Lean 4.35 third audit binary unavailable")
    if subprocess.check_output([str(harness.LEAN),"--version"],text=True).find("4.35.0")<0:
        raise ValueError("independent proof process is not Lean 4.35.0")
    cases=[(module,name) for module,(_,names) in harness.LOCKED.items() for name in names]
    locked=analysis._locked_roots(harness.LOCKED_SOURCE,harness.CONFIG)
    if set(cases)!=locked or len(cases)!=10:
        raise ValueError("TEST theorem selector changed from committed manifest")
    sources=harness.module_sources()
    # Prepare an independent destination only after all original case files are present.
    for index,(module,theorem) in enumerate(cases):
        for policy in harness.POLICIES:
            path=harness.OUT/"primary"/f"{index:02d}"/policy/"case.json"
            if not path.is_file():
                raise FileNotFoundError("missing original primary case: "+str(path))
    out.mkdir(parents=True)
    rows=[]
    start=time.perf_counter()
    for index,(module,theorem) in enumerate(cases):
        source=sources[module]
        original_sha=sha(source.encode("utf8"))
        for policy in harness.POLICIES:
            folder=harness.OUT/"primary"/f"{index:02d}"/policy
            receipt=json.loads((folder/"case.json").read_text(encoding="utf8"))
            if not (receipt["policy"]==policy and receipt["theorem"]==theorem and
                    receipt["original_source_sha256"]==original_sha and
                    receipt["budget"]==harness.PRIMARY):
                raise ValueError("source/case/policy/budget changed in original scorer receipt")
            calls=receipt["logical_evaluations"]
            if not isinstance(calls,int) or calls<0 or calls>harness.PRIMARY["evaluations"]:
                raise ValueError("score exceeded original registered request budget")
            if receipt["grammar_receipt"]["theorem"]!=theorem or receipt["grammar_receipt"]["ablation_original_20"] is not False:
                raise ValueError("wrong grammar or source in original primary")
            valid=bool(receipt["accepted"])
            if valid and (receipt["reason"]!="independently_certified" or
                          receipt["elapsed_seconds"]>harness.PRIMARY["seconds"]):
                raise ValueError("late or uncertified source received original proof credit")
            line=dict(phase="primary",index=index,module=module,theorem=theorem,
                policy=policy,accepted=valid,
                logical_evaluations=calls,elapsed_seconds=receipt["elapsed_seconds"],
                reason=receipt["reason"],original_source_sha256=original_sha,
                original_candidate_action_count=len(receipt["calls"]),
                independent_third_valid=False,first_source_sha256=None,
                second_source_sha256=None,third_source_sha256=None,
                allowed_axioms=[])
            if valid:
                actions=receipt["actions"]
                if not actions or len(actions)>harness.PRIMARY["transitions"]:
                    raise ValueError("accepted original proof outside registered transition cap")
                harness.grammar.set_context(source,theorem,ablation=False)
                if not all(harness.grammar.valid_script(action) for action in actions):
                    raise ValueError("accepted script outside shared legal candidate grammar")
                first=receipt["certificate"];second=receipt["replay"]
                if not(first and second and first["accepted"] and second["accepted"]):
                    raise ValueError("accepted proof lacks original paired compiler certificates")
                one=(folder/"certificate.lean").read_bytes()
                two=(folder/"independent_replay.lean").read_bytes()
                rebuilt=render_source(source,theorem,actions).encode("utf8")
                if one!=two or one!=rebuilt:
                    raise ValueError("original reference theorem proof or certificates differ")
                certsha=sha(one)
                if certsha!=first["source_sha256"] or certsha!=second["source_sha256"]:
                    raise ValueError("paired original compiler source checksum mismatch")
                proof_dir=out/"third_sources"/f"{index:02d}"/policy
                proof_dir.mkdir(parents=True)
                third_file=proof_dir/"third.lean";third_file.write_bytes(rebuilt)
                result=subprocess.run([str(harness.LEAN),str(third_file.resolve())],
                    cwd=harness.SRC,env=harness.make_env(),
                    capture_output=True,text=True,timeout=30,check=False)
                log=result.stdout+result.stderr
                (proof_dir/"third.log").write_text(log,encoding="utf8")
                gate,axioms,why=axiom_inventory(log)
                if result.returncode!=0 or not gate or "error:" in log or "sorryAx" in log:
                    raise AssertionError("independent original Lean theorem rejected: "+why+log[-600:])
                if axioms!=first["axioms"] or axioms!=second["axioms"]:
                    raise AssertionError("transitive theorem axioms changed between original and third source")
                line.update(first_source_sha256=certsha,
                    second_source_sha256=certsha,third_source_sha256=sha(third_file.read_bytes()),
                    third_compiler_log_sha256=sha(log.encode("utf8")),
                    allowed_axioms=axioms,independent_third_valid=True,
                    structural_script_used=any(
                        not harness.grammar.OLD.valid_script(s) for s in actions))
                (proof_dir/"third_receipt.json").write_text(json.dumps(line,sort_keys=True,indent=2)+"\n")
            rows.append(line)
            print("RAM17_THIRD_PRIMARY",index,policy,int(valid),flush=True)
    if len(rows)!=160:
        raise AssertionError("incomplete 16-controller x 10-case original primary")
    registered=analysis.analyse(rows,harness.LOCKED_SOURCE,harness.CONFIG)
    registered.update(independent_audit_duration_seconds=round(time.perf_counter()-start,3),
       source_mode="entire original theorem and module reconstructed for third Lean compile",
       original_test_untouched_by_training=True)
    (out/"RAM17_PRIMARY_THIRD_AUDIT.jsonl").write_text(
        "".join(json.dumps(x,sort_keys=True)+"\n" for x in rows),encoding="utf8")
    (out/"RAM17_REGISTERED_RESULT.json").write_text(
        json.dumps(registered,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("RAM17_THIRD_ALL_CASES",registered["cases"],"certified",
          sum(int(x["accepted"]) for x in rows),"criterion",
          registered["registered_recurrent_specific_criterion_passed"])
    return registered

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    audit(args.out.resolve())
