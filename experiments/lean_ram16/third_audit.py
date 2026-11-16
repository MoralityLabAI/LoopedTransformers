"""RAM-16 independent authentic third Lean4.35 train/DEV audit and gate.

A standalone source-only verification job. Reconstruct every first attempt from
the original MathCan bytes and frozen source-header-only grammar. Third-compile
all double-certified complete proofs in fresh Lean processes, re-observe each
unique positive root goal and verify source/axiom byte identity. Do not open
prospective TEST modules or award labels to uncompleted macros.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,re,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"lean_ram08"))
sys.path.insert(0,str(HERE))  # RAM16 collect.py takes precedence over RAM08 collect.py.
from runtime import ResourceGuard
from probe_session import Session
import grammar
from collect import protected_training_checkout,select_original_theorems,jsonwrite,jsonappend,sha_file
from native_replay import render_source,compile_once,axiom_inventory,sha


def verify_independent(mathcan:Path,first:Path,out:Path,lean:str):
    if out.exists():raise FileExistsError("independent audit output must be new")
    conf=protected_training_checkout(mathcan)
    initial=json.loads((first/"first_collection_receipt.json").read_text())
    if initial["status"]!="complete_pending_independent_third_audit":
        raise ValueError("original compiler collector did not complete every TRAIN/DEV module")
    original_rows=(first/"first_source_effects.jsonl").read_bytes()
    if hashlib.sha256(original_rows).hexdigest()!=initial["source_effects_sha256"]:
        raise AssertionError("original compiler observation rows changed")
    rows=[json.loads(line) for line in original_rows.decode().splitlines() if line.strip()]
    if len(rows)!=initial["candidate_compilations"]:
        raise AssertionError("first-run source attempt count mismatch")
    advertised=[json.loads(line) for line in
         (first/"two_compile_positive_candidates.jsonl").read_text().splitlines()
         if line.strip()]
    if len(advertised)!=initial["positive_source_twice_count"]:
        raise AssertionError("original announced positive script count mismatch")
    out.mkdir(parents=True)
    scratch=mathcan/"RAM16ThirdScratch"
    if scratch.exists():raise FileExistsError("same source already used for independent audit")
    scratch.mkdir()
    root_cache={}
    validated=[]
    status=dict(campaign="RAM16-new-structural-family-authentic-lean-source-gate",
        original_train_source=conf["train_source_commit"],phase="train_and_dev_only",
        original_first_source_receipt_sha256=sha_file(first/"first_collection_receipt.json"),
        first_effect_rows_sha256=initial["source_effects_sha256"],
        source_heldout_test_accessed=False,positive_label_types=["verified_source_closed"],
        unknown_actions_reclassified_as_negative=0,status="auditing",gate_admitted=False)
    jsonwrite(out/"independent_gate.json",status)
    start=time.perf_counter()
    with ResourceGuard(out/"resource_samples.jsonl") as guard:
        expected_counter=collections.Counter()
        for entry in rows:
            role=entry["role"]
            if role not in ("train","dev"):raise AssertionError("protected TEST source entered collection")
            name=entry["source_file"].removeprefix("MathCan/").removesuffix(".lean")
            if name not in conf["source_modules"][role]:
                raise AssertionError("not an original source-selected train root")
            path=mathcan/entry["source_file"]
            original_bytes=path.read_bytes()
            if grammar.git_blob(original_bytes)!=conf["source_modules"][role][name]:
                raise AssertionError("original train source Git blob mismatches")
            original=original_bytes.decode("utf8")
            theorems=select_original_theorems(original,conf["selection"][role+"_first"])
            theorem=entry["theorem"]
            if theorem not in theorems:raise AssertionError("not one of the frozen first-six source declarations")
            action=entry["candidate"]
            actions=grammar.generate(original,theorem)
            index=entry["action_index"]
            if index not in range(len(actions)) or action!=actions[index].as_dict():
                raise AssertionError("candidate action generated from reference proof or changed ordering")
            key=(entry["source_file"],theorem)
            case=first/"native_source_compiler_artifacts"/role/f'{entry["root_index"]:03d}'/f'{index:03d}'
            raw=render_source(original,theorem,[action["script"]]).encode()
            if sha(raw)!=entry["source_candidate_sha256"]:
                raise AssertionError("first source proof changed")
            original_first=entry["first_compiler"]
            if (case/"first.lean").read_bytes()!=raw or sha_file(case/"first.log")!=original_first["log_sha256"]:
                raise AssertionError("original compiler first source/log SHA mismatch")
            if not entry["full_source_positive"]:
                if entry["source_status"] not in ("compiler_timeout_unknown","unverified_nonclosure_censored"):
                    raise AssertionError("attempted to turn a censored result into a negative")
                if entry["second_compiler"] is not None and (case/"independent"/"second.lean").read_bytes()!=raw:
                    raise AssertionError("second partial original source modified")
                continue
            expected_counter[(role,theorem,entry["source_file"],action["family"])]+=1
            if original_first["status"]!="accepted" or entry["second_compiler"]["status"]!="accepted":
                raise AssertionError("first-run positive lacks two full source compiler passes")
            src2=case/"independent"/"second.lean"
            log2=case/"independent"/"second.log"
            if src2.read_bytes()!=raw or sha_file(log2)!=entry["second_compiler"]["log_sha256"]:
                raise AssertionError("second source/compilation differs")
            ok1,ax1,why1=axiom_inventory((case/"first.log").read_text())
            ok2,ax2,why2=axiom_inventory(log2.read_text())
            if not(ok1 and ok2) or ax1!=ax2 or ax1!=original_first["axioms"]:
                raise AssertionError("previous compiler transitive axiom inventory invalid: "+why1+why2)
            if key not in root_cache:
                folder=out/"replayed_original_root_goals"/f"{len(root_cache):03d}"
                with Session(lean,original,theorem,folder,guard,timeout=15.) as sess:
                    native=sess.initial
                root_cache[key]=native
            state=root_cache[key]
            first_state=entry["original_before_state"]
            if (state["goals"],state["definitions"])!=(first_state["goals"],first_state["definitions"]):
                raise AssertionError("first native original source goal not reproduced by independent compiler")
            third=compile_once(mathcan,raw,scratch/role/f'{entry["root_index"]:03d}'/f'{index:03d}'/"third",15.0)
            if third["status"]!="accepted" or third["source_sha256"]!=sha(raw) or third["axioms"]!=ax1:
                raise AssertionError("third compiler failed original source/axiom proof")
            positive=dict(source_file=entry["source_file"],theorem=theorem,
                role=role,before=state,script=action["script"],family=action["family"],
                source_header_sha256=entry["original_target_statement_sha256"],
                original_root_source_sha256=entry["original_source_sha256"],
                complete_proof_source_sha256=sha(raw),
                first_log_sha256=original_first["log_sha256"],
                second_log_sha256=entry["second_compiler"]["log_sha256"],
                independent_third_log_sha256=third["log_sha256"],
                allowed_axioms=ax1,label=3,delta=1,
                genuine_original_compiler_goal=True,source_compiles=3,
                source_visible_witnesses=action["witnesses"],heldout_TEST_accessed=False)
            validated.append(positive)
            jsonappend(out/"independently_certified_positive_actions.jsonl",positive)
            print("RAM16_INDEPENDENT_THIRD",role,entry["source_file"],theorem,index,action["family"],flush=True)
        verified_keys={(x["source_file"],x["theorem"],x["script"]) for x in validated}
        if len(verified_keys)!=len(validated):
            raise AssertionError("duplicate original theorem/action positives")
        if len(validated)!=len(advertised):
            raise AssertionError("original first-run positive count differs after third audit")
        original_lookup={(x["source_file"],x["theorem"],x["action_script"]) for x in advertised}
        if original_lookup!=verified_keys:
            raise AssertionError("independent original positive script inventory does not match")
        train=[x for x in validated if x["role"]=="train"]
        roots_nat={(x["source_file"],x["theorem"]) for x in train if x["family"]=="nat_induction"}
        roots_prior={(x["source_file"],x["theorem"]) for x in train if x["family"]=="rewrite_prior"}
        all_roots={(x["source_file"],x["theorem"]) for x in train}
        gate=conf["gate"]
        admitted=(len(roots_nat)>=gate["min_distinct_nat_roots"]
              and len(roots_prior)>=gate["min_distinct_prior_roots"]
              and len(train)>=gate["min_new_macro_positives"]
              and len(all_roots)>=gate["min_distinct_train_positive_roots"])
        family=collections.Counter(x["family"] for x in train)
        gramian=sum(x["source_file"]=="MathCan/GramianTransfer.lean" for x in train)
        status.update(status="passed_compiler_replay_gate_admitted" if admitted else
                    "passed_compiler_replay_gate_failed",gate_admitted=admitted,
            all_triple_compiled_positives=len(validated),train_positives=len(train),
            dev_positives=len(validated)-len(train),train_distinct_positive_roots=len(all_roots),
            train_nat_induction_roots=len(roots_nat),train_prior_rewrite_roots=len(roots_prior),
            train_positive_family_counts=dict(family),gramian_train_positives=gramian,
            gramian_positive_fraction=(gramian/len(train)) if train else None,
            strict_original_provenance=True,actual_native_states_reobserved=len(root_cache),
            third_source_compilations=len(validated),
            validated_positive_data_sha256=sha_file(out/"independently_certified_positive_actions.jsonl")
               if validated else None,
            spent_seconds=round(time.perf_counter()-start,3),ram08_pef_cases=0,ram10_new_cases=0,
            unmet_gate_requirements={
                "nat_roots":max(0,gate["min_distinct_nat_roots"]-len(roots_nat)),
                "prior_roots":max(0,gate["min_distinct_prior_roots"]-len(roots_prior)),
                "positives":max(0,gate["min_new_macro_positives"]-len(train)),
                "distinct_roots":max(0,gate["min_distinct_train_positive_roots"]-len(all_roots))},
            resource_guard=guard.receipt())
        jsonwrite(out/"independent_gate.json",status)
        import shutil
        shutil.copytree(scratch,out/"third_compiler_sources_and_logs")
        print("RAM16_ADMISSION_GATE",json.dumps({k:v for k,v in status.items() if k!="resource_guard"},sort_keys=True),flush=True)
        if not admitted:
            return False
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mathcan",type=Path,required=True)
    p.add_argument("--first",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    admitted=verify_independent(a.mathcan.resolve(),a.first.resolve(),a.out.resolve(),a.lean)
    if not admitted:
        raise SystemExit("TRAIN data gate failed; prospective TEST remains sealed and no model may fit")
