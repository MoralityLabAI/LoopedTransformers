"""RAM-06 separately marked, Lean-verified typed-exact auxiliary curriculum.

All labels are actual independent Lean4.35 tactic interpretations.
These 18 small synthetic theorems never count as original MathCan proofs.
Original 27-root compiler labels remain preserved in a distinct folder.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil

import ram06_core as c
from runtime import ResourceGuard,Session,ProbeTimeout

SOURCE="RAM06/ExactCurriculum.lean"
TRAIN=(
 "exact_train_prop","exact_train_prop_two","exact_train_nat_equality",
 "exact_train_int_symm","exact_train_nat_symm","exact_train_conj_left",
 "exact_train_conj_right","exact_train_implication",
 "exact_train_implication_reverse","exact_train_two_arguments",
 "exact_train_iff_forward","exact_train_iff_backward",
)
DEVELOPMENT=(
 "exact_dev_inequality","exact_dev_nat_reverse","exact_dev_conjunction",
 "exact_dev_forward","exact_dev_iff_reverse","exact_dev_two_arguments",
)
NEEDED_TRAIN_POSITIVE_EXACT=10
NEEDED_DEV_POSITIVE_EXACT=4
NEEDED_SPLIT_NEGATIVE_EXACT=4

def check_fixture(config:dict,source:Path):
    assert set(TRAIN).isdisjoint(DEVELOPMENT)
    assert len(TRAIN)==12 and len(DEVELOPMENT)==6
    declared=list(__import__("re").finditer(
        r"^theorem\s+([A-Za-z_0-9']+)",source.read_text(encoding="utf8"),
        __import__("re").M))
    assert tuple(x.group(1) for x in declared)==TRAIN+DEVELOPMENT
    assert config["curriculum"]=={
        "source_file":SOURCE,"train":list(TRAIN),
        "development":list(DEVELOPMENT)}
    assert all(SOURCE+":"+name in config["development"] for name in DEVELOPMENT)
    assert all(SOURCE+":"+name not in config["development"] for name in TRAIN)

def collect_and_replay(lean:str,fixture:Path,out:Path,config:dict,guard):
    check_fixture(config,fixture)
    original=fixture.read_text(encoding="utf8")
    rows=[];all_initial={}
    out.mkdir(parents=True,exist_ok=False)
    # Original reference fixture must compile with no sorry/admit.
    import subprocess
    check=subprocess.run([lean,str(fixture.resolve())],capture_output=True,
                         text=True,timeout=90)
    if check.returncode or "error:" in check.stdout+check.stderr or "sorryAx" in check.stdout+check.stderr:
        raise AssertionError("source exact curriculum did not compile as an original theorem fixture")
    for i,name in enumerate(TRAIN+DEVELOPMENT):
        with Session(lean,original,name,out/"collect"/f"{i:02d}",guard) as server:
            before=server.initial
            assert before["accepted"] and len(before["goals"])==1
            all_initial[name]=c.signature(before["goals"])
            family_scripts=c.candidates(before)
            assert any(f=="exact_typed" for f,s in family_scripts),name
            for family,script in family_scripts:
                try:
                    after=server.query([script],config["collector"]["request_seconds"])
                except ProbeTimeout:
                    c.append_json(out/"unknown.jsonl",dict(name=name,script=script,classification="unknown"))
                    continue
                row=dict(source_file=SOURCE,theorem=name,development=name in DEVELOPMENT,
                   prefix=[],before=before,after=after,family=family,script=script,
                   label=c.label(before,after),
                   delta=max(-3,min(3,len(before["goals"])-len(after.get("goals",[]))))
                         if after["accepted"] else 0,
                   state_sha256=c.signature(before["goals"]),
                   source_sha256=c.sha(original),
                   provenance="generated_verified_auxiliary")
                rows.append(row)
        print("RAM06_SYNTH_COLLECT",name,len(rows),flush=True)
    counts={}
    for split,names in (("train",set(TRAIN)),("development",set(DEVELOPMENT))):
        sub=[r for r in rows if r["theorem"] in names]
        wins=[r for r in sub if r["family"]=="exact_typed" and r["label"]>0]
        neg=[r for r in sub if r["family"]=="exact_typed" and r["label"]==0]
        counts[split]=dict(rows=len(sub),positive_typed=len(wins),
             positive_distinct_theorems=len({r["theorem"] for r in wins}),
             negative_typed=len(neg),
             labels=dict(Counter(r["label"] for r in sub)))
        assert len(wins)>=(NEEDED_TRAIN_POSITIVE_EXACT if split=="train" else NEEDED_DEV_POSITIVE_EXACT)
        assert len({r["theorem"] for r in wins})>=(NEEDED_TRAIN_POSITIVE_EXACT if split=="train" else NEEDED_DEV_POSITIVE_EXACT)
        assert len(neg)>=NEEDED_SPLIT_NEGATIVE_EXACT
    assert len(set(all_initial.values()))==18,"cross-split synthetic states collide"
    raw="".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows)
    (out/"exact_curriculum_actions.jsonl").write_text(raw,encoding="utf8")
    c.write_json(out/"collection_receipt.json",dict(source_file=SOURCE,
        fixture_sha256=c.digest(fixture),total_observations=len(rows),
        source_theorems=18,split_counts=counts,status="collected",
        raw_sha256=c.digest(out/"exact_curriculum_actions.jsonl")))
    # Independent Lean witness context, fresh server per theorem, never
    # reuse the collector's process, state, or its recorded acceptance.
    groups=defaultdict(list)
    for r in rows:groups[r["theorem"]].append(r)
    verified=0;problems=[]
    for i,name in enumerate(TRAIN+DEVELOPMENT):
        with Session(lean,original,name,out/"independent"/f"{i:02d}",guard) as independent:
            fresh=independent.initial
            if c.signature(fresh["goals"])!=all_initial[name]:
                problems.append([name,"root_state_signature_changed"])
            for row in groups[name]:
                actual=independent.query([row["script"]],config["collector"]["request_seconds"])
                if (actual["accepted"]!=row["after"]["accepted"] or
                    c.label(fresh,actual)!=row["label"] or
                    (actual["accepted"] and
                      c.signature(actual["goals"])!=c.signature(row["after"]["goals"]))):
                    problems.append([name,row["script"],"verdict_disagreement"])
                verified+=1
    report=dict(status="passed" if not problems else "failed",
         separately_generated=True,not_an_original_mathcan_proof=True,
         checked=verified,disagreements=problems,
         observations_sha256=c.digest(out/"exact_curriculum_actions.jsonl"),
         original_fixture_sha256=c.digest(fixture),
         split_counts=counts,number_independent_compiler_sessions=18)
    c.write_json(out/"independent_replay.json",report)
    if problems:raise AssertionError("synthetic train/dev compiler replay disagreed: "+repr(problems[:5]))
    print("RAM06_TYPED_EXACT_VERIFIED",json.dumps(report,sort_keys=True),flush=True)
    return rows,report


def combine(real:Path,synthetic:Path,out:Path,config_path:Path,config:dict):
    out.mkdir(parents=True,exist_ok=False)
    natural=c.read_rows(real/"transitions.jsonl")
    aux=c.read_rows(synthetic/"exact_curriculum_actions.jsonl")
    orig_collect=json.loads((real/"collection_report.json").read_text())
    orig_audit=json.loads((real/"independent_audit.json").read_text())
    synth_audit=json.loads((synthetic/"independent_replay.json").read_text())
    assert orig_audit["status"]==synth_audit["status"]=="passed"
    assert orig_collect["status"]=="complete"
    assert orig_audit["checked"]==len(natural)
    assert synth_audit["checked"]==len(aux)
    assert all(r["source_file"].startswith("MathCan/") for r in natural)
    assert all(r["source_file"]==SOURCE and r["provenance"]=="generated_verified_auxiliary" for r in aux)
    training_states={r["state_sha256"] for r in natural+aux if not r["development"]}
    development_states={r["state_sha256"] for r in natural+aux if r["development"]}
    if training_states & development_states:
        raise AssertionError("original/synthetic exact goal state crossover between train/dev")
    all_rows=natural+aux
    (out/"transitions.jsonl").write_text(
        "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in all_rows),encoding="utf8")
    combined_hash=c.digest(out/"transitions.jsonl")
    receipt=dict(status="complete",config_sha256=c.digest(config_path),
         dataset_sha256=combined_hash,source_only_natural=orig_collect,
         original_mathcan_rows=len(natural),
         new_verified_auxiliary_rows=len(aux),
         natural_data_sha256=c.digest(real/"transitions.jsonl"),
         synthetic_data_sha256=c.digest(synthetic/"exact_curriculum_actions.jsonl"),
         synthetic_independent_audit_sha256=c.digest(synthetic/"independent_replay.json"),
         model_training_contains_synthetic_examples=True,
         synthetic_results_excluded_from_original_theorem_coverage=True)
    c.write_json(out/"collection_report.json",receipt)
    c.write_json(out/"independent_audit.json",dict(
         status="passed",checked=len(all_rows),
         dataset_sha256=combined_hash,config_sha256=c.digest(config_path),
         original_real_lean_replays=len(natural),
         independent_synthetic_lean_replays=len(aux),
         original_mathcan_source_labels_immutable=True,
         auxiliary_curriculum_provenance="generated_verified_auxiliary"))
    print("RAM06_MERGED_VERIFIED_TRAIN",json.dumps({
        "natural_rows":len(natural),"synthetic_rows":len(aux),
        "total":len(all_rows),"sha256":combined_hash},sort_keys=True),flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real",type=Path,required=True)
    p.add_argument("--fixture",type=Path,default=Path(__file__).with_name("ExactCurriculum.lean"))
    p.add_argument("--lean",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--merged",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    a=p.parse_args()
    config=json.loads(a.config.read_text(encoding="utf8"))
    with ResourceGuard(a.out.parent/"ram06_exact_resource_log.jsonl") as guard:
        collect_and_replay(a.lean,a.fixture,a.out,config,guard)
        combine(a.real,a.out,a.merged,a.config,config)
if __name__=="__main__":main()
