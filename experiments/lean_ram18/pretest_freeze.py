"""Create a prospective checkpoint lock from RAM-18 TRAIN-only source receipts.

This command opens zero TEST modules. Its output is a *candidate lock* until
committed on the source branch and independently checked before TEST checkout.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/"config.json").read_text())
ARCHS=("loop64","flat71")
OBJECTIVES=("task_only","task_plus_structural")
SEEDS=(17,29,43)

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda:inp.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf8"))

def candidate(training:Path, third:Path)->dict:
    report=read_json(training/"training_report.json")
    gate=read_json(third/"independent_gate.json")
    if not (gate["status"]=="passed_compiler_replay_gate_admitted" and gate["gate_admitted"] is True):
        raise ValueError("source macro admission gate failed")
    if gate["source_heldout_test_accessed"] is not False:
        raise ValueError("TRAIN-only gate touched original TEST")
    if not (report["status"]=="complete" and report["fits"]==12 and
            report["whole_module_TEST_touched"] is False and
            report["heldout_proof_cases_scored"]==0):
        raise ValueError("twelve-fit training report incomplete or contaminated by TEST")
    if report["seeds"]!=list(SEEDS) or report["optimizer_updates_each"]!=400:
        raise ValueError("original paired-seed / update budget differs")
    if report["source_split"]["source_train_positive_sha256"]!=gate["validated_positive_data_sha256"]:
        raise ValueError("TRAIN supervised positive data SHA differs from third audit")
    heldout={f"MathCan/{x}.lean" for x in CONFIG["source_modules"]["test"]}
    if set(report["source_split"]["prospective_test_modules_excluded"])!=heldout:
        raise ValueError("historical RAM-08 task labels were not filtered by all TEST modules")
    expected={f"{arch}_{obj}_s{seed}" for arch in ARCHS for obj in OBJECTIVES for seed in SEEDS}
    if len(report["model_fits"])!=12:
        raise ValueError("missing original paired fit")
    checked={};starts={};sizes={}
    for row in report["model_fits"]:
        name=f"{row['architecture']}_{row['objective']}_s{row['seed']}"
        if name not in expected or name in checked:
            raise ValueError("duplicate or unregistered model fit")
        if row["training_steps"]!=400 or row["test_cases_scored"]!=0:
            raise ValueError("fit deviates from original optimizer or TEST freeze")
        initial=row["starting_state_sha256"]
        pair=(row["architecture"],row["seed"])
        if pair in starts and starts[pair]!=initial:
            raise ValueError("paired objective initial states do not match")
        starts[pair]=initial
        if len(initial)!=64:
            raise ValueError("invalid initial tensor fingerprint")
        sizes[row["architecture"]]=row["parameters"]
        expected_macro=2560 if row["objective"]=="task_plus_structural" else 0
        if row["macro_examples_sampled"]!=expected_macro:
            raise ValueError("registered 10 percent positive sampling changed")
        usage=row["macro_balance_receipt"]
        if not (usage["verified_positive_only"] and usage["total_draws"]==expected_macro):
            raise ValueError("nonverified or wrong-count curriculum actions")
        counts=list(usage["by_family"].values())
        if set(usage["by_family"])!=set(("nat_induction","simp_prior","rewrite_prior")):
            raise ValueError("wrong structural macro family roster")
        if max(counts)-min(counts)>1:
            raise ValueError("failed family-level balancing")
        for fam,mods in usage["by_module"].items():
            for mod in mods:
                if mod in heldout or mod.removeprefix("MathCan/").removesuffix(".lean") not in CONFIG["source_modules"]["train"]:
                    raise ValueError("heldout or unregistered module entered source curriculum")
        for label in ("best","last"):
            file=training/"fits"/name/(label+".pt")
            if not file.is_file() or digest(file)!=row[f"{label}_checkpoint_sha256"]:
                raise ValueError(f"checkpoint SHA mismatch: {name}:{label}")
        checked[name]=dict(architecture=row["architecture"],objective=row["objective"],
                           seed=row["seed"],best_step=row["best_step"],
                           best_validation_loss=row["best_validation_loss"],
                           parameters=row["parameters"],
                           starting_state_sha256=initial,
                           best_checkpoint=row["best_checkpoint"],
                           best_checkpoint_sha256=row["best_checkpoint_sha256"],
                           last_checkpoint_sha256=row["last_checkpoint_sha256"],
                           macro_example_count=row["macro_examples_sampled"])
    if set(checked)!=expected or set(starts)!={(a,s) for a in ARCHS for s in SEEDS}:
        raise ValueError("incomplete source-only paired experimental grid")
    if abs(sizes["loop64"]-sizes["flat71"])/sizes["loop64"]>.015:
        raise ValueError("recurrent and flat models differ by over 1.5 percent")
    return dict(schema="RAM18-pretest-checkpoint-candidate-v1",
        evidence_status="TRAIN-only candidate; commit before TEST name selection",
        all_twelve_models_sha256={k:v["best_checkpoint_sha256"] for k,v in sorted(checked.items())},
        all_best_and_last=checked,paired_initial_states={"_".join(map(str,k)):v for k,v in sorted(starts.items())},
        train_source_commit=CONFIG["train_source_commit"],
        test_source_commit_sealed=CONFIG["test_source_commit"],
        independently_triple_compiled_positive_sha256=gate["validated_positive_data_sha256"],
        original_training_report_sha256=digest(training/"training_report.json"),
        original_independent_gate_sha256=digest(third/"independent_gate.json"),
        old_task_compiler_dataset_sha256=CONFIG["original_ram08_transitions_sha256"],
        original_ram08_training_zip_sha256=CONFIG["original_ram08_zip_sha256"],
        test_modules=sorted(CONFIG["source_modules"]["test"]),
        original_test_goals_accessed=0, test_cases_scored=0,
        checkpoint_lock_committed=False)

def run(training:Path,third:Path,out:Path):
    if out.exists():raise FileExistsError("candidate lock must use a new output")
    result=candidate(training,third)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--training",required=True,type=Path)
    p.add_argument("--third",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    x=run(a.training,a.third,a.out)
    print("RAM18_PRETEST_CANDIDATE",len(x["all_twelve_models_sha256"]),"test_scored",x["test_cases_scored"])
