"""AF-33 RETROSPECTIVE original-receipt late STOP diagnosis.

Runs solely over archived original source-locked episodes and fitted HEAD receipts.
No model fitting, changed threshold, inference call or new source graph generation.
The three original registered scientific verdicts are immutable inputs here.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

SOURCE_RUN=44056762277
SOURCE_COMMIT="d4b0e70ee8db5a526901896a6db8bbabb3aee2fd"
SEEDS=(17,29,43,61,73,89,103,131)
ARMS=("BAL_STOP","J12_STOP")
HELD=("held_depth5","held_depth6")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def stat(values):
    if not values: return None
    return {
        "count":len(values),
        "minimum":min(values),"maximum":max(values),
        "mean":sum(values)/len(values),
    }


def analyze_seed(source,seed):
    source=Path(source)
    receipt,audit=load(source/"evaluation.json"),load(source/"audit.json")
    raw=(source/"episodes.jsonl").read_bytes()
    if (receipt["experiment"]!="AF-33" or receipt["seed"]!=seed or
        audit["seed"]!=seed or receipt["episodes"]!=8448 or
        audit["independently_replayed_episodes"]!=8448 or
        audit["episodes_sha256"]!=sha(raw) or
        receipt["episodes_sha256"]!=audit["episodes_sha256"] or
        not audit["no_false_accepted_proofs"] or
        not audit["independent_learned_stop_refit"] or
        not audit["all_action_decisions_recomputed"]):
        raise AssertionError("source AF33 scored episode/audit digest mismatch")
    heads={
        arm:load(source/("stop_"+arm.lower()+".json"))
        for arm in ARMS
    }
    for arm,m in heads.items():
        blob=(source/("stop_"+arm.lower()+".json")).read_bytes()
        if (m["seed"]!=seed or
            m["variant"]!=("BAL" if arm=="BAL_STOP" else "J12") or
            sha(blob)!=receipt["stop_models_sha256"][arm] or
            sha(blob)!=audit["stop_models_sha256"][arm]):
            raise AssertionError("frozen original STOP model identity/provenance changed")
    episode_rows=[json.loads(line) for line in raw.splitlines()]
    by_cohort=defaultdict(list)
    for r in episode_rows:
        if r["arm"] not in ARMS or r["budget_mode"]!="strict":
            continue
        if not r["history"]:
            raise AssertionError("AF33 source learned STOP episode lacks root trace")
        by_cohort[(r["split"],r["arm"])].append(r)
    if set(by_cohort)!={(split,arm) for split in HELD for arm in ARMS}:
        raise AssertionError("AF33 retrospective strict split/arm roster incomplete")
    groups={}
    for split in HELD:
        groups[split]={}
        for arm in ARMS:
            rows=by_cohort[(split,arm)]
            if len(rows)!=192:raise AssertionError("source held arms must include 192 complete cases")
            pos=[r for r in rows if r["designed_reachable"]]
            neg=[r for r in rows if not r["designed_reachable"]]
            if (len(pos),len(neg))!=(96,96):raise AssertionError("AF33 matched label count changed")
            stop_model=heads[arm]
            threshold=stop_model["dev_threshold"]
            positive_stop_steps=Counter()
            positive_prefix_probs=defaultdict(list)
            negative_root_probs=defaultdict(list)
            negative_root_stops=Counter()
            for r in pos:
                for step,h in enumerate(r["history"]):
                    prob=h["work"].get("learned_STOP_probability")
                    if prob is None or not 0<=prob<=1:
                        raise AssertionError("original positive STOP feature probability missing")
                    if h["work"].get("learned_STOP_threshold")!=threshold:
                        raise AssertionError("positive STOP threshold drift")
                    positive_prefix_probs[str(step)].append(prob)
                    if h["action"]==14:
                        positive_stop_steps[str(step)]+=1
            for r in neg:
                p=r["history"][0]["work"].get("learned_STOP_probability")
                if p is None or not 0<=p<=1:
                    raise AssertionError("original negative-root STOP probability missing")
                negative_root_probs[str(r["swap_position"])].append(p)
                if r["initial_stop"]:
                    negative_root_stops[str(r["swap_position"])]+=1
            expected=receipt["scores"]["strict_"+split.removeprefix("held_")][arm]
            if (sum(r["solved"] for r in pos)!=expected["positive_proofs"] or
                sum(r["initial_stop"] for r in neg)!=expected["negative_correct_initial_stop"] or
                sum(r["initial_stop"] for r in pos)!=expected["positive_premature_initial_stop"]):
                raise AssertionError("original registered source score mismatch")
            if sum(negative_root_stops.values())!=expected["negative_correct_initial_stop"]:
                raise AssertionError("posthoc negative STOP group count drift")
            groups[split][arm]={
                "positive_proofs":expected["positive_proofs"],
                "correct_negative_initial_STOP":expected["negative_correct_initial_stop"],
                "incorrect_positive_initial_STOP":expected["positive_premature_initial_stop"],
                "threshold_DEV_frozen":threshold,
                "DEV_selected_negative_STOP":stop_model["dev_choice"]["correct_negative_STOP"],
                "DEV_positive_continued":stop_model["dev_choice"]["provable_continued"],
                "positive_earliest_STOP_step_count":dict(positive_stop_steps),
                "positive_probabilities_by_step":{
                    k:stat(v) for k,v in sorted(positive_prefix_probs.items(),
                                               key=lambda x:int(x[0]))
                },
                "negative_root_probabilities_by_swap_position":{
                    k:stat(v) for k,v in sorted(negative_root_probs.items())
                },
                "correct_negative_stop_by_swap_position":{
                    str(j):negative_root_stops[str(j)] for j in (1,2,3)
                },
                "original_source_paired_AUC":expected["root_diagnostics"]["paired_AUC"],
                "original_source_Brier":expected["root_diagnostics"]["brier"],
            }
    return {
        "seed":seed,
        "original_episode_sha256":sha(raw),
        "original_model_sha256":receipt["model_sha256"],
        "frozen_STOP_models_sha256":receipt["stop_models_sha256"],
        "held":groups,
    }


def diagnose(root,out):
    root,out=Path(root),Path(out)
    if out.exists():raise FileExistsError("retrospective output already exists")
    byseed={str(seed):analyze_seed(root/f"af33-seed-{seed}",seed) for seed in SEEDS}
    summary={}
    for split in HELD:
        summary[split]={}
        for arm in ARMS:
            rows=[byseed[str(seed)]["held"][split][arm] for seed in SEEDS]
            all_steps=Counter()
            for row in rows:all_steps.update(row["positive_earliest_STOP_step_count"])
            summary[split][arm]={
                "completed_proofs_total":sum(x["positive_proofs"] for x in rows),
                "correct_negative_initial_STOP_total":sum(x["correct_negative_initial_STOP"] for x in rows),
                "incorrect_positive_initial_STOP_total":sum(x["incorrect_positive_initial_STOP"] for x in rows),
                "DEV_thresholds_by_seed":{str(seed):byseed[str(seed)]["held"][split][arm]["threshold_DEV_frozen"] for seed in SEEDS},
                "positive_earliest_STOP_steps_across_seeds":dict(all_steps),
                "held_root_AUC_by_seed":{str(seed):byseed[str(seed)]["held"][split][arm]["original_source_paired_AUC"] for seed in SEEDS},
            }
    result={
        "experiment":"AF-33",
        "stage":"RETROSPECTIVE_ONLY_AFTER_SOURCE_LOCKED_ORIGINAL_SCIENCE",
        "original_source_run":SOURCE_RUN,
        "original_source_commit":SOURCE_COMMIT,
        "fitted_models_rerun":False,
        "thresholds_changed":False,
        "heldout_episodes_regenerated":False,
        "original_position_balanced_result":"POSITION_BALANCED_STOP_TRANSFER_FAIL",
        "original_absolute_bal_result":"BAL_STOP_ABSOLUTE_FAIL",
        "original_strict_ranker_result":"STRICT_RANKER_SAFEGUARD_PASS",
        "total_original_learned_episodes_across_source_archive":8448*8,
        "original_source_checkpoint_seeds":list(SEEDS),
        "per_seed":byseed,
        "pooled":summary,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=diagnose(a.root,a.out)
    print(json.dumps({"stage":result["stage"],"pooled":result["pooled"]}))


if __name__=="__main__":
    main()
