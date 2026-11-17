"""AF-32 RETROSPECTIVE: inspect saved original source-locked STOP root scores.

Run this ONLY after the official scored experiment. No optimizer, new data,
threshold fitting, inference call, or source graph regeneration occurs here.
This cannot change the registered STRICT PASS / learned-STOP FAIL verdict.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

SEEDS=(17,29,43,61,73,89,103,131)
DEPTHS=("held_depth5","held_depth6")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def analyze_seed(folder,seed):
    folder=Path(folder)
    receipt=json.loads((folder/"evaluation.json").read_text())
    audit=json.loads((folder/"audit.json").read_text())
    stop=json.loads((folder/"stop_model.json").read_text())
    raw=(folder/"episodes.jsonl").read_bytes()
    if (receipt["experiment"]!="AF-32" or receipt["seed"]!=seed or
        audit["seed"]!=seed or stop["seed"]!=seed or
        receipt["episodes_sha256"]!=sha(raw) or
        audit["episodes_sha256"]!=sha(raw) or
        receipt["episodes"]!=7680 or
        audit["independently_replayed_episodes"]!=7680 or
        not audit["no_false_accepted_proofs"]):
        raise AssertionError("AF32 original source/episode/proof audit invalid")
    roots=defaultdict(dict)
    count=0
    for line in raw.decode().splitlines():
        r=json.loads(line)
        if r["budget_mode"]!="strict" or r["arm"]!="LRANK_STOP":
            continue
        count+=1
        if not r["history"] or not r["history"][0]["work"]:
            raise AssertionError("AF32 original STOP root work absent")
        w=r["history"][0]["work"]
        prob=w.get("learned_STOP_probability")
        threshold=w.get("learned_STOP_threshold")
        if not isinstance(prob,(int,float)) or not 0<=prob<=1 or threshold!=stop["dev_threshold"]:
            raise AssertionError("AF32 original learned-only STOP root probability invalid")
        key=(r["split"],r["pair_id"])
        label=bool(r["designed_reachable"])
        if label in roots[key]:
            raise AssertionError("AF32 duplicate source matched pair member")
        roots[key][label]={
            "prob":float(prob),
            "swap_position":r["swap_position"],
            "cycle_length":r["cycle_length"],
            "root_STOP":r["initial_stop"],
            "case_id":r["case_id"],
            "public_sha":r["public_graph_sha256"]
        }
    if count!=384 or len(roots)!=192:
        raise AssertionError("AF32 incomplete original paired STOP-root roster")
    results={}
    for split in DEPTHS:
        pairs=[v for (s,p),v in roots.items() if s==split]
        if len(pairs)!=96:
            raise AssertionError("AF32 STOP held split pair count changed")
        wins=ties=losses=0
        correct_neg=wrong_pos=0
        byswap={str(i):{"pairs":0,"correct_neg_STOP":0,"positive_false_STOP":0}
                for i in (1,2,3)}
        for pair in pairs:
            if set(pair)!={False,True}:
                raise AssertionError("AF32 positive/negative counterpart missing")
            neg,pos=pair[False],pair[True]
            if (neg["swap_position"]!=pos["swap_position"] or
                neg["cycle_length"]!=pos["cycle_length"] or
                neg["public_sha"]==pos["public_sha"]):
                raise AssertionError("source pair matcher or public graph hash invalid")
            gap=neg["prob"]-pos["prob"]
            if gap>1e-12:wins+=1
            elif gap< -1e-12:losses+=1
            else:ties+=1
            correct_neg+=int(neg["root_STOP"])
            wrong_pos+=int(pos["root_STOP"])
            b=byswap[str(pos["swap_position"])]
            b["pairs"]+=1
            b["correct_neg_STOP"]+=int(neg["root_STOP"])
            b["positive_false_STOP"]+=int(pos["root_STOP"])
        result={
            "pairs":len(pairs),"positive":96,"negative":96,
            "signed_negative_minus_positive_wins":wins,
            "ties":ties,"losses":losses,
            "paired_AUC":(wins+ties/2)/96,
            "correct_negative_immediate_STOP":correct_neg,
            "false_positive_immediate_STOP":wrong_pos,
            "by_swap_position":byswap,
        }
        official=receipt["scores"]["strict_"+split.removeprefix("held_")]["LRANK_STOP"]
        if (official["negative_correct_initial_stop"]!=correct_neg or
            official["positive_premature_initial_stop"]!=wrong_pos):
            raise AssertionError("post-hoc metric diverges from original audited STOP score")
        results[split]=result
    return {
        "seed":seed,"original_episodes_sha256":sha(raw),
        "original_STOP_model_sha256":sha((folder/"stop_model.json").read_bytes()),
        "DEV_frozen_threshold":stop["dev_threshold"],
        "DEV_correct_negative_immediate_STOP":stop["dev_choice"]["correct_negative_STOP"],
        "DEV_positive_root_continue":stop["dev_choice"]["provable_continued"],
        "held":results,
    }


def summarize(root,out):
    root,out=Path(root),Path(out)
    if out.exists():
        raise FileExistsError("immutable posthoc analysis destination exists")
    byseed={str(seed):analyze_seed(root/("af32-seed-"+str(seed)),seed)
            for seed in SEEDS}
    totals={}
    for split in DEPTHS:
        totals[split]={
            "paired_AUCs_by_seed":{s:x["held"][split]["paired_AUC"] for s,x in byseed.items()},
            "total_negative_initial_STOP":sum(x["held"][split]["correct_negative_immediate_STOP"]
                                              for x in byseed.values()),
            "total_false_positive_initial_STOP":sum(x["held"][split]["false_positive_immediate_STOP"]
                                                    for x in byseed.values()),
        }
    result={
        "experiment":"AF-32",
        "stage":"RETROSPECTIVE_ONLY_AFTER_ORIGINAL_SOURCE_LOCKED_SCIENTIFIC_SCORING",
        "original_scored_run":44043230390,
        "original_scored_commit":"21de716b0a370274cd482bee41e102069d4dd064",
        "official_registered_ranker_result":"STRICT_RULE_SELECTION_PASS_8_OF_8_UNCHANGED",
        "official_registered_learned_STOP_result":"LEARNED_STOP_STRICT_FAIL_0_OF_8_UNCHANGED",
        "source_rerun_or_new_fitting":False,
        "original_eight_seed_stop_receipts":byseed,
        "totals":totals,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=summarize(a.root,a.out)
    print(json.dumps({
        "stage":r["stage"],
        "thresholds":sorted({v["DEV_frozen_threshold"] for v in
                            r["original_eight_seed_stop_receipts"].values()}),
        "d5":r["totals"]["held_depth5"],
        "d6":r["totals"]["held_depth6"],
    }))


if __name__=="__main__":
    main()
