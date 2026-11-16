"""AF-28 source-verified eight-seed prospective STOP calibration verdict.

All eight source/frozen-model/fit/heldout/audit receipts required. The only
registered positive result is the prospective two-depth CAL-vs-DD criterion.
THR and the public symbolic solvers remain complete comparison controls.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from statistics import median

from . import gym,calibrate,evaluate as ex

REQUIRED=6
MIN_NEGATIVE_GAIN=7
MAX_POSITIVE_LOSS=4
KEYS=("cases","provable","unprovable","solved",
      "first_stop_positive","premature_stop_positive",
      "correct_initial_stop_negative","false_accepted_proofs",
      "neural_model_nodes","symbolic_expansions",
      "symbolic_rule_examinations")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def paired(a,b):
    if set(a)!=set(b):
        raise ValueError("AF28 paired case roster differs")
    return {
        "a_only":sum(a[k] and not b[k] for k in a),
        "b_only":sum(b[k] and not a[k] for k in a),
        "both":sum(a[k] and b[k] for k in a),
        "neither":sum(not a[k] and not b[k] for k in a),
        "n":len(a)
    }


def calibration_score(rows,arm):
    name=f"{arm}_probability"
    B=sum((float(r[name])-int(r["actual_STOP_label"]))**2
          for r in rows)/len(rows)
    logloss=0
    import math
    for r in rows:
        p=min(1-1e-12,max(1e-12,float(r[name])))
        y=int(r["actual_STOP_label"])
        logloss-=y*math.log(p)+(1-y)*math.log(1-p)
    logloss/=len(rows)
    ece=0
    bins=[]
    for i in range(10):
        cases=[r for r in rows if min(int(10*r[name]),9)==i]
        if not cases:
            bins.append({"bin":i,"n":0})
            continue
        confidence=sum(r[name] for r in cases)/len(cases)
        accuracy=sum(r["actual_STOP_label"] for r in cases)/len(cases)
        ece+=len(cases)/len(rows)*abs(confidence-accuracy)
        bins.append({"bin":i,"n":len(cases),"mean_probability":confidence,
                     "observed_stop_rate":accuracy})
    return {
        "n":len(rows),"brier":B,"logloss":logloss,
        "ECE_10bins":ece,"bins":bins
    }


def aggregate(root):
    root=Path(root)
    expected_source=calibrate.source_hashes()
    protocol=sha((calibrate.HERE/"PROTOCOL.json").read_bytes())
    manifest=gym.manifest()["manifest_sha256"]
    missing=[];seeds={}
    pooled={
        split:{arm:{k:0 for k in KEYS} for arm in ex.ARMS}
        for split in ex.SPLITS
    }
    family_results={
        split:{fam:{arm:{"solved":0,"provable":0,"unprovable":0,
                         "correct_initial_stop_negative":0}
               for arm in ex.ARMS}
               for fam in gym.TEMPLATES[split]}
        for split in ex.SPLITS
    }
    all_root={split:[] for split in ex.SPLITS}
    pairs={
        split:{
            key:{"a_only":0,"b_only":0,"both":0,"neither":0,"n":0}
            for key in (
                "CAL_vs_DD_positive","CAL_vs_DD_negative_STOP",
                "CAL_vs_THR_negative_STOP",
            )
        } for split in ex.SPLITS
    }
    for seed in gym.SEEDS:
        folder=root/f"af28-seed-{seed}"
        calfolder=folder/"calibration"
        ev=folder/"evaluation.json"
        audited=folder/"independent_audit.json"
        episodes=folder/"episodes.jsonl"
        roots=folder/"root_probabilities.jsonl"
        fit=calfolder/"calibrator.json"
        if not all(p.is_file() for p in (ev,audited,episodes,roots,fit)):
            missing.append(seed)
            continue
        r=json.loads(ev.read_text())
        a=json.loads(audited.read_text())
        c=json.loads(fit.read_text())
        if (r["experiment"]!="AF-28" or r["seed"]!=seed or r["smoke"]
            or r["training_updates_source_model"]!=0
            or r["all_arms"]!=list(ex.ARMS)
            or r["source_hashes"]!=expected_source
            or c["source_hashes"]!=expected_source
            or a["source_hashes"]!=expected_source
            or r["protocol_sha256"]!=protocol
            or c["source_protocol_sha256"]!=protocol
            or r["manifest_sha256"]!=manifest
            or c["source_manifest_sha256"]!=manifest
            or a["manifest_sha256"]!=manifest
            or r["calibrator_receipt_sha256"]!=sha(fit.read_bytes())
            or a["calibrator_receipt_sha256"]!=r["calibrator_receipt_sha256"]
            or r["episodes_sha256"]!=sha(episodes.read_bytes())
            or r["held_root_sha256"]!=sha(roots.read_bytes())
            or a["episodes_sha256"]!=r["episodes_sha256"]
            or a["held_root_sha256"]!=r["held_root_sha256"]
            or a["evaluation_sha256"]!=sha(ev.read_bytes())
            or a["original_checkpoint"]!=r["original_af23_checkpoint"]
            or c["original_af23_checkpoint"]!=r["original_af23_checkpoint"]
            or not a["verified"] or a["false_accepted_proofs"]!=0
            or a["audited_episodes"]!=192*2*len(ex.ARMS)
            or a["heldout_initial_probability_receipts"]!=384
            or c["heldout_labels_seen"] or c["heldout_source_features_seen"]):
            raise ValueError(f"AF28 source/mode/calibration audit discrepancy seed {seed}")
        rows=[json.loads(x) for x in episodes.read_text().splitlines()]
        rootrows=[json.loads(x) for x in roots.read_text().splitlines()]
        if len(rows)!=a["audited_episodes"] or len(rootrows)!=384:
            raise ValueError("AF28 original full case/held probability count missing")
        index={(row["split"],row["mode"],row["case_id"]):row for row in rows}
        if len(index)!=len(rows):
            raise ValueError("repeated original case/mode/physical source receipt")
        passed={}
        for split in ex.SPLITS:
            scores=r["scores"][split]
            for arm in ex.ARMS:
                data=scores[arm]
                if (data["cases"],data["provable"],data["unprovable"])!=(192,80,112):
                    raise ValueError("full original positive/negative cohort altered")
                for k in KEYS:
                    pooled[split][arm][k]+=data[k]
            dd=scores["DD"]
            cal=scores["CAL"]
            thr=scores["THR"]
            passed[split]={
                "registered_pass":bool(
                    cal["correct_initial_stop_negative"]>=
                        dd["correct_initial_stop_negative"]+MIN_NEGATIVE_GAIN
                    and cal["solved"]>=dd["solved"]-MAX_POSITIVE_LOSS
                    and cal["false_accepted_proofs"]==0
                    and dd["false_accepted_proofs"]==0
                ),
                "CAL_solved":cal["solved"],"DD_solved":dd["solved"],
                "THR_solved":thr["solved"],
                "CAL_negative_correct_STOP":cal["correct_initial_stop_negative"],
                "DD_negative_correct_STOP":dd["correct_initial_stop_negative"],
                "THR_negative_correct_STOP":thr["correct_initial_stop_negative"],
                "CAL_minus_DD_positive":cal["solved"]-dd["solved"],
                "CAL_minus_DD_negative_STOP":cal["correct_initial_stop_negative"]-
                                             dd["correct_initial_stop_negative"],
            }
            for fam in gym.TEMPLATES[split]:
                cs=[v for v in gym.cases_for(split) if v.family==fam]
                for arm in ex.ARMS:
                    source=[index[(split,arm,c.id)] for c in cs]
                    item=family_results[split][fam][arm]
                    item["solved"]+=sum(v["solved"] for v in source)
                    item["provable"]+=sum(v["goal_truth"] for v in source)
                    item["unprovable"]+=sum(not v["goal_truth"] for v in source)
                    item["correct_initial_stop_negative"]+=sum(
                        v["initial_stop"] and not v["goal_truth"] for v in source
                    )
            for aarm,barm,category in (
                ("CAL","DD","positive"),("CAL","DD","negative_STOP"),
                ("CAL","THR","negative_STOP"),
            ):
                if category=="positive":
                    source=[c for c in gym.cases_for(split) if c.designed_positive]
                    p1={c.id:bool(index[(split,aarm,c.id)]["solved"]) for c in source}
                    p2={c.id:bool(index[(split,barm,c.id)]["solved"]) for c in source}
                else:
                    source=[c for c in gym.cases_for(split) if not c.designed_positive]
                    p1={c.id:bool(index[(split,aarm,c.id)]["initial_stop"]) for c in source}
                    p2={c.id:bool(index[(split,barm,c.id)]["initial_stop"]) for c in source}
                pair=paired(p1,p2)
                label=f"{aarm}_vs_{barm}_{category}"
                for k,v in pair.items():
                    pairs[split][label][k]+=v
            held=[row for row in rootrows if row["split"]==split]
            if len(held)!=192:
                raise ValueError("heldout root probability split incomplete")
            all_root[split].extend(held)
            passed[split]["calibration_metrics"]={
                arm:calibration_score(held,arm) for arm in ("CAL","THR")
            }
        seeds[str(seed)]={
            "both_depths_pass":all(p["registered_pass"] for p in passed.values()),
            "depths":passed,
            "calibrators":{
                arm:{
                    "threshold":c["classifiers"][arm]["dev"]["chosen_threshold"],
                    "train_feature_indices":c["classifiers"][arm]["feature_indices"],
                    "weights":c["classifiers"][arm]["weights_bias_first"],
                    "dev_positive_continue":c["classifiers"][arm]["dev"]["dev_chosen_positive_continue"],
                    "dev_negative_stop":c["classifiers"][arm]["dev"]["dev_chosen_correct_negative_STOP"],
                } for arm in ("CAL","THR")
            },
            "audited_episodes":a["audited_episodes"],
            "source_neural_nodes":a["independent_neural_node_receipts"],
            "checkpoint_forward_replays":a["actual_frozen_checkpoint_forward_replays"],
            "train_dev_source_replays":a["train_dev_source_feature_replays"],
            "original_checkpoint":r["original_af23_checkpoint"],
        }
    winner=sum(s["both_depths_pass"] for s in seeds.values())
    verdict=("INCONCLUSIVE" if missing else
             "CALIBRATED_STOP_TRANSFER_PASS" if winner>=REQUIRED else
             "CALIBRATED_STOP_TRANSFER_FAIL")
    # Secondary threshold-only control can itself be stronger on exact same
    # heldout cases; this never silently changes the primary registration.
    thr_vs_cal={
        split:{
            "CAL_negative_STOP":pooled[split]["CAL"]["correct_initial_stop_negative"],
            "THR_negative_STOP":pooled[split]["THR"]["correct_initial_stop_negative"],
            "CAL_positive_solved":pooled[split]["CAL"]["solved"],
            "THR_positive_solved":pooled[split]["THR"]["solved"],
            "multifeature_only_benefit_supported":(
                pooled[split]["CAL"]["correct_initial_stop_negative"]>
                pooled[split]["THR"]["correct_initial_stop_negative"]
                and pooled[split]["CAL"]["solved"]>=
                    pooled[split]["THR"]["solved"]-32
            ),
        } for split in ex.SPLITS
    }
    return {
        "experiment":"AF-28",
        "registered_scientific_verdict":verdict,
        "required_seeds_both_depths":REQUIRED,
        "both_depths_seed_pass":winner,
        "qualified_seeds":len(seeds),"missing_seeds":missing,
        "protocol_sha256":protocol,"source_manifest_sha256":manifest,
        "source_hashes":expected_source,
        "original_frozen_af23_run":43812708916,
        "parent_af27_verdict":"STOP_TRANSFER_FAIL",
        "per_seed":seeds,"pooled":pooled,
        "families":family_results,"paired":pairs,
        "root_initial_calibration":{
            split:{
                arm:calibration_score(all_root[split],arm)
                for arm in ("CAL","THR")
            } for split in ex.SPLITS
        },
        "THR_comparison":thr_vs_cal,
        "verification":{
            "audited_episodes":sum(s["audited_episodes"] for s in seeds.values()),
            "independent_model_nodes":sum(s["source_neural_nodes"] for s in seeds.values()),
            "checkpoint_forward_replays":sum(s["checkpoint_forward_replays"] for s in seeds.values()),
            "train_dev_feature_replays":sum(s["train_dev_source_replays"] for s in seeds.values()),
            "false_accepted_proofs":0,"original_checkpoint_optimizer_updates":0,
        },
        "scope":"Grammar-disjoint finite synthetic positive-Horn STOP calibration. No equal-FLOP inference or autonomous Lean theorem proving claim."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=aggregate(a.root)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "verdict":r["registered_scientific_verdict"],
        "passed":r["both_depths_seed_pass"],
        "qualified":r["qualified_seeds"],
        "missing":r["missing_seeds"],
    }))


if __name__=="__main__":
    main()
