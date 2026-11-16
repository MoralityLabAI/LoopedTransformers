"""AF-27 eight-checkpoint source-bound STOP transfer and rule-gap aggregate.

This is a genuinely new-grammar evaluation with a frozen registered rule,
using exact per-case source and independent host audit receipts. Missing,
invalid, or partial seeds yield INCONCLUSIVE and cannot be silently excluded.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from statistics import median

from . import gym,evaluate as ex

REQUIRED=6
MIN_NEGATIVE_GAIN=7
MAX_POSITIVE_LOSS=4
PRIMARY=("dd","vd")
SUMMARY_KEYS=(
    "cases","provable","unprovable","solved",
    "initial_stop_provable","premature_stop_provable",
    "correct_initial_stop_unprovable","false_accepted_proofs",
    "neural_model_node_evaluations","symbolic_node_expansions",
    "symbolic_rule_examinations",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def matched_truth(a,b):
    if set(a)!=set(b):
        raise ValueError("paired source case IDs differ")
    return {
        "a_only":sum(bool(a[k]) and not bool(b[k]) for k in a),
        "b_only":sum(bool(b[k]) and not bool(a[k]) for k in a),
        "both":sum(bool(a[k]) and bool(b[k]) for k in a),
        "neither":sum(not bool(a[k]) and not bool(b[k]) for k in a),
        "paired_n":len(a),
    }


def gap_stats(rows):
    def packed(values):
        values=[x for x in values if x is not None]
        if not values:
            return {"n":0}
        return {
            "n":len(values),"median":median(values),
            "min":min(values),"max":max(values),
            "under_1e_5":sum(v<1e-5 for v in values),
            "under_0_01":sum(v<.01 for v in values),
        }
    return {
        "n":len(rows),
        "direct_top_optimal":sum(r["root_rule_gap"]["direct_top_rule_optimal"] for r in rows),
        "vi2_top_optimal":sum(r["root_rule_gap"]["vi2_top_rule_optimal"] for r in rows),
        "direct_optimal_vi2_nonoptimal":sum(
            r["root_rule_gap"]["direct_vs_vi2_regret_disagreement"] for r in rows
        ),
        "vi2_optimal_direct_nonoptimal":sum(
            r["root_rule_gap"]["vi2_vs_direct_regret_disagreement"] for r in rows
        ),
        "rank_disagreement":sum(
            not r["root_rule_gap"]["same_top_legal_rule"] for r in rows
        ),
        "interior_stop_bind_edges":sum(
            sum(p["interior_stop_binds"]
                for p in r["root_rule_gap"]["value_paths"].values())
            for r in rows
        ),
        "direct_margin":packed([r["root_rule_gap"]["direct"]["margin"] for r in rows]),
        "vi2_margin":packed([r["root_rule_gap"]["vi2"]["margin"] for r in rows]),
        "direct_stop_minus_best":packed([
            r["root_rule_gap"]["direct"]["stop_minus_top_rule"] for r in rows
        ]),
        "vi2_stop_minus_best":packed([
            r["root_rule_gap"]["vi2"]["stop_minus_top_rule"] for r in rows
        ]),
    }


def aggregate(root):
    root=Path(root)
    hashes=ex.source_hashes()
    protocol=sha((ex.HERE/"PROTOCOL.json").read_bytes())
    manifest=gym.manifest()["manifest_sha256"]
    missing=[];per_seed={}
    pooled={
        split:{arm:{k:0 for k in SUMMARY_KEYS} for arm in ex.ARMS}
        for split in gym.SPLITS
    }
    families={
        split:{fam:{arm:{"solved":0,"provable":0}
                    for arm in ex.ARMS} for fam in gym.FAMILIES}
        for split in gym.SPLITS
    }
    all_gaps={split:[] for split in gym.SPLITS}
    paired_by_depth={
        split:{
            "provable_dd_vs_vd":{"a_only":0,"b_only":0,
                                 "both":0,"neither":0,"paired_n":0},
            "unprovable_correct_stop_vd_vs_dd":{"a_only":0,"b_only":0,
                                "both":0,"neither":0,"paired_n":0},
        } for split in gym.SPLITS
    }
    for seed in gym.SEEDS:
        directory=root/f"af27-seed-{seed}"
        scorefile=directory/"evaluation.json"
        episodefile=directory/"episodes.jsonl"
        gapsfile=directory/"root_gaps.jsonl"
        auditfile=directory/"independent_audit.json"
        if not all(x.is_file() for x in
                   (scorefile,episodefile,gapsfile,auditfile)):
            missing.append(seed)
            continue
        record=json.loads(scorefile.read_text())
        verified=json.loads(auditfile.read_text())
        if (
            record["experiment"]!="AF-27" or record["seed"]!=seed
            or record["smoke"] or record["training_updates"]!=0
            or record["all_arms"]!=list(ex.ARMS)
            or record["source_hashes"]!=hashes
            or verified["source_hashes"]!=hashes
            or record["protocol_sha256"]!=protocol
            or record["structural_manifest_sha256"]!=manifest
            or verified["manifest_sha256"]!=manifest
            or record["episodes_sha256"]!=sha(episodefile.read_bytes())
            or record["root_gaps_sha256"]!=sha(gapsfile.read_bytes())
            or verified["evaluation_sha256"]!=sha(scorefile.read_bytes())
            or verified["episodes_sha256"]!=record["episodes_sha256"]
            or verified["root_gaps_sha256"]!=record["root_gaps_sha256"]
            or not verified["verified"] or verified["false_accepted_proofs"]!=0
            or record["original_af23_checkpoint"]!=verified["original_af23_checkpoint"]
            or record["episode_count"]!=192*2*len(ex.ARMS)
            or verified["audited_episodes"]!=record["episode_count"]
            or record["root_gap_count"]!=80*len(gym.SPLITS)
            or verified["root_gap_rows_checked"]!=record["root_gap_count"]
            or verified["sampled_original_torch_forwards"]<8
        ):
            raise ValueError(f"AF27 seed {seed} original source/audit evidence failed")
        episodes=[json.loads(line) for line in episodefile.read_text().splitlines()]
        gaprows=[json.loads(line) for line in gapsfile.read_text().splitlines()]
        if len(episodes)!=record["episode_count"] or len(gaprows)!=record["root_gap_count"]:
            raise ValueError("missing original AF27 full episodes or gap diagnostics")
        by_case={}
        for row in episodes:
            key=(row["split"],row["mode"],row["case_id"])
            if key in by_case:
                raise ValueError("duplicate independently audited episode")
            by_case[key]=row
            if row["goal_truth"]:
                family=families[row["split"]][row["family"]][row["mode"]]
                family["provable"]+=1
                family["solved"]+=int(row["solved"])
        seed_depth={}
        for split in gym.SPLITS:
            scores=record["scores"][split]
            for arm in ex.ARMS:
                v=scores[arm]
                if (v["cases"],v["provable"],v["unprovable"])!=(192,80,112):
                    raise ValueError("AF27 source case balance changed")
                for k in SUMMARY_KEYS:
                    pooled[split][arm][k]+=v[k]
            dd=scores["dd"]
            vd=scores["vd"]
            passed=bool(
                vd["correct_initial_stop_unprovable"] >=
                    dd["correct_initial_stop_unprovable"]+MIN_NEGATIVE_GAIN
                and vd["solved"]>=dd["solved"]-MAX_POSITIVE_LOSS
                and vd["false_accepted_proofs"]==0
                and dd["false_accepted_proofs"]==0
            )
            provable={a:{} for a in PRIMARY}
            negative_stop={a:{} for a in PRIMARY}
            for c in gym.cases_for(split):
                for a in PRIMARY:
                    row=by_case.get((split,a,c.id))
                    if row is None:
                        raise ValueError("missing exact matched source graph")
                    if c.designed_positive:
                        provable[a][c.id]=bool(row["solved"])
                    else:
                        negative_stop[a][c.id]=bool(row["initial_stop"])
            comparisons={
                "provable_dd_vs_vd":matched_truth(provable["dd"],provable["vd"]),
                "unprovable_correct_stop_vd_vs_dd":matched_truth(
                    negative_stop["vd"],negative_stop["dd"]
                ),
            }
            for name,part in comparisons.items():
                for k,v in part.items():
                    paired_by_depth[split][name][k]+=v
            seed_depth[split]={
                "registered_pass":passed,
                "dd_positive_solved":dd["solved"],
                "vd_positive_solved":vd["solved"],
                "dd_correct_negative_stop":dd["correct_initial_stop_unprovable"],
                "vd_correct_negative_stop":vd["correct_initial_stop_unprovable"],
                "positive_difference":vd["solved"]-dd["solved"],
                "negative_stop_difference":vd["correct_initial_stop_unprovable"]-
                                            dd["correct_initial_stop_unprovable"],
                "paired_transitions":comparisons,
            }
            current_gaps=[r for r in gaprows if r["split"]==split]
            if len(current_gaps)!=80:
                raise ValueError("incomplete per-depth source root gaps")
            all_gaps[split].extend(current_gaps)
        per_seed[str(seed)]={
            "both_depths_pass":all(x["registered_pass"] for x in seed_depth.values()),
            "depths":seed_depth,
            "original_af23_checkpoint":record["original_af23_checkpoint"],
            "audited_episodes":verified["audited_episodes"],
            "audited_neural_nodes":verified["audited_neural_nodes"],
            "sampled_torch_forwards":verified["sampled_original_torch_forwards"],
            "sampled_near_ties":verified["sampled_near_tie_action_replays"],
        }
    both=sum(s["both_depths_pass"] for s in per_seed.values())
    if missing:
        verdict="INCONCLUSIVE"
    elif both>=REQUIRED:
        verdict="STOP_TRANSFER_PASS"
    else:
        verdict="STOP_TRANSFER_FAIL"
    return {
        "experiment":"AF-27",
        "registered_scientific_verdict":verdict,
        "required_seeds":REQUIRED,
        "eligible_seeds":len(per_seed),
        "seeds_passing_both_depths":both,
        "missing_seeds":missing,
        "protocol_sha256":protocol,"manifest_sha256":manifest,
        "source_hashes":hashes,
        "original_frozen_af23_run":43812708916,
        "af26_parent_registered_outcome":"RULE_ONLY",
        "primary":"VD=VI2 STOP+Direct rank versus DD=Direct STOP+Direct rank",
        "per_seed":per_seed,
        "pooled":pooled,
        "family_results":families,
        "paired_by_depth":paired_by_depth,
        "root_gap_diagnostics":{
            split:gap_stats(all_gaps[split]) for split in gym.SPLITS
        },
        "verification":{
            "audited_proof_episodes":sum(x["audited_episodes"] for x in per_seed.values()),
            "reconstructed_neural_nodes":sum(x["audited_neural_nodes"] for x in per_seed.values()),
            "original_frozen_torch_replays":sum(x["sampled_torch_forwards"] for x in per_seed.values()),
            "near_tie_torch_replays":sum(x["sampled_near_ties"] for x in per_seed.values()),
            "false_accepted_proofs":0,
        },
        "scope":"Synthetic novel public Horn grammar only. Node-matched BFS matches evaluated-state count, not GPU FLOPs, CPU latency or algorithmic operations.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=aggregate(a.root)
    a.out.write_text(json.dumps(
        result,sort_keys=True,indent=2,allow_nan=False
    )+"\n")
    print(json.dumps({
        "verdict":result["registered_scientific_verdict"],
        "seed_pass":result["seeds_passing_both_depths"],
        "missing":result["missing_seeds"],
        "root_gap_n":{k:v["n"] for k,v in result["root_gap_diagnostics"].items()},
    }))


if __name__=="__main__":
    main()
