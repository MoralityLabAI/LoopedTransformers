"""AF-26 source-bound eight-seed factorial attribution from audited receipts.

The outcome is a preregistered descriptive *in-gym* recovery classification,
not a new proof-search capability claim. All source-matched DD/VV results must
reproduce the original AF-25 official proof counts exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.activation_funnel_af25 import gym
from . import factor, evaluate as ev

TARGET_SPLIT="held_depth5"
TARGET_FAMILY="decoy_and_shared_overcount"
REQUIRED_SEEDS=tuple(gym.SEEDS)
METRICS=("solved","provable","unprovable","first_stop_positive",
         "later_premature_stop","early_stop_positive","first_stop_negative",
         "false_accepted_proofs","neural_forward_nodes","rule_actions")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def matched_pairs(a,b):
    """Paired proof completion among identical case keys of one original seed."""
    if set(a)!=set(b):
        raise ValueError("paired cases changed")
    return {
        "a_only":sum(a[k] and not b[k] for k in a),
        "b_only":sum(b[k] and not a[k] for k in a),
        "both":sum(a[k] and b[k] for k in a),
        "neither":sum(not a[k] and not b[k] for k in a),
        "paired_n":len(a),
    }


def aggregate(root, original_file):
    root=Path(root)
    original=ev.original_aggregate(original_file)
    source=ev.hashes()
    protocol=sha((ev.HERE/"PROTOCOL.json").read_bytes())
    manifest=gym.manifest()["manifest_sha256"]
    missing=[]
    per_seed={}
    pooled={split:{arm:{k:0 for k in METRICS} for arm in factor.ARMS}
            for split in gym.SPLITS}
    family={
        split:{fam:{arm:{"solved":0,"provable":0,
                         "first_stop_positive":0,
                         "later_premature_stop":0}
                         for arm in factor.ARMS}
               for fam in gym.FAMILIES}
        for split in gym.SPLITS
    }
    pooled_pairs={
        comparison:{"a_only":0,"b_only":0,"both":0,"neither":0,"paired_n":0}
        for comparison in ("dv_vs_vv","vd_vs_vv","dd_vs_vv","nd_vs_dd","nv_vs_vv")
    }
    for seed in REQUIRED_SEEDS:
        folder=root/f"af26-seed-{seed}"
        receipt_path=folder/"evaluation.json"
        episode_path=folder/"episodes.jsonl"
        audit_path=folder/"independent_audit.json"
        if not all(p.is_file() for p in (receipt_path,episode_path,audit_path)):
            missing.append(seed)
            continue
        rec=json.loads(receipt_path.read_text())
        audited=json.loads(audit_path.read_text())
        expected_source=original["per_seed"][str(seed)]
        if (rec["experiment"]!="AF-26" or rec["seed"]!=seed or rec["smoke"]
            or audited["seed"]!=seed or audited["smoke"] or not audited["verified"]
            or audited["false_accepted_proofs"]!=0
            or not audited["original_af25_direct_vi2_parity"]
            or rec["source_hashes"]!=source or audited["source_hashes"]!=source
            or rec["protocol_sha256"]!=protocol
            or rec["af25_aggregate_sha256"]!=ev.PARENT_AGG_SHA
            or rec["af25_structural_manifest_sha256"]!=manifest
            or audited["evaluation_sha256"]!=sha(receipt_path.read_bytes())
            or audited["episodes_sha256"]!=sha(episode_path.read_bytes())
            or rec["episodes_sha256"]!=audited["episodes_sha256"]
            or rec["training_updates"]!=0
            or rec["all_arms"]!=list(factor.ARMS)
            or audited["original_af23_checkpoint"]!=rec["original_af23_checkpoint"]
            or rec["original_af23_checkpoint"]["zip_sha256"]!=
                   expected_source["original_zip_sha256"]
            or audited["episodes_audited"]!=192*len(gym.SPLITS)*len(factor.ARMS)
            or rec.get("dd_vv_controls_source") !=
                "archived immutable AF25 proof episodes; not counterfactual rerolls"
            or rec.get("crossed_arms_source") !=
                "fresh full closed-loop checkpoint forwards"
            or audited.get("original_af25_control_receipt")!=
                rec.get("original_af25_control_receipt")
            or rec.get("original_af25_control_receipt",{}).get("artifact_zip_sha256")!=
                ev.AF25_EPISODE_ZIPS[seed]
            or audited.get("original_control_actions_exact",0)<=0):
            raise ValueError(f"AF26 seed {seed}: source/checkpoint/independent audit mismatch")
        ev.check_original_parity(rec["scores"],original,seed)
        rows=[json.loads(l) for l in episode_path.read_text().splitlines()]
        indices={}
        for row in rows:
            key=(row["split"],row["family"],row["mode"],row["case_id"])
            if key in indices:
                raise ValueError("duplicate seeded factorial case")
            indices[key]=row
        if len(indices)!=192*2*len(factor.ARMS):
            raise ValueError("incomplete registered factorial case set")
        per_family={}
        for split in gym.SPLITS:
            for arm in factor.ARMS:
                v=rec["scores"][split][arm]
                if v["provable"]!=80 or v["unprovable"]!=112:
                    raise ValueError("positive/negative case roster moved")
                for k in METRICS:
                    pooled[split][arm][k]+=v[k]
            for fam in gym.FAMILIES:
                cr=[c for c in gym.cases_for(split) if c.family==fam and c.designed_positive]
                if len(cr)!=(48 if fam==TARGET_FAMILY else 32):
                    raise ValueError("AF25 hard/easy case count changed")
                per_family[(split,fam)]={}
                for arm in factor.ARMS:
                    selected=[indices[(split,fam,arm,c.id)] for c in cr]
                    item={
                        "solved":sum(bool(row["solved"]) for row in selected),
                        "provable":len(selected),
                        "first_stop_positive":sum(bool(row["initial_stop"]) for row in selected),
                        "later_premature_stop":sum(bool(row["early_stop"] and
                              not row["initial_stop"]) for row in selected),
                    }
                    per_family[(split,fam)][arm]=item
                    for k in item:
                        family[split][fam][arm][k]+=item[k]
        hard_cases=[c for c in gym.cases_for(TARGET_SPLIT)
                    if c.family==TARGET_FAMILY and c.designed_positive]
        paired_map={
            arm:{c.id:bool(indices[(TARGET_SPLIT,TARGET_FAMILY,arm,c.id)]["solved"])
                 for c in hard_cases}
            for arm in factor.ARMS
        }
        local_pairs={
            name:matched_pairs(paired_map[a],paired_map[b])
            for name,a,b in (
                ("dv_vs_vv","dv","vv"),
                ("vd_vs_vv","vd","vv"),
                ("dd_vs_vv","dd","vv"),
                ("nd_vs_dd","nd","dd"),
                ("nv_vs_vv","nv","vv"),
            )
        }
        for name,part in local_pairs.items():
            for k,v in part.items():
                pooled_pairs[name][k]+=v
        local_hard={arm:per_family[(TARGET_SPLIT,TARGET_FAMILY)][arm]["solved"]
                    for arm in factor.ARMS}
        per_seed[str(seed)]={
            "hard_d5":local_hard,
            "hard_d5_contrast":factor.factor_contrast(local_hard),
            "hard_d5_discordance":local_pairs,
            "depths":{
                split:{
                    arm:{k:rec["scores"][split][arm][k] for k in METRICS}
                    for arm in factor.ARMS
                }
                for split in gym.SPLITS
            },
            "original_checkpoint":rec["original_af23_checkpoint"],
            "source_evaluation_sha256":sha(receipt_path.read_bytes()),
            "proof_episodes_audited":audited["episodes_audited"],
            "planner_nodes_audited":audited["planner_nodes_reconstructed"],
            "torch_forward_replays":audited["torch_forward_replays"],
            "near_tie_forward_replays":audited["near_tie_numerical_forward_replays"],
            "original_control_actions_exact":audited["original_control_actions_exact"],
            "original_af25_source_zip":rec["original_af25_control_receipt"]["artifact_zip_sha256"],
        }
    primary={arm:family[TARGET_SPLIT][TARGET_FAMILY][arm]["solved"]
             for arm in factor.ARMS}
    if missing:
        verdict="INCONCLUSIVE"
        primary_contrast=None
    else:
        verdict=factor.outcome(primary)
        primary_contrast=factor.factor_contrast(primary)
    if not missing:
        # Stronger source-parity check against the exact original pooled
        # counts, including all cases and the registered hard-family gap.
        for split in gym.SPLITS:
            for arm,origin in (("dd","direct"),("vv","vi2")):
                if pooled[split][arm]["solved"] != original["pooled"][split][origin]["solved"]:
                    raise ValueError(f"AF25 pooled source replay mismatch {split}/{arm}")
        for orig_arm,arm in (("direct","dd"),("vi2","vv")):
            if primary[arm] != original["per_family"][TARGET_SPLIT][TARGET_FAMILY][orig_arm]["solved"]:
                raise ValueError("original AF25 hard-family primary comparator mismatch")
    return {
        "experiment":"AF-26",
        "registered_factorial_outcome":verdict,
        "science_scope":"In-gym STOP/rule source-policy attribution only, no general proof-search advantage claim",
        "original_af25_scientific_verdict":"NEURAL_ADVANTAGE_NOT_SHOWN",
        "original_af25_aggregate_sha256":ev.PARENT_AGG_SHA,
        "source_hashes":source,
        "protocol_sha256":protocol,
        "structural_manifest_sha256":manifest,
        "required_seeds":list(REQUIRED_SEEDS),
        "qualified_seeds":len(per_seed),
        "missing_seeds":missing,
        "hard_d5_factorial":primary,
        "hard_d5_primary_contrast":primary_contrast,
        "hard_d5_half_recovery_threshold":55,
        "hard_d5_pair_discordance":pooled_pairs,
        "per_seed":per_seed,
        "pooled":pooled,
        "per_family":family,
        "verification":{
            "episodes":sum(x["proof_episodes_audited"] for x in per_seed.values()),
            "nodes":sum(x["planner_nodes_audited"] for x in per_seed.values()),
            "torch_forward_replays":sum(x["torch_forward_replays"] for x in per_seed.values()),
            "near_tie_forward_replays":sum(x["near_tie_forward_replays"] for x in per_seed.values()),
            "exact_archived_source_control_steps":sum(x["original_control_actions_exact"] for x in per_seed.values()),
            "false_accepted_proofs":0,
        },
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--original-af25",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=aggregate(a.root,a.original_af25)
    a.out.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":r["registered_factorial_outcome"],
        "original_dd_vv_parity":not r["missing_seeds"],
        "hard":r["hard_d5_factorial"],
        "contrast":r["hard_d5_primary_contrast"],
        "replays":r["verification"],
    }))


if __name__=="__main__":
    main()
