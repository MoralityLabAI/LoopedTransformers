"""AF-33 source-locked prospective eight-seed two-verdict aggregation.

Scientific outcomes are independent:
- RANK_CERT absolute proof completion + >=10/96 best equal-certifier gain
- LRANK_STOP proof completion + learned-only negative STOP without certificate
Original frozen AF23 Direct is an on-identical-graph diagnostic, never scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import evaluate, gym, ranker
# Independent aggregation only reads direct source receipts; importing direct
# would pull in Torch unnecessarily and block a pure-Python audit runner.
DIRECT_ARMS=("AF23_DIRECT_NATIVE","AF23_DIRECT_FORCED","AF23_DIRECT_CERT")

COHORT_KEYS=tuple(evaluate.cohort_key(b,s) for b,s in evaluate.COHORTS)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root,out):
    root,out=Path(root),Path(out)
    if out.exists():
        raise FileExistsError("AF33 official source aggregate must be immutable")
    manifest=gym.manifest()["manifest_sha256"]
    protocol=digest(Path(__file__).parent/"PROTOCOL.json")
    all_seed={};ranker_source_hashes=set();per_seed_STOP_source_hashes={}
    for seed in ranker.SEEDS:
        a=root/f"af33-seed-{seed}"
        d=root/f"af33-direct-seed-{seed}"
        learned,la=load(a/"evaluation.json"),load(a/"audit.json")
        direct_eval,da=load(d/"direct_evaluation.json"),load(d/"direct_audit.json")
        model=load(a/"model.json")
        stop_model={
            arm:load(a/("stop_"+arm.lower()+".json"))
            for arm in evaluate.LEARNED_STOP_ARMS
        }
        if any(x["seed"]!=seed for x in (learned,la,direct_eval,da,model,*stop_model.values())):
            raise AssertionError("AF33 source seed identity mismatch")
        if (learned["manifest_sha256"]!=manifest or
            la["source_manifest_sha256"]!=manifest or
            direct_eval["source_manifest_sha256"]!=manifest or
            da["source_manifest_sha256"]!=manifest):
            raise AssertionError("AF33 held pair graph source mismatch")
        if learned["protocol_sha256"]!=protocol or direct_eval["protocol_sha256"]!=protocol:
            raise AssertionError("AF33 preregistered protocol drift")
        for name,field in (
            ("model.json","model_sha256"),
            ("episodes.jsonl","episodes_sha256"),
        ):
            if digest(a/name)!=learned[field] or digest(a/name)!=la[field]:
                raise AssertionError(f"AF33 learned source artifact integrity drift {seed}:{name}")
        for arm in evaluate.LEARNED_STOP_ARMS:
            name="stop_"+arm.lower()+".json"
            if digest(a/name)!=learned["stop_models_sha256"][arm] or digest(a/name)!=la["stop_models_sha256"][arm]:
                raise AssertionError(f"AF33 learned STOP source artifact integrity drift {seed}:{arm}")
        if digest(d/"direct_episodes.jsonl")!=direct_eval["direct_episodes_sha256"] or \
           digest(d/"direct_episodes.jsonl")!=da["direct_episodes_sha256"]:
            raise AssertionError("AF33 original Direct source proof archive hash drift")
        if (learned["episodes"]!=192*len(evaluate.COHORTS)*len(evaluate.ARMS) or
            la["independently_replayed_episodes"]!=learned["episodes"] or
            direct_eval["direct_episode_count"]!=192*len(evaluate.COHORTS)*len(DIRECT_ARMS) or
            da["independent_physical_proof_episodes"]!=direct_eval["direct_episode_count"]):
            raise AssertionError("AF33 full held policy episodes missing")
        if not all(la[k] for k in (
            "fresh_train_refit","independent_learned_stop_refit",
            "all_public_physical_transitions_valid",
            "all_action_decisions_recomputed",
            "all_scores_independently_aggregated",
            "no_false_accepted_proofs"
        )):
            raise AssertionError("AF33 source ranker/STOP independent audit missing")
        if not all(da[k] for k in (
            "all_original_direct_actions_reconstructed",
            "all_original_host_transitions_replayed",
            "all_original_scores_reaggregated",
            "no_false_accepted_proofs"
        )):
            raise AssertionError("AF33 original frozen Direct independent audit missing")
        if da["sampled_genuine_original_torch_forward_decisions"]<12 or \
           da["maximum_original_score_absolute_drift"]>1e-4:
            raise AssertionError("AF33 original checkpoint Torch replay insufficient")
        fresh_rank=evaluate.gate(learned["scores"])
        fresh_bal=evaluate.balanced_stop_gate(learned["scores"])
        fresh_abs=evaluate.absolute_bal_stop_gate(learned["scores"])
        if (fresh_rank!=learned["registered_seed_gate"] or
            fresh_rank!=la["registered_seed_gate"] or
            fresh_bal!=learned["balanced_stop_seed_gate"] or
            fresh_bal!=la["balanced_stop_seed_gate"] or
            fresh_abs!=learned["absolute_bal_stop_seed_gate"] or
            fresh_abs!=la["absolute_bal_stop_seed_gate"]):
            raise AssertionError("AF33 registered science gate source drift")
        if direct_eval["original_model_parameters"]!=6036 or \
           direct_eval["original_checkpoint_parameter_updates"]!=0:
            raise AssertionError("original frozen checkpoint integrity changed")
        ranker_source_hashes.add(model["training_row_sha256"])
        # STOP's ten features include this seed's trained ranker logit, so
        # its train/dev *feature rows* legitimately differ across seeds,
        # although the source physical graph roster is identical.
        per_seed_STOP_source_hashes[str(seed)]={
            arm:{
                "train":sub["train_sha256"],
                "dev":sub["dev_sha256"],
                "ranker_source":sub["ranker_training_sha256"],
                "train_rows":sub["train_rows"],
                "dev_threshold":sub["dev_threshold"],
            } for arm,sub in stop_model.items()
        }
        if any(sub["ranker_training_sha256"]!=model["training_row_sha256"]
               for sub in stop_model.values()):
            raise AssertionError("source STOP head fitted against foreign ranker")

        all_seed[str(seed)]={
            "primary_ranker_gate":fresh_rank,
            "balanced_stop_gate":fresh_bal,
            "absolute_bal_stop_gate":fresh_abs,
            "learned_scores":learned["scores"],
            "original_direct_scores":direct_eval["scores"],
            "source_model_sha256":learned["model_sha256"],
            "source_STOP_models_sha256":learned["stop_models_sha256"],
            "source_learned_episodes_sha256":learned["episodes_sha256"],
            "source_original_episodes_sha256":direct_eval["direct_episodes_sha256"],
            "learned_audit_sha256":digest(a/"audit.json"),
            "direct_audit_sha256":digest(d/"direct_audit.json"),
            "original_checkpoint_provenance":da["original_checkpoint_provenance"],
            "original_model_forward_nodes":da["original_neural_nodes_from_receipt"],
            "original_genuine_torch_replay_count":da["sampled_genuine_original_torch_forward_decisions"],
            "original_genuine_forward_max_drift":da["maximum_original_score_absolute_drift"],
            "learned_episode_count":learned["episodes"],
            "original_direct_episode_count":direct_eval["direct_episode_count"],
        }
    if len(ranker_source_hashes)!=1:
        raise AssertionError("AF33 eight linear rankers used different source physical TRAIN graph/action rows")
    rank_pass=sum(x["primary_ranker_gate"]["both_depths_pass"]
                  for x in all_seed.values())
    bal_pass=sum(x["balanced_stop_gate"]["both_depths_pass"]
                 for x in all_seed.values())
    abs_pass=sum(x["absolute_bal_stop_gate"]["both_depths_pass"]
                 for x in all_seed.values())
    totals={}
    for key in COHORT_KEYS:
        totals[key]={}
        for arm in (*evaluate.ARMS,*DIRECT_ARMS):
            src="learned_scores" if arm in evaluate.ARMS else "original_direct_scores"
            totals[key][arm]={
                metric:sum(all_seed[str(seed)][src][key][arm][metric]
                           for seed in ranker.SEEDS)
                for metric in ("positive_proofs","negative_correct_initial_stop",
                               "positive_premature_initial_stop","false_accepted_proofs",
                               "positive_root_first_off_optimal","rule_examinations")
            }
            if arm in DIRECT_ARMS:
                totals[key][arm]["neural_forward_nodes"]=sum(
                    all_seed[str(seed)][src][key][arm]["original_neural_forward_nodes"]
                    for seed in ranker.SEEDS)
    outdata={
        "experiment":"AF-33",
        "source_audit":"PASS_ALL_EIGHT_LEARNED_AND_FROZEN_DIRECT",
        "source_registry":"source-locked first-run HEAD and original checkpoint zip provenance",
        "ranker_safeguard_verdict":("STRICT_RANKER_SAFEGUARD_PASS" if rank_pass>=6
                                  else "STRICT_RANKER_SAFEGUARD_FAIL"),
        "position_balanced_stop_verdict":("POSITION_BALANCED_STOP_TRANSFER_PASS" if bal_pass>=6
                                         else "POSITION_BALANCED_STOP_TRANSFER_FAIL"),
        "absolute_bal_stop_verdict":("BAL_STOP_ABSOLUTE_PASS" if abs_pass>=6
                                    else "BAL_STOP_ABSOLUTE_FAIL"),
        "ranker_qualified_seeds":rank_pass,
        "position_balanced_qualified_seeds":bal_pass,
        "absolute_bal_stop_qualified_seeds":abs_pass,
        "required_seeds":6,
        "source_seeds":list(ranker.SEEDS),
        "paired_graphs_per_held_depth":96,
        "held_cycle_family_positive_count_each":{"2":48,"3":48},
        "budget_modes":list(evaluate.BUDGET_MODES),
        "cohort_keys":list(COHORT_KEYS),
        "source_manifest_sha256":manifest,
        "protocol_sha256":protocol,
        "ranker_training_row_sha256":next(iter(ranker_source_hashes)),
        "per_seed_STOP_ranker_conditional_train_dev_hashes":per_seed_STOP_source_hashes,
        "per_seed_stop_root_metrics":{
            key:{arm:{
                str(seed):all_seed[str(seed)]["learned_scores"][key][arm]["root_diagnostics"]
                for seed in ranker.SEEDS}
                for arm in evaluate.LEARNED_STOP_ARMS}
            for key in COHORT_KEYS
        },
        "per_seed":all_seed,
        "totals":totals,
        "independently_replayed_complete_episodes":sum(
            x["learned_episode_count"]+x["original_direct_episode_count"]
            for x in all_seed.values()),
        "interpretation":"AF33 primary is certificate-free BAL_STOP vs position-3-unexposed J12_STOP at strict source d budget with independently frozen DEV threshold for each arm. Full closure remains exact; RANK_CERT negatives use a public certificate. Held motif changes GOAL conclusion arity/root-diversion topology; 96 held pairs per depth repeat under eight seeds. Source rule exams and model-node forwards are incomparable algorithmic units.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(outdata,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return outdata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=aggregate(a.root,a.out)
    print(json.dumps({
        "ranker_safeguard":result["ranker_safeguard_verdict"],
        "ranker_seeds":result["ranker_qualified_seeds"],
        "balanced_STOP":result["position_balanced_stop_verdict"],
        "balanced_STOP_seeds":result["position_balanced_qualified_seeds"],
        "absolute_BAL_STOP":result["absolute_bal_stop_verdict"],
        "absolute_BAL_STOP_seeds":result["absolute_bal_stop_qualified_seeds"],
        "audited_episodes":result["independently_replayed_complete_episodes"],
        "strict_and_slack_source_rosters":list(COHORT_KEYS),
        "audit":result["source_audit"],
    }))


if __name__=="__main__":
    main()
