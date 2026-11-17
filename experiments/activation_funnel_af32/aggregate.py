"""AF-32 source-locked prospective eight-seed two-verdict aggregation.

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
        raise FileExistsError("AF32 official source aggregate must be immutable")
    manifest=gym.manifest()["manifest_sha256"]
    protocol=digest(Path(__file__).parent/"PROTOCOL.json")
    all_seed={};ranker_source_hashes=set();per_seed_STOP_source_hashes={}
    for seed in ranker.SEEDS:
        a=root/f"af32-seed-{seed}"
        d=root/f"af32-direct-seed-{seed}"
        learned,la=load(a/"evaluation.json"),load(a/"audit.json")
        direct_eval,da=load(d/"direct_evaluation.json"),load(d/"direct_audit.json")
        model=load(a/"model.json")
        stop_model=load(a/"stop_model.json")
        if any(x["seed"]!=seed for x in (learned,la,direct_eval,da,model,stop_model)):
            raise AssertionError("AF32 source seed identity mismatch")
        if (learned["manifest_sha256"]!=manifest or
            la["source_manifest_sha256"]!=manifest or
            direct_eval["source_manifest_sha256"]!=manifest or
            da["source_manifest_sha256"]!=manifest):
            raise AssertionError("AF32 held pair graph source mismatch")
        if learned["protocol_sha256"]!=protocol or direct_eval["protocol_sha256"]!=protocol:
            raise AssertionError("AF32 preregistered protocol drift")
        for name,field in (
            ("model.json","model_sha256"),
            ("stop_model.json","stop_model_sha256"),
            ("episodes.jsonl","episodes_sha256"),
        ):
            if digest(a/name)!=learned[field] or digest(a/name)!=la[field]:
                raise AssertionError(f"AF32 learned source artifact integrity drift {seed}:{name}")
        if digest(d/"direct_episodes.jsonl")!=direct_eval["direct_episodes_sha256"] or \
           digest(d/"direct_episodes.jsonl")!=da["direct_episodes_sha256"]:
            raise AssertionError("AF32 original Direct source proof archive hash drift")
        if (learned["episodes"]!=192*len(evaluate.COHORTS)*len(evaluate.ARMS) or
            la["independently_replayed_episodes"]!=learned["episodes"] or
            direct_eval["direct_episode_count"]!=192*len(evaluate.COHORTS)*len(DIRECT_ARMS) or
            da["independent_physical_proof_episodes"]!=direct_eval["direct_episode_count"]):
            raise AssertionError("AF32 full held policy episodes missing")
        if not all(la[k] for k in (
            "fresh_train_refit","independent_learned_stop_refit",
            "all_public_physical_transitions_valid",
            "all_action_decisions_recomputed",
            "all_scores_independently_aggregated",
            "no_false_accepted_proofs"
        )):
            raise AssertionError("AF32 source ranker/STOP independent audit missing")
        if not all(da[k] for k in (
            "all_original_direct_actions_reconstructed",
            "all_original_host_transitions_replayed",
            "all_original_scores_reaggregated",
            "no_false_accepted_proofs"
        )):
            raise AssertionError("AF32 original frozen Direct independent audit missing")
        if da["sampled_genuine_original_torch_forward_decisions"]<12 or \
           da["maximum_original_score_absolute_drift"]>1e-4:
            raise AssertionError("AF32 original checkpoint Torch replay insufficient")
        fresh_rank=evaluate.gate(learned["scores"])
        fresh_stop=evaluate.learned_stop_gate(learned["scores"])
        if fresh_rank != learned["registered_seed_gate"] or \
           fresh_rank!=la["registered_seed_gate"] or \
           fresh_stop!=learned["learned_stop_seed_gate"] or \
           fresh_stop!=la["learned_stop_seed_gate"]:
            raise AssertionError("AF32 registered seed gates source drift")
        if direct_eval["original_model_parameters"]!=6036 or \
           direct_eval["original_checkpoint_parameter_updates"]!=0:
            raise AssertionError("original frozen checkpoint integrity changed")
        ranker_source_hashes.add(model["training_row_sha256"])
        # STOP's ten features include this seed's trained ranker logit, so
        # its train/dev *feature rows* legitimately differ across seeds,
        # although the source physical graph roster is identical.
        per_seed_STOP_source_hashes[str(seed)]={
            "train":stop_model["train_sha256"],
            "dev":stop_model["dev_sha256"],
            "ranker_source":stop_model["ranker_training_sha256"],
        }
        if stop_model["ranker_training_sha256"]!=model["training_row_sha256"]:
            raise AssertionError("source STOP head fit against foreign ranker")

        all_seed[str(seed)]={
            "primary_ranker_gate":fresh_rank,
            "learned_only_STOP_gate":fresh_stop,
            "learned_scores":learned["scores"],
            "original_direct_scores":direct_eval["scores"],
            "source_model_sha256":learned["model_sha256"],
            "source_STOP_model_sha256":learned["stop_model_sha256"],
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
        raise AssertionError("AF32 eight linear rankers used different source physical TRAIN graph/action rows")
    rank_pass=sum(x["primary_ranker_gate"]["both_depths_pass"]
                  for x in all_seed.values())
    stop_pass=sum(x["learned_only_STOP_gate"]["both_depths_pass"]
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
        "experiment":"AF-32",
        "source_audit":"PASS_ALL_EIGHT_LEARNED_AND_FROZEN_DIRECT",
        "source_registry":"source-locked first-run HEAD and original checkpoint zip provenance",
        "ranker_primary_verdict":("STRICT_RULE_SELECTION_PASS" if rank_pass>=6
                                  else "STRICT_RULE_SELECTION_FAIL"),
        "learned_only_STOP_verdict":("LEARNED_STOP_STRICT_PASS" if stop_pass>=6
                                     else "LEARNED_STOP_STRICT_FAIL"),
        "ranker_qualified_seeds":rank_pass,
        "learned_only_STOP_qualified_seeds":stop_pass,
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
        "per_seed":all_seed,
        "totals":totals,
        "independently_replayed_complete_episodes":sum(
            x["learned_episode_count"]+x["original_direct_episode_count"]
            for x in all_seed.values()),
        "interpretation":"STRICT primary gate and certificate-free learned STOP evaluated under exact source shortest d rule firings; slack d+1 separately reports source-registered sensitivity. Full public closure is exact. Model training uses source TRAIN only; certified negatives use symbolic closure. Eight seeds repeat the same 96 held pairs per depth. Rule examination and original model-node forward work are separate units.",
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
        "ranker":result["ranker_primary_verdict"],
        "ranker_seeds":result["ranker_qualified_seeds"],
        "learned_STOP":result["learned_only_STOP_verdict"],
        "learned_STOP_seeds":result["learned_only_STOP_qualified_seeds"],
        "audited_episodes":result["independently_replayed_complete_episodes"],
        "strict_and_slack_source_rosters":list(COHORT_KEYS),
        "audit":result["source_audit"],
    }))


if __name__=="__main__":
    main()
