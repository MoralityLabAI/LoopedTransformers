"""GG-05P post-hoc forensic audit of *gold scoring sample* influence.

After GG-05's registered primary result was observed, the historical six-
cue scoring panel showed concentrated baseline CE. This separate exploratory
analysis reconstructs actual native report-conditioned SGD and evaluates
every 2-of-128 deletion from the *fixed gold scoring set*. Report gradients,
model weights, and original four-cue queries stay unchanged.

A deletion only changes which post-SGD gold examples contribute to the
scoring average. This is a conditional scoring-sensitivity identity, not
a retraining experiment or a population-generalization certificate.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any
import json
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, make_panel, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    Query, query_splits, _tensor_rows, _rows_hash,
)
from research_gym.interp.goal_guarding_gradient_conflict import (
    SIGN_EPS, query_gradients, dot64, norm64,
)


HISTORICAL_SIX_CUE_SEED=68001
PANEL_SIZE=128
SINGLE_STEP_LR=0.05
REPORT_SEEDS=(11,17)


def native_from_gg05_checkpoint(path:Path,
                                source_digest:str)->EvaluationBeliefLooper:
    """Load the original native class and verify frozen GG-05 source state."""
    record=torch.load(path,map_location="cpu",weights_only=False)
    if record.get("protocol")!="RMP-GG-01":
        raise ValueError("native GG-01 source checkpoint protocol mismatch")
    model=EvaluationBeliefLooper().eval()
    model.load_state_dict(record["model"],strict=True)
    if _weights_digest(model)!=source_digest:
        raise ValueError("GG-05 source model hash differs from saved outcome ledger")
    return model


def one_step_per_example(model:EvaluationBeliefLooper,row:Query,
                         panel,lr:float=SINGLE_STEP_LR)->Tensor:
    """Physical native SGD, vectorized independent gold per-example losses.

    Same starting model/gradient for each action. Gold CE is separately
    evaluated after the actual SGD optimizer step.
    """
    if lr!=SINGLE_STEP_LR:
        raise ValueError("post-hoc diagnostic uses registered original SGD LR")
    grads,_=query_gradients(model,row)
    per_arm={}
    for name in ("truth","flip"):
        trial=deepcopy(model).eval()
        for parameter,g in zip(trial.parameters(),grads[name]):
            parameter.grad=g.detach().clone()
        torch.optim.SGD(trial.parameters(),lr=lr,weight_decay=0,
                        momentum=0).step()
        with torch.no_grad():
            out=trial(panel.observations,panel.context)[0][:,-1,:]
            losses=F.cross_entropy(out,panel.labels,reduction="none")
        per_arm[name]=losses.detach().double()
    return per_arm["flip"]-per_arm["truth"]


def deletion_count(effect_matrix:Tensor,deleted:tuple[int,...],
                   epsilon:float=SIGN_EPS)->int:
    """Number of source queries with negative post-SGD gold gap."""
    if effect_matrix.ndim!=2 or not 2<=effect_matrix.shape[1]:
        raise ValueError("expected source queries by gold-scoring examples matrix")
    if len(set(deleted))!=len(deleted) or any(
        type(k) is not int or not 0<=k<effect_matrix.shape[1]
        for k in deleted
    ):
        raise ValueError("gold deletion indices must be distinct and valid")
    if len(deleted)>=effect_matrix.shape[1]:
        raise ValueError("cannot delete every gold-scoring sample")
    keep=torch.ones(effect_matrix.shape[1],dtype=torch.bool)
    keep[list(deleted)]=False
    gap=effect_matrix[:,keep].double().mean(dim=1)
    return int((gap<-epsilon).sum())


def all_two_deletions(effect_matrix:Tensor,
                      epsilon:float=SIGN_EPS)->dict[str,Any]:
    """Complete exact finite roster of C(N,2) scoring-set interventions."""
    if (effect_matrix.ndim!=2 or effect_matrix.shape[0]<1
            or not 3<=effect_matrix.shape[1]<=256
            or not torch.isfinite(effect_matrix).all()):
        raise ValueError("deletion census expects a finite [queries, gold] matrix")
    n=effect_matrix.shape[1]
    values={}
    # Deleting samples from a mean is equivalent to subtracting their
    # per-example *physical loss* differences from the full sum.
    total=effect_matrix.double().sum(dim=1)
    for i,j in combinations(range(n),2):
        gap=(total-effect_matrix[:,i]-effect_matrix[:,j])/(n-2)
        values[(i,j)]=int((gap < -epsilon).sum())
    order=sorted(values.values())
    quantiles={str(q):order[int(q*(len(order)-1))]
               for q in (0,.01,.1,.5,.9,.99,1)}
    return {
        "gold_examples":n,
        "source_queries":effect_matrix.shape[0],
        "pairs_enumerated":len(values),
        "minimum_negative_gap_queries":min(order),
        "maximum_negative_gap_queries":max(order),
        "number_pairs_eliminating_all_negative_gaps":sum(x==0 for x in order),
        "quantiles_of_negative_gap_count":quantiles,
        "pairs_attaining_minimum":[[i,j] for (i,j),count in values.items()
                                  if count==min(order)],
        "full_counts_by_pair":values,
    }


def audit_one(seed:int,checkpoint:Path,decomposition:Path,
              out_dir:Path)->dict[str,Any]:
    if seed not in REPORT_SEEDS:
        raise ValueError("GG-05P only replays original model seeds")
    torch.set_num_threads(1)
    old=json.loads(decomposition.read_text())
    rows=query_splits()["heldout"]
    if old["seed"]!=seed or _rows_hash(rows)!=old["query_manifest_sha256"]:
        raise ValueError("GG-05 model/query manifest failed source verification")
    model=native_from_gg05_checkpoint(
        checkpoint,old["source_model_sha256"])
    baseline_hash=_weights_digest(model)
    gold=make_panel(PANEL_SIZE,6,HISTORICAL_SIX_CUE_SEED)
    with torch.no_grad():
        logits=model(gold.observations,gold.context)[0][:,-1,:]
        loss=F.cross_entropy(logits,gold.labels,reduction="none")
        incorrect=(logits.argmax(-1)!=gold.labels)
    if abs(float(loss.mean())-old["baseline_gold_cross_entropy"]["old6"])>1e-6:
        raise AssertionError("reconstructed old gold loss differs from physical ledger")
    top=torch.argsort(loss,descending=True).tolist()
    top_two=tuple(sorted(top[:2]))
    per_query=[]
    for row,source in zip(rows,old["physical_query_readouts"]):
        if row.uid!=source["uid"]:
            raise AssertionError("GG-05 query identities changed")
        per=one_step_per_example(model,row,gold)
        actual=float(per.mean())
        expected=source["panel_readouts"]["0.0500"]["old6"]["gap"]
        # float32 native F.cross_entropy mean can accumulate a slightly
        # different rounding error from a mean of 128 per-example CE
        # values. A float64 direct mean of per-example CE has this
        # explicit numerically bounded tolerance.
        if abs(actual-expected)>1e-5:
            raise AssertionError("replayed SGD gap differs from the original")
        per_query.append(per)
    effects=torch.stack(per_query)
    original_reversals=int((effects.mean(1)<-SIGN_EPS).sum())
    recorded_reversals={11:6,17:35}[seed]
    if original_reversals!=recorded_reversals:
        raise AssertionError("historical sign reversal failed physical replay")
    full=all_two_deletions(effects)
    count_after=deletion_count(effects,top_two)
    if full["full_counts_by_pair"][top_two]!=count_after:
        raise AssertionError("exhaustive deletion and independent mask disagree")

    out_dir.mkdir(parents=True,exist_ok=True)
    ledger=out_dir/f"gg05p_seed{seed}_loss_contributions.json"
    ledger.write_text(json.dumps({
        "seed":seed,"checkpoint_sha256":baseline_hash,
        "source_gg05_ledger_sha256":sha256(decomposition.read_bytes()).hexdigest(),
        "original_six_cue_gold_seed":HISTORICAL_SIX_CUE_SEED,
        "per_example_gold_ce_at_baseline":[float(x) for x in loss.tolist()],
        "per_example_wrong_at_baseline":[bool(x) for x in incorrect.tolist()],
        "report_per_example_flip_minus_truth_sgd_ce_effect":[
            [float(v) for v in row.tolist()] for row in effects
        ],
        "source_query_manifest_sha256":old["query_manifest_sha256"],
    },sort_keys=True,indent=2,allow_nan=False)+"\n")
    sorted_pair=tuple(sorted(top_two))
    n=full["pairs_enumerated"]
    rank=sum(v<=count_after for v in full["full_counts_by_pair"].values())
    return {
        "seed":seed,
        "post_hoc_after_primary_GG05_outcome":True,
        "original_gold_examples":PANEL_SIZE,
        "original_gold_baseline_ce":float(loss.mean()),
        "baseline_misclassified_examples":int(incorrect.sum()),
        "top_two_baseline_loss_indices":list(top_two),
        "top_two_fraction_of_total_baseline_loss":
            float(loss[list(top_two)].sum()/loss.sum()),
        "top_five_fraction_of_total_baseline_loss":
            float(loss[top[:5]].sum()/loss.sum()),
        "original_negative_gap_queries":original_reversals,
        "negative_gap_queries_after_removing_hardest_two":count_after,
        "negative_gap_queries_after_removing_hardest_one":
            deletion_count(effects,(top[0],)),
        "negative_gap_queries_after_removing_second_hardest":
            deletion_count(effects,(top[1],)),
        "full_two_deletion_census":{
            k:v for k,v in full.items() if k!="full_counts_by_pair"},
        "top_two_rank_count_at_least_as_effective":rank,
        "top_two_fraction_at_least_as_effective":rank/n,
        "source_model_weights_unchanged":
            _weights_digest(model)==baseline_hash,
        "per_example_ledger_filename":ledger.name,
        "per_example_ledger_sha256":sha256(ledger.read_bytes()).hexdigest(),
        "hardest_example_details":[{
            "index":i,
            "bits":gold.observations[i].tolist(),
            "prior_and_ratio":[str(x) for x in gold.parameters[i]],
            "correct_label":int(gold.labels[i]),
            "actual_model_prediction":int(logits[i].argmax()),
            "baseline_ce":float(loss[i]),
        } for i in top[:5]],
    }


def run_posthoc_gg05(*,source_dir:Path,output:Path,
                    seeds:tuple[int,int]=REPORT_SEEDS)->dict[str,Any]:
    if seeds!=REPORT_SEEDS:
        raise ValueError("GG-05P requires both original model seeds")
    root={
        "protocol":"RMP-GG-05P-POSTHOC",
        "registered_primary_GG05_unchanged":True,
        "known_historical_six_cue_gold_only":True,
        "original_sgd_lr":SINGLE_STEP_LR,
        "finite_sign_epsilon":SIGN_EPS,
        "source_query_count":48,
        "gold_sample_count":128,
        "exact_two_deletion_pairs":8128,
        "results":[],
    }
    for seed in seeds:
        result=audit_one(
            seed,
            source_dir/f"gg01_seed{seed}_trained.pt",
            source_dir/f"gg05_seed{seed}_physical_decomposition.json",
            output.parent)
        root["results"].append(result)
    output.write_text(json.dumps(root,sort_keys=True,indent=2,
                                 allow_nan=False)+"\n")
    return root
