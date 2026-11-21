"""NEC-03H: source-frozen fallback site selector and 900-step development training.

NEC-03G, including eight failed development roots, remains immutable.
The 900-step trainer reinitializes native models from the original seeds,
rather than continuing an old checkpoint without its optimizer state.
The additive control is independently initialized. All selection is on
training contexts and all confirmation-world indices are rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.core import build_graph, decode, discover, node
from mechinterp.neuron_communities.pef07_adapter import context_attributions
from mechinterp.neuron_communities.nec03g_worlds import (
    SCHEMA as WORLD_SCHEMA, DEV_COUNT, context_catalog, target_bits,
    additive_control_truth, encoded, digest, registry, worlds,
)
from research.pef07.pef07 import Looper, as_tensor, model_digest

H_SCHEMA="NEC-03H/source-site-fallback-900steps-development-v2"
MODEL_STEPS=900
SAE_STEPS=480
ATTR_CONTEXTS=80
FALLBACK_VISITS="third-and-fourth-focal-input-positions"
FALLBACK_NEURONS_PER_VISIT=2
FALLBACK_MIN_CONTEXT_FREQ=2
COMPETENCE_FLOOR=0.90
MINIMUM_PHYSICAL_SIGNAL=0.25


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as src:
        for b in iter(lambda: src.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def training_world(index: int, *, calib: bool=False) -> dict:
    if not 0 <= index < DEV_COUNT:
        raise PermissionError("NEC-03H forbids confirmation cohort access")
    if calib and index not in range(8):
        raise PermissionError("calibration restricted to opened dev-000..007")
    if not calib and index not in range(8,16):
        raise PermissionError("new pilot restricted to untouched dev-008..015")
    w=worlds()[index]
    if w["cohort"]!="development" or registry()["dev_full_world_specs"][index]!=w:
        raise AssertionError("frozen NEC-03G world registry moved")
    return w


def train_native_900(world, catalog, out: Path, *, arm: str, steps: int=MODEL_STEPS):
    if world["cohort"]!="development" or arm not in ("composed","additive"):
        raise PermissionError("invalid native teacher or sealed world")
    if steps<1 or steps>MODEL_STEPS:
        raise ValueError("only fixed 900-step native schedule or short engineering smoke")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed=world["trunk_seed"]+(100000 if arm=="additive" else 0)
    torch.manual_seed(seed)
    model=Looper().train()
    original=model_digest(model)
    source=as_tensor(catalog["train"])
    truth=target_bits(source,world,control=arm=="additive")
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0)
    gen=torch.Generator().manual_seed(seed*7919+59)
    loss_start=loss_end=None
    for _ in range(steps):
        idx=torch.randint(len(source),(128,),generator=gen)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(source[idx]),truth[idx])
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite native 900-step training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if loss_start is None:
            loss_start=float(loss.detach())
        loss_end=float(loss.detach())
    model.eval()
    frozen=model_digest(model)
    out.mkdir(parents=True,exist_ok=True)
    path=out/f"{arm}.native900.pt"
    torch.save({"world_id":world["world_id"],
                "world_sha256":world["world_sha256"],
                "teacher":arm,"seed":seed,"steps":steps,
                "model":model.state_dict(),
                "model_sha256":frozen},path)
    receipt={"world_sha256":world["world_sha256"],
             "teacher":arm,"training_seed":seed,"steps":steps,
             "source_training_only":True,
             "initial_weights_sha256":original,
             "final_model_sha256":frozen,
             "checkpoint_name":path.name,"checkpoint_sha256":sha_file(path),
             "optimizer":"AdamW(lr=0.003,weight_decay=0.0),batch=128,clip_norm=1",
             "same_seed_and_optimizer_stream_as_NEC03G_first_450_steps":True,
             "reinitialized_from_seed_with_complete_optimizer_stream":True,
             "train_loss_initial":loss_start,"train_loss_final":loss_end,
             "source_native_weights_frozen_before_evaluation":True}
    return model,receipt


def source_site_selection(model, catalog, world):
    """Source-only Louvain, with deterministic focal-visit fallback.

    Fallback selects the two most frequently top-two-attributed non-hub
    GELU channels at visits immediately following the 3rd and 4th focal
    bits. It retains a record of missing graph communities and never
    consults labels or development outcomes.
    """
    if world["cohort"]!="development":
        raise PermissionError("confirmation source-site selection prohibited")
    groups=[catalog["train"][i:i+16]
            for i in range(0,ATTR_CONTEXTS*16,16)]
    if len(groups)!=ATTR_CONTEXTS:
        raise AssertionError("source attribution requires 80 full contexts")
    before=model_digest(model)
    rows=[context_attributions(model,as_tensor(group),top_k=2)
          for group in groups]
    graph=build_graph(rows,hub_rate=0.9,min_pair_count=3,min_phi=0.12)
    candidates=discover(graph,max_size=12,max_candidates=8,seed=0)
    accepted=[]
    for cand in candidates:
        counts=Counter(decode(x)[0] for x in cand["nodes"])
        if all(0<n<=8 for n in counts.values()):
            accepted.append(cand)
    fallback_used=False
    fallback_metadata=None
    candidate=accepted[0] if accepted else None
    if candidate is None:
        fallback_used=True
        scheduled=sorted(world["focal_positions"])[2:4]
        keys=[]
        visits_receipt={}
        for visit in scheduled:
            available=[
                (key,int(graph["frequency"].get(key,0)))
                for j in range(64)
                for key in [node(visit,j)]
                if key not in set(graph["hubs"])
                   and int(graph["frequency"].get(key,0))>=FALLBACK_MIN_CONTEXT_FREQ
            ]
            available.sort(key=lambda pair:(-pair[1],decode(pair[0])[1]))
            chosen=available[:FALLBACK_NEURONS_PER_VISIT]
            visits_receipt[str(visit)]={
                "eligible_source_nonhub_channels":len(available),
                "top_selection":[{"node":k,"frequency":n} for k,n in chosen],
            }
            keys.extend(k for k,n in chosen)
        if len(keys)==2*FALLBACK_NEURONS_PER_VISIT:
            canonical=",".join(sorted(keys,key=decode))
            candidate={"id":"nec03h-fallback-"+hashlib.sha256(
                canonical.encode()).hexdigest()[:12],
                "nodes":sorted(keys,key=decode),
                "visits":scheduled,"source_selection":"focal-visit-nonhub-freq",
                "graph_edges":0,"graph_mean_phi":None}
        fallback_metadata={
            "strategy":FALLBACK_VISITS,
            "focal_positions":list(world["focal_positions"]),
            "scheduled_visits":scheduled,
            "channels_per_visit":FALLBACK_NEURONS_PER_VISIT,
            "min_source_context_frequency":FALLBACK_MIN_CONTEXT_FREQ,
            "visit_frequencies":visits_receipt,
            "selection_succeeded":candidate is not None,
        }
    if model_digest(model)!=before:
        raise AssertionError("source attribution changed frozen model")
    info={
        "schema":H_SCHEMA,
        "source_contexts":ATTR_CONTEXTS,
        "source_attribution_top_k":2,
        "candidate_roster_size_before_fallback":len(candidates),
        "valid_source_graph_candidates":len(accepted),
        "graph_has_qualifying_community":bool(accepted),
        "fallback_used":fallback_used,
        "fallback":fallback_metadata,
        "graph":graph,
        "candidate":candidate,
        "source_selection_uses_labels":False,
        "development_or_reserved_contexts_exposed_for_selection":False,
        "graph_parameters":{"hub_rate":0.90,"min_joint":3,
                            "min_phi":0.12,"max_size":12,
                            "max_candidates":8,
                            "max_occurrences_per_visit":8},
    }
    return candidate,graph,info


@torch.no_grad()
def score_clean(model,rows,world,arm):
    correct=0
    total=0
    for i in range(0,len(rows),128):
        bits=as_tensor(rows[i:i+128])
        truth=target_bits(bits,world,control=(arm=="additive"))
        correct+=int((model.logits(bits).argmax(1)==truth).sum())
        total+=len(truth)
    if total!=768:
        raise AssertionError("development score must use 48 complete context quartets")
    return {"correct":correct,"total":total,
            "accuracy":correct/total}


def run_calibration(index: int, out: Path, *, steps=MODEL_STEPS):
    w=training_world(index,calib=True)
    catalog,source=context_catalog(w)
    out.mkdir(parents=True,exist_ok=True)
    (out/"world_spec.json").write_bytes(encoded(w))
    (out/"context_source_manifest.json").write_bytes(encoded(source))
    both={}
    for arm in ("composed","additive"):
        model,trained=train_native_900(w,catalog,out,arm=arm,steps=steps)
        candidate,graph,selection=source_site_selection(model,catalog,w)
        raw=encoded({"world_id":w["world_id"],"model_sha256":
                     trained["final_model_sha256"],"source_sha256":source["sha256"],
                     "selection":selection})
        frozen_path=out/f"{arm}.selection.json"
        frozen_path.write_bytes(raw)
        score=score_clean(model,catalog["development"],w,arm)
        both[arm]={"trained":trained,"source_selection_sha256":sha_file(frozen_path),
                   "source_selection_kind":"graph" if candidate and not selection["fallback_used"]
                   else "fallback" if candidate else "none",
                   "old_graph_community_found":selection["graph_has_qualifying_community"],
                   "candidate_found":candidate is not None,
                   "candidate_id":candidate["id"] if candidate else None,
                   "source_fallback_used":selection["fallback_used"],
                   "clean":score}
    result={"schema":H_SCHEMA,"phase":"900-step-training-calibration-on-already-opened-roots",
            "index":index,"world_id":w["world_id"],
            "world_sha256":w["world_sha256"],"world_source_sha256":source["sha256"],
            "registered_world_population_sha256":registry()["registry_sha256"],
            "training_steps":steps,
            "full_calibration_recipe":steps==MODEL_STEPS,
            "calibration_world_was_already_opened_in_NEC03G":True,
            "source_candidate_fallback_frozen_before_dev_readout":True,
            "new_development_worlds_evaluated":0,
            "reserved_contexts_evaluated":0,
            "confirmation_worlds_trained_or_scored":0,
            "arms":both,
            "claim_class":"development-training-competence-calibration-only"}
    (out/"calibration.json").write_bytes(encoded(result))
    print(json.dumps({"world":w["world_id"],"steps":steps,
                      "scores":{k:v["clean"]["correct"] for k,v in both.items()},
                      "source_site":{k:v["source_selection_kind"] for k,v in both.items()},
                      "confirmation_worlds":0},sort_keys=True),flush=True)
    return result


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--development-world-index",type=int,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=MODEL_STEPS)
    a=ap.parse_args(argv)
    run_calibration(a.development_world_index,a.out,steps=a.steps)


if __name__=="__main__":
    main()
