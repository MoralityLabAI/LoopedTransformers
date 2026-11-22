"""NEC-03V-B independent original-source behavioral replay.

All 12 recurrent visits and both teacher arms are independently executed via
a fresh MLP-down pre-forward hook (without calling NEC03V scorer or its donor
construction). Restore exactly the original source native/SAE checkpoints.
Rebuild each full-affine donor and physically matched group/sham removal,
then verify every recorded source-only logit and original teacher label.
"""
from __future__ import annotations

import argparse
from hashlib import sha1,sha256
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack
from .nec03v_source_attest import attest,canonical,sha256_file

ROOTS=tuple(range(8))
ARMS=("composed","additive")
PANEL=tuple(range(24,32))
MODES=("full_affine","group_025","sham_025","group_05",
       "sham_05","group_1","sham_1")
SCHEMA="NEC-03V/source-only-group-removal-behavior/v1"


def lock_check():
    base=Path(__file__).parent
    entry=json.loads((base/"NEC03V_BEHAVIOR_METHOD_LOCK.json").read_text("utf-8"))
    original=json.loads((base/"NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json").read_text("utf-8"))
    if (entry.get("schema")!="NEC-03V/frozen-original-source-behavior-method/v1"
            or entry.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"
            or entry.get("new_training_updates")!=0
            or entry.get("recruited_or_confirmation_worlds_allowed") is not False
            or entry.get("allowed_original_roots")!=list(ROOTS)
            or original.get("fixed_ITT_zero_roots")!=[3,5]
            or original.get("source_groups")!=list(PANEL)):
        raise AssertionError("NEC03V independent original-source registration changed")
    for path,blob in {**entry["git_blob_sha1"],
                      **entry["dependency_git_blob_sha1"]}.items():
        contents=(base.parents[1]/path).read_bytes()
        digest=sha1(b"blob "+str(len(contents)).encode()+b"\0"+contents).hexdigest()
        if digest!=blob:
            raise AssertionError("NEC03V method byte identity not registered")
    return original


def original_source_forward(model,bits,donors,gates):
    """A separate implementation of the physical source donor trace."""
    import torch
    visits=0
    def alter(_module,arguments):
        nonlocal visits
        visit=visits
        visits+=1
        if visit not in donors:
            return None
        original=arguments[0]
        if tuple(original.shape)!=(len(bits),2,64):
            raise ValueError("source recurrence memory token changed shape")
        proposed=original.clone()
        proposed[:,1,:]=original[:,1,:]*gates[visit][None,:]+donors[visit]
        return (proposed,)
    hook=model.block.mlp_down.register_forward_pre_hook(alter)
    try:
        with torch.no_grad():
            logits=model.logits(bits)
    finally:
        hook.remove()
    if visits!=12 or tuple(logits.shape)!=(32,2):
        raise ValueError("independent source native recurrent history changed")
    return logits


def independent_donor(frozen,source_cases,registered,mode):
    import torch
    sites=frozen["source_sites"]
    n=32
    scale=frozen["scale"]
    sae=frozen["sae"]
    down=frozen["model"].block.mlp_down.weight.detach()
    donors={}
    gates={}
    for visit in sorted(sites):
        token=str(visit)
        code=torch.zeros((n,128),dtype=down.dtype)
        for row_index,item in enumerate(source_cases):
            for feature,value in item["visits"][token]:
                code[row_index,int(feature)]=float(value)
        mask=torch.ones(64,dtype=down.dtype)
        for hub in frozen["source_hubs"].get(visit,()):
            mask[int(hub)]=0
        base=((code@sae.decoder.weight.T+sae.decoder.bias)*scale+frozen["mean"])*mask
        removal=torch.zeros_like(base)
        if mode!="full_affine":
            kind="source" if mode.startswith("group") else "sham"
            for j,case in enumerate(registered):
                fit=case["visits"][token]
                if fit["effective_zero_pair"]:
                    continue
                ids=fit["source" if kind=="source" else "sham"]
                gain=fit[kind+"_gain"]
                raw=torch.zeros(64,dtype=down.dtype)
                for idx in ids:
                    raw+=code[j,idx]*sae.decoder.weight[:,idx]*scale*mask
                physical=raw*float(gain)
                independent32=physical@down.T
                expected=fit["effective_source32" if kind=="source" else "effective_sham32"]
                if max(abs(a-b) for a,b in zip(independent32.tolist(),expected))>2e-4:
                    raise ValueError("independently built source removal violates 32D physical comparator")
                if not 0<=float(gain)<=1.0000001:
                    raise ValueError("source perturbation amplified after registration")
                removal[j]=physical
        fraction=(.25 if mode.endswith("025") else
                  .5 if mode.endswith("05") else
                  0. if mode=="full_affine" else 1.)
        donors[visit]=base-fraction*removal
        gates[visit]=1-mask
    return donors,gates


def verify(root:int,source:Path,t:Path,reset:Path|None,pack:Path,
           physical:Path,report_dir:Path):
    if type(root) is not int or root not in ROOTS:
        raise PermissionError("independent behavior replay accepts original eight roots only")
    panel=lock_check()
    trusted=attest(root,source,t,reset)
    pin=panel["roots"][str(root)]
    meta,shards,manifest_sha=read_pack(pack)
    physical_json=json.loads(physical.read_text("utf-8"))
    summary_file=report_dir/"source_behavior_report.json"
    summary=json.loads(summary_file.read_text("utf-8"))
    raw_file=report_dir/"source_behavior_raw.jsonl"
    if (meta["world_index"]!=root
            or manifest_sha!=pin["source_pack_manifest_sha256"]
            or physical_json["frozen_group_selection_sha256"]!=pin["frozen_group_selection_sha256"]
            or sha256_file(physical)!=pin["physical_preflight_sha256"]
            or summary.get("schema")!=SCHEMA
            or summary.get("world_index")!=root
            or summary.get("source_pack_manifest_sha256")!=manifest_sha
            or summary.get("registered_source_panel_lock_sha256")!=sha256_file(
                Path(__file__).with_name("NEC03V_BEHAVIOR_SOURCE_PANEL_LOCK.json"))
            or summary.get("registered_behavior_method_lock_sha256")!=sha256_file(
                Path(__file__).with_name("NEC03V_BEHAVIOR_METHOD_LOCK.json"))
            or summary.get("raw_behavior_sha256")!=sha256_file(raw_file)
            or summary.get("original_NEC03T_report_sha256")!=trusted["original_nec03t_report_sha256"]
            or summary.get("source_only_native_forward_calls")!=112
            or summary.get("new_optimizer_updates")!=0
            or summary.get("recruited_or_embargoed_world_queries")!=0
            or summary.get("development_evaluation_or_reserved_queries")!=0
            or summary.get("old_confirmation_queries")!=0
            or summary.get("scientific_gate")!="REDESIGN_OR_PILOT_ONLY"):
        raise ValueError("independent source behavior result receipt or raw hash differs")
    data=[json.loads(line) for line in raw_file.read_text("utf-8").splitlines()
          if line.strip()]
    if len(data)!=16:
        raise ValueError("source behavioral results require eight groups per original model arm")
    by_id={(r["arm"],r["source_group"]):r for r in data}
    physical_lookup={(r["arm"],r["group"],r["mask"]):r
                     for r in physical_json["case_receipts"]}
    if len(by_id)!=16 or len(physical_lookup)!=512:
        raise ValueError("source case/arm identities duplicate or missing")
    original_meta=json.loads((t/"nec03t_source_necessity_report.json").read_text("utf-8"))
    import torch
    if torch.version.cuda is not None or getattr(torch.version,"hip",None) is not None:
        raise RuntimeError("independent replay requires CPU-only PyTorch")
    from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
    from mechinterp.neuron_communities.nec03l_five_input_worlds import labels
    from research.pef07.pef07 import as_tensor,model_digest

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    counts={}
    forward_cost={}
    max_logit_diff=0.
    for arm in ARMS:
        frozen=load_source_arm(root,arm,source,reset=reset)
        original_model_digest=model_digest(frozen["model"])
        if (original_model_digest!=trusted["source_checkpoint_receipts"][arm]["native_sha256"]
                or frozen["sae_sha256"]!=trusted["source_checkpoint_receipts"][arm]["sae_sha256"]):
            raise ValueError("original model and SAE differ in independent source replay")
        raw_old=t/original_meta["arms"][arm]["raw_file"]
        if sha256_file(raw_old)!=original_meta["arms"][arm]["raw_sha256"]:
            raise ValueError("original NEC03T case-arm archive changed")
        old=[json.loads(s) for s in raw_old.read_text().splitlines() if s.strip()]
        if len(old)!=8:
            raise ValueError("original full-affine source case count changed")
        counts[arm]={name:0 for name in MODES}
        forward_cost[arm]=0
        with torch.no_grad():
            for index,g in enumerate(PANEL):
                case=by_id[arm,g]
                previous=old[index]
                expected_inputs=frozen["catalog"]["train"][g*32:(g+1)*32]
                if (previous["source_group"]!=g
                        or [tuple(v) for v in previous["input_histories"]]!=expected_inputs
                        or [tuple(v) for v in case["source_histories"]]!=expected_inputs
                        or case["focal_mask_order"]!=list(range(32))):
                    raise ValueError("original source case ownership diverged")
                bits=as_tensor(expected_inputs)
                truth=labels(bits,frozen["world"],control=(arm=="additive")).tolist()
                if truth!=case["labels"] or truth!=previous["true_teacher_labels"]:
                    raise ValueError("independent original-source teacher labels disagree")
                rows=shards[arm]["cases"][index*32:(index+1)*32]
                phys=[physical_lookup[arm,g,m] for m in range(32)]
                if ([sha256(canonical(r).encode()).hexdigest() for r in phys]
                        !=case["registered_physical_case_hashes"]):
                    raise ValueError("source group physical case identity changed")
                if case["eligible"] != [r["eligible_all_visits"] for r in phys]:
                    raise ValueError("original source case validity changed")
                for mode in MODES:
                    donors,gates=independent_donor(frozen,rows,phys,mode)
                    actual=original_source_forward(frozen["model"],bits,donors,gates)
                    expected=torch.tensor(case["logits"][mode],dtype=actual.dtype)
                    if expected.shape!=(32,2):
                        raise ValueError("model response has wrong original source-case shape")
                    error=float((actual-expected).abs().max())
                    max_logit_diff=max(error,max_logit_diff)
                    if not torch.allclose(actual,expected,atol=1e-4,rtol=2e-5):
                        raise ValueError("independently replayed source behavioral logits changed")
                    if mode=="full_affine":
                        original_logits=torch.tensor(previous["logits"]["full_affine"],dtype=actual.dtype)
                        if not torch.allclose(actual,original_logits,atol=1e-3,rtol=2e-5):
                            raise ValueError("original NEC03T full-affine baseline changed")
                    counts[arm][mode]+=sum(
                        int(a==b) for a,b in zip(actual.argmax(1).tolist(),truth))
                    forward_cost[arm]+=1
        if model_digest(frozen["model"])!=original_model_digest:
            raise ValueError("independent source reviewer altered frozen original model")
        if counts[arm]!=summary["correctness_totals"][arm]:
            raise ValueError("registered source correct-count ledger differs in independent replay")
    observed=((counts["composed"]["sham_1"]-counts["composed"]["group_1"])
              -(counts["additive"]["sham_1"]-counts["additive"]["group_1"]))/256
    if not math.isclose(observed,summary["raw_root_composed_minus_additive_correctness_DID"],
                        abs_tol=1e-9):
        raise ValueError("independent root primary source contrast changed")
    if (summary["registered_ITT_root_effect"]!=(0. if root in (3,5) else observed)
            or summary["original_fixed_ITT_zero"]!=(root in (3,5))):
        raise ValueError("fixed ITT zero or registered source root effect changed")
    return {
        "schema":"NEC-03V/independent-original-source-group-removal-behavior/v1",
        "world_index":root,
        "world_id":trusted["world_id"],
        "independent_behavior_replay_passed":True,
        "original_source_case_arm_records":512,
        "independent_replayed_native_forwards":sum(forward_cost.values()),
        "original_pilot_native_forwards":summary["source_only_native_forward_calls"],
        "maximum_absolute_native_logit_replay_error":max_logit_diff,
        "independent_correctness_totals":counts,
        "original_primary_ITT_effect":0. if root in (3,5) else observed,
        "raw_source_root_effect":observed,
        "source_behavior_raw_sha256":sha256_file(raw_file),
        "source_behavior_report_sha256":sha256_file(summary_file),
        "new_optimizer_updates":0,
        "new_recruited_or_embargoed_root_reads":0,
        "development_eval_reserved_or_old_confirmation_reads":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "population_claim_permitted":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--nec03t-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--geometry-pack",type=Path,required=True)
    p.add_argument("--physical-preflight",type=Path,required=True)
    p.add_argument("--source-behavior",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        raise ValueError("independent source behavior audit output must be new")
    receipt=verify(a.world_index,a.source_root,a.nec03t_root,a.reset_root,
                   a.geometry_pack,a.physical_preflight,a.source_behavior)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(canonical(receipt)+"\n",encoding="utf-8")
    print(canonical(receipt))
    return receipt


if __name__=="__main__":
    main()
