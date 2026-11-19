"""DEUTERIUM-ADAPTER-01: post-result source-only audit of tied-block VPD caching.

The pinned stock param_decomp ComponentModel caches recurrent-module inputs
under ONE module-path dictionary key. This experiment compares that cache to
a separately hooked four-pass ledger, and checks whether its CI mask is reused
at each application of the *same* weight matrix. The original VPD result and
its score/test source bytes are frozen and are never recomputed here.

Smoke: excluded untrained seed 61076. Trained mode: exact first LGA-02 teachers
on eight pre-existing original training strings from seed 31041 only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data, tensor_int64_sha
from mechinterp.loopscope05.fit_and_freeze import load_frozen
from mechinterp.deuterium.native_trm.preflight import MODEL_SHA, META_SHA, sha256
from mechinterp.deuterium.native_trm.fit_and_select import config

SCHEMA="DEUTERIUM-ADAPTER-01/reused-module-cache-source-only-v1"
TARGET="block.chan_mix.2"
N_LOOPS=4
MODEL_DIM=48
WIDTH=128
COMPONENTS=16
TRAINED_SEEDS=(103,211,307)
SMOKE_SEED=61076
TRAIN_SEED=31041
N_SOURCES=8
ORIGINAL_UPSTREAM_COMMIT="2df0092d5be4d8e4748b3c66ef8325dedef17549"


def record_one(model, x, source_id, source_kind, fitted_checkpoint=None):
    from param_decomp.models.batch_and_loss_fns import run_batch_first_element
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.models.components import make_mask_infos
    from param_decomp.utils.module_utils import expand_module_patterns

    cfg=config(SMOKE_SEED,smoke=True)
    model.eval().requires_grad_(False)
    if model.n_steps!=4 or sum(p.numel() for p in model.parameters())!=44001:
        raise RuntimeError("original four-pass model changed")
    target=model.get_submodule(TARGET)
    cm=ComponentModel(
        target_model=model,run_batch=run_batch_first_element,
        module_path_info=expand_module_patterns(model,cfg.all_module_info),
        ci_config=cfg.ci_config,sigmoid_type=cfg.sigmoid_type,
    )
    cm.eval().requires_grad_(False)
    importance_source="DIAGNOSTIC_UNTRAINED_CI"
    fitted_checkpoint_sha=None
    if fitted_checkpoint is not None:
        fitted_checkpoint=Path(fitted_checkpoint)
        state=torch.load(fitted_checkpoint,map_location="cpu",weights_only=True)
        for name,value in model.state_dict().items():
            saved=state.get("target_model."+name)
            if saved is None or not torch.equal(saved,value):
                raise RuntimeError("original saved VPD checkpoint targets different teacher weights")
        cm.load_state_dict(state,strict=True)
        importance_source="ORIGINAL_STAGEB_FITTED_VPD_CI"
        fitted_checkpoint_sha=sha256(fitted_checkpoint)
    native_input=[]
    def native_pre_hook(module,args):
        native_input.append(args[0].detach().clone())

    hook=target.register_forward_pre_hook(native_pre_hook)
    try:
        with torch.no_grad():
            first=cm((x,),cache_type="input")
    finally:
        hook.remove()
    if len(native_input)!=4 or set(first.cache)!={TARGET}:
        raise RuntimeError("stock VPD tied-module input cache/actual pass count unexpected")
    from_cache=first.cache[TARGET]
    if tuple(from_cache.shape)!=(len(x),81,WIDTH):
        raise RuntimeError("stock VPD module activation tensor shape drift")
    max_last_error=float((from_cache-native_input[-1]).abs().max())
    if max_last_error!=0:
        raise RuntimeError("single-key VPD cache is not actually last-call tensor")

    with torch.no_grad():
        final_ci=cm.calc_causal_importances(first.cache,sampling=cfg.sampling)
        selected_mask=final_ci.lower_leaky[TARGET]
        if tuple(selected_mask.shape)!=(len(x),81,COMPONENTS):
            raise RuntimeError("CI mask does not match source batch/token shapes")
        per_depth_masks=[]
        for acts in native_input:
            values=cm.ci_fn({TARGET:acts})[TARGET]
            per_depth_masks.append(cm.lower_leaky_fn(values).detach().clone())
    if not bool(torch.isfinite(selected_mask).all()):
        raise RuntimeError("nonfinite last-pass component importance")
    mask_last_error=float((selected_mask-per_depth_masks[-1]).abs().max())
    if mask_last_error>1e-7:
        raise RuntimeError("final CI did not use the last recurrent call's input")

    # The parent model is called with one fixed mask dictionary throughout all
    # native F applications. Catch actual calls to LinearComponents to prove
    # that the mask consumed at depths 0..3 is the same tensor.
    component=cm.components[TARGET]
    seen_masks=[]
    def mask_pre_hook(module,args,kwargs):
        seen_masks.append(kwargs.get("mask"))
    component_hook=component.register_forward_pre_hook(mask_pre_hook,with_kwargs=True)
    try:
        with torch.no_grad():
            masked_native=cm((x,),mask_infos=make_mask_infos({TARGET:selected_mask}))
    finally:
        component_hook.remove()
    if len(seen_masks)!=4:
        raise RuntimeError("masked original recurrent module did not execute four times")
    if not all(torch.equal(m,selected_mask) for m in seen_masks):
        raise RuntimeError("CI mask differs from selected final-pass mask at some application")
    if not all(m is seen_masks[0] for m in seen_masks):
        raise RuntimeError("CI masks were reallocated between tied recurrent applications")
    differences=[float((per_depth_masks[k]-selected_mask).abs().mean()) for k in range(4)]
    max_differences=[float((per_depth_masks[k]-selected_mask).abs().max()) for k in range(4)]
    input_rms=[
        float((native_input[k]-native_input[-1]).square().mean().sqrt())
        for k in range(4)
    ]
    state_bytes=int(native_input[0].numel()*native_input[0].element_size())
    return {
        "model_id":source_id,
        "source_kind":source_kind,
        "importance_network_source":importance_source,
        "fitted_vpd_checkpoint_sha256":fitted_checkpoint_sha,
        "n_model_weights":44001,
        "recurrent_target":TARGET,
        "n_original_tied_module_calls":len(native_input),
        "stock_cache_keys":list(first.cache),
        "stock_cache_values_per_module":1,
        "last_cached_native_input_error":max_last_error,
        "last_cached_CI_mask_error":mask_last_error,
        "component_mask_used_at_all_tied_depths":True,
        "all_four_mask_uses_are_same_tensor_object":True,
        "mean_absolute_per_depth_mask_mismatch_against_last_mask":differences,
        "max_absolute_per_depth_mask_mismatch_against_last_mask":max_differences,
        "physical_input_rms_distance_from_last_pass":input_rms,
        "cached_single_pass_f32_bytes_per_batch":state_bytes,
        "all_four_inputs_f32_bytes_per_batch":4*state_bytes,
        "original_native_final_logits_shape":list(first.output.shape),
        "masked_module_native_final_logits_shape":list(masked_native.shape),
        "upstream_param_decomp_commit":ORIGINAL_UPSTREAM_COMMIT,
        "first_731057_heldout_generated":False,
        "calibration_731055_opened":False,
        "scientific_inference":"SOURCE_CODE_ADAPTER_BEHAVIOR_ONLY",
    }


def run(out,mode,teacher_bundle=None,teacher_meta=None,fitted_root=None):
    torch.set_num_threads(2)
    t0=time.perf_counter()
    if mode=="smoke":
        torch.manual_seed(SMOKE_SEED)
        x,_=fixed_data(N_SOURCES,SMOKE_SEED)
        roster={"mode":"smoke_untrained","source_seed":SMOKE_SEED,
                "source_input_sha256":tensor_int64_sha(x),"models":{}}
        roster["models"]["untrained_"+str(SMOKE_SEED)]=record_one(
            TinyRecursiveModel(48,4),x,SMOKE_SEED,"untrained_source_only")
    elif mode=="trained":
        if teacher_bundle is None or teacher_meta is None or fitted_root is None:
            raise ValueError("trained old-source audit requires pinned LGA-02 bundle, metadata and original fitted VPD root")
        if sha256(teacher_bundle)!=MODEL_SHA or sha256(teacher_meta)!=META_SHA:
            raise RuntimeError("original LGA-02 trained source artifact SHA mismatch")
        meta,bundle=load_frozen(teacher_bundle,teacher_meta)
        if meta.get("mode")!="full" or meta.get("teacher_init_seeds")!=list(TRAINED_SEEDS):
            raise RuntimeError("original trained teacher source lineage missing")
        x,_=fixed_data(N_SOURCES,TRAIN_SEED)
        roster={"mode":"old_train_reuse","source_seed":TRAIN_SEED,
                "source_input_sha256":tensor_int64_sha(x),"models":{}}
        for seed in TRAINED_SEEDS:
            model=TinyRecursiveModel(48,4)
            model.load_state_dict(bundle["models"][str(seed)],strict=True)
            checkpoint=Path(fitted_root)/f"seed_{seed}"/"vpd"/"model_64.pth"
            receipt=json.loads((Path(fitted_root)/f"seed_{seed}"/"seed_result.json").read_text())
            if (receipt.get("teacher_seed")!=seed or
                receipt.get("vpd",{}).get("checkpoint_sha256")!=sha256(checkpoint) or
                receipt.get("heldout_generated") is not False or
                receipt.get("selected_before_heldout") is not True):
                raise RuntimeError("original Stage B source-frozen learned VPD CI provenance missing")
            roster["models"][str(seed)]=record_one(
                model,x,seed,"old_original_train_only",fitted_checkpoint=checkpoint)
            roster["models"][str(seed)]["original_stageb_seed_result_sha256"]=sha256(
                Path(fitted_root)/f"seed_{seed}"/"seed_result.json")
            roster["models"][str(seed)]["original_teacher_state_tensor_sha256"]=meta[
                "teacher_receipts"][str(seed)]["model_state_sha256"]
    else:
        raise ValueError("source-only modes are smoke or trained")
    if not all(q["n_original_tied_module_calls"]==4 and
               q["component_mask_used_at_all_tied_depths"] for q in roster["models"].values()):
        raise RuntimeError("registered native tied-block cache instrumentation gate failed")
    roster.update({
        "schema":SCHEMA,
        "status":"SOURCE_ONLY_PASS",
        "time_seconds":time.perf_counter()-t0,
        "torch":torch.__version__,
        "python":platform.python_version(),
        "test_source_seed_731057_status":"ALREADY_EXPOSED_IN_ORIGINAL_ACTIONS_RUN_44465490737",
        "new_test_source_generated":False,
        "trained_model_steering_scored":False,
        "n_model_roots":len(roster["models"]),
    })
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(roster,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return roster


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["smoke","trained"],required=True)
    parser.add_argument("--teacher-bundle",type=Path)
    parser.add_argument("--teacher-meta",type=Path)
    parser.add_argument("--fitted-root",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    result=run(a.out,a.mode,a.teacher_bundle,a.teacher_meta,a.fitted_root)
    print("DEUTERIUM_RECURRENT_COMPONENT_CACHE_AUDIT",result["status"],result["mode"],flush=True)


if __name__=="__main__":
    main()
