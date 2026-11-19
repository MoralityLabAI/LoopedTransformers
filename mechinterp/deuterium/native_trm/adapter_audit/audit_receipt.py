"""Independent stdlib-only receipt audit, no trained model or source generation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA="DEUTERIUM-ADAPTER-01/reused-module-cache-source-only-v1"
PASSES=4
TARGET="block.chan_mix.2"
UPSTREAM="2df0092d5be4d8e4748b3c66ef8325dedef17549"


def audit(source,output):
    d=json.loads(Path(source).read_text())
    errors=[]
    def must(ok,name):
        if not ok:errors.append(name)
    must(d.get("schema")==SCHEMA and d.get("status")=="SOURCE_ONLY_PASS","source_only_schema")
    must(d.get("trained_model_steering_scored") is False and
         d.get("new_test_source_generated") is False and
         d.get("test_source_seed_731057_status")==
         "ALREADY_EXPOSED_IN_ORIGINAL_ACTIONS_RUN_44465490737","no_new_heldout_or_steering")
    must(d.get("source_seed")==(
        61076 if d.get("mode")=="smoke_untrained" else 31041),"source_seed")
    must(d.get("source_input_sha256") and len(d["source_input_sha256"])==64,"source_checksum")
    records=d.get("models",{})
    expected_keys={"untrained_61076"} if d.get("mode")=="smoke_untrained" else {"103","211","307"}
    must(set(records)==expected_keys,"source_only_model_roster")
    measured={}
    for key,item in records.items():
        prefix=key+":"
        must(item.get("recurrent_target")==TARGET and item.get("n_model_weights")==44001,prefix+"native_target")
        must(item.get("n_original_tied_module_calls")==PASSES and
             item.get("stock_cache_values_per_module")==1 and
             item.get("stock_cache_keys")==[TARGET],prefix+"four_inputs_one_cache")
        must(item.get("last_cached_native_input_error")==0 and
             item.get("last_cached_CI_mask_error")<=1e-7,prefix+"final_input_CI")
        must(item.get("component_mask_used_at_all_tied_depths") is True and
             item.get("all_four_mask_uses_are_same_tensor_object") is True,prefix+"reused_mask")
        must(item.get("upstream_param_decomp_commit")==UPSTREAM,prefix+"pinned_optimizer")
        for field in ("mean_absolute_per_depth_mask_mismatch_against_last_mask",
                      "max_absolute_per_depth_mask_mismatch_against_last_mask",
                      "physical_input_rms_distance_from_last_pass"):
            vals=item.get(field,[])
            must(len(vals)==4 and all(type(x) in (float,int) and math.isfinite(x) and x>=0
                                      for x in vals),prefix+field)
        must(item.get("cached_single_pass_f32_bytes_per_batch",0)>0 and
             item.get("all_four_inputs_f32_bytes_per_batch")==
             4*item.get("cached_single_pass_f32_bytes_per_batch",0),
             prefix+"four_depth_memory_count")
        must(item.get("original_native_final_logits_shape")==[8,81,10] and
             item.get("masked_module_native_final_logits_shape")==[8,81,10],
             prefix+"full_native_outputs")
        measured[key]={
          "mean_mask_mismatch":item.get("mean_absolute_per_depth_mask_mismatch_against_last_mask"),
          "input_rms_from_last":item.get("physical_input_rms_distance_from_last_pass"),
        }
    r={
        "schema":"DEUTERIUM-ADAPTER-01/stdlib-receipt-audit-v1",
        "status":"PASS" if not errors else "FAIL",
        "errors":errors,
        "torch_imported":"torch" in sys.modules,
        "source_only":True,
        "n_model_roots":len(records),
        "measured":measured,
        "first_heldout_exposure_unchanged":True,
        "source_file_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "conclusion":"ONE_PATH_CACHE_LAST_PASS_AND_REUSED_CI_MASK",
        "scientific_scope":"ADAPTER_ENGINEERING_DIAGNOSTIC_ONLY",
    }
    if r["torch_imported"]:
        r["status"]="FAIL"
        r["errors"].append("torch_imported")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    if r["status"]!="PASS":
        raise RuntimeError("source-only adapter receipt audit failed: "+str(errors))
    return r


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    d=audit(a.source,a.out)
    print("DEUTERIUM_SOURCE_ONLY_RECURRENT_CACHE_AUDIT",d["status"],flush=True)


if __name__=="__main__":
    main()
