"""LS-SAE04 independent stdlib-only replay of sealed calibration and native JVP outputs.

No Torch or NumPy is imported. The source calibration job has separately
validated the original NPZ file SHA and arrays before model inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_jvp_v1 import audit as JVP_AUD
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as NEW

ORIGINAL_CHECKPOINT="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
PROTOCOL_COMMIT="9e4aa45635b3a2dc6358926aac3ae10e54aaa9d3"
SCHEMA="LS-SAE04/preheldout-source-sealed-native-512D-JVP-v1"
FIT_SCHEMA="LS-SAE04/preheldout-source-frozen-PCA48-top4-SAE96-v1"


def close(a,b):
    if a is None or b is None:return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-5+3e-5*max(abs(a),abs(b))


def verify_source(r):
    if (r["schema"]!=SCHEMA or r["preregistration_commit"]!=PROTOCOL_COMMIT
        or r["source_checkpoint_sha256"]!=ORIGINAL_CHECKPOINT
        or r["source_control_code"]!="LS-SAE03 frozen direction/seed and forward-JVP engine"
        or r["radius"]!=.2 or r["depths"]!=[0,1,2]
        or r["rank_one_null_budget"]!=128):
        raise ValueError("original LS-SAE04 source/protocol metadata changed")
    m=r["calibration_manifest"]
    if (m["schema"]!=FIT_SCHEMA or m["protocol_commit"]!=PROTOCOL_COMMIT
        or m["preheldout_calibration_only"] is not True
        or m["original_checkpoint_file_sha256"]!=ORIGINAL_CHECKPOINT
        or m["calibration_seed_base"]!=NEW.BASES["fit_30"]
        or m["calibration_sample_count"]!=NEW.SIZES["fit_30"]
        or m["calibration_roster_sha256"]!=OLD.fingerprint(NEW.roster("fit_30"))
        or not math.isfinite(m["training_final_loss"])):
        raise ValueError("calibration original fit source metadata differs")
    for key in ("npz_file_sha256","fit_sha256","physical_dictionary_sha256"):
        if len(m[key])!=64 or any(c not in "0123456789abcdef" for c in m[key]):
            raise ValueError("calibration NPZ/weight checksum malformed")
    source=hashlib.sha256((json.dumps(m,sort_keys=True,indent=2)+"\n").encode()).hexdigest()
    if source!=r["calibration_manifest_file_sha256"]:
        raise ValueError("embedded manifest changed after archived original source fit")
    if r["mode"] not in ("pilot","full") or r["full_protocol"]!=(r["mode"]=="full"):
        raise ValueError("pilot/full source protocol designation differs")
    expected=NEW.verify_disjoint()
    if r["all_registered_roster_sha256"]!=expected:
        raise ValueError("registered calibration, pilot or heldout roster SHA changed")


def verify_family(r,family):
    from_name=("pilot_"+family.split("_")[1] if r["mode"]=="pilot" else family)
    source_roster=NEW.roster(from_name)
    saved=r["new_test_rosters"][family]
    if saved!=source_roster or OLD.fingerprint(saved)!=r["all_registered_roster_sha256"][from_name]:
        raise ValueError("test/engineering cohort changed or collided")
    block=r["heldout"][family]
    if block["source_family"]!=from_name:
        raise ValueError("pilot/test source cohort misidentified")
    rows=block["per_puzzle"]
    if len(rows)!=len(saved):
        raise ValueError("one hidden-state JVP receipt required per fresh Sudoku")
    for j,(row,sudoku) in enumerate(zip(rows,saved)):
        if row["calibration_manifest_sha256"]!=r["calibration_manifest_file_sha256"]:
            raise ValueError("per-puzzle source fit no longer points to sealed calibration")
        JVP_AUD.validate_case(row,sudoku,family,j)
    reconstructed=JVP_AUD.summary(rows,family)
    reported=block["statistics"]
    for key,value in reconstructed.items():
        observed=reported[key]
        if isinstance(value,list):
            if len(value)!=len(observed) or any(not close(a,b) for a,b in zip(value,observed)):
                raise ValueError("paired bootstrap bounds do not replay")
        elif isinstance(value,float):
            if not close(value,observed):
                raise ValueError("registered JVP statistic differs "+key)
        elif value!=observed:
            raise ValueError("registered event/gate differs "+key)


def audit(receipt):
    verify_source(receipt)
    if set(receipt["new_test_rosters"])!={"test_30","test_34"} or set(receipt["heldout"])!={"test_30","test_34"}:
        raise ValueError("source-registered 30/34-hole cohorts missing")
    for family in ("test_30","test_34"):
        verify_family(receipt,family)
    joint=(receipt["full_protocol"] and
           all(receipt["heldout"][f]["statistics"]["registered_success"] for f in ("test_30","test_34")))
    if joint!=receipt["registered_success_both_families"]:
        raise ValueError("registered two-family scientific gate modified")
    return {
      "schema":"LS-SAE04/independent-source-sealed-stdlib-JVP-replay-v1",
      "original_model_sha_and_preheldout_sealed_fit_manifest_checked":True,
      "fit_pilot_and_fresh_test_Sudoku_rosters_checked":True,
      "all97_token_jvp_512D_goal_readout_and_revgrad_replayed":True,
      "128_rank1_nulls_and_registered_bootstrap_gates_replayed":True,
      "n_fresh_puzzles":sum(len(x["per_puzzle"]) for x in receipt["heldout"].values()),
      "full_protocol":receipt["full_protocol"],
      "registered_success_both_families":joint,
      "limitation":"Independent numerical receipt replay; original calibration NPZ bytes are source-checked in model job, and no pretrained Torch forward passes run here."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    source=args.input.read_bytes()
    report=audit(json.loads(source))
    report["raw_json_sha256"]=hashlib.sha256(source).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
