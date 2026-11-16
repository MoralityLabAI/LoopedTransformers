"""LS-SAE04 source-calibration-immutable native 512D Jacobian-vector experiment.

Reuses the LS-SAE03 already CPU-tested full 97×512 JVP measurement geometry;
the only change is a separately preregistered, FIT-ONLY preheldout calibration
artifact and genuinely fresh LS-SAE04 test/pilot solution families.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from mechinterp.loopscope02.scripts import trm_sudoku as TS
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT
from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_jvp_v1 import run_native as MEASURE
from mechinterp.sae_causal_transfer.native_jvp_v1 import geometry as G
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as NEW
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as PACK

SCHEMA="LS-SAE04/preheldout-source-sealed-native-512D-JVP-v1"


def run(*,manifest_path,npz_path,pilot=False,out=None):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    fit,manifest,manifest_bytes_sha=PACK.load_pack(npz_path,manifest_path)
    # Original source and saved artifact must both agree before new hidden states.
    model,source_sha=TS.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if source_sha!=manifest["original_checkpoint_file_sha256"]:
        raise ValueError("original pretrained source file SHA mismatch")
    model.eval().requires_grad_(False)
    # All independent solution rosters are source-locked and model-free.
    full_roster_manifests=NEW.verify_disjoint()
    cohorts=("pilot_30","pilot_34") if pilot else ("test_30","test_34")
    observed={}
    selected_rosters={}
    for source_family,score_family in zip(cohorts,("test_30","test_34")):
        rosters=NEW.roster(source_family)
        selected_rosters[score_family]=rosters
        stages,anchors=FIT.stage_one(model,rosters,retain_zl=True)
        cases=[]
        for p,row in enumerate(rosters):
            pos=NEW.median_blank(row["x"])
            reduced=fit.encode(stages[p,pos])
            case=MEASURE.measure_case(model,anchors[p],reduced,fit,row,score_family,pos)
            case.update({"puzzle_index":p,"puzzle_seed":row["seed"],
                         "calibration_manifest_sha256":manifest_bytes_sha})
            case["sae_event"]=G.attenuation_event(case["directions"]["sae"])
            case["random_event"]=G.attenuation_event(case["directions"]["random"])
            cases.append(case)
            print("LS_SAE04_JVP",source_family,p,
                  case["sae_event"],case["random_event"],flush=True)
        stats=G.family_score(cases,score_family)
        observed[score_family]={"source_family":source_family,
                                "per_puzzle":cases,"statistics":stats}
        print("LS_SAE04_REGISTERED",source_family,json.dumps(stats,sort_keys=True),flush=True)
    result={
       "schema":SCHEMA,
       "preregistration_commit":PACK.PROTOCOL_COMMIT,
       "mode":"pilot" if pilot else "full",
       "calibration_manifest":manifest,
       "calibration_manifest_file_sha256":manifest_bytes_sha,
       "source_checkpoint_sha256":source_sha,
       "all_registered_roster_sha256":full_roster_manifests,
       "new_test_rosters":selected_rosters,
       "source_control_code":"LS-SAE03 frozen direction/seed and forward-JVP engine",
       "radius":G.RADIUS,
       "depths":list(G.DEPTHS),
       "rank_one_null_budget":G.N_RANK_NULL,
       "heldout":observed,
       "full_protocol":not pilot,
       "registered_success_both_families":bool(
           not pilot and all(observed[f]["statistics"]["registered_success"]
                             for f in ("test_30","test_34"))),
       "scope":"Native full pretrained 7M LoopedTRM; checkpoint and calibration weights source-locked before unseen test."
    }
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--calibration-manifest",type=Path,required=True)
    p.add_argument("--calibration-npz",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=run(manifest_path=a.calibration_manifest,npz_path=a.calibration_npz,
          pilot=a.pilot,out=a.out)
    print(json.dumps({"output":str(a.out),"mode":r["mode"],
                      "success":r["registered_success_both_families"],
                      "family":{k:v["statistics"] for k,v in r["heldout"].items()}},indent=2))


if __name__=="__main__":
    main()
