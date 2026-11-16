"""AF-28 non-scoring diagnostic for Torch-batch feature replay variance.

Uses only original source train/dev rows, checkpoint, and training/dev model
features. NEVER opens heldout episodes or labels. Pure numerical diagnostic.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from experiments.activation_funnel_af25 import evaluate as original
from . import calibrate,calibration,gym


def compare(seed,original_dir,calibrator_dir,out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,provenance=original.load_frozen_original(original_dir,seed)
    frozen=calibrate.load_calibrator(calibrator_dir,seed,provenance)
    report={"experiment":"AF-28","seed":seed,
            "diagnostic":"train_dev_source_feature_fp32_delta_only",
            "heldout_data_seen":False,"new_model_training":False,"split":{}}
    for split in ("train","dev"):
        source_path=Path(calibrator_dir)/f"{split}_features.jsonl"
        source=[json.loads(x) for x in source_path.read_text().splitlines()]
        reread=calibrate.generate_features(model,split,split=="train")
        if len(source)!=len(reread):
            raise ValueError("diagnostic source roster count mismatch")
        deltas=[]
        examples=[]
        stability={arm:0 for arm in ("CAL","THR")}
        probability_max={arm:0. for arm in ("CAL","THR")}
        for a,b in zip(source,reread):
            if any(a[k]!=b[k] for k in
                   ("split","case_id","family","state","public_graph_sha256","stop_target")):
                raise ValueError("diagnostic original source/label changed")
            diff=np.abs(np.asarray(a["features"])-np.asarray(b["features"]))
            deltas.append(diff)
            if diff.max()>1e-5 and len(examples)<12:
                examples.append({"id":a["case_id"],"state":a["state"],
                                 "field_deltas":diff.tolist(),
                                 "source_features":a["features"],
                                 "replay_features":b["features"]})
            for arm in ("CAL","THR"):
                ca=frozen["classifiers"][arm]
                pa=calibration.predict(ca,a["features"])
                pb=calibration.predict(ca,b["features"])
                probability_max[arm]=max(probability_max[arm],abs(pa-pb))
                t=ca["dev"]["chosen_threshold"]
                stability[arm]+=int((pa>=t)!=(pb>=t))
        delta=np.asarray(deltas)
        report["split"][split]={
            "rows":len(source),
            "max_by_feature":delta.max(axis=0).tolist(),
            "p99_by_feature":np.quantile(delta,.99,axis=0).tolist(),
            "abs_deltas_above_by_threshold":{
                str(v):int((delta>v).any(axis=1).sum())
                for v in (1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3)
            },
            "max_abs_overall":float(delta.max()),
            "calibrator_max_probability_change":probability_max,
            "calibrated_action_flip_counts":stability,
            "first_12_offenders":examples,
        }
    p=Path(out)
    p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=gym.SEEDS,required=True)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--calibrator-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=compare(a.seed,a.original_zip_dir,a.calibrator_dir,a.out)
    print(json.dumps({split:{"max":v["max_abs_overall"],
                              "over_1e5":v["abs_deltas_above_by_threshold"]["1e-05"],
                              "action_flips":v["calibrated_action_flip_counts"]}
                      for split,v in result["split"].items()}))


if __name__=="__main__":
    main()
