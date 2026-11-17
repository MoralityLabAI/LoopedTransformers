"""AF-29 train/dev-only STOP calibration, eight fixed original checkpoints.

The original recurrent model and Direct rule head are frozen. The sole newly
fitted parameters are a per-source-seed six-input linear logistic STOP gate
and an equally budgeted one-input margin logistic control. Calibration
thresholds are frozen from development root cases before heldout scoring.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch

from experiments.activation_funnel_af25 import evaluate as original
from . import gym,transfer_planner,calibration

HERE=Path(__file__).resolve().parent
SEEDS=gym.SEEDS
SOURCE_FILES=(
    "PROTOCOL.json","gym.py","transfer_planner.py","calibration.py",
    "calibrate.py","evaluate.py","audit.py","aggregate.py",
    "tests/test_structure.py","tests/test_calibration.py","README.md",
    "../../.github/workflows/activation-funnel-af29-preflight.yml",
    "../../.github/workflows/activation-funnel-af29.yml",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {s:sha((HERE/s).read_bytes()) for s in SOURCE_FILES}


def write_json(path,data):
    p=Path(path)
    p.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")


def generate_features(model,split,include_successors=False):
    if split not in ("train","dev"):
        raise ValueError("STOP fitting may access train or dev, never heldout")
    samples=[]
    for c in gym.cases_for(split):
        states=gym.training_states(c) if include_successors else (gym.initial_facts(c),)
        for state in states:
            if gym.GOAL in state or not gym.enabled(c,state):
                continue
            samples.append((c,state))
    rows=[]
    for start in range(0,len(samples),32):
        part=samples[start:start+32]
        plans=transfer_planner.decisions(
            model,[c for c,s in part],[s for c,s in part]
        )
        for (c,state),plan in zip(part,plans):
            features=calibration.feature_row(plan,state,gym.enabled(c,state))
            # Exact reachability labels are source-approved only for TRAIN/DEV.
            truth=gym.shortest_steps(c,state)
            if (truth is not None)!=c.designed_positive:
                raise ValueError("positive Horn reachability changed during source-only sampling")
            rows.append({
                "experiment":"AF-29","split":split,
                "case_id":c.id,"family":c.family,
                "state":sorted(state),
                "public_graph_sha256":gym.digest(gym.public_features(c)),
                "stop_target":int(truth is None),
                "features":features,
                "source_node_count":plan["trace"]["nodes_forward_count"],
            })
    if not rows:
        raise ValueError("source calibration features missing")
    if split=="dev" and len(rows)!=192:
        raise ValueError("development threshold candidates must be exact 192 roots")
    if split=="train" and len({r["case_id"] for r in rows})!=192:
        raise ValueError("training features must cover all 192 original cases")
    return rows


def calibrate(seed,original_zip_dir,out_dir):
    if seed not in SEEDS:
        raise ValueError("unregistered frozen AF29 source checkpoint")
    out=Path(out_dir)
    if out.exists():
        raise FileExistsError("per-seed calibration output must be fresh")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest=gym.manifest()
    model,provenance=original.load_frozen_original(original_zip_dir,seed)
    train_rows=generate_features(model,"train",include_successors=True)
    dev_rows=generate_features(model,"dev",include_successors=False)
    classifiers=calibration.make_models(train_rows,dev_rows)
    trainraw=("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                       for r in train_rows)).encode()
    devraw=("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n"
                       for r in dev_rows)).encode()
    (out/"train_features.jsonl").write_bytes(trainraw)
    (out/"dev_features.jsonl").write_bytes(devraw)
    data={
        "experiment":"AF-29","seed":seed,"trained_source_model":False,
        "new_original_model_updates":0,
        "source_hashes":source_hashes(),
        "source_protocol_sha256":sha((HERE/"PROTOCOL.json").read_bytes()),
        "source_manifest_sha256":manifest["manifest_sha256"],
        "original_af23_checkpoint":provenance,
        "train_feature_sha256":sha(trainraw),
        "dev_feature_sha256":sha(devraw),
        "train_feature_rows":len(train_rows),
        "dev_feature_rows":len(dev_rows),
        "classifiers":classifiers,
        "heldout_labels_seen":False,
        "heldout_source_features_seen":False,
        "status":"FITTED_USING_TRAIN_DEV_ONLY_AND_FROZEN_FOR_HELDOUT",
    }
    write_json(out/"calibrator.json",data)
    return data


def load_calibrator(path,seed,provenance):
    path=Path(path)
    data=json.loads((path/"calibrator.json").read_text())
    if (data["experiment"]!="AF-29" or data["seed"]!=seed or
        data["source_hashes"]!=source_hashes() or
        data["source_protocol_sha256"]!=sha((HERE/"PROTOCOL.json").read_bytes()) or
        data["source_manifest_sha256"]!=gym.manifest()["manifest_sha256"] or
        data["original_af23_checkpoint"]!=provenance or
        data["train_feature_sha256"]!=sha((path/"train_features.jsonl").read_bytes()) or
        data["dev_feature_sha256"]!=sha((path/"dev_features.jsonl").read_bytes()) or
        data["dev_feature_rows"]!=192 or
        data["new_original_model_updates"]!=0 or
        data["heldout_labels_seen"] or
        data["heldout_source_features_seen"]):
        raise ValueError("AF29 calibration source, checkpoint or development leak")
    # Independently regenerate the entire training loss and dev threshold
    # from the frozen, source-pinned feature rows and the pure logistic fitter.
    train=[json.loads(line) for line in
           (path/"train_features.jsonl").read_text().splitlines()]
    dev=[json.loads(line) for line in
         (path/"dev_features.jsonl").read_text().splitlines()]
    exact=calibration.make_models(train,dev)
    for arm in ("CAL","THR"):
        recorded=data["classifiers"][arm]
        if recorded["feature_indices"]!=exact[arm]["feature_indices"] or (
            abs(recorded["dev"]["chosen_threshold"]-
                exact[arm]["dev"]["chosen_threshold"])>1e-12):
            raise ValueError("AF29 calibration feature/threshold receipt mismatch")
        import numpy as np
        if not np.allclose(recorded["weights_bias_first"],
                           exact[arm]["weights_bias_first"],atol=1e-9,rtol=0):
            raise ValueError("AF29 logistic fit coefficients unreproducible")
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--original-zip-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=calibrate(a.seed,a.original_zip_dir,a.out)
    print(json.dumps({
        "seed":a.seed,"training_examples":result["train_feature_rows"],
        "calibrated_thresholds":{
            arm:result["classifiers"][arm]["dev"]["chosen_threshold"]
            for arm in ("CAL","THR")
        },"heldout_used":False
    }))


if __name__=="__main__":
    main()
