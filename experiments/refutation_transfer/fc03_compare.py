"""FC-03: preregistered tied-versus-untied, cell-compute-matched LR-01.

Public FC-03 preregistration was committed at 650bd2cc5f0fd628784f672248c3a3e362e9dcdf
BEFORE any FC-03 training or score. The CI workflow is split into an isolated
training job (FC-02 train+dev repository only) and a subsequent evaluation
job (FC-03 public test repository plus immutable weight artifacts). The model
is a finite Horn-closure recognizer, not a natural-language proof analyzer.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time

import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import formal_student as fs

FC02_SHA="7aca2bc3fc5586c5b7997fba868bd179db761121"
FC02_DATA_SHA="8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f"
FC03_SOURCE_SHA="c859f04cda1d142d4eb560148414b82d53f80705"
FC03_DATA_SHA="ddc04ec8553a24269b16b53e27d28b521bc2f928978bfd5ee8a9e5fc36463fdb"
PREREG_SHA="650bd2cc5f0fd628784f672248c3a3e362e9dcdf"

STEPS=350
BATCH=48
SEEDS=(17,29)
TRAIN_KINDS=("tied4_width24","untied4_width24","untied4_width12")


class UntiedReach(nn.Module):
    """Same LR-01 graph cell operations, separate weights at each pass.

    At equal width and pass count, exactly the same computational operators
    and graph inputs as tied LR-01. Parameter count grows fourfold in cell.
    """
    def __init__(self,loops:int,width:int):
        super().__init__()
        self.loops=loops
        self.atom_in=nn.Linear(fs.ATOM_F,width)
        self.rule_in=nn.Linear(fs.RULE_F,width)
        self.blocks=nn.ModuleList([fs.TypedLoop(width) for _ in range(loops)])
        self.atom_out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))

    def forward(self,b):
        a=self.atom_in(b["a"])
        r=self.rule_in(b["r"])
        results=[]
        for block in self.blocks:
            a,r=block(a,r,b["prem"],b["head"])
            results.append(self.atom_out(a).squeeze(-1))
        return results


def make_model(kind:str):
    if kind=="tied4_width24":
        model=fs.LoopReach(4,width=24)
        expected=11521
    elif kind=="untied4_width24":
        model=UntiedReach(4,width=24)
        expected=45073
    elif kind=="untied4_width12":
        model=UntiedReach(4,width=12)
        expected=11593
    else:
        raise ValueError("unregistered model kind")
    actual=sum(p.numel() for p in model.parameters())
    if actual!=expected:
        raise AssertionError(f"parameter formula mismatch {kind}: {actual} != {expected}")
    return model


def digest_tasks(fc02,rows):
    return hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in rows)).hexdigest()


def load_source_training(root:Path):
    """This path must be the older FC-02 SHA; no FC-03 tests are accessible."""
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.formal_curriculum import generator as fc02
    rows=fc02.generate()
    fc02.audit(rows)
    h=digest_tasks(fc02,rows)
    if h!=FC02_DATA_SHA:
        raise AssertionError("source training data hash differs from registration")
    train=[t for t in rows if t["split"]=="train"]
    dev=[t for t in rows if t["split"]=="dev"]
    if (len(train),len(dev))!=(1152,192):
        raise AssertionError("FC-02 split altered")
    return train,dev,h


def _train_one(train:list[dict],kind:str,seed:int):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    net=make_model(kind)
    full=fs.encode(train)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    g=torch.Generator().manual_seed(seed+6154)  # identical minibatches per arm
    start=time.perf_counter()
    losses=[]
    for step in range(STEPS):
        idx=torch.randint(len(train),(BATCH,),generator=g)
        batch=fs.batch_select(full,idx)
        opt.zero_grad(set_to_none=True)
        net.train()
        logits=net(batch)
        loss=fs.objective(logits,batch)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step()
        if step in (0,STEPS//4,STEPS//2,STEPS-1):
            losses.append([step+1,round(float(loss.detach()),6)])
    net.eval()
    return net,{"kind":kind,"seed":seed,"steps":STEPS,"batch_size":BATCH,
                "learning_rate":.002,"weight_decay":.01,
                "param_count":sum(p.numel() for p in net.parameters()),
                "cell_invocations_train":4,
                "train_seconds":round(time.perf_counter()-start,3),
                "losses":losses}


@torch.no_grad()
def predict_blind(model,tasks:list[dict],passes:int=4,batch_size:int=48):
    """Model input is strictly a whitelisted graph, with dummy gold targets.

    We time only network forwards, excluding preprocessing and verifier calls.
    Gold labels / known closures are read later by the evaluation scorer.
    """
    model.eval()
    if passes!=4 and not isinstance(model,fs.LoopReach):
        raise ValueError("inference extra passes belong to tied model only")
    old_loops=model.loops
    model.loops=passes
    output=[]
    forward_seconds=0.0
    try:
        for off in range(0,len(tasks),batch_size):
            batch=tasks[off:off+batch_size]
            blind=[]
            for t in batch:
                blind.append({key:t[key] for key in ("case_id","atoms","facts","rules","goal")}|{
                    "label":"refuted","reachable":[]})
            features=fs.encode(blind)
            start=time.perf_counter()
            logits=model(features)[-1]
            forward_seconds+=time.perf_counter()-start
            probs=logits.sigmoid()
            for t,p in zip(batch,probs):
                n=len(t["atoms"])
                assignment={atom:bool(float(p[j])>=.5)
                            for j,atom in enumerate(t["atoms"])}
                g=t["goal"]
                verdict="established" if assignment[g] else "refuted"
                output.append({
                    "case_id":t["case_id"],
                    "predicted":verdict,
                    "goal_probability":round(float(p[t["atoms"].index(g)]),8),
                    "assignment":assignment,
                })
    finally:
        model.loops=old_loops
    return output,round(1000*forward_seconds/len(tasks),6)


def train(root:Path,out:Path):
    torch.set_num_threads(2)
    train,dev,data_hash=load_source_training(root)
    out.mkdir(parents=True,exist_ok=True)
    manifest={
        "experiment":"FC03",
        "phase":"train_only_before_test_checkout",
        "protocol_sha":PREREG_SHA,
        "fc02_source_sha":FC02_SHA,
        "fc02_data_sha256":data_hash,
        "train_examples":len(train),
        "development_examples":len(dev),
        "fc03_test_data_present":False,
        "fixed_training_hyperparameters":{"steps":STEPS,"batch":BATCH,"seed":list(SEEDS),
                                          "lr":.002,"weight_decay":.01},
        "models":[],
    }
    for kind in TRAIN_KINDS:
        for seed in SEEDS:
            model,meta=_train_one(train,kind,seed)
            tag=f"{kind}_s{seed}"
            f=out/(tag+".pt")
            torch.save(model.state_dict(),f)
            meta["tag"]=tag
            meta["checkpoint_sha256"]=hashlib.sha256(f.read_bytes()).hexdigest()
            # Allowed development metrics are diagnostic only, no model selection.
            dev_pred,dev_ms=predict_blind(model,dev)
            # This reads dev gold only after the checkpoint has been written.
            meta["development_accuracy"]=round(
                sum(p["predicted"]==t["label"] for p,t in zip(dev_pred,dev))/len(dev),6)
            meta["development_forward_ms_per_case"]=dev_ms
            manifest["models"].append(meta)
            print("TRAINED",json.dumps({k:meta[k] for k in (
                "tag","param_count","train_seconds","development_accuracy")}),flush=True)
    f=out/"train_manifest.json"
    f.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("TRAIN_RECEIPT",f)


def load_frozen_test(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc03 import generator as fc03
    from research.refutation_bench.formal_curriculum import generator as fc02
    from research.refutation_bench.certificates import verify_countermodel
    rows=fc03.generate_tests()
    audit=fc03.audit_tests(rows)
    h=digest_tasks(fc02,rows)
    if h!=FC03_DATA_SHA:
        raise AssertionError(f"heldout corpus digest changed: {h}")
    if audit["splits"]!={"test_same":192,"test_deep":192}:
        raise AssertionError("test split count shifted")
    return rows,verify_countermodel,h


def _certify_predictions(tasks:list[dict],preds:list[dict],verify):
    """Separate post-inference pass: reads test formulas and verifies proposals."""
    result=[]
    for t,p in zip(tasks,preds):
        if t["case_id"]!=p["case_id"]:
            raise AssertionError("prediction order / ID mismatch")
        report=dict(p)
        success=False
        if p["predicted"]=="refuted":
            prob=t["formal"]
            success=verify(prob["premises"],prob["conclusion"],p["assignment"])["verified"]
        report["countermodel_verified"]=success
        result.append(report)
    return result


def scores(tasks,predictions):
    a=Counter()
    cert=0
    cert_pred=0
    bytes_total=0
    for task,p in zip(tasks,predictions):
        a[(task["label"],p["predicted"])]+=1
        cert+=bool(p["countermodel_verified"] and task["label"]=="refuted")
        cert_pred+=bool(p["countermodel_verified"])
        bytes_total+=len(json.dumps(p,sort_keys=True).encode())
    good=len([t for t in tasks if t["label"]=="established"])
    bad=len(tasks)-good
    tp=a["refuted","refuted"]
    fp=a["established","refuted"]
    fn=a["refuted","established"]
    tn=a["established","established"]
    f1_refuted=(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    f1_valid=(2*tn)/(2*tn+fp+fn) if 2*tn+fp+fn else 0
    return {
        "total":len(tasks),"correct":tp+tn,
        "accuracy":round((tp+tn)/len(tasks),6),
        "macro_f1":round((f1_refuted+f1_valid)/2,6),
        "false_refutations":fp,
        "false_refutation_rate":round(fp/good,6),
        "certified_invalid":cert,
        "certified_recall_invalid":round(cert/bad,6),
        "verified_negative_proposals":cert_pred,
        "negative_proposals":tp+fp,
        "verified_proposal_precision":round(cert_pred/(tp+fp),6) if tp+fp else None,
        "mean_response_bytes":round(bytes_total/len(tasks),3),
        "confusion":{f"{a0}->{b0}":a[a0,b0]
                     for a0 in ("established","refuted")
                     for b0 in ("established","refuted")},
    }


def paired(tasks,pred_tied,pred_untied):
    b=c=0
    for task,p,q in zip(tasks,pred_tied,pred_untied):
        if not (task["case_id"]==p["case_id"]==q["case_id"]):
            raise AssertionError("mismatched paired observations")
        tp=p["predicted"]==task["label"]
        up=q["predicted"]==task["label"]
        b+=tp and not up
        c+=up and not tp
    return {"tied_right_untied_wrong":b,"untied_right_tied_wrong":c,
            "net_tied_advantage":b-c}


def evaluate(root:Path,train_dir:Path,out:Path):
    torch.set_num_threads(2)
    manifest=json.loads((train_dir/"train_manifest.json").read_text())
    if manifest["fc02_data_sha256"]!=FC02_DATA_SHA:
        raise AssertionError("training source hash mismatch")
    if manifest.get("fc03_test_data_present") is not False:
        raise AssertionError("training had test source material")
    if {m["tag"] for m in manifest["models"]}!={f"{kind}_s{seed}"
            for kind in TRAIN_KINDS for seed in SEEDS}:
        raise AssertionError("model recipe count differs")
    rows,verifier,data_hash=load_frozen_test(root)
    out.mkdir(parents=True,exist_ok=True)
    model_scores=[]
    prediction_map={}
    meta_by_tag={m["tag"]:m for m in manifest["models"]}
    for kind in TRAIN_KINDS:
        for seed in SEEDS:
            tag=f"{kind}_s{seed}"
            meta=meta_by_tag[tag]
            weight_path=train_dir/(tag+".pt")
            if hashlib.sha256(weight_path.read_bytes()).hexdigest()!=meta["checkpoint_sha256"]:
                raise AssertionError("trained checkpoint changed after its training receipt")
            net=make_model(kind)
            net.load_state_dict(torch.load(weight_path,map_location="cpu",weights_only=True))
            passes_to_check=(4,8,12) if kind=="tied4_width24" else (4,)
            for passes in passes_to_check:
                for split in ("test_same","test_deep"):
                    tasks=[t for t in rows if t["split"]==split]
                    # Important: neural inference happens first, without gold labels.
                    raw,forward_ms=predict_blind(net,tasks,passes)
                    # Gold labels/logic checker are consulted only afterwards.
                    verified=_certify_predictions(tasks,raw,verifier)
                    score=scores(tasks,verified)
                    label=f"{tag}_p{passes}_{split}"
                    f=out/(label+".jsonl")
                    f.write_text("".join(json.dumps(p,sort_keys=True)+"\n" for p in verified))
                    score.update(tag=tag,kind=kind,seed=seed,passes=passes,split=split,
                                 param_count=meta["param_count"],cell_invocations=passes,
                                 inference_model_ms_per_case=forward_ms,
                                 prediction_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                    model_scores.append(score)
                    prediction_map[(kind,seed,passes,split)]=verified
                    print("TEST",json.dumps({key:score[key] for key in (
                        "tag","passes","split","correct","total","accuracy",
                        "false_refutations","certified_invalid",
                        "inference_model_ms_per_case")}),flush=True)
    contrasts={}
    for seed in SEEDS:
        for split in ("test_same","test_deep"):
            tasks=[t for t in rows if t["split"]==split]
            a=prediction_map["tied4_width24",seed,4,split]
            for kind in ("untied4_width24","untied4_width12"):
                b=prediction_map[kind,seed,4,split]
                contrasts[f"{seed}_{split}_tied_vs_{kind}"]=paired(tasks,a,b)
    result={
        "experiment":"FC03_pre_registered",
        "prereg_sha":PREREG_SHA,
        "fc02_train_data_sha256":FC02_DATA_SHA,
        "fc03_test_data_sha256":data_hash,
        "test_generator_commit":FC03_SOURCE_SHA,
        "training_manifest_sha256":hashlib.sha256((train_dir/"train_manifest.json").read_bytes()).hexdigest(),
        "test_collections":{"test_same":192,"test_deep":192},
        "n_test_families":4,
        "seed_list":list(SEEDS),
        "primary_contrast":"tied4_width24 versus untied4_width24 at equal four cell invocations and equal width",
        "parameter_control":"tied4_width24 versus untied4_width12 at approximate parameter parity",
        "compute_notes":"Same width and cell count give same operator arithmetic but not identical memory/runtime; measured inference per-case times provided.",
        "test_dataset_public":True,
        "test_labels_accessed_only_after_model_inference":True,
        "symbolic_oracle_accuracy":1.0,
        "constant_verdict_accuracy":.5,
        "scores":model_scores,
        "paired_primary_and_sensitivity":contrasts,
        "process_peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3),
    }
    f=out/"fc03_test_report.json"
    f.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("FC03_REPORT",f)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--fc02-root",type=Path)
    p.add_argument("--fc03-root",type=Path)
    p.add_argument("--train-dir",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="train":
        if args.fc02_root is None or args.fc03_root is not None:
            raise ValueError("training job must only receive frozen FC02 path")
        train(args.fc02_root,args.out)
    else:
        if args.fc03_root is None or args.train_dir is None:
            raise ValueError("evaluation needs FC03 source and frozen checkpoints")
        evaluate(args.fc03_root,args.train_dir,args.out)


if __name__=="__main__":
    main()
