"""NEC-03N-DIAG: isolated ancestry mismatch diagnostic, not a training result.

Recreate the *original* source-only 900-step training on only the two
failed, previously opened five-input worlds. Compare frozen old checkpoint
tensors, original/replay training losses and source scores WITHOUT training
a new 1800 checkpoint or changing the preregistered NEC-03N gate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,context_catalog,labels,
)
from research.pef07.pef07 import Looper,as_tensor,model_digest


def file_sha(path):
    h=sha256()
    with path.open("rb") as src:
        for item in iter(lambda: src.read(1<<20),b""):
            h.update(item)
    return h.hexdigest()


def diagnose(index,original,out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if index not in (2,3):
        raise PermissionError("diagnose only the two already-opened failed NEC03N roots")
    w=worlds()[index]
    source,source_manifest=context_catalog(w)
    old=json.loads((original/"source_pilot_receipt.json").read_text())
    if (old["world_sha256"]!=w["world_sha256"]
        or old["source_manifest_sha256"]!=source_manifest["source_sha256"]
        or old["development_evaluation_context_groups_scored"]!=0
        or old["reserved_context_groups_scored"]!=0
        or old["original_confirmation_worlds_scored"]!=0):
        raise AssertionError("frozen old world context membership changed")
    diagnostics={}
    for arm in ("composed","additive"):
        receipt=old["arms"][arm]["native"]
        checkpoint=original/receipt["checkpoint_file"]
        if file_sha(checkpoint)!=receipt["checkpoint_sha256"]:
            raise AssertionError("original 900-step checkpoint bytes altered")
        loaded=torch.load(checkpoint,map_location="cpu",weights_only=True)
        seed=w["native_model_seed"]+(100000 if arm=="additive" else 0)
        torch.manual_seed(seed)
        model=Looper().train()
        initial=model_digest(model)
        x=as_tensor(source["train"])
        y=labels(x,w,control=arm=="additive")
        opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0.)
        rng=torch.Generator().manual_seed(seed*7919+59)
        first=last=None
        for step in range(1,901):
            ids=torch.randint(len(x),(128,),generator=rng)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model.logits(x[ids]),y[ids])
            if not torch.isfinite(loss):
                raise AssertionError("non-finite 900-step replay diagnostic")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if first is None: first=float(loss.detach())
            last=float(loss.detach())
        digest=model_digest(model)
        differences={}
        exact=True
        for name,tensor in model.state_dict().items():
            old_tensor=loaded["model"][name]
            if not torch.equal(tensor.detach(),old_tensor):
                exact=False
            diff=(tensor.detach()-old_tensor).abs()
            differences[name]={
                "max_absolute":float(diff.max()),
                "mean_absolute":float(diff.mean()),
                "same_bytes":bool(torch.equal(tensor.detach(),old_tensor)),
            }
        model.eval()
        score=0
        with torch.no_grad():
            for start in range(0,len(x),128):
                score+=int((model.logits(x[start:start+128]).argmax(1)
                            ==y[start:start+128]).sum())
        diagnostics[arm]={
            "original_seed":seed,
            "original_start_digest":receipt["initial_model_weights_sha256"],
            "replayed_start_digest":initial,
            "original_900_digest":receipt["final_model_weights_sha256"],
            "replayed_900_digest":digest,
            "original_first_batch_loss":receipt["initial_train_loss"],
            "replayed_first_batch_loss":first,
            "original_900_last_batch_loss":receipt["final_train_loss"],
            "replayed_900_last_batch_loss":last,
            "difference_step900_last_batch_loss":last-receipt["final_train_loss"],
            "original_source_correct":receipt["source_correct"],
            "replayed_source_correct":score,
            "exact_parameter_equality":exact,
            "max_tensor_absolute_error":max(k["max_absolute"] for k in differences.values()),
            "mismatched_parameter_names":[k for k,v in differences.items() if not v["same_bytes"]],
            "per_parameter_difference":differences,
            "root_original_checkpoint_sha256":receipt["checkpoint_sha256"],
        }
    report={
        "schema":"NEC-03N-DIAG/posthoc-step900-reproduction-receipt/v1",
        "world_id":w["world_id"],
        "world_index":index,
        "original_world_spec_sha256":w["world_sha256"],
        "original_source_manifest_sha256":source_manifest["source_sha256"],
        "old_run":44894980345,
        "failed_registered_1800_run":44895917012,
        "source_only":True,
        "new_1800_checkpoint_created":False,
        "native_model_source_steps":900,
        "source_histories_used":2560,
        "development_evaluation_contexts_scored":0,
        "reserved_contexts_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "four_input_confirmation_worlds_scored":0,
        "arms":diagnostics,
        "scientific_status":"diagnostic only, registered NEC03N failure preserved",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"milestone_diagnostic.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"world":w["world_id"],
                      "composed_loss_difference":diagnostics["composed"]["difference_step900_last_batch_loss"],
                      "additive_loss_difference":diagnostics["additive"]["difference_step900_last_batch_loss"],
                      "composed_exact":diagnostics["composed"]["exact_parameter_equality"],
                      "additive_exact":diagnostics["additive"]["exact_parameter_equality"],
                      "source_only":True},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--old-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    diagnose(args.index,args.old_root,args.out)
