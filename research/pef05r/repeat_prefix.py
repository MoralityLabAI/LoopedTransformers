"""PEF-05R same-runner source-only prefix reproducibility smoke.

Repeat unmodified first 125 PEF-05 optimizer updates twice on the
same GitHub runner and test *bitwise* equality of parameter/optimizer
hashes at frozen steps. No PEF-05 downstream probe or heldout scoring.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import torch
from research.pef03.pef03 import BitMemoryLooper, hash_weights, tensor
from research.pef05.pef05 import BATCH, _hash_optimizer, pair_split, train_update


def repeated_prefix(seed:int, steps:int=125):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    catalog,_=pair_split()
    train=tensor(catalog["train"])
    checkpoints=(0,1,10,50,steps)
    out=[]
    for attempt in range(2):
        torch.manual_seed(seed)
        model=BitMemoryLooper().train()
        opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0)
        gen=torch.Generator(device="cpu").manual_seed(seed*7919+59)
        sampled=torch.randint(len(train),(steps,BATCH),generator=gen)
        checkpoints_here=[]
        checkpoints_here.append({"step":0,"weights_sha256":hash_weights(model),
                                "optimizer_sha256":_hash_optimizer(opt)})
        for update in range(1,steps+1):
            train_update(model,opt,train[sampled[update-1]],
                         arm="baseline",seed=seed,step=update)
            if update in checkpoints:
                checkpoints_here.append({
                    "step":update,
                    "weights_sha256":hash_weights(model),
                    "optimizer_sha256":_hash_optimizer(opt),
                })
        out.append(checkpoints_here)
    return {
        "seed":seed,
        "max_optimizer_steps":steps,
        "same_runner_two_repeats_bitwise_equal":out[0]==out[1],
        "paired_run_hashes":out,
        "torch_version":str(torch.__version__),
        "cpu_capability":torch.backends.cpu.get_cpu_capability(),
        "mkldnn_enabled":torch.backends.mkldnn.enabled,
        "torch_num_threads":torch.get_num_threads(),
        "host_platform":platform.platform(),
        "same_runner_result_not_a_new_preregistered_pef05_outcome":True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result={"campaign":"PEF-05R","repeated_prefix_smoke":[
        repeated_prefix(seed) for seed in (101,127)
    ]}
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
       "same_runner_seed_replay":[
         {"seed":r["seed"],
          "bitwise_repeatability":r["same_runner_two_repeats_bitwise_equal"],
          "cpu_capability":r["cpu_capability"]}
         for r in result["repeated_prefix_smoke"]
       ]
    },indent=2))


if __name__=="__main__":
    main()
