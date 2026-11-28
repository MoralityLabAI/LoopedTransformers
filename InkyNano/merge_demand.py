"""Join independently verified commodity shards for a single arm and horizon."""
import argparse
import json
from pathlib import Path
import numpy as np
from gym_manifest import entries
from inventory import REV


def merge(folders,arm,horizon):
    parts=[]
    roots=set()
    families=set()
    for root in folders:
        manifest=json.loads((root/"manifest.json").read_text())
        if manifest["revision"]!=REV or manifest["files"]!=entries(root):
            raise ValueError("invalid manifest: "+str(root))
        receipt=json.loads((root/"online_receipt.json").read_text())
        if receipt["status"]!="complete" or not receipt["arms"][arm]["complete"]:
            raise ValueError("partial shard")
        with np.load(root/arm/f"demand_H{horizon}.npz",allow_pickle=False) as z:
            data=dict(z)
        if int(data["horizon"])!=horizon:
            raise ValueError("wrong horizon")
        r=set(data["root"]); f=set(data["family"])
        if roots&r or families&f:
            raise ValueError("root/family duplicated across shards")
        if set(data["split"])-{"train","dev"}:
            raise ValueError("invalid split")
        for key in ("root","family"):
            if set(data[key][data["split"]=="train"])&set(data[key][data["split"]=="dev"]):
                raise ValueError("split leakage")
        roots|=r;families|=f;parts.append(data)
    if not parts:
        raise ValueError("no shards")
    return {k:(parts[0][k] if k=="horizon" else np.concatenate([p[k] for p in parts],axis=0)) for k in parts[0]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=["generated","reference_forced"],required=True)
    p.add_argument("--horizon",type=int,choices=[0,32,64,128],required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("folders",type=Path,nargs="+")
    a=p.parse_args()
    data=merge(a.folders,a.arm,a.horizon)
    np.savez_compressed(a.out,**data)
    print(json.dumps({"roots":len(set(data["root"])),"samples":len(data["root"]),"horizon":a.horizon}))


if __name__=="__main__":
    main()
