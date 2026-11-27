"""Run a small independent cache roundtrip test on the pinned external wrapper."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from .core import UPSTREAM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream",type=Path,required=True)
    a=p.parse_args()
    if subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM:
        raise RuntimeError("unexpected upstream revision")
    if subprocess.check_output(["git","-C",str(a.upstream),"status","--porcelain"],text=True).strip():
        raise RuntimeError("upstream checkout must be clean")
    sys.path.insert(0,str(a.upstream.resolve()))
    import torch
    from rse_looper.looping import snapshot_cache_layer, restore_cache_layer

    results=[]
    for initial in (False,True):
        keys=torch.tensor([1.]) if initial else None
        values=torch.tensor([2.]) if initial else None
        conv={0:torch.tensor([3.])} if initial else {}
        cache=SimpleNamespace(layers=[SimpleNamespace(keys=keys,values=values)],conv_states=conv)
        snap=snapshot_cache_layer(cache,0)
        cache.layers[0].keys=torch.tensor([7.])
        cache.layers[0].values=torch.tensor([8.])
        cache.conv_states[0]=torch.tensor([9.])
        restore_cache_layer(cache,0,snap)
        ok=(torch.equal(cache.layers[0].keys,keys) and torch.equal(cache.layers[0].values,values)
            and torch.equal(cache.conv_states[0],torch.tensor([3.]))) if initial else (
                cache.layers[0].keys is None and cache.layers[0].values is None and 0 not in cache.conv_states)
        results.append({"initially_populated":initial,"roundtrip_exact":bool(ok)})
    print(json.dumps({"evidence_class":"synthetic_cache_contract","upstream":UPSTREAM,"cases":results},indent=2))


if __name__=="__main__":
    main()
